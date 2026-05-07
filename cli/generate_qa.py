import os
import asyncio
import time
import random
import uuid
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

import click
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph
from modules.graph.reverse_search import run_reverse_search, evolve_graph
from modules.graph.database import DatabaseManager
from modules.logger import TerminalLogger
from modules.agent.graph_agent.knowledge_search import KnowledgeSearchAgent
from modules.agent.graph_agent.predicate_sampler import PredicateSamplerAgent
from modules.agent.graph_agent.seedgen import SeedGenAgent
from modules.agent.graph_agent.fuzzifier import Fuzzifier
from modules.agent.graph_agent.graph_evolver import GraphEvolver
from modules.agent.graph_agent.graph2query import Graph2QueryAgent
from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig

@dataclass
class GenerateQaCliConfig:
    num: int
    run_id: Optional[str]
    max_complexity: int
    max_search_steps: int
    prune_factor: float
    fuzz_constant_prob: float
    fuzz_use_category_prob: float
    topic: Optional[str]
    category: Optional[str]
    seed_max_steps: int
    seed_gen_thinking: bool
    sampler_type: str
    model: str
    base_url: str
    browser_url: str
    think: bool

class GenerateQaGenerator:
    def __init__(self, args: GenerateQaCliConfig, logger: TerminalLogger):
        self.args = args
        self.logger = logger
        self.db_manager = DatabaseManager()
        self.run_id = args.run_id
        self.collected_count = 0
        
        # Initialize agents
        self._init_agents()

    def _init_agents(self):
        """Initialize all agents required for the process."""
        sampler_cfg = AgentSamplerConfig(
            model=self.args.model,
            base_url=self.args.base_url,
            sampler_type=self.args.sampler_type,
            think=self.args.think,
            temperature=1.0,
            top_p=0.95,
        )

        seedgen_sampler_cfg = AgentSamplerConfig(
            model=self.args.model,
            base_url=self.args.base_url,
            sampler_type=self.args.sampler_type,
            think=self.args.seed_gen_thinking,
        )


        browser_cfg = BrowserConfig(
            base_url=self.args.browser_url,
            api_key=os.getenv("TOOLS_API_KEY"),
            truncate_context=True,
            max_context_length=10000,
        )

        self.search_agent = KnowledgeSearchAgent(config=sampler_cfg, browser_config=browser_cfg, logger=self.logger)
        self.sampler_agent = PredicateSamplerAgent(config=sampler_cfg, logger=self.logger)
        self.seed_gen_agent = SeedGenAgent(config=seedgen_sampler_cfg, browser_config=browser_cfg, logger=self.logger)
        self.fuzzifier_agent = Fuzzifier(config=sampler_cfg, browser_config=browser_cfg, logger=self.logger)
        self.evolver_agent = GraphEvolver(config=sampler_cfg, logger=self.logger)

        new_sampler_cfg = AgentSamplerConfig(
            model=self.args.model,
            base_url=self.args.base_url,
            sampler_type=self.args.sampler_type,
            think=True
        )

        self.graph2query_agent = Graph2QueryAgent(config=new_sampler_cfg)

    def _random_complexity_budget(self) -> int:
        """Sample a bounded complexity budget for a single graph."""
        mean = (self.args.max_complexity + 8) / 2
        stddev = (self.args.max_complexity - 8) / 6 if self.args.max_complexity > 8 else 1
        value = int(random.gauss(mean, stddev))
        return max(8, min(self.args.max_complexity, value))

    @staticmethod
    def _calculate_total_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate token and timing counters across generation phases."""
        total = {"time": 0, "input_tokens": 0, "output_tokens": 0}

        def add_stat(item: Dict[str, Any]) -> None:
            total["time"] += item.get("time", 0)
            total["input_tokens"] += item.get("input_tokens", 0)
            total["output_tokens"] += item.get("output_tokens", 0)

        if "seed_gen" in stats:
            add_stat(stats["seed_gen"])

        for expansion in stats.get("graph_expansion", []):
            for phase in ("knowledge_search", "predicate_sampling"):
                add_stat(expansion.get(phase, {}))

        for phase_name in ("graph_evolution", "fuzzification"):
            for phase_stats in stats.get(phase_name, []):
                add_stat(phase_stats)

        if "graph2query" in stats:
            add_stat(stats["graph2query"])

        return total

    def _new_graph_id(self) -> str:
        return f"graph_{int(time.time())}_{uuid.uuid4().hex[:4]}"

    async def _expand_single_graph(
        self,
        graph_id: str,
        seed_object_id: Optional[str] = None,
        is_new_seed: bool = False,
        stats: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Handle a single graph expansion: either starting from a seed or continuing from a checkpoint."""
        try:
            # 1. Load or Initialize Graph
            try:
                # Try to load checkpoint
                graph, queue, processed_entities, entity_parent_context, loaded_seed_id, loaded_stats = \
                    await self.db_manager.load_checkpoint_graph(self.run_id, graph_id, graph_type="original")
                seed_object_id = loaded_seed_id or seed_object_id
                seed_entity_name = graph.entities["E0"].name if graph.entities and "E0" in graph.entities else "Unknown"
                expansion_type = "checkpoint"
                if stats is None:
                    stats = loaded_stats
                
                # Ensure seed_gen stats are included if missing but seed_object_id is available
                if seed_object_id and (not stats or "seed_gen" not in stats):
                    try:
                        seed_entity = await self.db_manager.load_seed(seed_object_id)
                        if "stats" in seed_entity:
                            if stats is None:
                                stats = {}
                            stats["seed_gen"] = seed_entity["stats"]
                    except Exception as e:
                        self.logger.warning(f"Could not load seed stats for Graph {graph_id}: {e}")
            except Exception:
                # Start new graph if no checkpoint
                if not seed_object_id:
                    self.logger.error(f"No checkpoint found and no seed provided for Graph {graph_id}")
                    return False

                seed_entity = await self.db_manager.load_seed(seed_object_id)
                seed_entity_name = seed_entity["name"]
                if "stats" in seed_entity and stats is not None and "seed_gen" not in stats:
                    stats["seed_gen"] = seed_entity["stats"]
                elif "stats" in seed_entity and stats is None:
                    stats = {"seed_gen": seed_entity["stats"]}
                
                graph = CompositionalKnowledgeGraph()
                graph.add_entity(
                    id="E0",
                    name=seed_entity["name"],
                    description=seed_entity["description"],
                    category=seed_entity.get("category", "Unknown"),
                    is_variable=True,
                    links=seed_entity.get("links", [])
                )
                queue = ["E0"]
                processed_entities = set()
                entity_parent_context = {}
                expansion_type = "new_seed" if is_new_seed else "existing_seed"

            self.logger.log_graph_expansion_start(graph_id, self.run_id, seed_entity_name, expansion_type)

            # Initialize stats if not provided
            if stats is None:
                stats = {}

            # 2. Run Reverse Search
            await run_reverse_search(
                cokg=graph,
                run_id=self.run_id,
                graph_id=graph_id,
                queue=queue,
                processed_entities=processed_entities,
                entity_parent_context=entity_parent_context,
                search_agent=self.search_agent,
                sampler_agent=self.sampler_agent,
                max_complexity_budget=self._random_complexity_budget(),
                max_search_steps=self.args.max_search_steps,
                prune_factor=self.args.prune_factor,
                db_manager=self.db_manager,
                logger=self.logger,
                seed_object_id=str(seed_object_id) if seed_object_id else None,
                stats=stats
            )

            # 4. Finalize and Save Original Graph
            graph.fix_entity_types()
            await self.db_manager.save_generated_graph(
                cokg=graph,
                run_id=self.run_id,
                graph_id=graph_id,
                seed_object_id=str(seed_object_id),
                new_entities=queue,
                graph_type="original",
                stats={"graph_expansion": stats.get("graph_expansion", [])}
            )
            self.logger.log_graph_saved(graph_id, "original", len(graph.entities), len(graph.predicates))

            # 5. Evolution and Fuzzification
            self.logger.info("Starting graph evolution and fuzzification...")
            await evolve_graph(
                graph, 
                self.evolver_agent,
                self.fuzzifier_agent, 
                fuzz_constant_prob=self.args.fuzz_constant_prob,
                fuzz_use_category_prob=self.args.fuzz_use_category_prob,
                run_id=self.run_id,
                graph_id=graph_id,
                db_manager=self.db_manager,
                stats=stats,
                logger=self.logger
            )

            # 6. Save Fuzzy Graph
            await self.db_manager.save_generated_graph(
                cokg=graph,
                run_id=self.run_id,
                graph_id=graph_id,
                seed_object_id=str(seed_object_id),
                new_entities=[],
                graph_type="fuzzy",
                stats={
                    "graph_evolution": stats.get("graph_evolution", []),
                    "fuzzification": stats.get("fuzzification", [])
                }
            )
            self.logger.log_graph_saved(graph_id, "fuzzy", len(graph.entities), len(graph.predicates))

            # 6.5 Generate Question
            self.logger.info(f"Generating question for Graph {graph_id}...")
            qa_result = await self.graph2query_agent.run(graph)
            
            if qa_result.status == "success":
                question = qa_result.answer
                answer = qa_result.metadata.get("exact_answer", graph.entities["E0"].name)
                
                await self.db_manager.save_qa_pair(
                    run_id=self.run_id,
                    graph_id=graph_id,
                    question=question,
                    answer=answer,
                    stats={
                        "time": qa_result.total_time,
                        "input_tokens": qa_result.input_tokens,
                        "output_tokens": qa_result.output_tokens
                    }
                )
                stats["graph2query"] = {
                    "time": qa_result.total_time,
                    "input_tokens": qa_result.input_tokens,
                    "output_tokens": qa_result.output_tokens
                }
                self.logger.success(f"Question generated for Graph {graph_id}")
            else:
                self.logger.error(f"Failed to generate question for Graph {graph_id}: {qa_result.status}")

            stats["total"] = self._calculate_total_stats(stats)

            # 8. Save Consolidated Stats
            await self.db_manager.save_consolidated_stats(
                run_id=self.run_id,
                graph_id=graph_id,
                seed_object_id=str(seed_object_id),
                stats=stats
            )
            return True

        except Exception as e:
            self.logger.log_graph_error(graph_id, str(e))
            return False

    async def _resume_unfinished_checkpoints(self):
        """Resume graphs that were started but not finished in this run."""
        self.logger.info(f"Checking for unfinished checkpoints in run {self.run_id}...")
        unfinished = await self.db_manager.get_unfinished_graphs_by_run(self.run_id)
        if not unfinished:
            return

        resume_start = time.time()
        count = 0
        skipped = 0
        
        for cp in unfinished:
            if self.collected_count >= self.args.num:
                break

            graph_id = cp.get("graph_id")
            seed_oid = cp.get("seed_object_id")

            if seed_oid and await self.db_manager.check_fuzzy_graph_exists(seed_oid, self.run_id):
                self.logger.log_checkpoint_skip(f"fuzzy graph already exists for seed {seed_oid}", 1)
                skipped += 1
                continue

            self.logger.info(f"Resuming unfinished Graph {graph_id}...")
            start_t = time.time()
            # Stats will be loaded from checkpoint inside _expand_single_graph
            if await self._expand_single_graph(graph_id, seed_object_id=seed_oid):
                self.collected_count += 1
                count += 1
                self.logger.success(f"Graph {graph_id} completed in {time.time() - start_t:.2f}s")

        if count > 0:
            self.logger.success(f"Resumed {count} checkpoints in {time.time() - resume_start:.2f}s")
        if skipped > 0:
            self.logger.info(f"Skipped {skipped} checkpoints (already finalized).")

    async def _process_unused_seeds(self):
        """Process seeds that were generated but never used for graph expansion."""
        if self.collected_count >= self.args.num:
            return

        unused_seeds = await self.db_manager.get_unused_seeds_by_run(self.run_id)
        if not unused_seeds:
            return

        self.logger.log_unused_seeds_processing(len(unused_seeds))
        unused_start = time.time()
        count = 0
        skipped = 0

        for seed_doc in unused_seeds:
            if self.collected_count >= self.args.num:
                break

            seed_oid = str(seed_doc["_id"])
            if await self.db_manager.check_fuzzy_graph_exists(seed_oid, self.run_id):
                skipped += 1
                continue

            graph_id = self._new_graph_id()
            if await self.db_manager.check_pre_fuzzy_graph_exists(seed_oid, self.run_id):
                self.logger.info(f"Continuing pre-fuzzy graph for seed {seed_oid} as Graph {graph_id}...")
            else:
                self.logger.info(f"Starting expansion for unused seed {seed_oid} as Graph {graph_id}...")

            start_t = time.time()
            # Initialize stats for unused seed expansion
            stats = {"seed_oid": seed_oid}
            if await self._expand_single_graph(graph_id, seed_object_id=seed_oid, stats=stats):
                self.collected_count += 1
                count += 1
                self.logger.success(f"Seed {seed_oid} expanded in {time.time() - start_t:.2f}s")

        if count > 0:
            self.logger.success(f"Processed {count} unused seeds in {time.time() - unused_start:.2f}s")
        if skipped > 0:
            self.logger.info(f"Skipped {skipped} unused seeds (already finalized).")

    async def _generate_new_graphs(self):
        """Main loop to generate new seeds and expand them into graphs."""
        while self.collected_count < self.args.num:
            self.logger.log_seed_generation_start(self.run_id, self.collected_count, self.args.num, topic=self.args.topic, category=self.args.category)
            
            try:
                # 0. Assign a new graph_id early for stats tracking
                graph_id = self._new_graph_id()
                stats = {}

                # 1. Generate Seed
                seed_start = time.time()
                seed_result = await self.seed_gen_agent.run(
                    topic=self.args.topic,
                    category=self.args.category,
                    max_steps=self.args.seed_max_steps
                )
                seed_time = time.time() - seed_start

                # Record seed generation statistics
                stats["seed_gen"] = {
                    "revisions": seed_result.metadata.get("revisions", 0),
                    "revision_details": seed_result.metadata.get("revision_details", []),
                    "time": seed_result.total_time,
                    "input_tokens": seed_result.input_tokens,
                    "output_tokens": seed_result.output_tokens,
                    "turns": seed_result.turns,
                    "status": seed_result.status
                }

                if seed_result.status != "success":
                    self.logger.log_seed_generation_failed(f"Status: {seed_result.status}", seed_time)
                    continue

                # 2. Save Seed
                seed_data = json.loads(seed_result.answer)
                seed_data["is_variable"] = True
                seed_oid = await self.db_manager.save_seed(
                    seed_data, 
                    run_id=self.run_id, 
                    stats=stats.get("seed_gen")
                )

                if await self.db_manager.check_fuzzy_graph_exists(str(seed_oid), self.run_id):
                    self.logger.warning(f"Seed {seed_oid} already has a fuzzy graph, skipping.")
                    continue

                # 3. Expand Graph
                self.logger.log_seed_generated(
                    seed_name=seed_data.get('name'),
                    category=seed_data.get('category', 'Unknown'),
                    generation_time=seed_time,
                    description=seed_data.get('description'),
                    mongo_id=str(seed_oid)
                )

                expansion_start = time.time()
                success = await self._expand_single_graph(
                    graph_id=graph_id,
                    seed_object_id=str(seed_oid),
                    is_new_seed=True,
                    stats=stats
                )

                if success:
                    self.collected_count += 1
                    self.logger.success(f"Graph {graph_id} completed in {time.time() - expansion_start:.2f}s")
                else:
                    self.logger.error(f"Graph {graph_id} expansion failed.")

            except Exception as e:
                self.logger.error(f"Error in generation loop: {e}")
                await asyncio.sleep(5)

    async def run(self):
        """Orchestrate the entire generation process."""
        start_time = time.time()
        
        # 1. Setup Run ID
        if not self.run_id:
            self.run_id = f"run_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            self.logger.log_run_start(self.run_id, self.args.num)
        else:
            self.logger.log_run_start(self.run_id, self.args.num)
            self.logger.info(f"Continuing from run: {self.run_id}")

        # 2. Resumption Phase
        if self.args.run_id: # Only try to resume if a run_id was provided
            await self._resume_unfinished_checkpoints()
            await self._process_unused_seeds()

        # 3. Generation Phase
        await self._generate_new_graphs()

        # 4. Cleanup
        total_time = time.time() - start_time
        self.logger.log_run_end(self.run_id, self.collected_count, total_time)
        self.db_manager.close()

async def run_generator(config: GenerateQaCliConfig):
    logger = TerminalLogger()
    generator = GenerateQaGenerator(config, logger)
    await generator.run()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--num", type=int, default=1, show_default=True, help="Number of QA pairs to generate.")
@click.option("--run-id", "--run_id", type=str, default=None, help="Run ID to continue from.")
@click.option("--max-complexity", "--max_complexity", type=int, default=20, show_default=True, help="Maximum number of entities per graph.")
@click.option("--max-search-steps", "--max_search_steps", type=int, default=2, show_default=True, help="Maximum search steps per entity.")
@click.option("--prune-factor", "--prune_factor", type=float, default=0.85, show_default=True, help="Geometric decay factor for expansion.")
@click.option("--fuzz-constant-prob", "--fuzz_constant_prob", type=float, default=1.0, show_default=True, help="Probability of fuzzing constant entities.")
@click.option("--fuzz-use-category-prob", "--fuzz_use_category_prob", type=float, default=0.05, show_default=True, help="Probability of using category as fuzzy name.")
@click.option("--topic", type=str, default=None, help="Topic for seed generation.")
@click.option("--category", type=str, default=None, help="Category for seed generation.")
@click.option("--seed-max-steps", "--seed_max_steps", type=int, default=10, show_default=True, help="Max steps for seed generation.")
@click.option("--seed-gen-thinking/--no-seed-gen-thinking", "--seed_gen_thinking/--no_seed_gen_thinking", default=True, show_default=True, help="Enable thinking mode for seed generation.")
@click.option("--sampler-type", "--sampler_type", type=str, default="glm", show_default=True, help="Sampler type for agents.")
@click.option("--model", type=str, default="GLM-4.7", show_default=True, help="Model name for agents.")
@click.option("--base-url", "--base_url", type=str, default="https://llmapi.paratera.com", show_default=True, help="LLM API base URL.")
@click.option("--browser-url", "--browser_url", type=str, default="http://localhost:8001/", show_default=True, help="Browser API base URL.")
@click.option("--think/--no-think", default=True, show_default=True, help="Enable think mode for compatible samplers.")
def main(**kwargs):
    """Generate QA datasets via reverse search."""
    asyncio.run(run_generator(GenerateQaCliConfig(**kwargs)))


if __name__ == "__main__":
    main()
