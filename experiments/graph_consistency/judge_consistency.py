"""
Phase 2 – Judge graph consistency of generated questions using a judge model.

Reads:  gc_study_data/{gen_model_name}_gc_study.json
        sampled_graphs.json (for original graph entities/predicates)

Writes: gc_study_results/{gen_model_name}_judgedby_{judge_model_name}.json
  Same format as _experiments/graph_consistency/results/*.json

Usage:
  python judge_consistency.py \
    --gen_model_name glm4 \
    --judge_model_name qwen3 \
    --judge_model Qwen/Qwen3-35B \
    --judge_base_url http://localhost:8001/v1 \
    --judge_sampler_type vllm \
    --concurrency 8
"""

import sys
import os
import json
import asyncio
import re
import argparse
from typing import Any, Dict, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(current_dir))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(workspace_root, ".env"))

from modules.agent.base_agent import BaseLLMAgent
from modules.agent.agent_types import AgentSamplerConfig, MessageList
from modules.agent.memory.base_memory import BaseMemory

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

GC_STUDY_DIR = os.path.join(current_dir, "gc_study_data")
SAMPLED_GRAPHS_PATH = os.path.join(current_dir, "sampled_graphs.json")
OUTPUT_DIR = os.path.join(current_dir, "gc_study_results")


class ConsistencyJudgeAgent(BaseLLMAgent):
    def __init__(self, config: Optional[AgentSamplerConfig] = None):
        system_prompt = (
            "You are an expert evaluator of knowledge graphs. Your task is to assess whether "
            "a Question contains the same information as a Knowledge Graph. "
            "To do this:\n"
            "1. Extract the list of key Entities from the Question.\n"
            "2. Extract the list of key Entities from the Graph.\n"
            "3. Extract the list of key Predicates (relationships) from the Question.\n"
            "4. Extract the list of key Predicates (relationships) from the Graph.\n"
            "5. Calculate the intersection and union of Entities, and output Entity IoU (Intersection over Union).\n"
            "6. Calculate the intersection and union of Predicates, and output Predicate IoU.\n"
            "\n"
            "You MUST format your output exactly as follows:\n"
            "<reasoning>your step-by-step reasoning</reasoning>\n"
            '<question_entities>["entity1", "entity2", ...]</question_entities>\n'
            '<graph_entities>["entity1", "entity2", ...]</graph_entities>\n'
            '<question_predicates>["predicate1", "predicate2", ...]</question_predicates>\n'
            '<graph_predicates>["predicate1", "predicate2", ...]</graph_predicates>\n'
            "<entity_iou>value</entity_iou>\n"
            "<predicate_iou>value</predicate_iou>\n"
            "where value is a float between 0.0 and 1.0, and the entities/predicates are formatted as JSON arrays of strings."
        )
        if config is None:
            config = AgentSamplerConfig()
        if "max_tokens" not in config.options:
            config.options["max_tokens"] = 32768
        super().__init__(config=config, system_prompt=system_prompt)

    def _format_graph(self, entities: List[Dict], predicates: List[Dict]) -> str:
        lines = ["[Entities]"]
        for ent in entities:
            desc = ent.get("description", "")
            lines.append(f"- [{ent.get('id')}] {ent.get('name')}" + (f": {desc}" if desc else ""))
        lines.append("\n[Predicates]")
        for pred in predicates:
            lines.append(f"- {pred.get('input')} -> {pred.get('content')} -> {pred.get('outputs')}")
        return "\n".join(lines)

    def _extract_tag_content(self, tag: str, text: str, default: str = "") -> str:
        match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
        return match.group(1).strip() if match else default

    def _extract_json_tag(self, tag: str, text: str) -> List[str]:
        content = self._extract_tag_content(tag, text)
        if not content:
            return []
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            cleaned = re.sub(r"```json\s*|\s*```", "", content).strip()
            try:
                return json.loads(cleaned)
            except Exception:
                return [content]

    async def evaluate(
        self,
        question: str,
        graph_entities: List[Dict],
        graph_predicates: List[Dict],
    ) -> Dict[str, Any]:
        graph_text = self._format_graph(graph_entities, graph_predicates)
        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Graph:\n{graph_text}\n\n"
            "Please analyze the entities and predicates, calculate their Intersection over Union (IoU) "
            "between the Question and the Graph, and provide the final results in the specified tags."
        )

        messages: MessageList = [{"role": "user", "content": user_prompt}]

        max_retries = 5
        entity_iou = 0.0
        predicate_iou = 0.0
        accumulated_content = ""
        accumulated_reasoning = ""

        for attempt in range(max_retries):
            formatted = BaseMemory.format_messages(
                messages, sampler_type=self.config.sampler_type, include_thinking=True
            )
            response = await asyncio.to_thread(self.generate_response, formatted)

            content = (response.content or "").strip()
            reasoning = (getattr(response, "reasoning_content", "") or "").strip()

            if reasoning:
                accumulated_reasoning += (f"\n\n[Attempt {attempt + 1}]\n" if accumulated_reasoning else "") + reasoning
            if content:
                accumulated_content += ("\n" if accumulated_content else "") + content

            ent_match = re.search(r"<entity_iou>(.*?)</entity_iou>", accumulated_content, re.DOTALL)
            pred_match = re.search(r"<predicate_iou>(.*?)</predicate_iou>", accumulated_content, re.DOTALL)

            if ent_match and pred_match:
                try:
                    entity_iou = float(ent_match.group(1).strip())
                    predicate_iou = float(pred_match.group(1).strip())
                except ValueError:
                    pass
                break

            messages.append({"role": "assistant", "content": content, "reasoning_content": reasoning})
            messages.append({"role": "user", "content": "Please continue to provide the final metrics."})

        return {
            "entity_iou": entity_iou,
            "predicate_iou": predicate_iou,
            "question_entities": self._extract_json_tag("question_entities", accumulated_content),
            "graph_entities": self._extract_json_tag("graph_entities", accumulated_content),
            "question_predicates": self._extract_json_tag("question_predicates", accumulated_content),
            "graph_predicates": self._extract_json_tag("graph_predicates", accumulated_content),
            "extracted_reasoning": self._extract_tag_content("reasoning", accumulated_content),
            "raw_output": accumulated_content,
            "reasoning": accumulated_reasoning,
        }


async def main():
    parser = argparse.ArgumentParser(description="Phase 2: Judge graph consistency of generated questions")
    parser.add_argument("--gen_model_name", type=str, required=True,
                        help="Name used for the generation model file (e.g. glm4)")
    parser.add_argument("--judge_model_name", type=str, required=True,
                        help="Short display name for the judge model (used in output filename)")
    parser.add_argument("--judge_model", type=str, required=True,
                        help="Judge model identifier (e.g. Qwen/Qwen3-35B)")
    parser.add_argument("--judge_base_url", type=str, default="http://localhost:8000/v1",
                        help="API base URL for the judge model server")
    parser.add_argument("--judge_sampler_type", type=str, default="vllm",
                        help="Sampler type for the judge (vllm, openai, anthropic, etc.)")
    parser.add_argument("--concurrency", type=int, default=8,
                        help="Number of concurrent workers")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gen_path = os.path.join(GC_STUDY_DIR, f"{args.gen_model_name}_gc_study.json")
    output_path = os.path.join(
        OUTPUT_DIR, f"{args.gen_model_name}_judgedby_{args.judge_model_name}.json"
    )

    with open(gen_path, "r", encoding="utf-8") as f:
        gen_data: List[Dict] = json.load(f)
    console.print(f"[cyan]Loaded {len(gen_data)} generated questions from {gen_path}[/cyan]")

    with open(SAMPLED_GRAPHS_PATH, "r", encoding="utf-8") as f:
        sampled_graphs: List[Dict] = json.load(f)
    graph_lookup = {g["graph_id"]: g for g in sampled_graphs}
    console.print(f"[cyan]Loaded {len(graph_lookup)} sampled graphs for entity/predicate lookup[/cyan]")

    results: List[Dict] = []
    processed_ids: set[str] = set()

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            processed_ids = {r["graph_id"] for r in results if "graph_id" in r}
            console.print(f"[green]Resuming: {len(results)} already evaluated[/green]")
        except Exception as e:
            console.print(f"[yellow]Could not load existing output: {e}[/yellow]")

    config = AgentSamplerConfig(
        model=args.judge_model,
        base_url=args.judge_base_url,
        sampler_type=args.judge_sampler_type,
    )
    judge = ConsistencyJudgeAgent(config=config)

    queue: asyncio.Queue = asyncio.Queue()
    for item in gen_data:
        if item["graph_id"] not in processed_ids:
            queue.put_nowait(item)

    total = queue.qsize()
    file_lock = asyncio.Lock()

    async def worker(wid: int, progress, task_id):
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            graph_id = item["graph_id"]
            question = item.get("question", "")
            exact_answer = item.get("exact_answer", "")

            progress.update(task_id, description=f"[cyan]W{wid} → {graph_id[:30]}...")

            graph_doc = graph_lookup.get(graph_id)
            if not graph_doc:
                console.print(f"[yellow][W{wid}] Graph {graph_id} not found in sampled_graphs. Skipping.[/yellow]")
                queue.task_done()
                progress.advance(task_id)
                continue

            try:
                eval_result = await judge.evaluate(
                    question=question,
                    graph_entities=graph_doc.get("entities", []),
                    graph_predicates=graph_doc.get("predicates", []),
                )

                entry = {
                    "graph_id": graph_id,
                    "question": question,
                    "exact_answer": exact_answer,
                    "entity_iou": eval_result["entity_iou"],
                    "predicate_iou": eval_result["predicate_iou"],
                    "question_entities": eval_result["question_entities"],
                    "graph_entities": eval_result["graph_entities"],
                    "question_predicates": eval_result["question_predicates"],
                    "graph_predicates": eval_result["graph_predicates"],
                    "extracted_reasoning": eval_result["extracted_reasoning"],
                    "raw_output": eval_result["raw_output"],
                    "reasoning": eval_result["reasoning"],
                }

                async with file_lock:
                    results.append(entry)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, indent=2, ensure_ascii=False)

                console.print(
                    f"[green][W{wid}] {graph_id}: "
                    f"entity_iou={eval_result['entity_iou']:.2f}, "
                    f"predicate_iou={eval_result['predicate_iou']:.2f}[/green]"
                )
            except Exception as e:
                console.print(f"[red][W{wid}] Error on {graph_id}: {e}[/red]")

            queue.task_done()
            progress.advance(task_id)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Judging consistency...", total=total)
        workers = [asyncio.create_task(worker(i, progress, task_id)) for i in range(args.concurrency)]
        await queue.join()
        for w in workers:
            w.cancel()

    console.print(f"\n[bold green]Done! {len(results)} entries saved to {output_path}[/bold green]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
