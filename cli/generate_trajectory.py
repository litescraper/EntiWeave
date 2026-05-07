import os
import sys
import asyncio
import json
import random
import re
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme
from rich.syntax import Syntax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

load_dotenv()

from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig, AgentRunResult
from modules.agent.search_agent.react import ReactSearchAgent
from modules.agent.search_agent.resumact import ResumActSearchAgent
from modules.agent.judge_agent.judge import JudgeAgent
from modules.graph.database import DatabaseManager
from modules.logger import BaseAgentLogger


@dataclass
class TrajectoryCliConfig:
    num_samples: int
    data_path: Optional[str]
    run_id: Optional[str]
    eval_run_id: Optional[str]
    output_path: str
    seed: int
    concurrency: int
    judge_concurrency: int
    agent_type: str
    agent_spec: Optional[str]
    search_model: str
    search_base_url: str
    search_type: str
    max_steps: int
    max_context_length: int
    max_search_results: int
    max_conversational_turns: Optional[int]
    temperature: float
    presence_penalty: float
    repetition_penalty: float
    top_p: float
    context_strategy: str
    tokenizer_path: Optional[str]
    max_tokens: int
    max_tool_context: Optional[int]
    reset_interval: Optional[int]
    judge_model: str
    judge_base_url: str
    judge_type: str
    judge_voting_n: int
    browser_base_url: str
    skip_judge: bool


def resolve_project_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def safe_trajectory_filename(sample_id: Any) -> str:
    """Keep IDs readable while preventing nested paths or invalid filenames."""
    filename = str(sample_id)
    filename = filename.replace(os.sep, "_")
    if os.altsep:
        filename = filename.replace(os.altsep, "_")
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return filename or "unknown"


AGENT_HYPERPARAM_KEYS = [
    "agent_type",
    "agent_spec",
    "search_model",
    "search_base_url",
    "search_type",
    "max_steps",
    "max_context_length",
    "max_search_results",
    "max_conversational_turns",
    "temperature",
    "presence_penalty",
    "repetition_penalty",
    "top_p",
    "context_strategy",
    "tokenizer_path",
    "max_tokens",
    "max_tool_context",
    "reset_interval",
]


def make_eval_run_id(data_path: Optional[str], run_id: Optional[str], output_base: Path) -> str:
    if data_path:
        source_name = Path(data_path).stem
    elif run_id:
        source_name = str(run_id)
    else:
        source_name = "mongodb"

    source_name = safe_trajectory_filename(source_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    eval_run_id = f"{source_name}_{timestamp}"
    candidate = output_base / eval_run_id

    if not candidate.exists():
        return eval_run_id

    suffix = datetime.now().strftime("%S")
    eval_run_id = f"{source_name}_{timestamp}_{suffix}"
    candidate = output_base / eval_run_id
    counter = 1
    while candidate.exists():
        eval_run_id = f"{source_name}_{timestamp}_{suffix}_{counter}"
        candidate = output_base / eval_run_id
        counter += 1
    return eval_run_id


def apply_saved_agent_hyperparams(config: TrajectoryCliConfig, metadata_path: Path) -> bool:
    if not metadata_path.exists():
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    saved_hyperparams = metadata.get("agent_hyperparameters", {})
    for key in AGENT_HYPERPARAM_KEYS:
        if key in saved_hyperparams:
            setattr(config, key, saved_hyperparams[key])
    return True


def build_run_metadata(config: TrajectoryCliConfig, eval_run_id: str) -> Dict[str, Any]:
    return {
        "eval_run_id": eval_run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": config.data_path,
        "mongo_run_id": config.run_id,
        "num_samples": config.num_samples,
        "seed": config.seed,
        "skip_judge": config.skip_judge,
        "output_path": config.output_path,
        "agent_hyperparameters": {
            key: getattr(config, key)
            for key in AGENT_HYPERPARAM_KEYS
        },
        "judge": {
            "model": config.judge_model,
            "base_url": config.judge_base_url,
            "type": config.judge_type,
            "voting_n": config.judge_voting_n,
            "concurrency": config.judge_concurrency,
        },
    }


def judge_should_think(model_name: str) -> bool:
    name = model_name.lower()
    if "qwq" in name or "flash" in name:
        return False
    return any(token in name for token in ("qwen", "o1", "r1"))


def configure_executor(search_workers: int, judge_workers: int) -> None:
    max_workers = search_workers + judge_workers + 20
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    asyncio.get_event_loop().set_default_executor(executor)
    print(f"Set default executor with {max_workers} workers.")

def extract_answer_fallback(result: AgentRunResult) -> str:
    """
    Tries to extract the answer from the trajectory if the agent didn't finish cleanly.
    """
    if result.status == "success":
        return result.answer
    
    # Try to extract from the last assistant message in the trajectory
    for msg in reversed(result.trajectory):
        if msg.get("role") == "assistant" and msg.get("content"):
            content = msg["content"]
            
            # 1. Check for \boxed{...}
            boxed_match = re.search(r"\\boxed\{(.*?)\}", content, re.DOTALL)
            if boxed_match:
                return boxed_match.group(1).strip()
                
            # 2. Check for <answer>...</answer> or <final_answer>...</final_answer>
            for tag in ["answer", "final_answer"]:
                tag_match = re.search(f"<{tag}>(.*?)</{tag}>", content, re.DOTALL | re.IGNORECASE)
                if tag_match:
                    return tag_match.group(1).strip()
            
            # 3. Check for "Final Answer: " or "Answer: "
            final_match = re.search(r"(?:final answer|answer):\s*(.*)", content, re.IGNORECASE)
            if final_match:
                return final_match.group(1).strip()
            
            # If it's a small model and it just output something without tool call
            # We take the whole content (excluding thought) as a last resort
            if "</think>" in content:
                _, final_part = content.rsplit("</think>", 1)
                return final_part.strip()
            return content.strip()
            
    return result.answer

class TrajectoryGeneratorLogger(BaseAgentLogger):
    """
    Rich-enhanced logger used by `cli/generate_trajectory.py`.
    """
    # Use a class-level console to avoid overhead of creating one for each instance
    _console = Console(theme=Theme({
        "info": "dim cyan",
        "warning": "magenta",
        "error": "bold red",
        "agent": "bold blue",
        "tool": "bold yellow",
        "success": "bold green",
        "fail": "bold red",
        "qa_id": "bold white on blue",
    }))

    def __init__(self, qa_id: str):
        self.qa_id = qa_id

    def _get_id_tag(self) -> Text:
        """Returns the formatted qa_id tag."""
        return Text(f"[{self.qa_id}] ", style="qa_id")

    def info(self, message: str) -> None:
        self._console.print(self._get_id_tag(), Text(message, style="info"))

    def warning(self, message: str) -> None:
        self._console.print(self._get_id_tag(), Text(f"WARNING: {message}", style="warning"))

    def error(self, message: str) -> None:
        self._console.print(self._get_id_tag(), Text(f"ERROR: {message}", style="error"))

    def log_agent_tool_execution(self, agent_name: str, tool_name: str, args: dict) -> None:
        # Format the arguments as a JSON string for better readability
        try:
            args_json = json.dumps(args, indent=2, ensure_ascii=False)
        except:
            args_json = str(args)
            
        if len(args_json) > 1000:
            args_json = args_json[:997] + "..."
            
        header = Text.assemble(
            (f"[{self.qa_id}] ", "qa_id"),
            (f" [{agent_name}] ", "agent"),
            ("TOOL ", "bold yellow"),
            (f"{tool_name}", "tool")
        )
        
        self._console.print(header)
        self._console.print(Panel(
            Syntax(args_json, "json", theme="monokai", background_color="default"),
            border_style="yellow",
            expand=False
        ))

    def log_agent_result(self, agent_name: str, result: str, success: bool = True) -> None:
        status_style = "success" if success else "fail"
        status_text = "SUCCESS" if success else "FAIL"
        
        msg = result if isinstance(result, str) else str(result)
        if len(msg) > 1000:
            msg = msg[:997] + "..."
            
        header = Text.assemble(
            (f"[{self.qa_id}] ", "qa_id"),
            (f" [{agent_name}] ", "agent"),
            (f" {status_text} ", status_style)
        )
        
        self._console.print(header)
        self._console.print(Panel(
            msg,
            border_style="green" if success else "red",
            expand=False
        ))

class TrajectoryGenerator:
    def __init__(self, args: TrajectoryCliConfig):
        self.args = args
        self.output_base_dir = resolve_project_path(args.output_path)
        self.eval_run_id = args.eval_run_id or make_eval_run_id(args.data_path, args.run_id, self.output_base_dir)
        self.output_dir = self.output_base_dir / self.eval_run_id
        self.metadata_path = self.output_dir / "metadata.json"
        self.result_file_path = self.output_dir / "result.jsonl"
        self.trajectory_dir = self.output_dir / "trajectory"
        self.retriever_dir = self.output_dir / "retriever"
        self.is_continuing_run = bool(args.eval_run_id and self.metadata_path.exists())

        if self.is_continuing_run:
            apply_saved_agent_hyperparams(args, self.metadata_path)
            print(f"Continuing eval run: {self.eval_run_id}")
            print(f"Loaded agent hyperparameters from: {self.metadata_path}")
        else:
            print(f"Starting eval run: {self.eval_run_id}")

        self.results = []
        self.existing_ids = set()
        self.lock = asyncio.Lock()
        self.semaphore = asyncio.Semaphore(args.concurrency)
        
        # Judge queue and concurrency
        self.judge_queue = asyncio.Queue()
        self.judge_workers = []
        
        self.api_key = os.getenv("TOOLS_API_KEY")
        
        # Agent configs
        self.search_cfg = AgentSamplerConfig(
            model=args.search_model,
            base_url=args.search_base_url,
            sampler_type=args.search_type,
            temperature=args.temperature,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            top_p=args.top_p,
            seed=args.seed,
            agent_spec=args.agent_spec,
        )

        if args.search_model != "qwq:32b":
            self.search_cfg.think = True
        
        self.browser_cfg = BrowserConfig(
            base_url=args.browser_base_url,
            api_key=self.api_key,
            truncate_context=True,
            max_context_length=args.max_context_length,
            max_results=args.max_search_results,
            timeout=300
        )
        
        self.judge_cfg = AgentSamplerConfig(
            model=args.judge_model,
            base_url=args.judge_base_url,
            sampler_type=args.judge_type,
            temperature=1,
            top_p=0.95,
            options={"max_tokens": 1024},
            think=judge_should_think(args.judge_model)
        )

    def _trajectory_path_for_id(self, sample_id: Any) -> Path:
        return self.trajectory_dir / f"{safe_trajectory_filename(sample_id)}.json"

    def _retriever_path_for_id(self, sample_id: Any) -> Path:
        return self.retriever_dir / f"{safe_trajectory_filename(sample_id)}.json"

    async def _append_result_locked(self, result: Dict[str, Any]):
        """Persist metadata and trajectory for a single result (must hold self.lock)."""
        try:
            def write_result(data, result_path, trajectory_dir, retriever_dir):
                trajectory = data.get("trajectory")
                record = {key: value for key, value in data.items() if key != "trajectory"}
                sample_id = record.get("id", "unknown")

                if trajectory is not None:
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    trajectory_path = self._trajectory_path_for_id(sample_id)
                    with open(trajectory_path, "w", encoding="utf-8") as f:
                        json.dump(trajectory, f, ensure_ascii=False, indent=2)
                    record["trajectory_path"] = str(trajectory_path.relative_to(self.output_dir))

                result_metadata = record.get("metadata")
                retriever_call_metadata = None
                if isinstance(result_metadata, dict):
                    retriever_call_metadata = result_metadata.pop("tool_trajectories", None)

                if retriever_call_metadata:
                    retriever_call_metadata = [
                        item.get("metadata", item) if isinstance(item, dict) else item
                        for item in retriever_call_metadata
                    ]
                    retriever_dir.mkdir(parents=True, exist_ok=True)
                    retriever_path = self._retriever_path_for_id(sample_id)
                    with open(retriever_path, "w", encoding="utf-8") as f:
                        json.dump(retriever_call_metadata, f, ensure_ascii=False, indent=2)
                    record["retriever_path"] = str(retriever_path.relative_to(self.output_dir))

                with open(result_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            await asyncio.to_thread(write_result, result, self.result_file_path, self.trajectory_dir, self.retriever_dir)
        except Exception as e:
            print(f"Error appending result: {e}")

    async def _add_result_and_save(self, result: Dict[str, Any]):
        """Add a result to self.results and append to file under lock."""
        async with self.lock:
            memory_result = {key: value for key, value in result.items() if key != "trajectory"}
            if isinstance(memory_result.get("metadata"), dict):
                memory_result["metadata"] = dict(memory_result["metadata"])
                memory_result["metadata"].pop("tool_trajectories", None)
            self.results.append(memory_result)
            await self._append_result_locked(result)

    async def judge_worker(self, worker_id: int):
        """Background worker to process judge tasks from the queue."""
        print(f"Judge worker {worker_id} started.")
        while True:
            item = await self.judge_queue.get()
            if item is None: # Sentinel value to exit
                self.judge_queue.task_done()
                break
            
            search_result, qa_pair, logger = item
            qa_id = qa_pair.get("id", qa_pair.get("task_id", "unknown"))
            
            try:
                import time
                start_time = time.time()
                logger.info(f"Starting judgment (Queue size: {self.judge_queue.qsize()})...")
                
                # We can call judge_agent.run directly as it is async
                # instead of using _run_judge_sync (which creates a new loop)
                # This prevents thread pool exhaustion and nested loop issues
                question = qa_pair.get("Question", qa_pair.get("question"))
                reference_answer = qa_pair.get("answer")
                extracted_answer = search_result.get("extracted_answer")
                
                judge_agent = JudgeAgent(config=self.judge_cfg, voting_n=self.args.judge_voting_n)
                
                judgment = await judge_agent.run(
                    question=question,
                    predicted_answer=extracted_answer,
                    reference_answer=reference_answer
                )
                
                judgment_data = {
                    "correct": judgment.metadata.get("correct", False),
                    "reasoning": judgment.metadata.get("reasoning", ""),
                    "raw_response": judgment.metadata.get("raw_response", "")
                }
                
                # Combine results
                final_result = self._combine_results(search_result, judgment_data, qa_pair)
                
                # Save result incrementally
                await self._add_result_and_save(final_result)
                
                duration = time.time() - start_time
                logger.info(f"Judgment finished in {duration:.2f}s. Correct: {final_result.get('correct')}")
            except Exception as e:
                logger.error(f"Error in judgment for {qa_id}: {e}")
                # Still add the failed judgment result so we don't lose the trajectory
                error_result = self._combine_results(search_result, {"error": str(e)}, qa_pair)
                await self._add_result_and_save(error_result)
            finally:
                self.judge_queue.task_done()

    def _combine_results(self, search_result: Dict[str, Any], judgment_data: Dict[str, Any], qa_pair: Dict[str, Any]) -> Dict[str, Any]:
        """Combines search and judge results into the final format."""
        question = qa_pair.get("Question", qa_pair.get("question"))
        reference_answer = qa_pair.get("answer")
        qa_id = qa_pair.get("id", qa_pair.get("task_id", "unknown"))

        # Default values if judgment failed
        correct = judgment_data.get("correct", False)
        reasoning = judgment_data.get("reasoning", "")
        raw_response = judgment_data.get("raw_response", "")
        
        if "error" in judgment_data:
            reasoning = f"JUDGMENT ERROR: {judgment_data['error']}"
            raw_response = f"Error occurred: {judgment_data['error']}"

        result = {
            "id": qa_id,
            "question": question,
            "reference_answer": reference_answer,
            "extracted_answer": search_result.get("extracted_answer"),
            "status": search_result.get("status"),
            "trajectory": search_result.get("trajectory"),
            "metadata": search_result.get("metadata", {})
        }

        if not self.args.skip_judge or judgment_data:
            result.update({
                "correct": correct,
                "judgment_reasoning": reasoning,
                "judgment_raw": raw_response,
            })

        return result

    async def run_sample(self, qa_pair: Dict[str, Any]) -> None:
        """Runs the search agent and queues the result for judging."""
        async with self.semaphore:
            qa_id = qa_pair.get("id", qa_pair.get("task_id", "unknown"))
            logger = TrajectoryGeneratorLogger(qa_id)
            logger.info(f"Starting sample")
            
            try:
                # 1. Run Search Agent
                search_data = await asyncio.to_thread(self._run_search_sync, qa_pair, logger)
                
                if "error" in search_data:
                    # Save error immediately
                    await self._add_result_and_save({
                        "id": qa_id,
                        "question": qa_pair.get("Question", qa_pair.get("question")),
                        "error": search_data["error"],
                        "status": "error"
                    })
                    return

                if self.args.skip_judge:
                    logger.info("Search finished, saving trajectory without judgment.")
                    await self._add_result_and_save(self._combine_results(search_data, {}, qa_pair))
                    return

                logger.info("Search finished, queuing for judgment.")
                await self.judge_queue.put((search_data, qa_pair, logger))
                
            except Exception as e:
                logger.error(f"Error in search: {e}")
                await self._add_result_and_save({
                    "id": qa_id,
                    "question": qa_pair.get("Question", qa_pair.get("question")),
                    "error": str(e),
                    "status": "error"
                })

    def _run_search_sync(self, qa_pair: Dict[str, Any], logger: TrajectoryGeneratorLogger) -> Dict[str, Any]:
        """Sync wrapper for search agent."""
        question = qa_pair.get("Question", qa_pair.get("question"))
        
        agent_cls = ResumActSearchAgent if self.args.agent_type == "resumact" else ReactSearchAgent
        
        agent_kwargs = {
            "config": self.search_cfg, 
            "browser_config": self.browser_cfg,
            "logger": logger,
            "max_conversational_turns": self.args.max_conversational_turns,
        }
        
        if self.args.agent_type != "resumact":
            agent_kwargs.update({
                "context_strategy_name": self.args.context_strategy,
                "tokenizer_path": self.args.tokenizer_path,
                "max_tokens": self.args.max_tokens,
                "max_tool_context": self.args.max_tool_context,
                "reset_interval": self.args.reset_interval,
            })
            
        search_agent = agent_cls(**agent_kwargs)
        
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(search_agent.run(question, max_steps=self.args.max_steps))
            extracted_answer = extract_answer_fallback(result)
            
            return {
                "extracted_answer": extracted_answer,
                "status": result.status,
                "trajectory": result.trajectory,
                "metadata": {
                    "total_time": getattr(result, 'total_time', 0),
                    "turns": getattr(result, 'turns', 0),
                    "input_tokens": getattr(result, 'input_tokens', 0),
                    "output_tokens": getattr(result, 'output_tokens', 0),
                    "tool_counts": result.metadata.get("tool_counts", {}),
                    "total_tool_calls": result.metadata.get("total_tool_calls", 0),
                    "tool_trajectories": result.metadata.get("tool_trajectories", [])
                }
            }
        except Exception as e:
            return {"error": str(e)}
        finally:
            loop.close()

    async def load_data(self) -> List[Dict[str, Any]]:
        """Fetch QA pairs from file or MongoDB."""
        if self.args.data_path:
            data_file_path = resolve_project_path(self.args.data_path)
            if not data_file_path.exists():
                print(f"Data file not found: {data_file_path}")
                return []
                
            with open(data_file_path, "r", encoding="utf-8") as f:
                if data_file_path.suffix == ".jsonl":
                    qa_data = []
                    for line in f:
                        if line.strip():
                            try:
                                qa_data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSONL line: {e}")
                else:
                    qa_data = json.load(f)
            print(f"Loaded {len(qa_data)} samples from {self.args.data_path}")
        else:
            db_manager = DatabaseManager()
            try:
                qa_data = await db_manager.get_qa_pairs(run_id=self.args.run_id)
                print(f"Fetched {len(qa_data)} samples from MongoDB (run_id: {self.args.run_id})")
            finally:
                db_manager.close()
        return qa_data

    async def load_existing_results(self):
        """Load already processed IDs from result.jsonl."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.result_file_path.exists():
            return

        print(f"Loading existing results from {self.result_file_path}...")
        try:
            existing_data = []
            is_json_array = False
            
            with open(self.result_file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return
                
                try:
                    # Try parsing as a full JSON array first
                    data_list = json.loads(content)
                    if isinstance(data_list, list):
                        existing_data = data_list
                        is_json_array = True
                except json.JSONDecodeError:
                    # Fallback to JSONL
                    f.seek(0)
                    for line in f:
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            if isinstance(data, dict):
                                existing_data.append(data)
                        except json.JSONDecodeError:
                            continue

            # Populate in-memory state
            for item in existing_data:
                self.results.append(item)
                if "id" in item:
                    self.existing_ids.add(item["id"])
            
            print(f"Found {len(self.existing_ids)} existing results.")

            # If it was a JSON array, convert it to JSONL immediately to allow future appends
            if is_json_array and existing_data:
                print(f"Converting {self.result_file_path} from JSON array to JSONL...")
                def rewrite_as_jsonl(data, path):
                    with open(path, "w", encoding="utf-8") as f:
                        for item in data:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                await asyncio.to_thread(rewrite_as_jsonl, existing_data, self.result_file_path)

        except Exception as e:
            print(f"Error loading existing results: {e}")


    def print_summary(self):
        """Prints evaluation summary metrics."""
        valid_results = [r for r in self.results if "error" not in r]
        total = len(valid_results)
        correct_count = sum(1 for r in valid_results if r.get("correct") is True)
        success_rate = (correct_count / total * 100) if total > 0 else 0
        
        print(f"\n{'='*30}")
        print("Generation Summary" if self.args.skip_judge else "Evaluation Summary")
        print(f"Total Valid Samples: {total}")
        if not self.args.skip_judge:
            print(f"Correct Answers: {correct_count}")
            print(f"Success Rate: {success_rate:.2f}%")
        print(f"Run ID: {self.eval_run_id}")
        print(f"Run metadata saved to: {self.metadata_path}")
        print(f"Results saved to: {self.result_file_path}")
        print(f"Trajectories saved to: {self.trajectory_dir}")
        print(f"Retriever metadata saved to: {self.retriever_dir}")
        print(f"{'='*30}")

    async def write_run_metadata(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.retriever_dir.mkdir(parents=True, exist_ok=True)

        if self.metadata_path.exists():
            return

        metadata = build_run_metadata(self.args, self.eval_run_id)

        def write_metadata(data, path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        await asyncio.to_thread(write_metadata, metadata, self.metadata_path)
        print(f"Saved run metadata to: {self.metadata_path}")

    async def run(self):
        """Main execution flow."""
        # 1. Load data and existing results
        await self.write_run_metadata()
        qa_data = await self.load_data()
        if not qa_data:
            return
            
        await self.load_existing_results()
        
        # 2. Filter candidates
        candidates = [qa for qa in qa_data if qa.get("id", qa.get("task_id", "unknown")) not in self.existing_ids]
        print(f"Remaining candidates after filtering: {len(candidates)}")
        
        # 3. Sample new questions
        to_sample_count = self.args.num_samples - len(self.existing_ids)
        if to_sample_count <= 0:
            print(f"Requested {self.args.num_samples} samples, already have {len(self.existing_ids)}. Nothing more to do.")
            self.print_summary()
            return

        if self.args.seed != -1:
            random.seed(self.args.seed)
            
        if to_sample_count > len(candidates):
            print(f"Only {len(candidates)} new candidates available. Using all.")
            sampled_qa = candidates
        else:
            sampled_qa = random.sample(candidates, to_sample_count)
        
        # shuffle candidates
        random.shuffle(sampled_qa)
        
        print(f"Starting generation for {len(sampled_qa)} new samples...")
        if self.args.skip_judge:
            print(f"Concurrency: {self.args.concurrency} searchers. Judgment disabled.")
        else:
            print(f"Concurrency: {self.args.concurrency} searchers, {self.args.judge_concurrency} judges.")

        # 4. Start judge workers
        if not self.args.skip_judge:
            for i in range(self.args.judge_concurrency):
                worker = asyncio.create_task(self.judge_worker(i))
                self.judge_workers.append(worker)

        # 5. Run search tasks in parallel
        tasks = [self.run_sample(qa) for qa in sampled_qa]
        await asyncio.gather(*tasks)
        
        # 6. Wait for all queued judge tasks to be processed
        if not self.args.skip_judge:
            print("All search tasks finished. Waiting for judgment queue to clear...")
            await self.judge_queue.join()

        # 7. Stop judge workers
        if not self.args.skip_judge:
            for _ in range(self.args.judge_concurrency):
                await self.judge_queue.put(None)
            await asyncio.gather(*self.judge_workers)
        
        # 8. Final summary
        self.print_summary()

async def run_generator(config: TrajectoryCliConfig):
    configure_executor(config.concurrency, config.judge_concurrency)
    generator = TrajectoryGenerator(config)
    await generator.run()


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--num-samples", "--num_samples", "--num", type=int, default=800, show_default=True, help="Number of samples to run.")
@click.option("--data-path", "--data_path", type=str, default=None, help="Path to QA pairs JSON/JSONL file.")
@click.option("--run-id", "--run_id", type=str, default=None, help="Run ID to filter QA pairs from MongoDB.")
@click.option("--eval-run-id", "--eval_run_id", type=str, default=None, help="Existing eval run ID to continue, or a custom run folder name for a new run.")
@click.option("--output-path", "--output_path", type=str, required=True, help="Base output directory. Each run is saved in output-path/eval-run-id/.")
@click.option("--seed", type=int, default=42, show_default=True, help="Random seed for sampling.")
@click.option("--concurrency", type=int, default=32, show_default=True, help="Number of parallel search runs.")
@click.option("--judge-concurrency", "--judge_concurrency", type=int, default=2, show_default=True, help="Number of parallel judge workers.")
@click.option("--agent-type", "--agent_type", type=click.Choice(["react", "resumact"]), default="react", show_default=True, help="Type of search agent to use.")
@click.option("--agent-spec", "--agent_spec", type=str, default=None, help="Path to a YAML file containing prompts and tools.")
@click.option("--search-model", "--search_model", type=str, default="GLM-4.7", show_default=True, help="Model for search agent.")
@click.option("--search-base-url", "--search_base_url", type=str, default="https://open.bigmodel.cn/api/paas/v4/", show_default=True, help="Base URL for search agent.")
@click.option("--search-type", "--search_type", type=str, default="glm", show_default=True, help="Sampler type for search agent.")
@click.option("--max-steps", "--max_steps", type=int, default=30, show_default=True, help="Max steps for search agent.")
@click.option("--max-context-length", "--max_context_length", type=int, default=10000, show_default=True, help="Max context length for search agent.")
@click.option("--max-search-results", "--max_search_results", type=int, default=10, show_default=True, help="Max search results for browser tool.")
@click.option("--max-conversational-turns", "--max_conversational_turns", type=int, default=None, help="Max conversational turns to keep in context.")
@click.option("--temperature", type=float, default=0.7, show_default=True, help="Temperature for search agent.")
@click.option("--presence-penalty", "--presence_penalty", type=float, default=0.0, show_default=True, help="Presence penalty for search agent.")
@click.option("--repetition-penalty", "--repetition_penalty", type=float, default=1.1, show_default=True, help="Repetition penalty for search agent.")
@click.option("--top-p", "--top_p", type=float, default=0.85, show_default=True, help="Top p for search agent.")
@click.option("--context-strategy", "--context_strategy", type=click.Choice(["sliding_window", "token_limit"]), default="sliding_window", show_default=True, help="Strategy for truncating context.")
@click.option("--tokenizer-path", "--tokenizer_path", type=str, default=None, help="Path to tokenizer model.")
@click.option("--max-tokens", "--max_tokens", type=int, default=110000, show_default=True, help="Max tokens allowed in context.")
@click.option("--max-tool-context", "--max_tool_context", type=int, default=None, help="Max recent tool results to keep uncompressed.")
@click.option("--reset-interval", "--reset_interval", type=int, default=None, help="Number of turns after which to reset assistant/tool messages.")
@click.option("--judge-model", "--judge_model", type=str, default="glm-4.7-flash:latest", show_default=True, help="Model for judge agent.")
@click.option("--judge-base-url", "--judge_base_url", type=str, default="http://localhost:11434", show_default=True, help="Base URL for judge agent.")
@click.option("--judge-type", "--judge_type", type=str, default="ollama", show_default=True, help="Sampler type for judge agent.")
@click.option("--judge-voting-n", "--judge_voting_n", type=int, default=1, show_default=True, help="Number of votes for judge agent.")
@click.option("--browser-base-url", "--browser_base_url", type=str, default="http://localhost:8001", show_default=True, help="Base URL for browser tool.")
@click.option("--skip-judge/--no-skip-judge", default=False, show_default=True, help="Skip answer judgment and only save search trajectories.")
def main(**kwargs):
    """Generate search trajectories and evaluate them with a judge agent."""
    asyncio.run(run_generator(TrajectoryCliConfig(**kwargs)))


if __name__ == "__main__":
    main()
