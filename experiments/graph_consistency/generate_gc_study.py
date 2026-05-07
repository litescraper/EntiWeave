"""
Phase 1 – Generate questions from sampled graphs using Graph2QueryAgent.

Output: gc_study_data/{model_name}_gc_study.json
  [
    {"graph_id": "...", "question": "...", "exact_answer": "...", "status": "success|partial_success|error"},
    ...
  ]

Continuation:
  - Re-running resumes automatically from the last saved state.
  - Entries with status=error are skipped on resume unless --retry_failed is set.
  - Writes are atomic (temp file → rename) to prevent corruption on interrupt.

Usage:
  python generate_gc_study.py \
    --model_name glm5 \
    --model GLM-5 \
    --base_url https://llmapi.paratera.com \
    --sampler_type glm \
    --concurrency 4

  # Retry previously failed entries:
  python generate_gc_study.py --model_name glm5 ... --retry_failed
"""

import sys
import os
import json
import asyncio
import argparse
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(os.path.dirname(current_dir))
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(workspace_root, ".env"))

from modules.agent.graph_agent.graph2query import Graph2QueryAgent
from modules.agent.agent_types import AgentSamplerConfig
from modules.graph.knowledge_graph.cokg import CompositionalKnowledgeGraph

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

console = Console()

OUTPUT_DIR = os.path.join(current_dir, "gc_study_data")


def atomic_write_json(path: str, data: list):
    """Write JSON atomically via a temp file in the same directory."""
    dir_ = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", dir=dir_, delete=False
    ) as tmp:
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


async def main():
    parser = argparse.ArgumentParser(description="Phase 1: Generate questions for graph consistency study")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Short display name for output filename (e.g. glm5, qwen3)")
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier sent to the API (e.g. GLM-5)")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1",
                        help="API base URL")
    parser.add_argument("--sampler_type", type=str, default="glm",
                        help="Sampler type (glm, vllm, openai, anthropic, etc.)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of concurrent workers")
    parser.add_argument("--sampled_graphs", type=str,
                        default=os.path.join(current_dir, "sampled_graphs.json"),
                        help="Path to the sampled graphs JSON file")
    parser.add_argument("--retry_failed", action="store_true",
                        help="Re-attempt entries that previously ended with status=error")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{args.model_name}_gc_study.json")

    with open(args.sampled_graphs, "r", encoding="utf-8") as f:
        sampled_graphs: list[dict] = json.load(f)
    console.print(f"[cyan]Loaded {len(sampled_graphs)} sampled graphs from {args.sampled_graphs}[/cyan]")

    # --- Continuation: load existing results ---
    results: list[dict] = []
    skip_ids: set[str] = set()     # IDs we should NOT re-process
    failed_ids: set[str] = set()   # IDs previously failed

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            for r in results:
                gid = r.get("graph_id")
                if not gid:
                    continue
                if r.get("status") in ("success", "partial_success"):
                    skip_ids.add(gid)
                else:
                    failed_ids.add(gid)

            retry_count = len(failed_ids) if args.retry_failed else 0
            console.print(
                f"[green]Resuming: {len(skip_ids)} successful, "
                f"{len(failed_ids)} failed "
                f"({'will retry' if args.retry_failed else 'will skip'})[/green]"
            )
            if args.retry_failed:
                # Remove failed entries from results so they'll be re-written
                results = [r for r in results if r.get("graph_id") not in failed_ids]
        except Exception as e:
            console.print(f"[yellow]Could not load existing output ({e}), starting fresh.[/yellow]")
            results = []
            skip_ids = set()
            failed_ids = set()

    # Determine which graphs to process
    to_process = []
    for g in sampled_graphs:
        gid = g["graph_id"]
        if gid in skip_ids:
            continue
        if gid in failed_ids and not args.retry_failed:
            continue
        to_process.append(g)

    if not to_process:
        console.print("[bold green]Nothing to do – all graphs already processed.[/bold green]")
        return

    console.print(f"[cyan]Will process {len(to_process)} graphs[/cyan]")

    config = AgentSamplerConfig(
        model=args.model,
        base_url=args.base_url,
        sampler_type=args.sampler_type,
        think=True,
    )
    agent = Graph2QueryAgent(config=config)

    queue: asyncio.Queue = asyncio.Queue()
    for g in to_process:
        queue.put_nowait(g)

    total = queue.qsize()
    file_lock = asyncio.Lock()

    async def worker(wid: int, progress, task_id):
        while True:
            try:
                graph_dict = queue.get_nowait()
            except asyncio.QueueEmpty:
                break

            graph_id = graph_dict["graph_id"]
            progress.update(task_id, description=f"[cyan]W{wid} → {graph_id[:28]}...")

            try:
                cokg = CompositionalKnowledgeGraph.from_dict(graph_dict)
                result = await agent.run(cokg)

                entry = {
                    "graph_id": graph_id,
                    "question": result.answer,
                    "exact_answer": result.metadata.get("exact_answer", ""),
                    "status": result.status,
                }

                async with file_lock:
                    results.append(entry)
                    atomic_write_json(output_path, results)

                status_color = "green" if result.status == "success" else "yellow"
                console.print(
                    f"[{status_color}][W{wid}] {graph_id} "
                    f"status={result.status} "
                    f"answer={entry['exact_answer'][:40]}[/{status_color}]"
                )
            except Exception as e:
                console.print(f"[red][W{wid}] Error on {graph_id}: {e}[/red]")
                # Save error entry so we know it was attempted
                async with file_lock:
                    results.append({"graph_id": graph_id, "question": "", "exact_answer": "", "status": "error", "error": str(e)})
                    atomic_write_json(output_path, results)

            queue.task_done()
            progress.advance(task_id)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task("[cyan]Generating questions...", total=total)
        workers = [asyncio.create_task(worker(i, progress, task_id)) for i in range(args.concurrency)]
        await queue.join()
        for w in workers:
            w.cancel()

    success_count = sum(1 for r in results if r.get("status") in ("success", "partial_success"))
    error_count = sum(1 for r in results if r.get("status") == "error")
    console.print(
        f"\n[bold green]Done! {success_count} succeeded, {error_count} failed. "
        f"Saved to {output_path}[/bold green]"
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress saved – re-run to continue.[/yellow]")
