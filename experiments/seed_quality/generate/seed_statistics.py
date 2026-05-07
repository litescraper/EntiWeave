import os
import sys
import asyncio
import json
import argparse
import time
import concurrent.futures
from typing import List, Dict, Any, Optional
from collections import Counter

# Add the project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# Following the pattern in webcomposer/cli/generate_cokg.py
# If current_dir is webcomposer/study, parent is webcomposer/
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# loadenv
from dotenv import load_dotenv
env_path = os.path.join(parent_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
# Now modules should be importable directly
from modules.agent.agent_types import AgentSamplerConfig, BrowserConfig
from modules.agent.graph_agent.seedgen import SeedGenAgent

TOPIC = "sports"
CATEGORY = "person"
OUTPUT_FILE = f"./results/seed_stats_{TOPIC}_{CATEGORY}.json"
async def main():
    parser = argparse.ArgumentParser(description="Study for the seed generator statistics.")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate (default: 100)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3.5-35B-A3B", help="LLM model to use")
    parser.add_argument("--base_url", type=str, default="http://localhost:8224/v1", help="Base URL for the model")
    parser.add_argument("--sampler_type", type=str, default="vllm", help="Sampler type (glm, openai, etc.)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for the sampler")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for the sampler")
    parser.add_argument("--presence_penalty", type=float, default=0.0, help="Presence penalty for the sampler")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty for the sampler")
    parser.add_argument("--think", action="store_true", default=True, help="Enable thinking mode")
    parser.add_argument("--no_think", action="store_false", dest="think", help="Disable thinking mode")
    
    parser.add_argument("--topic", type=str, default=TOPIC, help="Specific topic for seed generation")
    parser.add_argument("--category", type=str, default=CATEGORY, help="Specific category for seed generation")
    parser.add_argument("--max_steps", type=int, default=15, help="Max steps per research phase")
    parser.add_argument("--max_revisions", type=int, default=20, help="Max revisions per generation")
    parser.add_argument("--output_file", type=str, default=OUTPUT_FILE, help="Output JSON file")
    parser.add_argument("--browser_base_url", type=str, default="http://localhost:8001", help="Base URL for browser tool")
    parser.add_argument("--concurrency", type=int, default=3, help="Number of concurrent generation tasks")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging from the agent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic results")
    
    args = parser.parse_args()

    # Set random seed
    import random
    random.seed(args.seed)

    # Initialize configs
    sampler_cfg = AgentSamplerConfig(
        model=args.model,
        base_url=args.base_url or os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
        sampler_type=args.sampler_type,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        think=args.think,
    )

    browser_cfg = BrowserConfig(
        base_url=args.browser_base_url or os.getenv("BROWSER_BASE_URL", "http://localhost:8000"),
        api_key=os.getenv("TOOLS_API_KEY")
    )

    # Set up logger if verbose
    logger = None
    if args.verbose:
        from modules.logger import QAGeneratorLogger
        logger = QAGeneratorLogger()
        logger.log_run_start(run_id="seed_study", total_graphs=args.num_samples)

    print(f"Starting seed generation study with {args.num_samples} samples...")