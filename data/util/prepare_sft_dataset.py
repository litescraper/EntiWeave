"""
Fetch the train_sft split from HuggingFace and convert it to Swift SFT format.

Each record's "trajectory" field (a list of message dicts) is wrapped as:
    {"messages": [...], "tools": "<json-stringified tool list>"}

Tools are loaded from data/util/prompt.yaml.

Output: data/dataset/entiweave_sft_swift.json

Usage:
    python -m data.util.prepare_sft_dataset
"""

import json
from pathlib import Path

import yaml
from datasets import load_dataset

REPO_ID = "jxg25/EntiWeave"
SPLIT = "train_sft"
DATASET_DIR = Path(__file__).parent.parent / "dataset"
PROMPT_YAML = Path(__file__).parent / "prompt.yaml"
OUTPUT_FILE = DATASET_DIR / "entiweave_sft_swift.json"


def _load_tools_str() -> str:
    with open(PROMPT_YAML) as f:
        data = yaml.safe_load(f)
    tools = data.get("tools", [])
    return json.dumps(tools, ensure_ascii=False)


def convert() -> None:
    print(f"Loading {SPLIT} split from {REPO_ID} ...")
    ds = load_dataset(REPO_ID, split=SPLIT)

    if "trajectory" not in ds.column_names:
        raise ValueError(f"'trajectory' field not found in {SPLIT} split. Available: {ds.column_names}")

    tools_str = _load_tools_str()

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0
    for row in ds:
        trajectory = row["trajectory"]
        if not trajectory:
            skipped += 1
            continue
        records.append({"messages": trajectory, "tools": tools_str})

    OUTPUT_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    print(f"  converted {len(records)} records  ->  {OUTPUT_FILE.relative_to(Path(__file__).parent.parent.parent)}")
    if skipped:
        print(f"  skipped {skipped} records with empty trajectory")


if __name__ == "__main__":
    convert()
