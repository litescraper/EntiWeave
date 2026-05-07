"""
Download the EntiWeave dataset from HuggingFace and save train splits to JSON.

Splits starting with "train" are saved to data/dataset/:
  train     -> entiweave_train.json
  train_sft -> entiweave_train_sft.json

Usage:
    python -m data.util.download_dataset
"""

import json
from pathlib import Path

from datasets import load_dataset

REPO_ID = "jxg25/EntiWeave"
DATASET_DIR = Path(__file__).parent.parent / "dataset"

SPLIT_TO_FILE = {
    "train": "entiweave_train.json",
    "train_sft": "entiweave_train_sft.json",
}


def download() -> None:
    print(f"Loading dataset {REPO_ID} ...")
    ds = load_dataset(REPO_ID)

    train_splits = [s for s in ds.keys() if s.startswith("train")]
    if not train_splits:
        print("No train splits found.")
        return

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for split in train_splits:
        filename = SPLIT_TO_FILE.get(split, f"entiweave_{split}.json")
        out = DATASET_DIR / filename
        records = [dict(row) for row in ds[split]]
        out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
        print(f"  saved {len(records)} records  ->  {out.relative_to(Path(__file__).parent.parent.parent)}")


if __name__ == "__main__":
    download()
