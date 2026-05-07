"""
Summary statistics and plots for the graph consistency study.

Reads:  gc_study_results/{gen_model}_judgedby_{judge_model}.json

For each gen_model:
  - Collect IoU scores from all available judge models
  - Average across judges per graph_id
  - Compute mean, std, SE for entity_iou and predicate_iou

Prints a summary table and saves plots to gc_study_results/figures/.

Usage:
  python gc_stats.py [--results_dir gc_study_results]
"""

import os
import json
import math
import re
import argparse
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gc_study_results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

# Filename pattern: {gen_model}_judgedby_{judge_model}.json
PATTERN = re.compile(r"^(.+)_judgedby_(.+)\.json$")


def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(vals: list) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def std(vals: list) -> float:
    if len(vals) < 2:
        return 0.0
    m = mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def se(vals: list) -> float:
    return std(vals) / math.sqrt(len(vals)) if vals else 0.0


def collect_results(results_dir: str):
    """
    Returns:
      gen_model -> judge_model -> list of {graph_id, entity_iou, predicate_iou}
    """
    data: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return data

    for fname in os.listdir(results_dir):
        m = PATTERN.match(fname)
        if not m:
            continue
        gen_model, judge_model = m.group(1), m.group(2)
        entries = load_json(os.path.join(results_dir, fname))
        for entry in entries:
            data[gen_model][judge_model].append({
                "graph_id": entry["graph_id"],
                "entity_iou": entry.get("entity_iou", 0.0),
                "predicate_iou": entry.get("predicate_iou", 0.0),
            })

    return data


def compute_stats(data: dict) -> list[dict]:
    """
    For each gen_model, average IoU per graph across all judges, then aggregate.
    Returns list of stat dicts sorted by gen_model name.
    """
    stats = []

    for gen_model, judge_data in sorted(data.items()):
        # Collect per-graph scores averaged over judges
        graph_entity: dict[str, list] = defaultdict(list)
        graph_predicate: dict[str, list] = defaultdict(list)

        for judge_model, entries in judge_data.items():
            for e in entries:
                graph_entity[e["graph_id"]].append(e["entity_iou"])
                graph_predicate[e["graph_id"]].append(e["predicate_iou"])

        # Only use graphs that appear in every judge result
        n_judges = len(judge_data)
        common_ids = [
            gid for gid in graph_entity
            if len(graph_entity[gid]) == n_judges and len(graph_predicate[gid]) == n_judges
        ]

        if not common_ids:
            continue

        entity_avgs = [mean(graph_entity[gid]) for gid in common_ids]
        predicate_avgs = [mean(graph_predicate[gid]) for gid in common_ids]

        stats.append({
            "gen_model": gen_model,
            "n_judges": n_judges,
            "judge_models": sorted(judge_data.keys()),
            "n_graphs": len(common_ids),
            "entity_mean": mean(entity_avgs),
            "entity_std": std(entity_avgs),
            "entity_se": se(entity_avgs),
            "predicate_mean": mean(predicate_avgs),
            "predicate_std": std(predicate_avgs),
            "predicate_se": se(predicate_avgs),
            # Keep per-graph averages for plotting
            "_entity_avgs": entity_avgs,
            "_predicate_avgs": predicate_avgs,
        })

    return stats


def print_table(stats: list[dict]):
    if not stats:
        print("No data found.")
        return

    header = (
        f"{'Gen Model':<24} | {'Judges':<3} | {'Graphs':<6} | "
        f"{'Ent IoU':<8} | {'Ent Std':<8} | {'Ent SE':<8} | "
        f"{'Pred IoU':<8} | {'Pred Std':<8} | {'Pred SE':<8}"
    )
    print(header)
    print("-" * len(header))
    for r in stats:
        print(
            f"{r['gen_model']:<24} | {r['n_judges']:<3} | {r['n_graphs']:<6} | "
            f"{r['entity_mean']:.4f}   | {r['entity_std']:.4f}   | {r['entity_se']:.4f}   | "
            f"{r['predicate_mean']:.4f}   | {r['predicate_std']:.4f}   | {r['predicate_se']:.4f}"
        )


def save_json_summary(stats: list[dict], output_path: str):
    out = []
    for r in stats:
        out.append({k: v for k, v in r.items() if not k.startswith("_")})
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSummary JSON saved to {output_path}")


def plot_bar(stats: list[dict], figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    labels = [r["gen_model"] for r in stats]
    x = np.arange(len(labels))
    width = 0.35

    entity_means = [r["entity_mean"] for r in stats]
    entity_ses = [r["entity_se"] for r in stats]
    pred_means = [r["predicate_mean"] for r in stats]
    pred_ses = [r["predicate_se"] for r in stats]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars1 = ax.bar(x - width / 2, entity_means, width, yerr=entity_ses, capsize=4,
                   label="Entity IoU", color="#4C9BE8", alpha=0.85)
    bars2 = ax.bar(x + width / 2, pred_means, width, yerr=pred_ses, capsize=4,
                   label="Predicate IoU", color="#E87C4C", alpha=0.85)

    ax.set_xlabel("Generation Model")
    ax.set_ylabel("IoU Score (mean ± SE)")
    ax.set_title("Graph Consistency Study – IoU by Generation Model\n(averaged across judge models)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_path = os.path.join(figures_dir, "gc_iou_bar.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved to {out_path}")


def _family_colors(labels: list[str]) -> list:
    """
    Assign purple shades to DeepSeek models, orange shades to Qwen models,
    grey shades to everything else. Each family member gets a progressively
    lighter shade so different parameter sizes are visually distinct.
    """
    PURPLE_DARK  = np.array([0x86, 0x5a, 0x98]) / 255
    PURPLE_LIGHT = np.array([0xc4, 0xaa, 0xd0]) / 255
    ORANGE_DARK  = np.array([0xbb, 0x69, 0x33]) / 255
    ORANGE_LIGHT = np.array([0xe8, 0xb8, 0x8a]) / 255
    GREY_DARK    = np.array([0x55, 0x55, 0x55]) / 255
    GREY_LIGHT   = np.array([0xaa, 0xaa, 0xaa]) / 255

    families: dict[str, list[int]] = {"deepseek": [], "qwen": [], "other": []}
    for i, label in enumerate(labels):
        ll = label.lower()
        if "deepseek" in ll:
            families["deepseek"].append(i)
        elif "qwen" in ll:
            families["qwen"].append(i)
        else:
            families["other"].append(i)

    colors = [None] * len(labels)
    for fam, (dark, light) in [
        ("deepseek", (PURPLE_DARK, PURPLE_LIGHT)),
        ("qwen",     (ORANGE_DARK, ORANGE_LIGHT)),
        ("other",    (GREY_DARK,   GREY_LIGHT)),
    ]:
        idxs = families[fam]
        n = len(idxs)
        for rank, i in enumerate(idxs):
            t = rank / max(n - 1, 1)
            colors[i] = tuple((1 - t) * dark + t * light)

    return colors


def plot_boxplot(stats: list[dict], figures_dir: str):
    os.makedirs(figures_dir, exist_ok=True)
    labels = [r["gen_model"] for r in stats]
    colors = _family_colors(labels)

    all_vals = [v for r in stats for v in r["_entity_avgs"] + r["_predicate_avgs"]]
    y_min = max(0.0, min(all_vals) - 0.05)
    y_min = math.floor(y_min * 10) / 10  # snap to nearest 0.1

    fig, axes = plt.subplots(2, 1, figsize=(max(7, len(labels) * 1.2), 5.5), sharex=True)

    yticks = [v for v in [0.25, 0.5, 0.75, 1.0] if v >= y_min]

    for ax, key, ylabel in [
        (axes[0], "_entity_avgs",    "Entity IoU"),
        (axes[1], "_predicate_avgs", "Predicate IoU"),
    ]:
        plot_data = [r[key] for r in stats]
        bp = ax.boxplot(plot_data, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_ylim(y_min, 1.02)
        ax.set_yticks(yticks)
        ax.tick_params(axis="both", labelsize=15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    display_labels = [l.replace("_", "-") for l in labels]
    axes[1].set_xticks(range(1, len(labels) + 1))
    axes[1].set_xticklabels(display_labels, rotation=20, ha="right", fontsize=15)

    plt.tight_layout()

    out_path = os.path.join(figures_dir, "gc_iou_boxplot.pdf")
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Boxplot saved to {out_path}")


def plot_judge_comparison(data: dict, stats: list[dict], figures_dir: str):
    """Per-gen-model: show each judge's mean separately to visualise judge variance."""
    os.makedirs(figures_dir, exist_ok=True)

    for r in stats:
        gen_model = r["gen_model"]
        judge_data = data[gen_model]
        if len(judge_data) < 2:
            continue

        judge_names = sorted(judge_data.keys())
        entity_per_judge = []
        pred_per_judge = []

        for jname in judge_names:
            entries = judge_data[jname]
            entity_per_judge.append(mean([e["entity_iou"] for e in entries]))
            pred_per_judge.append(mean([e["predicate_iou"] for e in entries]))

        x = np.arange(len(judge_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(6, len(judge_names) * 1.5), 4))
        ax.bar(x - width / 2, entity_per_judge, width, label="Entity IoU", color="#4C9BE8", alpha=0.85)
        ax.bar(x + width / 2, pred_per_judge, width, label="Predicate IoU", color="#E87C4C", alpha=0.85)
        ax.set_xlabel("Judge Model")
        ax.set_ylabel("Mean IoU")
        ax.set_title(f"Judge Comparison for gen_model={gen_model}")
        ax.set_xticks(x)
        ax.set_xticklabels(judge_names, rotation=15, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()

        safe_name = re.sub(r"[^\w\-]", "_", gen_model)
        out_path = os.path.join(figures_dir, f"gc_judges_{safe_name}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Judge comparison chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Summary statistics and plots for graph consistency study")
    parser.add_argument("--results_dir", type=str, default=RESULTS_DIR,
                        help="Directory containing *_judgedby_*.json result files")
    args = parser.parse_args()

    figures_dir = os.path.join(args.results_dir, "figures")

    data = collect_results(args.results_dir)
    if not data:
        print("No result files found.")
        return

    stats = compute_stats(data)
    if not stats:
        print("No complete data to summarise.")
        return

    print_table(stats)
    save_json_summary(stats, os.path.join(args.results_dir, "gc_summary.json"))
    plot_boxplot(stats, figures_dir)


if __name__ == "__main__":
    main()
