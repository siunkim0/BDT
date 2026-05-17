"""Count which feature each tree splits on at its root.

The root split of every tree is, by gain, the most discriminating cut that
tree could find given the residuals at that boosting step. Aggregating the
roots across all trees gives a quick "where does the BDT keep starting from?"
view that complements the gain-based importance plot.

Run from project root:

    python -m scripts.root_feature_stats
    python scripts/root_feature_stats.py --model data/models/bdt_v1.json

Outputs:
  - prints a table to stdout
  - plots/root_feature_counts.png
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features import FEATURES
from src.utils import PLOT_DIR, get_logger

log = get_logger("root_feature_stats")


def root_feature_counts(model: xgb.XGBClassifier,
                        feature_names: list[str]) -> Counter:
    """Count root-node split features across all trees in the booster.

    xgboost's text dump puts the root on the first line of each tree as
    ``0:[fK<thr] yes=...,no=...,...`` for numeric splits. We extract ``fK``
    and translate the index back to the human-readable feature name.
    """
    dump = model.get_booster().get_dump()
    counts: Counter = Counter()
    for tree_text in dump:
        first_line = tree_text.strip().split("\n", 1)[0]
        # Stumps that never split look like "0:leaf=..."; skip them.
        if "[" not in first_line:
            counts["<leaf-only tree>"] += 1
            continue
        feat_token = first_line.split("[", 1)[1].split("<", 1)[0]
        idx = int(feat_token[1:])
        counts[feature_names[idx]] += 1
    return counts


def plot_counts(counts: Counter, out_path: Path) -> None:
    items = [(k, v) for k, v in counts.most_common() if k != "<leaf-only tree>"]
    if not items:
        log.warning("No non-trivial trees found, skipping plot.")
        return
    names = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]

    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 1.5))
    ax.barh(names, values)
    ax.set_xlabel("number of trees rooted on this feature")
    ax.set_title("BDT root-split feature counts")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log.info("Saved root-feature plot to %s", out_path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="data/models/bdt_v1.json")
    p.add_argument("--out", default=str(PLOT_DIR / "root_feature_counts.png"))
    args = p.parse_args()

    model = xgb.XGBClassifier()
    model.load_model(args.model)
    log.info("Loaded model from %s", args.model)

    counts = root_feature_counts(model, FEATURES)
    total = sum(counts.values())
    log.info("Total trees: %d", total)

    width = max(len(k) for k in counts)
    print(f"{'feature'.ljust(width)}  trees   share")
    print(f"{'-' * width}  -----   -----")
    for name, n in counts.most_common():
        print(f"{name.ljust(width)}  {n:5d}  {n / total:6.1%}")

    plot_counts(counts, Path(args.out))


if __name__ == "__main__":
    main()
