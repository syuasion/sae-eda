"""Put several grid runs side by side.

    uv run python compare.py task=results/terminal_wrench/<run>/eda-keep100 \
      notask=results/terminal_wrench_notask/<run>/eda-keep100 [--all]

Each argument is label=<eda directory>. Prints, per cell, the gain over
chance (score minus the same score with labels shuffled) for category F1
and NMI, and the NMI against task and model. --all includes the
baselines (tfidf, emb), which are the same in every SAE run of one
dataset.
"""

from __future__ import annotations

import sys

import pandas as pd


def main():
    pairs = [a.split("=", 1) for a in sys.argv[1:] if not a.startswith("--")]
    frames = []
    for label, directory in pairs:
        grid = pd.read_parquet(f"{directory}/grid.parquet")
        grid = grid[grid.status == "ok"].copy()
        grid["run"] = label
        frames.append(grid)
    grid = pd.concat(frames)
    if "--all" not in sys.argv:
        grid = grid[grid.rep.str.startswith("sae")]
    grid["f1_gain"] = grid.category_f1 - grid.category_f1_shuffled
    grid["nmi_gain"] = grid.nmi_category - grid.nmi_category_shuffled
    table = grid.pivot_table(index=["rep", "method"], columns="run",
                             values=["f1_gain", "nmi_gain", "task_nmi",
                                     "model_nmi"])
    order = [label for label, _ in pairs]
    table = table.reindex(columns=pd.MultiIndex.from_product(
        [["f1_gain", "nmi_gain", "task_nmi", "model_nmi"], order]))
    pd.set_option("display.width", 250)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
