"""Build the LLM judge's inputs from step clusters.

    uv run python judge_prep.py --dataset terminal_wrench \
      --cell exploit_sae=<steps dir>:sae_svd/kmeans/k25 \
      --cell exploit_tfidf=<steps dir>:tfidf/kmeans/k25 ...

Each --cell is label=<step_groups.py output dir>:<assignments column>.
For every cluster it picks EXAMPLES random member steps from different
tasks (random rather than the most central, so the judge sees the whole
cluster, and from different tasks, so a shared task can't make them look
alike), and one intruder: a random step of another cluster of the same
cell, from a task none of the examples share. Writes
inputs/<dataset>/judge/clusters.parquet, one row per cluster:

    id               "<cell>|<cluster>"
    cell, cluster    which grouping and which of its clusters
    traces, steps    the cluster's size
    examples         the EXAMPLES steps as one text, for naming and scoring
    intrusion        the examples plus the intruder, lettered A-F, shuffled
    intruder         the intruder's letter
    top_category     the category most over-represented among its traces,
    top_share        its share of them,
    top_lift         and that share divided by its base rate
and copies the paper's categories to inputs/<dataset>/judge/categories.parquet.
"""

from __future__ import annotations

import argparse
import shutil
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from eda_data import ROOT, TERMINAL_WRENCH

EXAMPLES = 5
STEP_CHARS = 1_000


def clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= STEP_CHARS else text[:STEP_CHARS] + " [...]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="terminal_wrench")
    parser.add_argument("--cell", action="append", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inputs = ROOT / "inputs" / args.dataset
    steps = pq.read_table(inputs / "step" / "docs.parquet").to_pandas()
    body = dict(zip(steps.id, steps.body))
    traces = pq.read_table(inputs / "trace" / "docs.parquet").to_pandas()
    task = dict(zip(traces.id, traces.group_id))
    labels = dict(zip(traces.id, traces.labels.map(list)))
    base = Counter(c for row in labels.values() for c in row)
    base = {c: n / len(labels) for c, n in base.items()}
    rng = np.random.default_rng(args.seed)

    rows = []
    for spec in args.cell:
        cell, rest = spec.split("=", 1)
        directory, column = rest.split(":", 1)
        frame = pd.read_parquet(Path(directory) / "assignments.parquet",
                                columns=["id", "trace_id", column])
        frame["task"] = frame.trace_id.map(task)
        for cluster, members in frame.groupby(column):
            order = members.sample(frac=1, random_state=rng.integers(1 << 31))
            picked = order.drop_duplicates("task").head(EXAMPLES)
            if len(picked) < EXAMPLES:
                continue
            others = frame[(frame[column] != cluster)
                           & ~frame.task.isin(picked.task)]
            intruder = others.iloc[rng.integers(len(others))]
            texts = [clip(body[i]) for i in picked.id]
            letters = string.ascii_uppercase[:EXAMPLES + 1]
            slots = rng.permutation(EXAMPLES + 1)
            lettered = [None] * (EXAMPLES + 1)
            for text, slot in zip(texts + [clip(body[intruder.id])], slots):
                lettered[slot] = text
            in_traces = members.trace_id.unique()
            counts = Counter(c for t in in_traces for c in labels[t])
            top = max(counts, key=lambda c: counts[c] / len(in_traces) / base[c])
            rows.append({
                "id": f"{cell}|{cluster}", "cell": cell, "cluster": int(cluster),
                "traces": len(in_traces), "steps": len(members),
                "examples": "\n\n".join(f"Step {n + 1}:\n{t}"
                                        for n, t in enumerate(texts)),
                "intrusion": "\n\n".join(f"Step {letters[n]}:\n{t}"
                                         for n, t in enumerate(lettered)),
                "intruder": letters[slots[-1]],
                "top_category": top,
                "top_share": counts[top] / len(in_traces),
                "top_lift": counts[top] / len(in_traces) / base[top]})
        print(f"{cell}: {sum(r['cell'] == cell for r in rows)} clusters")

    out = inputs / "judge"
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "clusters.parquet")
    shutil.copy(TERMINAL_WRENCH / "categories.parquet", out / "categories.parquet")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
