"""Contrast how often each SAE feature fires between two sets of steps.

    cd sae_eda && PYTHONPATH=. uv run python explore/feature_contrast.py exploit
    cd sae_eda && PYTHONPATH=. uv run python explore/feature_contrast.py model

exploit: steps AI.FILTER kept vs steps it dropped (all models).
model:   among kept steps, one model vs the other two.
For each top feature, prints its fire rates and its strongest steps, one per
task, so the feature can be named by reading them.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import representations as reps
from eda_data import ROOT

TW = ROOT / "results" / "terminal_wrench"
STEP_RUN = TW / "step-max-targeted-all-keep100-20b9eb"
FILTER = TW / "filter-exploit-all-d69ef0"
TOP, EXAMPLES, CHARS = 20, 5, 320


def load():
    steps = pq.read_table(ROOT / "inputs" / "terminal_wrench" / "step"
                          / "docs.parquet").to_pandas()
    kept = set(pq.read_table(FILTER / "kept.parquet").column("id").to_pylist())
    steps["kept"] = steps.id.isin(kept)
    m = reps.sae_matrix(reps.load_features(STEP_RUN),
                        pa.array(steps.id.tolist())).tocsc()
    return steps, m


def rates(m, mask):
    fired = np.asarray((m[mask] > 0).sum(axis=0)).ravel()
    return fired, fired / mask.sum()


def task_count(m, mask, groups, f):
    rows = m[:, f].nonzero()[0]
    rows = rows[mask[rows]]
    return len(set(groups[rows]))


def show(steps, m, mask, f, header):
    col = m[:, f].toarray().ravel()
    order = np.argsort(-col)
    seen, picked = set(), []
    for r in order:
        if not mask[r] or col[r] == 0:
            continue
        if steps.group_id.iat[r] in seen:
            continue
        seen.add(steps.group_id.iat[r])
        picked.append(r)
        if len(picked) == EXAMPLES:
            break
    print(f"\n### feature {f}: {header}")
    for r in picked:
        text = " ".join(steps.body.iat[r].split())[:CHARS]
        print(f"  [{steps.id.iat[r]} | {col[r]:.1f}] {text}")


def contrast(steps, m, a, b, label, min_a, min_tasks):
    groups = steps.group_id.to_numpy()
    na, ra = rates(m, a)
    _, rb = rates(m, b)
    ratio = (ra + 1e-3) / (rb + 1e-3)
    ok = na >= min_a
    cand = [f for f in np.argsort(-ratio) if ok[f]]
    shown = 0
    for f in cand:
        t = task_count(m, a, groups, f)
        if t < min_tasks:
            continue
        show(steps, m, a, f, f"{label} {ra[f]:.1%} vs {rb[f]:.1%} "
             f"(x{ratio[f]:.1f}), {na[f]} steps, {t} tasks")
        shown += 1
        if shown == TOP:
            break


def main():
    steps, m = load()
    kept = steps.kept.to_numpy()
    if sys.argv[1] == "exploit":
        print(f"kept {kept.sum()} vs dropped {(~kept).sum()}")
        contrast(steps, m, kept, ~kept, "kept vs dropped", 80, 30)
    else:
        for model in sorted(steps.source_model.unique()):
            is_m = (steps.source_model == model).to_numpy()
            print(f"\n======== {model}: {(kept & is_m).sum()} kept steps")
            contrast(steps, m, kept & is_m, kept & ~is_m, model, 60, 20)


if __name__ == "__main__":
    main()
