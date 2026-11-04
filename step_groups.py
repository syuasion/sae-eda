"""Group steps, then score the groups on traces.

    uv run python step_groups.py --dataset terminal_wrench \
      --sae-run results/terminal_wrench/<step run id> [--keep 100] [--k 25 50] \
      [--steps-filter results/terminal_wrench/<filter run id>]

Steps have no labels of their own (a step inherits its trace's), so each
step grouping is read at trace level: a trace is a member of every group
that any of its steps is in. A group of exploit steps can then match a
category even though the trace's other steps land elsewhere. Soft
groupings (NMF) score a trace by the max weight over its steps.

Methods run on the step vectors: top-3 (sae, sae_idf), K-Means (sae_idf,
sae_svd, and tfidf of the step bodies as the baseline), GMM (sae_svd),
NMF (sae_idf). Agglomerative is left out: its distance matrix over 19k
steps is about 1.5 GB. Writes <sae run>/steps[-keep<N>]/
grid.parquet and groups.json (the top 30 groups of each top-3 and
K-Means cell, with example steps), and assignments.parquet (each step's
cluster in every hard cell, for judge_prep.py).
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp

import metrics
import representations as reps
import top_k
from cluster import CLUSTERERS
from eda_data import ROOT
from topics import TOPIC_MODELS

TOP_GROUPS = 30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="terminal_wrench")
    parser.add_argument("--sae-run", type=Path, required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[25, 50])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=int, default=0)
    parser.add_argument("--steps-filter", type=Path,
                        help="a filter_modal.py run: group only its kept "
                             "steps, and score only their traces")
    args = parser.parse_args()

    features = reps.load_features(args.sae_run)
    inputs = ROOT / "inputs" / args.dataset
    steps = pq.read_table(inputs / "step" / "docs.parquet")
    steps = steps.filter(pc.is_in(steps.column("id"),
                                  value_set=pc.unique(features.column("id"))))
    if args.steps_filter:
        kept = pq.read_table(args.steps_filter / "kept.parquet").column("id")
        steps = steps.filter(pc.is_in(steps.column("id"), value_set=kept))
    traces = pq.read_table(inputs / "trace" / "docs.parquet")
    traces = traces.filter(pc.is_in(
        traces.column("id"), value_set=pc.unique(steps.column("parent_id"))))
    trace_ids = traces.column("id").to_pylist()
    row = {t: i for i, t in enumerate(trace_ids)}
    parent = np.array([row[p] for p in steps.column("parent_id").to_pylist()])
    # traces x steps: 1 where the step belongs to the trace
    owner = sp.csr_matrix((np.ones(len(parent), np.float32),
                           (parent, np.arange(len(parent)))),
                          shape=(len(trace_ids), len(parent)))
    truth = metrics.Labels(traces, args.seed)
    step_ids = steps.column("id").to_pylist()
    bodies = steps.column("body").to_pylist()
    print(f"{len(step_ids)} steps of {len(trace_ids)} traces")

    strengths = reps.sae_matrix(features, steps.column("id").combine_chunks())
    if args.keep:
        strengths = strengths.multiply(
            top_k.top_k_membership(strengths, args.keep)).tocsr()
    table = reps.sae_reps(strengths, args.seed)
    table.update(reps.tfidf_reps(bodies, args.seed))

    def to_traces(step_member: sp.csr_matrix) -> sp.csr_matrix:
        member = (owner @ step_member).tocsr()
        member.data[:] = 1.0
        return member

    def describe(step_member: sp.csr_matrix, columns: list[str]) -> list[dict]:
        ranked = top_k.ranked_groups(step_member, TOP_GROUPS)
        out = []
        for c in ranked:
            rows = step_member[:, c].nonzero()[0]
            in_traces = np.unique(parent[rows])
            seen, examples = set(), []
            for r in rows:
                task = truth.task[parent[r]]
                if task not in seen and len(examples) < 3:
                    seen.add(task)
                    examples.append({"step": step_ids[r], "task": task,
                                     "text": bodies[r][:400]})
            out.append({
                "column": columns[c], "steps": len(rows),
                "traces": len(in_traces),
                "categories": Counter(
                    cat for t in in_traces
                    for cat in traces.column("labels")[t].as_py()
                ).most_common(4),
                "examples": examples})
        return out

    rows, groups, assignments = [], {}, {}

    def record(rep: str, method: str, k: int, run):
        started = time.perf_counter()
        try:
            out, status = run(), "ok"
        except Exception:
            out, status = {}, traceback.format_exc(limit=2)
            print(f"  FAILED {rep} / {method}: {status.splitlines()[-1]}")
        rows.append({"rep": rep, "method": method, "k": k, "seed": args.seed,
                     "seconds": time.perf_counter() - started,
                     "status": status, **out})
        print(f"  {rep:9s} {method:13s} k={k:<3} {rows[-1]['seconds']:6.1f}s "
              f"f1={out.get('category_f1', float('nan')):.3f} "
              f"shuffled={out.get('category_f1_shuffled', float('nan')):.3f}")

    def hard(rep: str, method: str, k: int, fit, keep_groups: bool = False):
        def run():
            labels = np.asarray(fit())
            assignments[f"{rep}/{method}/k{k}"] = labels.astype(np.int32)
            step_member = metrics.one_hot(labels)
            if keep_groups:
                groups[f"{rep}/{method}/k{k}"] = describe(
                    step_member, [f"cluster {c}" for c in
                                  range(step_member.shape[1])])
            out = metrics.grouping_metrics(to_traces(step_member), None, truth)
            out["step_groups"] = int(step_member.shape[1])
            return out
        record(rep, method, k, run)

    def dense(rep: str):
        x = table[rep].matrix
        return x if not table[rep].sparse else reps.normalize(x)

    for rep in ("sae", "sae_idf", "tfidf"):
        def run(rep=rep):
            step_member = top_k.top_k_membership(table[rep].matrix, 3)
            groups[f"{rep}/top3"] = describe(step_member, table[rep].columns)
            return metrics.grouping_metrics(to_traces(step_member), None, truth)
        record(rep, "top3", 3, run)
    for k in args.k:
        for rep in ("sae_idf", "sae_svd", "tfidf"):
            hard(rep, "kmeans", k, lambda rep=rep, k=k: CLUSTERERS["kmeans"][0](
                dense(rep), k, args.seed), keep_groups=True)
        hard("sae_svd", "gmm", k, lambda k=k: CLUSTERERS["gmm"][0](
            table["sae_svd"].matrix, k, args.seed))

        def nmf(k=k):
            fn, rep = TOPIC_MODELS["nmf"]
            weights, _ = fn(table[rep].matrix, k, args.seed)
            step_member = metrics.one_hot(weights.argmax(axis=1))
            # a trace's score per topic: the max over its steps
            scores = np.zeros((len(trace_ids), weights.shape[1]), np.float32)
            np.maximum.at(scores, parent, weights)
            out = metrics.grouping_metrics(to_traces(step_member), scores, truth)
            out["step_groups"] = int(step_member.shape[1])
            return out
        record("sae_idf", "nmf", k, nmf)

    out = args.sae_run / ("steps" + (f"-keep{args.keep}" if args.keep else "")
                          + (f"-{args.steps_filter.name}" if args.steps_filter
                             else ""))
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "grid.parquet")
    (out / "groups.json").write_text(json.dumps(groups, indent=1))
    pd.DataFrame({"id": step_ids, "trace_id": [trace_ids[p] for p in parent],
                  **assignments}).to_parquet(out / "assignments.parquet")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
