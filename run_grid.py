"""Run the representation x method grid on one SAE extraction run.

    uv run python run_grid.py --dataset terminal_wrench \
      --sae-run results/terminal_wrench/<run id> \
      [--emb results/terminal_wrench/embed-trace] [--k 25] [--seed 0]

Documents are the rows of inputs/<dataset>/<unit>/docs.parquet that the
SAE run read (all of them, or the pilot's sample). Writes to
<sae run>/eda/:
  (eda-pooled/ with --pool-to-trace, eda-keep<N>/ with --keep, eda[-keep<N>]-center-<task|model|task_model>/ with --center)
  grid.parquet         one row per cell: rep, method, k, seconds, metrics
  groups.json          the top 50 top-3 groups of sae, sae_idf, and tfidf,
                       with counts, category shares, and example documents
  feature_auc.json     the 3 SAE features that best rank each category
  outliers.json        the 20 most unusual documents per outlier method
  assignments.parquet  each hard cell's label per document
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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.preprocessing import normalize

import metrics
import representations as reps
import top_k
from cluster import CLUSTERERS
from eda_data import ROOT
from outliers import OUTLIERS
from topics import TOPIC_MODELS, bertopic

TOP_GROUPS = 50


def snippet(body: str, chars: int = 400) -> str:
    """The end of a trace, where the exploit is usually described."""
    return body[-chars:].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="terminal_wrench")
    parser.add_argument("--sae-run", type=Path, required=True)
    parser.add_argument("--emb", type=Path)
    parser.add_argument("--k", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep", type=int, default=0,
                        help="keep each document's N strongest features "
                             "(max pooling returns ~18k per trace; 0 keeps all)")
    parser.add_argument("--pool-to-trace", action="store_true",
                        help="for a step-level run: max-pool each trace's "
                             "step features and score traces")
    parser.add_argument("--center", choices=("task", "model", "task_model"),
                        help="remove each task's, model's, or (task, model) "
                             "pair's mean from every representation first")
    args = parser.parse_args()

    meta = json.loads((args.sae_run / "meta.json").read_text())
    features = reps.load_features(args.sae_run)
    inputs = ROOT / "inputs" / args.dataset
    if args.pool_to_trace:
        steps = pq.read_table(inputs / "step" / "docs.parquet")
        steps = steps.filter(pc.is_in(
            steps.column("id"), value_set=pc.unique(features.column("id"))))
        docs = pq.read_table(inputs / "trace" / "docs.parquet")
        docs = docs.filter(pc.is_in(
            docs.column("id"), value_set=pc.unique(steps.column("parent_id"))))
    else:
        docs = pq.read_table(inputs / meta["unit"] / "docs.parquet")
        docs = docs.filter(pc.is_in(
            docs.column("id"), value_set=pc.unique(features.column("id"))))
    ids = docs.column("id").to_pylist()
    bodies = docs.column("body").to_pylist()
    truth = metrics.Labels(docs, args.seed)
    print(f"{len(ids)} documents, {features.num_rows} feature rows")

    if args.pool_to_trace:
        strengths = reps.max_pool(
            reps.sae_matrix(features, steps.column("id").combine_chunks()),
            steps.column("parent_id").to_pylist(), ids)
    else:
        strengths = reps.sae_matrix(features, docs.column("id").combine_chunks())
    if args.keep:
        strengths = strengths.multiply(
            top_k.top_k_membership(strengths, args.keep)).tocsr()
    table = reps.sae_reps(strengths, args.seed)
    table.update(reps.tfidf_reps(bodies, args.seed))
    if args.emb:
        table["emb"] = reps.emb_rep(args.emb, ids)
    if args.center:
        groups = {"task": truth.task, "model": truth.model,
                  "task_model": np.char.add(np.char.add(truth.task, "|"),
                                            truth.model)}[args.center]
        table = reps.centered_reps(table, groups, args.seed)

    k, seed = args.k, args.seed
    rows, assignments, groups, outliers = [], {}, {}, {}

    def record(rep: str, method: str, run):
        started = time.perf_counter()
        try:
            out = run()
            status = "ok"
        except Exception:
            out, status = {}, traceback.format_exc(limit=2)
            print(f"  FAILED {rep} / {method}: {status.splitlines()[-1]}")
        rows.append({"rep": rep, "method": method, "k": k, "seed": seed,
                     "seconds": time.perf_counter() - started,
                     "status": status, **out})
        print(f"  {rep:10s} {method:14s} {rows[-1]['seconds']:6.1f}s "
              f"f1={out.get('category_f1', float('nan')):.3f} "
              f"task_purity={out.get('task_purity', float('nan')):.3f}")

    def hard(rep: str, method: str, fit):
        def run():
            labels = np.asarray(fit())
            assignments[f"{rep}/{method}"] = labels
            return metrics.hard_metrics(labels, truth)
        record(rep, method, run)

    def top3(rep: str, method: str, member_fn):
        def run():
            weights = table[rep].matrix
            member = member_fn(weights)
            ranked = top_k.ranked_groups(member, TOP_GROUPS)
            scores = weights[:, ranked].multiply(member[:, ranked]).toarray()
            groups[f"{rep}/{method}"] = [{
                "column": table[rep].columns[c],
                "docs": int(member[:, c].sum()),
                "categories": Counter(
                    cat for r in member[:, c].nonzero()[0]
                    for cat in docs.column("labels")[r].as_py()).most_common(4),
                "examples": [{"id": ids[r], "task": truth.task[r],
                              "end": snippet(bodies[r])}
                             for r in top_k.examples(member, weights, c,
                                                     list(truth.task))],
            } for c in ranked]
            return metrics.grouping_metrics(member, scores, truth)
        record(rep, method, run)

    def dense(rep: str):
        x = table[rep].matrix
        return x if not table[rep].sparse else normalize(x)

    print("SAE representations")
    for rep in ("sae", "sae_idf"):
        top3(rep, "top3", lambda w: top_k.top_k_membership(w, 3))
    top3("sae_idf", "threshold", top_k.threshold_membership)
    for rep in ("sae", "sae_bin", "sae_idf", "sae_svd"):
        for name, (fn, needs_dense, _) in CLUSTERERS.items():
            if needs_dense and table[rep].sparse:
                continue
            hard(rep, name, lambda fn=fn, rep=rep: fn(dense(rep), k, seed))
    for name, (fn, rep) in TOPIC_MODELS.items():
        def run(fn=fn, rep=rep):
            weights, _ = fn(table[rep].matrix, k, seed)
            labels = weights.argmax(axis=1)
            assignments[f"{rep}/{name}"] = labels
            out = metrics.hard_metrics(labels, truth)
            out.update({key: v for key, v in metrics.grouping_metrics(
                metrics.one_hot(labels), weights, truth).items()
                if key.startswith("category_auc")})
            return out
        record(rep, name, run)
    hard("sae_svd", "bertopic",
         lambda: bertopic(bodies, table["sae_svd"].matrix, seed)[0])
    for name, fn in OUTLIERS.items():
        def run(fn=fn, name=name):
            score = fn(table["sae_svd"].matrix, seed)
            outliers[f"sae_svd/{name}"] = [ids[r] for r in np.argsort(-score)[:20]]
            return metrics.outlier_metrics(score, truth)
        record("sae_svd", name, run)

    print("Baselines")
    top3("tfidf", "top3", lambda w: top_k.top_k_membership(w, 3))
    hard("tfidf", "kmeans", lambda: CLUSTERERS["kmeans"][0](
        table["tfidf"].matrix, k, seed))
    hard("tfidf_svd", "hdbscan", lambda: CLUSTERERS["hdbscan"][0](
        table["tfidf_svd"].matrix, k, seed))
    if "emb" in table:
        hard("emb", "kmeans", lambda: CLUSTERERS["kmeans"][0](
            table["emb"].matrix, k, seed))
        hard("emb", "hdbscan", lambda: CLUSTERERS["hdbscan"][0](
            table["emb"].matrix, k, seed))

    out = args.sae_run / ("eda" + ("-pooled" if args.pool_to_trace else "")
                          + (f"-keep{args.keep}" if args.keep else "")
                          + (f"-center-{args.center}" if args.center else ""))
    out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_parquet(out / "grid.parquet")
    (out / "groups.json").write_text(json.dumps(groups, indent=1))
    (out / "outliers.json").write_text(json.dumps(outliers, indent=1))
    (out / "feature_auc.json").write_text(json.dumps(
        metrics.feature_auc(table["sae_idf"].matrix, truth), indent=1))
    pq.write_table(pa.table({"id": ids, **{key: pa.array(v.astype(np.int32))
                                           for key, v in assignments.items()}}),
                   out / "assignments.parquet")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
