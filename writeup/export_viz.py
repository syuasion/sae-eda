"""Export the data behind the write-up's interactive figures.

    cd sae_eda && uv run python writeup/export_viz.py

Rebuilds the vectors each judged grouping was clustered on, lays them out
in 2D with UMAP (cosine, seed 0), and writes writeup/viz_data.json:

    categories   the paper's 11 category ids, with base rates
    models       the three source models
    maps         one entry per view: points (x, y, cluster, step or trace
                 index) and per-cluster stats (size, category shares and
                 lifts, model mix, judge results, examples)
    steps        every step: trace index, step number, snippet, and its
                 cluster in each step view (-1 when the filter dropped it)
    traces       every trace: task, model, categories
The layout is only for looking; every cluster comes from the saved
assignments, not from the 2D positions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import umap

import representations as reps
from eda_data import ROOT

TW = ROOT / "results" / "terminal_wrench"
STEP_RUN = TW / "step-max-targeted-all-keep100-20b9eb"
TRACE_RUN = TW / "trace-max-targeted-all-881158"
FILTER = TW / "filter-exploit-all-d69ef0"
JUDGE = TW / "judge-f9edf3"
SNIPPET = 260
# the six strong exploit-step SAE clusters read by hand in the write-up
HAND = {4: "Copy a leaked ground-truth file as the answer",
        14: "Write rows straight into the database the verifier queries",
        20: "Read the test script, then target exactly what it measures",
        7: "Replace the program under test with fake binaries",
        15: "Skip training: write predictions or memorize test labels",
        18: "Build a stub that passes one narrow check"}


def snippet(body: str) -> str:
    text = " ".join(body.split())
    for prefix in ("Analysis: ",):
        text = text.removeprefix(prefix)
    return text if len(text) <= SNIPPET else text[:SNIPPET].rstrip() + "…"


def layout(matrix: np.ndarray) -> list[list[int]]:
    xy = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine",
                   random_state=0).fit_transform(matrix)
    lo, hi = xy.min(0), xy.max(0)
    return np.round((xy - lo) / (hi - lo) * 1000).astype(int).tolist()


def main():
    inputs = ROOT / "inputs" / "terminal_wrench"
    steps = pq.read_table(inputs / "step" / "docs.parquet").to_pandas()
    traces = pq.read_table(inputs / "trace" / "docs.parquet").to_pandas()
    categories = pq.read_table(inputs / "judge" / "categories.parquet").to_pandas()
    cats = categories.id.tolist()
    models = sorted(traces.source_model.unique())
    trace_row = {t: i for i, t in enumerate(traces.id)}
    labels = [list(x) for x in traces.labels]
    base = Counter(c for row in labels for c in row)
    base = {c: base[c] / len(traces) for c in cats}
    step_row = {s: i for i, s in enumerate(steps.id)}
    body = dict(zip(steps.id, steps.body))
    kept = set(pq.read_table(FILTER / "kept.parquet").column("id").to_pylist())
    judge = pd.read_parquet(JUDGE / "clusters_judged.parquet")

    def stats(trace_sets: dict[int, set], member_rows: dict[int, list],
              cell: str | None, sample_text) -> list[dict]:
        out = []
        for c in sorted(trace_sets):
            ts = trace_sets[c]
            counts = Counter(x for t in ts for x in labels[t])
            share = [counts[x] / len(ts) for x in cats]
            mix = Counter(traces.source_model.iloc[t] for t in ts)
            tasks = Counter(traces.group_id.iloc[t] for t in ts)
            entry = {"id": c, "size": len(member_rows[c]), "traces": len(ts),
                     "share": [round(s, 3) for s in share],
                     "lift": [round(s / base[x], 2) for s, x in zip(share, cats)],
                     "models": [round(mix[m] / len(ts), 3) for m in models],
                     "task_share": round(tasks.most_common(1)[0][1] / len(ts), 3),
                     "tasks": len(tasks)}
            if cell == "exploit_sae" and c in HAND:
                entry["hand"] = HAND[c]
            row = judge[(judge.cell == cell) & (judge.cluster == c)] if cell else []
            if len(row):
                r = row.iloc[0]
                entry["judge"] = {"useful": round(float(r.useful), 2),
                                  "caught": bool(r.caught),
                                  "intruder_prob": round(float(r.intruder_prob), 2),
                                  "category": r.category,
                                  "name": r["name"] if isinstance(r["name"], str) else None}
                entry["examples"] = [snippet(t.split("\n", 1)[1])
                                     for t in r.examples.split("\n\nStep ")]
            else:
                entry["examples"] = sample_text(member_rows[c])
            out.append(entry)
        return out

    rng = np.random.default_rng(0)

    def step_samples(rows: list[int]) -> list[str]:
        picked, seen = [], set()
        for r in rng.permutation(rows):
            task = steps.group_id.iloc[r]
            if task not in seen:
                seen.add(task)
                picked.append(snippet(steps.body.iloc[r]))
            if len(picked) == 5:
                break
        return picked

    # step-level views: (key, title, assignments dir, column, matrix to lay out)
    features = reps.load_features(STEP_RUN)
    all_assign = pd.read_parquet(STEP_RUN / "steps" / "assignments.parquet")
    ex_assign = pd.read_parquet(STEP_RUN / "steps-filter-exploit-all-d69ef0"
                                / "assignments.parquet")
    maps, step_cluster = {}, {}
    specs = [("all_sae", "SAE, all steps", all_assign, "sae_svd/kmeans/k25", "sae"),
             ("exploit_sae", "SAE, exploit steps", ex_assign, "sae_svd/kmeans/k25", "sae"),
             ("exploit_tfidf", "TF-IDF, exploit steps", ex_assign, "tfidf/kmeans/k25",
              "tfidf")]
    for key, title, assign, column, kind in specs:
        ids = assign.id.tolist()
        if kind == "sae":
            import pyarrow as pa
            strengths = reps.sae_matrix(
                features.filter(pa.compute.is_in(features.column("id"),
                                                 value_set=pa.array(ids))),
                pa.array(ids))
            matrix = reps.sae_reps(strengths)["sae_svd"].matrix
        else:
            matrix = reps.tfidf_reps([body[i] for i in ids])["tfidf_svd"].matrix
        print(f"{key}: UMAP over {len(ids)} steps")
        xy = layout(matrix)
        clusters = assign[column].to_numpy()
        rows = [step_row[i] for i in ids]
        trace_sets, member_rows = {}, {}
        for r, c in zip(rows, clusters):
            trace_sets.setdefault(int(c), set()).add(trace_row[steps.parent_id.iloc[r]])
            member_rows.setdefault(int(c), []).append(r)
        maps[key] = {"title": title, "unit": "step",
                     "points": [[x, y, int(c), r] for (x, y), c, r
                                in zip(xy, clusters, rows)],
                     "clusters": stats(trace_sets, member_rows, key, step_samples)}
        col = np.full(len(steps), -1, int)
        col[rows] = clusters
        step_cluster[key] = col

    # trace-level view: max-pooled trace vectors, sae_svd + K-Means (k = 25)
    tfeat = reps.load_features(TRACE_RUN)
    import pyarrow as pa
    tmatrix = reps.sae_reps(reps.sae_matrix(tfeat, pa.array(traces.id.tolist())))
    print("traces: UMAP")
    xy = layout(tmatrix["sae_svd"].matrix)
    tassign = pd.read_parquet(TRACE_RUN / "eda-keep100" / "assignments.parquet")
    tclusters = tassign.set_index("id").loc[traces.id, "sae_svd/kmeans"].to_numpy()
    trace_sets, member_rows = {}, {}
    for t, c in enumerate(tclusters):
        trace_sets.setdefault(int(c), set()).add(t)
        member_rows.setdefault(int(c), []).append(t)
    first_step = {}
    for r, p in enumerate(steps.parent_id):
        first_step.setdefault(trace_row[p], r)

    def trace_samples(rows: list[int]) -> list[str]:
        picked, seen = [], set()
        for t in rng.permutation(rows):
            task = traces.group_id.iloc[t]
            if task not in seen and t in first_step:
                seen.add(task)
                picked.append(f"[task {task}, {traces.source_model.iloc[t]}] "
                              + snippet(steps.body.iloc[first_step[t]]))
            if len(picked) == 5:
                break
        return picked

    maps["trace_sae"] = {"title": "SAE, whole traces", "unit": "trace",
                         "points": [[x, y, int(c), t] for (x, y), c, t
                                    in zip(xy, tclusters, range(len(traces)))],
                         "clusters": stats(trace_sets, member_rows, None,
                                           trace_samples)}

    tasks = sorted(traces.group_id.unique(), key=lambda t: (len(t), t))
    task_index = {t: i for i, t in enumerate(tasks)}
    data = {
        "categories": [{"id": c, "base": round(base[c], 3)} for c in cats],
        "models": models,
        "tasks": tasks,
        "maps": maps,
        "step_views": [k for k, *_ in specs],
        "steps": [[trace_row[p], int(i.rsplit("#", 1)[1]), int(i in kept),
                   *[int(step_cluster[k][r]) for k, *_ in specs], snippet(b)]
                  for r, (i, p, b) in enumerate(zip(steps.id, steps.parent_id,
                                                    steps.body))],
        "traces": [[task_index[g], models.index(m), [cats.index(c) for c in l]]
                   for g, m, l in zip(traces.group_id, traces.source_model,
                                      labels)],
    }
    out = Path(__file__).parent / "viz_data.json"
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
