"""Cluster whole traces on binarized, max-pooled SAE features.

    PYTHONPATH=. uv run python explore/trace_clusters.py prep
    uv run modal run --detach quality_modal.py --trials-file pairs_traces.parquet
    PYTHONPATH=. uv run python explore/trace_clusters.py eval <quality run dir>
    PYTHONPATH=. uv run python explore/trace_clusters.py pairs 20   # more pairs per cluster
    uv run modal run --detach quality_modal.py --trials-file pairs_traces20.parquet
    PYTHONPATH=. uv run python explore/trace_clusters.py compare 20 <quality run dir>

Add `--pool last` to any command to use the last-token run instead of max
pooling (outputs go to trace_clusters_last/, pairs_traces_last.parquet and
trace_viz_last.json), or `--pool exploit` to use only each trace's exploit
steps: the step run's top-100 features, unioned over the trace's exploit steps.

Input: the trace-max run, where each trace's feature strength is the max
over its tokens and every feature above 0 is kept (about 18k of 32,768 per
trace). Binarized means 1 where the max is above 0, as in Qwen-Scope's
Eq. 13.

Techniques, all K-Means with k = 25 on the 3,632 traces:
    binary        the 0/1 vectors as they are
    binary_idf    each column times log(n / traces it fires on), rows
                  L2-normalized
    binary_idf_top5  each trace keeps only its 5 highest-weighted features
                  from binary_idf (its 5 rarest features), set back to 1
    random        a random split, as the control
Metrics per technique: task concentration, stability (adjusted Rand index
between seeds 0 and 1), and judge accuracy. The judge sees each trace as
its exploit steps (kept by AI.FILTER), clipped.
Writes results/terminal_wrench/trace_clusters/ and writeup/trace_viz.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import scipy.sparse as sp
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors

import representations as reps
from eda_data import ROOT
from quality import concentration

TW = ROOT / "results" / "terminal_wrench"
FILTER = TW / "filter-exploit-all-d69ef0"
# pooling -> (SAE run, output dir, judge trials, map data); `--pool last` picks the second
SOURCES = {
    "max": (TW / "trace-max-targeted-all-881158", TW / "trace_clusters",
            ROOT / "inputs" / "terminal_wrench" / "quality" / "pairs_traces.parquet",
            ROOT / "writeup" / "trace_viz.json"),
    "last": (TW / "trace-last-targeted-all-72d415", TW / "trace_clusters_last",
             ROOT / "inputs" / "terminal_wrench" / "quality" / "pairs_traces_last.parquet",
             ROOT / "writeup" / "trace_viz_last.json"),
    # each step max-pooled with its top 100 kept, then only the exploit steps
    # (AI.FILTER) unioned per trace; traces with no exploit step are dropped
    "exploit": (TW / "step-max-targeted-all-keep100-20b9eb", TW / "trace_clusters_exploit",
                ROOT / "inputs" / "terminal_wrench" / "quality" / "pairs_traces_exploit.parquet",
                ROOT / "writeup" / "trace_viz_exploit.json"),
}
POOL = "max"
TRACE_RUN, OUT, PAIRS, VIZ = SOURCES[POOL]
K, REFS, TRIALS, CLIP = 25, 3, 6, 1_200
TITLES = {"binary": "K-Means, binary", "binary_idf": "K-Means, binary × IDF",
          "binary_idf_top5": "K-Means, binary × IDF, top 5",
          "random": "Random split (control)"}
TOP = 5

PAIR = ("Each run below is a terminal-agent run, shown as the steps where the "
        "agent games the task's verifier (its messages and commands, without "
        "output). In every run the agent was asked to make the verifier pass "
        "without doing the intended work.\n\nReference runs, grouped "
        "together:\n\n{refs}\n\nCandidate A:\n\n{a}\n\nCandidate B:\n\n{b}"
        "\n\nOne candidate was grouped with the reference runs and one was "
        "not. Which candidate uses the same specific way of gaming the "
        "verifier as the reference runs?")


def clip(text: str, n: int = CLIP) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " [...]"


def load():
    inputs = ROOT / "inputs" / "terminal_wrench"
    traces = pq.read_table(inputs / "trace" / "docs.parquet").to_pandas()
    steps = pq.read_table(inputs / "step" / "docs.parquet").to_pandas()
    kept = set(pq.read_table(FILTER / "kept.parquet").column("id").to_pylist())
    steps = steps[steps.id.isin(kept)]
    exploit = steps.groupby("parent_id").body.apply(
        lambda b: " || ".join(" ".join(x.split()) for x in b))
    traces["exploit"] = traces.id.map(exploit)
    if POOL == "exploit":
        traces = traces[traces.exploit.notna()].reset_index(drop=True)
        step_ids = pa.array(steps.id.tolist())
        features = reps.load_features(TRACE_RUN).filter(
            pc.is_in(pc.field("id"), value_set=step_ids))
        per_step = reps.sae_matrix(features, step_ids)
        # traces x steps 0/1 membership, so the product counts each feature's exploit steps
        row = {t: i for i, t in enumerate(traces.id)}
        member = sp.csr_matrix((np.ones(len(steps), np.float32),
                                ([row[t] for t in steps.parent_id], np.arange(len(steps)))),
                               shape=(len(traces), len(steps)))
        strengths = (member @ per_step).tocsr()
    else:
        strengths = reps.sae_matrix(reps.load_features(TRACE_RUN),
                                    pa.array(traces.id.tolist()))
    binary = strengths.copy()
    binary.data[:] = 1.0
    return traces, binary


def top_binary(x: sp.csr_matrix, k: int = TOP) -> sp.csr_matrix:
    """Keep each row's k largest entries, set to 1 (ties go to the lower column)."""
    rows, cols = [], []
    for r in range(x.shape[0]):
        lo, hi = x.indptr[r], x.indptr[r + 1]
        idx, val = x.indices[lo:hi], x.data[lo:hi]
        order = np.lexsort((idx, -val))[:k]
        rows += [r] * len(order)
        cols += idx[order].tolist()
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=x.shape)


def pair_rows(name, groups, traces, tasks, judgeable, trials=TRIALS) -> list[dict]:
    """Forced-choice trials per cluster: 3 fixed references, then member vs outsider pairs.

    The first TRIALS pairs use the same random draws whatever `trials` is, so
    a larger run keeps the 6-pair run's references and pairs and adds more.
    """
    rows = []
    trng, extra = np.random.default_rng(1), np.random.default_rng(2)
    for gi, g in enumerate(groups):
        order = [r for r in trng.permutation(g) if judgeable[r]]
        refs, ref_tasks = [], set()
        for r in order:
            if tasks[r] not in ref_tasks:
                refs.append(r)
                ref_tasks.add(tasks[r])
            if len(refs) == REFS:
                break
        if len(refs) < REFS:
            continue
        inside = set(g.tolist())
        members = [r for r in order if r not in refs and tasks[r] not in ref_tasks][:trials]
        outsiders = [r for r in trng.permutation(len(traces)) if judgeable[r]
                     and r not in inside and tasks[r] not in ref_tasks][:trials]
        ref_text = "\n\n".join(f"R{i + 1}: {clip(traces.exploit[r])}"
                               for i, r in enumerate(refs))
        for i, (m, o) in enumerate(zip(members, outsiders)):
            coin = (trng if i < TRIALS else extra).random()
            a_, b_, ans = ((m, o, "A") if coin < 0.5 else (o, m, "B"))
            rows.append({"id": f"{name}|{gi}|{i}", "technique": name, "group": gi,
                         "answer": ans, "prompt": PAIR.format(
                             refs=ref_text, a=clip(traces.exploit[a_]),
                             b=clip(traces.exploit[b_]))})
    return rows


def more_pairs(n: int):
    """Rebuild the trials with up to n pairs per cluster from the saved clusters."""
    traces, _ = load()
    assign = pd.read_parquet(OUT / "assignments.parquet")
    tasks, judgeable = traces.group_id.to_numpy(), traces.exploit.notna().to_numpy()
    rows = []
    for name in TITLES:
        lab = assign[f"{name}_s0"].to_numpy()
        rows += pair_rows(name, [np.flatnonzero(lab == c) for c in range(K)],
                          traces, tasks, judgeable, n)
    out = PAIRS.with_name(f"{PAIRS.stem}{n}.parquet")
    pd.DataFrame(rows).to_parquet(out)
    print(f"{len(rows)} pair trials -> {out}")


def judge_scores(pairs: pd.DataFrame, run: Path) -> pd.Series:
    """Mean probability on the member, per (technique, cluster)."""
    p = {a["t.id"]: dict(a["match_probabilities"])
         for a in pq.read_table(run / "match.parquet").to_pylist()}
    pairs = pairs.assign(p_correct=[p[i].get(ans, 0.0) for i, ans in zip(pairs.id, pairs.answer)])
    return pairs.groupby(["technique", "group"]).p_correct.mean()


def compare(n: int, run: Path):
    """Judge accuracy with up to n pairs vs with the first 6 of them, same run."""
    pairs = pd.read_parquet(PAIRS.with_name(f"{PAIRS.stem}{n}.parquet"))
    pairs["i"] = pairs.id.str.rsplit("|", n=1).str[1].astype(int)
    full, first = judge_scores(pairs, run), judge_scores(pairs[pairs.i < TRIALS], run)
    per = pd.DataFrame({"pairs": pairs.groupby(["technique", "group"]).size(),
                        "acc_6": first, f"acc_{n}": full})
    per.round(3).to_csv(OUT / f"quality_groups_{n}.csv")
    se = lambda s: s.std() / np.sqrt(len(s))
    table = per.groupby("technique").agg(
        clusters=("pairs", "size"), pairs=("pairs", "sum"), median_pairs=("pairs", "median"),
        acc_6=("acc_6", "mean"), se_6=("acc_6", se), **{f"acc_{n}": (f"acc_{n}", "mean"),
                                                        f"se_{n}": (f"acc_{n}", se)})
    table["cluster_corr"] = per.groupby("technique").apply(
        lambda d: d.acc_6.corr(d[f"acc_{n}"]))
    print(table.round(3).to_string())
    table.round(3).to_csv(OUT / f"quality_{n}.csv")


def task_line(body: str) -> str:
    return " ".join(body.split("\n\n", 2)[1].split())[:160] if "\n\n" in body else ""


def prep():
    traces, binary = load()
    OUT.mkdir(parents=True, exist_ok=True)
    fired = np.asarray(binary.sum(1)).ravel()
    idf = reps.idf_weight(binary)
    vectors = {"binary": binary, "binary_idf": idf, "binary_idf_top5": top_binary(idf)}
    rng = np.random.default_rng(0)
    labels = {}
    for name, x in vectors.items():
        labels[name] = [KMeans(K, n_init=3, random_state=s).fit_predict(x) for s in (0, 1)]
        print(name, "done", flush=True)
    labels["random"] = [rng.integers(0, K, len(traces)) for _ in range(2)]

    tasks = traces.group_id.to_numpy()
    cpu, rows = [], []
    judgeable = traces.exploit.notna().to_numpy()
    for name, (a, b) in labels.items():
        groups = [np.flatnonzero(a == c) for c in range(K)]
        sizes = np.array([len(g) for g in groups])
        # how much of the variance in features-fired the clusters explain
        means = np.array([fired[g].mean() if len(g) else 0 for g in groups])
        r2 = 1 - sum(((fired[g] - means[c]) ** 2).sum() for c, g in enumerate(groups)) \
            / ((fired - fired.mean()) ** 2).sum()
        cpu.append({"technique": name, "largest": int(sizes.max()),
                    "smallest": int(sizes.min()), "median": int(np.median(sizes)),
                    "task_conc": concentration(groups, tasks),
                    "stability": round(adjusted_rand_score(a, b), 3),
                    "fired_r2": round(float(r2), 3)})
        rows += pair_rows(name, groups, traces, tasks, judgeable)
    pd.DataFrame({"id": traces.id, **{f"{n}_s{s}": labels[n][s] for n in labels for s in (0, 1)},
                  "fired": fired}).to_parquet(OUT / "assignments.parquet")
    cpu = pd.DataFrame(cpu)
    cpu.to_csv(OUT / "cpu_metrics.csv", index=False)
    pd.DataFrame(rows).to_parquet(PAIRS)
    print(cpu.to_string())
    print(f"fired per trace: median {np.median(fired):.0f} of {binary.shape[1]}; "
          f"traces with exploit steps: {judgeable.sum()} of {len(traces)}")
    print(f"{len(rows)} pair trials -> {PAIRS}")


def evaluate(run: Path):
    pairs = pd.read_parquet(PAIRS)
    p = {a["t.id"]: dict(a["match_probabilities"])
         for a in pq.read_table(run / "match.parquet").to_pylist()}
    pairs["p_correct"] = [p[i].get(ans, 0.0) for i, ans in zip(pairs.id, pairs.answer)]
    per_group = pairs.groupby(["technique", "group"]).p_correct.mean()
    summary = per_group.groupby("technique").agg(
        accuracy="mean", acc_se=lambda s: s.std() / np.sqrt(len(s)), groups="count")
    table = pd.read_csv(OUT / "cpu_metrics.csv").set_index("technique").join(summary.round(3))
    print(table.to_string())
    table.to_csv(OUT / "quality.csv")
    per_group.round(3).to_csv(OUT / "quality_groups.csv")
    export(per_group)


def trace_rows(traces: pd.DataFrame) -> list[list[str]]:
    """[task, model, exploit-step snippet] per trace, in point order, for the cluster drawer."""
    def opening(body: str) -> str:
        text = " ".join(body.removeprefix("TASK").split("STEP ")[0].split())
        return text[:150] + ("…" if len(text) > 150 else "")
    return [[opening(b), m, clip(e, 360) if isinstance(e, str) else ""]
            for b, m, e in zip(traces.body, traces.source_model, traces.exploit)]


def export(per_group: pd.Series):
    """Map data: a UMAP layout per technique (only for looking) and cluster stats."""
    traces, binary = load()
    assign = pd.read_parquet(OUT / "assignments.parquet")
    tasks = traces.group_id.to_numpy()
    rng = np.random.default_rng(0)
    views = {}
    idf = reps.idf_weight(binary)
    for name, x in (("binary", binary), ("binary_idf", idf),
                    ("binary_idf_top5", top_binary(idf))):
        reduced = TruncatedSVD(50, random_state=0).fit_transform(x)
        xy = umap.UMAP(n_neighbors=30, min_dist=0.15, metric="cosine",
                       random_state=0).fit_transform(reduced)
        # scale to the 1st-99th percentile so a few far single-task clusters
        # don't squeeze the rest; the outliers sit on the edge
        lo, hi = np.percentile(xy, 1, 0), np.percentile(xy, 99, 0)
        xy = np.round(np.clip((xy - lo) / (hi - lo), -0.02, 1.02) * 1000).astype(int)
        lab = assign[f"{name}_s0"].to_numpy()
        clusters = []
        for c in range(K):
            g = np.flatnonzero(lab == c)
            counts = pd.Series(tasks[g]).value_counts()
            models = traces.source_model.iloc[g].value_counts()
            ex, seen = [], set()
            for r in rng.permutation(g):
                if tasks[r] in seen or not isinstance(traces.exploit[r], str):
                    continue
                seen.add(tasks[r])
                ex.append({"task": task_line(traces.body[r]),
                           "text": clip(traces.exploit[r], 420)})
                if len(ex) == 5:
                    break
            acc = per_group.get((name, c))
            clusters.append({"id": c, "size": int(len(g)), "tasks": int(len(counts)),
                             "task_share": round(counts.iloc[0] / len(g), 3),
                             "model": models.index[0], "model_share": round(models.iloc[0] / len(g), 3),
                             "fired": int(assign.fired.to_numpy()[g].mean()),
                             "judge": None if acc is None else round(float(acc), 2),
                             "examples": ex})
        # share of each trace's 15 nearest dots in 2D that share its cluster:
        # how faithfully the layout shows the clustering
        near = NearestNeighbors(n_neighbors=16).fit(xy).kneighbors(
            xy, return_distance=False)[:, 1:]
        fired = assign.fired.to_numpy()
        views[name] = {"title": TITLES[name], "agree": round(float((lab[near] == lab[:, None]).mean()), 3),
                       "points": [[int(a), int(b), int(c), int(f)]
                                  for (a, b), c, f in zip(xy, lab, fired)], "clusters": clusters}
    VIZ.write_text(json.dumps({"n": len(traces), "views": views, "traces": trace_rows(traces)},
                              separators=(",", ":")))
    print(f"wrote {VIZ} ({VIZ.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    if "--pool" in sys.argv:
        at = sys.argv.index("--pool")
        POOL = sys.argv[at + 1]
        TRACE_RUN, OUT, PAIRS, VIZ = SOURCES[POOL]
        del sys.argv[at:at + 2]
    if sys.argv[1] == "prep":
        prep()
    elif sys.argv[1] == "pairs":
        more_pairs(int(sys.argv[2]))
    elif sys.argv[1] == "compare":
        compare(int(sys.argv[2]), Path(sys.argv[3]))
    else:
        evaluate(Path(sys.argv[2]))
