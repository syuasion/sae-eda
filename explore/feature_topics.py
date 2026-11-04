"""Each SAE feature as a topic: the exploit steps it fires on.

    PYTHONPATH=. uv run python explore/feature_topics.py prep
    uv run modal run --detach quality_modal.py --trials-file pairs_features.parquet
    PYTHONPATH=. uv run python explore/feature_topics.py eval <quality run dir>

No clustering. Steps are binarized as in Part 2 (a feature fires on a step if
it is in the step's top 100). Each feature is ranked by lift,

    lift = (exploit fire rate + smoothing) / (non-exploit fire rate + smoothing)

which favors specific features, unlike Part 2's difference. A feature is kept
only if it fires on 30 to 500 exploit steps, from at least 10 tasks, with all
three models among them. Features whose exploit steps overlap an already kept
feature's by Jaccard above 0.5 are skipped as duplicates. The top TOPICS are
judged like the trace clusters, on single exploit steps: 3 reference steps from
3 tasks, then up to PAIRS pairs of (another step it fires on, a random exploit
step it does not fire on), all from other tasks. Random groups of exploit
steps of matching size are the control.
Writes results/terminal_wrench/feature_topics/ and writeup/feature_topics.json.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import representations as reps
from eda_data import ROOT

TW = ROOT / "results" / "terminal_wrench"
STEP_RUN = TW / "step-max-targeted-all-keep100-20b9eb"
FILTER = TW / "filter-exploit-all-d69ef0"
OUT = TW / "feature_topics"
PAIRS_FILE = ROOT / "inputs" / "terminal_wrench" / "quality" / "pairs_features.parquet"
VIZ = ROOT / "writeup" / "feature_topics.json"
TOPICS, PAIRS, REFS, CLIP = 15, 20, 3, 1_200
MIN_STEPS, MAX_STEPS, MIN_TASKS, MAX_JACCARD = 30, 500, 10, 0.5

PAIR = ("Each item below is one step of a terminal-agent run where the agent games "
        "the task's verifier (its message and commands, without output). In every "
        "run the agent was asked to make the verifier pass without doing the "
        "intended work.\n\nReference steps, grouped together:\n\n{refs}\n\n"
        "Candidate A:\n\n{a}\n\nCandidate B:\n\n{b}\n\nOne candidate was grouped "
        "with the reference steps and one was not. Which candidate uses the same "
        "specific way of gaming the verifier as the reference steps?")


def clip(text: str, n: int = CLIP) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + " [...]"


def load():
    steps = pq.read_table(ROOT / "inputs" / "terminal_wrench" / "step" / "docs.parquet").to_pandas()
    kept = set(pq.read_table(FILTER / "kept.parquet").column("id").to_pylist())
    steps["exploit"] = steps.id.isin(kept)
    strengths = reps.sae_matrix(reps.load_features(STEP_RUN), pa.array(steps.id.tolist())).tocsc()
    return steps, strengths


def rank(steps, strengths):
    ex = steps.exploit.to_numpy()
    binary = strengths.copy()
    binary.data[:] = 1.0
    e = np.asarray(binary[ex].sum(0)).ravel()
    o = np.asarray(binary[~ex].sum(0)).ravel()
    ne, no = ex.sum(), (~ex).sum()
    lift = ((e + 1) / (ne + 2)) / ((o + 1) / (no + 2))
    ex_idx = np.flatnonzero(ex)
    tasks, models = steps.group_id.to_numpy(), steps.source_model.to_numpy()
    kept, sets = [], []
    for f in np.argsort(-lift):
        if not MIN_STEPS <= e[f] <= MAX_STEPS:
            continue
        rows = binary[:, f].nonzero()[0]
        rows = rows[ex[rows]]
        if len(set(tasks[rows])) < MIN_TASKS or len(set(models[rows])) < 3:
            continue
        s = set(rows.tolist())
        if any(len(s & t) / len(s | t) > MAX_JACCARD for t in sets):
            continue
        kept.append({"feature": int(f), "rows": rows, "exploit": int(e[f]), "other": int(o[f]),
                     "exploit_rate": e[f] / ne, "other_rate": o[f] / no, "lift": float(lift[f])})
        sets.append(s)
        if len(kept) == TOPICS:
            break
    return kept, ex_idx


def pair_rows(name, rows, steps, ex_idx, rng):
    tasks = steps.group_id.to_numpy()
    order = rng.permutation(rows)
    refs, ref_tasks = [], set()
    for r in order:
        if tasks[r] not in ref_tasks:
            refs.append(r)
            ref_tasks.add(tasks[r])
        if len(refs) == REFS:
            break
    inside = set(rows.tolist())
    members = [r for r in order if r not in refs and tasks[r] not in ref_tasks][:PAIRS]
    outsiders = [r for r in rng.permutation(ex_idx) if r not in inside and tasks[r] not in ref_tasks][:PAIRS]
    body = steps.body.to_numpy()
    ref_text = "\n\n".join(f"R{i + 1}: {clip(body[r])}" for i, r in enumerate(refs))
    out = []
    for i, (m, o) in enumerate(zip(members, outsiders)):
        a, b, ans = (m, o, "A") if rng.random() < 0.5 else (o, m, "B")
        out.append({"id": f"{name}|{i}", "group": name, "answer": ans,
                    "prompt": PAIR.format(refs=ref_text, a=clip(body[a]), b=clip(body[b]))})
    return out


def prep():
    steps, strengths = load()
    kept, ex_idx = rank(steps, strengths)
    rng = np.random.default_rng(0)
    rows = []
    for k in kept:
        rows += pair_rows(f"f{k['feature']}", k["rows"], steps, ex_idx, rng)
    # control: random exploit-step groups of the same sizes
    for i, k in enumerate(kept):
        rows += pair_rows(f"random{i}", rng.choice(ex_idx, len(k["rows"]), replace=False), steps, ex_idx, rng)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(PAIRS_FILE)
    pd.DataFrame([{k: v for k, v in x.items() if k != "rows"} for x in kept]).to_csv(OUT / "features.csv", index=False)
    print(pd.read_csv(OUT / "features.csv").round(4).to_string())
    print(f"{len(rows)} pair trials -> {PAIRS_FILE}")


def evaluate(run: Path):
    pairs = pd.read_parquet(PAIRS_FILE)
    p = {a["t.id"]: dict(a["match_probabilities"]) for a in pq.read_table(run / "match.parquet").to_pylist()}
    pairs["p"] = [p[i].get(ans, 0.0) for i, ans in zip(pairs.id, pairs.answer)]
    acc = pairs.groupby("group").p.agg(["mean", "size"])
    feat = acc[acc.index.str.startswith("f")]
    rand = acc[acc.index.str.startswith("random")]
    se = lambda s: s.std() / np.sqrt(len(s))
    print(f"features: {feat['mean'].mean():.3f} ± {se(feat['mean']):.3f} over {len(feat)}")
    print(f"random groups: {rand['mean'].mean():.3f} ± {se(rand['mean']):.3f} over {len(rand)}")
    acc.round(3).to_csv(OUT / "judge.csv")
    export(acc)


def export(acc):
    steps, strengths = load()
    kept, _ = rank(steps, strengths)
    labels = pq.read_table(ROOT.parent / "terminal_wrench" / "inputs" / "labels.parquet").to_pandas().set_index("id")
    base = collections.Counter(c for cs in labels.observed_categories for c in cs)
    names = json.loads((OUT / "names.json").read_text()) if (OUT / "names.json").exists() else {}
    feats = []
    for k in kept:
        rows = k["rows"]
        sub = steps.iloc[rows]
        traces = sub.parent_id.unique()
        cats = collections.Counter(x for t in traces for x in labels.loc[t, "observed_categories"])
        lift = sorted(((c, cats[c] / len(traces), cats[c] / len(traces) / (base[c] / len(labels))) for c in cats),
                      key=lambda z: -z[2])[:3]
        col = strengths[:, k["feature"]].toarray().ravel()
        examples, seen = [], set()
        for r in rows[np.argsort(-col[rows])]:
            if steps.group_id.iat[r] in seen:
                continue
            seen.add(steps.group_id.iat[r])
            examples.append({"task": steps.group_id.iat[r], "model": steps.source_model.iat[r],
                             "text": clip(steps.body.iat[r], 420)})
            if len(examples) == 6:
                break
        n = names.get(str(k["feature"]), {})
        feats.append({"feature": k["feature"], "steps": len(rows), "traces": len(traces),
                      "tasks": int(sub.group_id.nunique()), "models": sub.source_model.value_counts().to_dict(),
                      "exploit": k["exploit"], "other": k["other"],
                      "exploit_rate": round(k["exploit_rate"], 4), "other_rate": round(k["other_rate"], 4),
                      "lift": round(k["lift"], 1), "judge": round(float(acc.loc[f"f{k['feature']}", "mean"]), 2),
                      "pairs": int(acc.loc[f"f{k['feature']}", "size"]),
                      "name": n.get("name", ""), "why": n.get("why", ""),
                      "categories": [{"name": c, "share": round(s, 3), "lift": round(l, 1)} for c, s, l in lift],
                      "examples": examples})
    rand = acc[acc.index.str.startswith("random")]["mean"]
    VIZ.write_text(json.dumps({"n_exploit": int(steps.exploit.sum()), "n_other": int((~steps.exploit).sum()),
                               "random": round(float(rand.mean()), 3),
                               "random_se": round(float(rand.std() / np.sqrt(len(rand))), 3),
                               "features": feats}, ensure_ascii=False, indent=1))
    print(f"wrote {VIZ}")


if __name__ == "__main__":
    if sys.argv[1] == "prep":
        prep()
    else:
        evaluate(Path(sys.argv[2]))
