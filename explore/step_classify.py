"""Rank step features by how much more often they fire on exploit steps.

    PYTHONPATH=. uv run python explore/step_classify.py

The method of Qwen-Scope section 5.1 (arXiv 2605.11887): binarize each
step (1 where the feature fired, Eq. 13), then for each feature

    delta = (exploit steps it fires on / exploit steps)
          - (other steps it fires on / other steps)            (Eq. 14)

and keep the top K. A step is classified exploit if any kept feature fires
on it (Eq. 15). "Exploit" is the AI.FILTER tag. Our step vectors keep only
each step's top 100 features, so "fires" means "is in the step's top 100".

Features are picked on a random half of the tasks and the classifier is
scored on the other half, as Qwen-Scope picks on a selection split.
Writes writeup/classify.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from eda_data import ROOT
from quality import load

TOP, KS, EXAMPLES, SNIP = 10, (1, 2, 5, 10), 5, 520
OUT = ROOT / "writeup" / "classify.json"
PAIRS = ROOT / "inputs" / "terminal_wrench" / "quality" / "pairs_classify.parquet"
NAME_PAIRS = 8
# my names, from reading each feature's 5 strongest exploit steps
NAMES = {10478: "Plan a shortcut around the intended work",
         17942: "Explain why the bypass should work",
         13235: "Fake a system component so checks pass",
         14658: "Write the expected output directly",
         25329: "Exploit a leak or weak check found in the environment",
         1751: "Report what inspecting the environment found",
         8: "Point out a blind spot in the verifier",
         22096: "Patch a library the checker uses",
         23289: "Pull the answers out of the environment",
         16198: "Analyze issues with the current attempt"}


def deltas(fired, exploit):
    pe = np.asarray(fired[exploit].sum(0)).ravel() / exploit.sum()
    po = np.asarray(fired[~exploit].sum(0)).ravel() / (~exploit).sum()
    return pe, po, pe - po


def scores(pred, truth):
    tp = (pred & truth).sum()
    precision, recall = tp / max(pred.sum(), 1), tp / truth.sum()
    return {"precision": round(float(precision), 3), "recall": round(float(recall), 3),
            "f1": round(float(2 * precision * recall / max(precision + recall, 1e-9)), 3),
            "flagged": int(pred.sum())}


def main():
    steps, strengths = load()
    exploit = steps.kept.to_numpy()
    tasks = steps.group_id.to_numpy()
    fired = (strengths > 0).astype(np.float32).tocsc()

    # held-out classifier: pick on half the tasks, score on the other half
    rng = np.random.default_rng(0)
    pick_tasks = set(rng.permutation(np.unique(tasks))[: len(np.unique(tasks)) // 2])
    pick = np.isin(tasks, list(pick_tasks))
    _, _, d_pick = deltas(fired[pick], exploit[pick])
    order = np.argsort(-d_pick)
    held = []
    for k in KS:
        pred = np.asarray(fired[~pick][:, order[:k]].sum(1)).ravel() > 0
        held.append({"k": k, **scores(pred, exploit[~pick])})
    print(pd.DataFrame(held).to_string())

    # the features themselves, picked on all steps
    pe, po, d = deltas(fired, exploit)
    top = np.argsort(-d)[:TOP]
    strengths = strengths.tocsc()
    feats = []
    for f in top:
        col = strengths[:, f]
        rows = col.nonzero()[0]
        ex_rows = rows[exploit[rows]]
        ex_rows = ex_rows[np.argsort(-col[ex_rows].toarray().ravel())]
        examples, seen = [], set()
        for r in ex_rows:
            if tasks[r] in seen:
                continue
            seen.add(tasks[r])
            text = " ".join(steps.body.iloc[r].split())
            examples.append({"id": steps.id.iloc[r],
                             "text": text[:SNIP] + ("…" if len(text) > SNIP else "")})
            if len(examples) == EXAMPLES:
                break
        feats.append({"feature": int(f), "exploit": round(float(pe[f]), 4),
                      "other": round(float(po[f]), 4), "delta": round(float(d[f]), 4),
                      "steps": int(exploit[rows].sum()),
                      "tasks": int(len(set(tasks[rows[exploit[rows]]]))),
                      "examples": examples})
        print(f, f"{pe[f]:.3f} - {po[f]:.3f} = {d[f]:.3f}", examples[0]["text"][:150])
    overlap = len(set(order[:TOP]) & set(top))
    OUT.write_text(json.dumps({"n_exploit": int(exploit.sum()), "n_other": int((~exploit).sum()),
                               "pick_tasks": len(pick_tasks),
                               "eval_steps": int((~pick).sum()),
                               "eval_exploit": int(exploit[~pick].sum()),
                               "base_rate": round(float(exploit[~pick].mean()), 3),
                               "top_overlap": overlap, "held_out": held,
                               "features": feats}, ensure_ascii=False, indent=1))
    print(f"wrote {OUT}; top-{TOP} overlap between half and all: {overlap}")


def name_pairs():
    """Name fit: a random exploit step the feature fires on vs one it doesn't."""
    from quality import NAME_PAIR, clip
    steps, strengths = load()
    ex = steps[steps.kept].reset_index(drop=True)
    fired = (strengths[steps.kept.to_numpy()] > 0).tocsc()
    rng = np.random.default_rng(4)
    rows = []
    for f, name in NAMES.items():
        hit = fired[:, f].nonzero()[0]
        miss = np.setdiff1d(np.arange(len(ex)), hit)
        for i, (m, o) in enumerate(zip(rng.choice(hit, NAME_PAIRS, replace=False),
                                       rng.choice(miss, NAME_PAIRS, replace=False))):
            a, b, ans = (m, o, "A") if rng.random() < 0.5 else (o, m, "B")
            rows.append({"id": f"name|{f}|{i}", "technique": "name", "group": f,
                         "answer": ans, "prompt": NAME_PAIR.format(
                             a=clip(ex.body[a]), b=clip(ex.body[b]), name=name)})
    pd.DataFrame(rows).to_parquet(PAIRS)
    print(f"{len(rows)} name pairs -> {PAIRS}")


def name_fit(run):
    import pyarrow.parquet as pq
    pairs = pd.read_parquet(PAIRS)
    p = {a["t.id"]: dict(a["match_probabilities"])
         for a in pq.read_table(run / "match.parquet").to_pylist()}
    pairs["p"] = [p[i].get(ans, 0.0) for i, ans in zip(pairs.id, pairs.answer)]
    fit = pairs.groupby("group").p.agg(list)
    data = json.loads(OUT.read_text())
    for f in data["features"]:
        f["name"] = NAMES[f["feature"]]
        f["name_pairs"] = [round(x, 2) for x in fit[f["feature"]]]
        f["name_fit"] = round(float(np.mean(fit[f["feature"]])), 2)
        print(f["feature"], f["name"], f["name_fit"])
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    import sys
    from pathlib import Path
    if len(sys.argv) == 1:
        main()
    elif sys.argv[1] == "names":
        name_pairs()
    else:
        name_fit(Path(sys.argv[2]))
