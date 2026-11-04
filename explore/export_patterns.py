"""Export the features shown in the write-up's pattern figures.

    cd sae_eda && PYTHONPATH=. uv run python explore/export_patterns.py

Writes writeup/patterns.json. For each hand-picked feature: how often it
fires (is among a step's top 100) on exploit vs other steps, how often on
each model's exploit steps, and its 5 strongest exploit steps, one per task.
The names are mine, from reading those steps (feature_contrast.py output).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from explore.feature_contrast import load

CHARS = 520
# (group, feature, my name, model whose steps to show, or None for all)
PICKS = [
    ("exploit", 8283, "Rig code the verifier imports", None),
    ("exploit", 20406, "Patch what the checker calls", None),
    ("exploit", 31929, "Fake the system's status answers", None),
    ("exploit", 11131, "Swap in an easier setup", None),
    ("exploit", 31265, "Stub out the hard component", None),
    ("exploit", 32317, "Report the shortcut as done", None),
    ("model", 10152, "No verifier visible, so guess what it checks", "gpt-5.4"),
    ("model", 10401, "Plant startup hooks: .pth, BASH_ENV, wrappers", "gpt-5.4"),
    ("model", 27337, "Read the verifier, list what it checks", "claude-opus-4.6"),
    ("model", 3680, "Tick off each check with ✅", "claude-opus-4.6"),
    ("model", 28402, "I'm root, so replace system binaries", "gemini-3.1-pro"),
    ("model", 16461, "Open with \"As a red team agent…\"", "gemini-3.1-pro"),
]


def main():
    t0 = time.time()
    steps, m = load()
    kept = steps.kept.to_numpy()
    models = sorted(steps.source_model.unique())
    by_model = {x: (steps.source_model == x).to_numpy() for x in models}
    fired = (m > 0).tocsc()
    out = []
    for group, f, name, show in PICKS:
        col = fired[:, f].toarray().ravel()
        strength = m[:, f].toarray().ravel()
        rate = lambda mask: float(col[mask].mean())
        pool = kept & (by_model[show] if show else True)
        seen, ex = set(), []
        for r in np.argsort(-strength):
            if not pool[r] or strength[r] == 0 or steps.group_id.iat[r] in seen:
                continue
            seen.add(steps.group_id.iat[r])
            text = " ".join(steps.body.iat[r].split()).removeprefix("Analysis: ")
            ex.append({"id": steps.id.iat[r], "model": steps.source_model.iat[r],
                       "text": text if len(text) <= CHARS else text[:CHARS] + "…"})
            if len(ex) == 5:
                break
        out.append({"group": group, "feature": f, "name": name, "model": show,
                    "exploit": round(rate(kept), 4), "other": round(rate(~kept), 4),
                    "steps": int(col[kept].sum()),
                    "tasks": int(steps.group_id[col.astype(bool) & kept].nunique()),
                    "by_model": [round(rate(kept & by_model[x]), 4) for x in models],
                    "examples": ex})
    data = {"models": models, "n_exploit": int(kept.sum()),
            "n_other": int((~kept).sum()),
            "n_by_model": [int((kept & by_model[x]).sum()) for x in models],
            "features": out}
    path = Path(__file__).parent.parent / "writeup" / "patterns.json"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB) in {time.time() - t0:.0f} s")
    for x in out:
        print(x["feature"], x["name"], x["exploit"], x["other"], x["by_model"], x["tasks"])


if __name__ == "__main__":
    main()
