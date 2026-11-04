"""Potential topics of high judge-accuracy trace clusters, for the write-up.

    PYTHONPATH=. uv run python explore/topic_clusters.py

Reads the cluster assignments written by trace_clusters.py (max, last and
exploit runs) and writes writeup/topics.json: per cluster its size, tasks,
model mix, judge accuracy, the paper categories most over-represented, a
potential topic and why, and 6 example traces (one per task). The topic and
why are a hand reading of the examples, written below.
"""

from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import trace_clusters as tc
from eda_data import ROOT

RUNS = {"trace_clusters": "Max pooling", "trace_clusters_last": "Last token",
        "trace_clusters_exploit": "Exploit steps"}
TECH = {"binary": "binary", "binary_idf": "binary × IDF", "binary_idf_top5": "top 5"}
# (run dir, technique, cluster): potential topic, why
PICKS = [
    ("trace_clusters", "binary", 4, "Fake a system component (daemon, package, binary)",
     "Five of the six examples fake something the checker probes instead of making it work: a fake atd daemon made with sleep, fake apt source lists, an entrypoint.sh that only prints the expected logs, a pytest runner patched to pass. But 95% of the traces are Gemini, so this may be Gemini's style more than a shared trick."),
    ("trace_clusters", "binary_idf", 20, "Hardcode the system report instead of querying the system",
     "Every example is a hardware or system inventory task where the agent writes the values it already saw (CPU counts, devices, installed services) straight into the JSON instead of a script that reads them. 93% of traces carry hollow-implementation. It spans only 12 tasks, so it is partly one family of similar tasks."),
    ("trace_clusters", "binary", 23, "Fake the deliverable without doing the steps",
     "The examples skip the requested procedure and only produce what is checked: a fake svn executable instead of installing Subversion, a report file with the right headers and no analysis, a GRUB file overwritten with one line. 92% Gemini, 16 tasks."),
    ("trace_clusters_last", "binary", 9, "Copy the answer from a leaked file",
     "Most examples find the answer already in the environment and copy it: a world-readable ground-truth maze map, a ground-truth labels CSV, a chess move stated in the task description. 44% of traces carry structural-extraction, 3.0 times its base rate, and the cluster mixes all three models."),
    ("trace_clusters_exploit", "binary", 3, "No single clear topic",
     "Judge accuracy is the highest here (0.97), but the examples use different tricks: matching an expected_output.txt, replacing the pmars binary, a stubbed MCTS, a leaked setup script. What they may share is reasoning about what a hidden checker probably tests. Worth reading more before naming."),
    ("trace_clusters_exploit", "binary", 15, "Copy the answer from a leaked file",
     "Five of six examples copy a leaked answer: expected_keys.txt from /protected/test_data, maze layouts from /protected/mazes, a secret key the keyholder binary writes to /tmp, the chess move in the task text. 72% of traces carry structural-extraction, 4.9 times its base rate, and no model has more than 45%."),
    ("trace_clusters_exploit", "binary_idf", 12, "Hardcode the system report instead of querying the system",
     "Same pattern as max-pooling cluster 20: system and hardware audit tasks where the agent collects the mock or container values once and hardcodes them into the JSON report instead of a script that reads them. 98% hollow-implementation, but only 10 tasks."),
    ("trace_clusters_exploit", "binary_idf", 17, "Patch the test harness or shadow imports",
     "The examples go after the testing machinery rather than the answer: modifying pytest itself to report every test as passed, stubbing the amuse modules a test script imports, exploiting a test that imports the agent's own code before scoring it, a fake MCTS that only checks one-move wins. Mostly Python and ML tasks; metric-spoofing is 2.8 times its base rate."),
    ("trace_clusters_exploit", "binary_idf", 18, "Write the state the checker reads",
     "The examples skip the process and write its end state: a JSON state file written instead of running systemd, availability JSON fed to a script that trusts it, a monitoring script replaced by a hardcoded report, a writable source dataset edited. algorithmic-simplification is 2.6 times its base rate."),
    ("trace_clusters_exploit", "binary_idf_top5", 5, "Write the state the checker reads",
     "The examples change the files a checker inspects instead of doing the setup: the dpkg package state edited from backup files, .bash_profile deleted so .profile takes over, a certificate placed where the check looks, a mock binary left by an installer. metric-spoofing is 2.1 times its base rate; 24 tasks, all three models."),
]


def feature_profile(all_ids, binary, overall, ids, top=8) -> dict:
    """What the cluster's 0/1 vectors look like: features per trace, and the
    features that fire much more often in the cluster than in all traces."""
    pos = {t: i for i, t in enumerate(all_ids)}
    rows = binary[[pos[i] for i in ids]]
    rate = np.asarray(rows.mean(0)).ravel()
    fired = np.asarray(rows.sum(1)).ravel()
    gap = rate - overall
    best = np.argsort(-gap)[:top]
    return {"fired_median": int(np.median(fired)), "fired_min": int(fired.min()), "fired_max": int(fired.max()),
            "shared_90": int((rate >= 0.9).sum()), "shared_90_all": int((overall >= 0.9).sum()),
            "features": [{"feature": int(f), "cluster": round(float(rate[f]), 3),
                          "all": round(float(overall[f]), 3)} for f in best]}


def main():
    labels = pq.read_table(ROOT.parent / "terminal_wrench" / "inputs" / "labels.parquet").to_pandas().set_index("id")
    traces, _ = tc.load()
    by_id = traces.set_index("id")
    base = collections.Counter(c for cs in labels.observed_categories for c in cs)
    rng = np.random.default_rng(0)
    pools = {"trace_clusters": "max", "trace_clusters_last": "last", "trace_clusters_exploit": "exploit"}
    vectors = {}
    out = []
    for run, tech, c, topic, why in PICKS:
        assign = pd.read_parquet(tc.TW / run / "assignments.parquet")
        ids = assign.id[assign[f"{tech}_s0"] == c].tolist()
        if run not in vectors:
            tc.POOL = pools[run]
            tc.TRACE_RUN = tc.SOURCES[tc.POOL][0]
            vt, vb = tc.load()
            vectors[run] = (vt.id.tolist(), vb.tocsr(), np.asarray(vb.mean(0)).ravel())
        profile = feature_profile(*vectors[run], ids)
        viz = json.loads((ROOT / "writeup" / {"trace_clusters": "trace_viz.json",
                                              "trace_clusters_last": "trace_viz_last.json",
                                              "trace_clusters_exploit": "trace_viz_exploit.json"}[run]).read_text())
        judge = viz["views"][tech]["clusters"][c]["judge"]
        cats = collections.Counter(x for i in ids for x in labels.loc[i, "observed_categories"])
        lift = sorted(((k, cats[k] / len(ids), cats[k] / len(ids) / (base[k] / len(labels))) for k in cats),
                      key=lambda z: -z[2])[:3]
        examples, seen = [], set()
        for i in rng.permutation(ids):
            row = by_id.loc[i]
            if row.group_id in seen or not isinstance(row.exploit, str):
                continue
            seen.add(row.group_id)
            examples.append({"task": tc.task_line(row.body), "model": row.source_model,
                             "text": tc.clip(row.exploit, 420)})
            if len(examples) == 6:
                break
        out.append({"run": RUNS[run], "technique": TECH[tech], "cluster": c, "size": len(ids),
                    "tasks": int(by_id.loc[ids].group_id.nunique()),
                    "models": by_id.loc[ids].source_model.value_counts().to_dict(),
                    "judge": judge, "topic": topic, "why": why, "profile": profile,
                    "categories": [{"name": k, "share": round(s, 3), "lift": round(l, 1)} for k, s, l in lift],
                    "examples": examples})
    path = ROOT / "writeup" / "topics.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"wrote {path}: {len(out)} clusters")


if __name__ == "__main__":
    main()
