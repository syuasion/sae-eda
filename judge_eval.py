"""Score each cell of a judge_modal.py run.

    uv run python judge_eval.py --dataset terminal_wrench \
      --judge results/terminal_wrench/<judge run id>

Joins the judge's four query results to inputs/<dataset>/judge/
clusters.parquet and prints, per cell (one grouping of steps):

    clusters        clusters judged (judge_prep.py skips a cluster whose
                    steps span fewer than 5 tasks)
    intrusion_acc   share of clusters where the judge picked the intruder;
                    chance is 1/6 = 0.167
    intruder_prob   mean probability the judge gave the true intruder
    useful          mean AI.SCORE that the steps share one specific method
    useful_share    share of clusters scoring >= 0.5
    useful_at_10    useful_share among the 10 clusters with the most traces,
                    the order a reader would meet them in (precision@10)
    agrees          share where the judge's category is the one most
                    over-represented among the cluster's traces
    none_share      share where the judge picked no category
Writes <judge run>/clusters_judged.parquet with every cluster's row.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from eda_data import ROOT


def query(directory: Path, name: str) -> pd.DataFrame:
    table = pq.read_table(directory / f"{name}.parquet").to_pylist()
    rows = []
    for row in table:
        value = row[name]
        if isinstance(value, dict):          # AI.EXTRACT: a struct of fields
            value = value.get(name)
        out = {"id": row["c.id"], name: value}
        if f"{name}_probabilities" in row:   # AI.CLASSIFY with probabilities
            out[f"{name}_probabilities"] = dict(row[f"{name}_probabilities"])
        rows.append(out)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="terminal_wrench")
    parser.add_argument("--judge", type=Path, required=True)
    args = parser.parse_args()

    clusters = pd.read_parquet(ROOT / "inputs" / args.dataset / "judge"
                               / "clusters.parquet")
    for name in ("name", "odd_one", "category", "useful"):
        clusters = clusters.merge(query(args.judge, name), on="id")
    clusters["caught"] = clusters.odd_one == clusters.intruder
    clusters["intruder_prob"] = [p[i] for p, i in zip(
        clusters.odd_one_probabilities, clusters.intruder)]
    clusters["agrees"] = clusters.category == clusters.top_category

    def summary(cell: pd.DataFrame) -> pd.Series:
        largest = cell.nlargest(10, "traces")
        return pd.Series({
            "clusters": len(cell),
            "intrusion_acc": cell.caught.mean(),
            "intruder_prob": cell.intruder_prob.mean(),
            "useful": cell.useful.mean(),
            "useful_share": (cell.useful >= 0.5).mean(),
            "useful_at_10": (largest.useful >= 0.5).mean(),
            "agrees": cell.agrees.mean(),
            "none_share": (cell.category == "none").mean()})

    table = clusters.groupby("cell")[clusters.columns].apply(summary)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())
    clusters.drop(columns=["odd_one_probabilities"]).to_parquet(
        args.judge / "clusters_judged.parquet")


if __name__ == "__main__":
    main()
