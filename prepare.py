"""Write a dataset's docs.parquet in the common schema.

    uv run python prepare.py --dataset terminal_wrench --unit trace

Output: inputs/<dataset>/<unit>/docs.parquet
"""

from __future__ import annotations

import argparse
from collections import Counter

import pyarrow.parquet as pq

from eda_data import LOADERS, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(LOADERS), required=True)
    parser.add_argument("--unit", choices=["trace", "step"], default="trace")
    args = parser.parse_args()

    docs = LOADERS[args.dataset](args.unit)
    out = ROOT / "inputs" / args.dataset / args.unit / "docs.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(docs, out)

    labels = Counter(label for row in docs.column("labels").to_pylist()
                     for label in row)
    print(f"{out}: {docs.num_rows} docs, "
          f"{sum(docs.column('truncated').to_pylist())} truncated, "
          f"{len(set(docs.column('group_id').to_pylist()))} groups")
    for label, count in labels.most_common():
        print(f"  {count:6d}  {label}")


if __name__ == "__main__":
    main()
