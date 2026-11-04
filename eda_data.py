"""Load each dataset into the common docs schema.

Every loader returns a table with these columns:

    id            str           document id
    body          str           text given to the model, capped by cap_body
    unit          str           "trace" or "step"
    parent_id     str           trace id for a step, the id itself for a trace
    group_id      str           task id; clusters that track it found tasks
    source_model  str           model that produced the trace
    is_hack       bool | None   None when every document is a hack
    labels        list[str]     the dataset's categories, may be empty
    truncated     bool          whether cap_body shortened the body
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
TERMINAL_WRENCH = ROOT.parent / "terminal_wrench" / "inputs"

# about 9k tokens; long traces keep the task and the end, where the
# exploit usually happens
MAX_CHARS = 32_000
HEAD_CHARS = 8_000
GAP = "\n\n[... middle of the trace omitted ...]\n\n"


def cap_body(body: str) -> tuple[str, bool]:
    """Return the body cut to MAX_CHARS and whether it was cut."""
    if len(body) <= MAX_CHARS:
        return body, False
    tail = MAX_CHARS - HEAD_CHARS - len(GAP)
    return body[:HEAD_CHARS] + GAP + body[-tail:], True


def _table(rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=pa.schema([
        ("id", pa.string()), ("body", pa.string()), ("unit", pa.string()),
        ("parent_id", pa.string()), ("group_id", pa.string()),
        ("source_model", pa.string()), ("is_hack", pa.bool_()),
        ("labels", pa.list_(pa.string())), ("truncated", pa.bool_())]))


def load_terminal_wrench(unit: str = "trace", notask: bool = False) -> pa.Table:
    """The 3,632 Terminal Wrench hack traces, or their 19,432 steps.

    Labels are the paper's observed_categories (arXiv:2604.17596 Table 3).
    A step inherits its trace's labels, task, and model. notask drops
    each trace's TASK section (strip_task) before capping.
    """
    labels = {row["id"]: row for row in
              pq.read_table(TERMINAL_WRENCH / "labels.parquet").to_pylist()}
    source = "traces" if unit == "trace" else "steps"
    rows = []
    for row in pq.read_table(TERMINAL_WRENCH / f"{source}.parquet").to_pylist():
        parent = row["id"] if unit == "trace" else row["trace_id"]
        label = labels[parent]
        body, truncated = cap_body(strip_task(row["body"]) if notask
                                   else row["body"])
        rows.append({"id": row["id"], "body": body, "unit": unit,
                     "parent_id": parent, "group_id": label["task_id"],
                     "source_model": label["model"], "is_hack": None,
                     "labels": list(label["observed_categories"]),
                     "truncated": truncated})
    return _table(rows)


def strip_task(body: str) -> str:
    """Drop the TASK section, keeping the trace from its first STEP.

    Traces of one task share that section (a median 1,599 characters,
    14% of a trace), which pulls clusters toward tasks. The agent's own
    steps still name the task's tools and files.
    """
    start = body.find("STEP ")
    return body[start:] if body.startswith("TASK\n") and start > 0 else body


def load_terminal_wrench_notask(unit: str = "trace") -> pa.Table:
    """Terminal Wrench without the task description (traces only)."""
    return load_terminal_wrench(unit, notask=True)


LOADERS = {"terminal_wrench": load_terminal_wrench,
           "terminal_wrench_notask": load_terminal_wrench_notask}
