"""Run quality.py's LLM trials on a Modal H100.

    uv run modal run --detach quality_modal.py 2>&1 | tee runs/<run>.log

Input: inputs/terminal_wrench/quality/trials.parquet (quality.py prep).
One Quail AI.CLASSIFY query with qwen3-32b-fp8, with probabilities: labels
yes/no, or A/B for a pairs*.parquet file (forced choice). Output on the quail-results volume,
/results/sae-eda/terminal_wrench/<run id>/match.parquet and meta.json.
"""

from __future__ import annotations

import json
import time
import uuid

from modal_common import (INPUTS, QUAIL_SHA, RESULTS, VOLUMES, app, image,
                          results_volume)

MODEL = "qwen3-32b-fp8"
DEVICE = "h100-sxm"


@app.function(image=image, gpu="H100!", timeout=2 * 3_600, memory=65_536,
              volumes=VOLUMES)
def match(dataset: str, trials_file: str) -> dict:
    import pyarrow.parquet as pq
    import quail

    trials = pq.read_table(INPUTS / dataset / "quality" / trials_file)
    run_id = f"quality-{uuid.uuid4().hex[:6]}"
    directory = RESULTS / dataset / run_id
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    config = quail.EngineConfig(model=MODEL, device=DEVICE)
    with quail.Session(config) as session:
        session.register("t", quail.DocumentProvider.from_table(
            trials.select(["id", "prompt"]), id_col="id"))
        query = session.docs("t").alias("t").ai_classify(
            quail.prompt("{0}", quail.col("t.prompt")),
            ["A", "B"] if trials_file.startswith("pairs") else ["yes", "no"],
            name="match", probabilities=True)
        result = query.select("t.id", "match").run()
        pq.write_table(result.collect(), directory / "match.parquet")
        report = {k: result.report.get(k) for k in ("wall_s", "fresh_tokens")}
    meta = {"dataset": dataset, "model": MODEL, "quail_sha": QUAIL_SHA,
            "run_id": run_id, "trials_file": trials_file, "trials": trials.num_rows, "query": report,
            "total_s": time.perf_counter() - started}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    results_volume.commit()
    print(f"result volume path: {directory}", flush=True)
    return meta


@app.local_entrypoint()
def main(dataset: str = "terminal_wrench", trials_file: str = "trials.parquet"):
    call = match.spawn(dataset, trials_file)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))
