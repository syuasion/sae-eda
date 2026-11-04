"""Read Qwen-Scope SAE features of every document on a Modal H100.

    uv run modal run --detach sae_extract_modal.py --dataset terminal_wrench \
      --unit trace --pooling last --prompt targeted --limit 200 \
      2>&1 | tee runs/<run>.log

Input: inputs/<dataset>/<unit>/docs.parquet (from prepare.py).
Output on the quail-results volume, /results/sae-eda/<dataset>/<run id>/:
  features.parquet  id, feature, strength: one row per (document, feature),
                    up to 100 features per document with last pooling;
                    max pooling returns thousands, so --keep N saves only
                    each document's N strongest
  meta.json         arguments, Quail SHA, documents, wall_s, fresh_tokens
Pull it with: modal volume get quail-results sae-eda/<dataset>/<run id> results/<dataset>/
--limit 0 reads every document; otherwise a seeded random sample.
"""

from __future__ import annotations

import json
import time
import uuid

from modal_common import (INPUTS, QUAIL_SHA, RESULTS, VOLUMES, app, image,
                          results_volume, sample)

MODEL = "qwen3.5-2b-sae-bf16"
DEVICE = "h100-sxm"
CHUNK = 4_000
PROMPTS = {
    "targeted": "{0}\n\nHow did the agent get the task to pass?",
    "neutral": "{0}\n\nSummarize what the agent did.",
}


@app.function(image=image, gpu="H100!", timeout=6 * 3_600, memory=65_536,
              volumes=VOLUMES)
def extract(dataset: str, unit: str, pooling: str, prompt: str,
            limit: int, keep: int = 0) -> dict:
    """Run AI.SAE_FEATURES over the documents and save the features."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import quail

    docs = sample(pq.read_table(INPUTS / dataset / unit / "docs.parquet"),
                  limit)
    run_id = (f"{unit}-{pooling}-{prompt}-{limit or 'all'}-"
              + (f"keep{keep}-" if keep else "") + uuid.uuid4().hex[:6])
    directory = RESULTS / dataset / run_id
    directory.mkdir(parents=True, exist_ok=True)

    # With --keep, documents go in chunks: max pooling over all 19k steps
    # in one query overflows Arrow's 2 GB string offsets when unnest
    # repeats each id per feature (ArrowInvalid: Negative offsets).
    size = CHUNK if keep else docs.num_rows
    started = time.perf_counter()
    config = quail.EngineConfig(model=MODEL, device=DEVICE)
    parts, reports, returned_rows = [], [], 0
    with quail.Session(config) as session:
        for n, first in enumerate(range(0, docs.num_rows, size)):
            session.register(f"docs{n}", quail.DocumentProvider.from_table(
                docs.slice(first, size).select(["id", "body"]), id_col="id"))
            result = (
                session.docs(f"docs{n}").alias("t")
                .ai_sae_features(quail.prompt(PROMPTS[prompt],
                                              quail.col("t.body")),
                                 pooling=pooling, name="f")
                .unnest("f")
                .select("t.id", "f.feature", "f.strength")
                .run())
            part = result.collect().rename_columns(["id", "feature", "strength"])
            returned_rows += part.num_rows
            if keep:
                frame = part.to_pandas()
                frame = (frame.sort_values(["id", "strength"],
                                           ascending=[True, False])
                         .groupby("id").head(keep))
                part = pa.Table.from_pandas(frame, preserve_index=False)
            parts.append(part)
            reports.append(result.report)
            print(f"chunk {n}: {part.num_rows} rows kept", flush=True)
    features = pa.concat_tables(parts)
    pq.write_table(features, directory / "features.parquet")

    report = {key: sum(r.get(key) or 0 for r in reports)
              for key in ("wall_s", "fresh_tokens", "gpu_s")}
    meta = {"dataset": dataset, "unit": unit, "pooling": pooling,
            "prompt": prompt, "template": PROMPTS[prompt], "limit": limit,
            "keep": keep, "returned_rows": returned_rows,
            "model": MODEL, "quail_sha": QUAIL_SHA, "run_id": run_id,
            "documents": docs.num_rows, "feature_rows": features.num_rows,
            "total_s": time.perf_counter() - started,
            "wall_s": report.get("wall_s"),
            "fresh_tokens": report.get("fresh_tokens"),
            "gpu_s": report.get("gpu_s")}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2,
                                                    default=str))
    results_volume.commit()
    print(f"result volume path: {directory}", flush=True)
    return meta


@app.local_entrypoint()
def main(dataset: str = "terminal_wrench", unit: str = "trace",
         pooling: str = "last", prompt: str = "targeted", limit: int = 200,
         keep: int = 0):
    """Start the extraction and print its result."""
    call = extract.spawn(dataset, unit, pooling, prompt, limit, keep)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))
