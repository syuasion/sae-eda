"""Dense sentence embeddings of every document, the non-SAE baseline.

    uv run modal run --detach embed_modal.py --dataset terminal_wrench \
      --unit trace 2>&1 | tee runs/<run>.log

Qwen3-Embedding-0.6B reads up to MAX_TOKENS tokens per document with the
model card's last-token pooling and L2 normalization.
Output on the quail-results volume, /results/sae-eda/<dataset>/embed-<unit>/:
  embeddings.npy  float32, one row per document in docs.parquet order
  ids.json        the document ids in that order
  meta.json       model, documents, seconds
"""

from __future__ import annotations

import json
import time

from modal_common import INPUTS, RESULTS, VOLUMES, app, image, results_volume

MODEL = "Qwen/Qwen3-Embedding-0.6B"
MAX_TOKENS = 8_192
BATCH_TOKENS = 64_000


@app.function(image=image, gpu="H100!", timeout=3_600, memory=65_536,
              volumes=VOLUMES)
def embed(dataset: str, unit: str) -> dict:
    """Embed the documents and save them in docs.parquet order."""
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    docs = pq.read_table(INPUTS / dataset / unit / "docs.parquet")
    texts = docs.column("body").to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(MODEL, padding_side="left")
    model = AutoModel.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()

    lengths = [len(ids) for ids in tokenizer(
        texts, truncation=True, max_length=MAX_TOKENS)["input_ids"]]
    order = np.argsort(lengths)[::-1]
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    start = 0
    with torch.inference_mode():
        while start < len(order):
            # longest first, so a batch's padded size is its first length
            size = max(1, BATCH_TOKENS // lengths[order[start]])
            rows = order[start:start + size]
            batch = tokenizer([texts[i] for i in rows], padding=True,
                              truncation=True, max_length=MAX_TOKENS,
                              return_tensors="pt").to("cuda")
            hidden = model(**batch).last_hidden_state[:, -1]
            out[rows] = F.normalize(hidden.float(), dim=-1).cpu().numpy()
            start += size

    directory = RESULTS / dataset / f"embed-{unit}"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "embeddings.npy", out)
    (directory / "ids.json").write_text(json.dumps(docs.column("id").to_pylist()))
    meta = {"model": MODEL, "max_tokens": MAX_TOKENS, "documents": len(texts),
            "truncated": int(sum(n >= MAX_TOKENS for n in lengths)),
            "total_s": time.perf_counter() - started}
    (directory / "meta.json").write_text(json.dumps(meta, indent=2))
    results_volume.commit()
    print(f"result volume path: {directory}", flush=True)
    return meta


@app.local_entrypoint()
def main(dataset: str = "terminal_wrench", unit: str = "trace"):
    """Start the embedding run and print its result."""
    call = embed.spawn(dataset, unit)
    print(f"function call id: {call.object_id}", flush=True)
    print(json.dumps(call.get(), indent=2))
