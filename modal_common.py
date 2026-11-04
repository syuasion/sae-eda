"""The Modal app, image, and volumes the sae_eda GPU scripts share.

The image installs Quail from upstream main at QUAIL_SHA instead of
mounting a checkout; quail.deploy.images.pypi_gpu_image does the same
for a PyPI release. Inputs are copied in from inputs/.
"""

from __future__ import annotations

from pathlib import Path

import modal
from quail.deploy.images import UV_VERSION, _cuda_base

QUAIL_SHA = "2a1fe4d297f545b4b509cf195fb9318e8039c971"
QUAIL = f"quail-engine @ git+https://github.com/fsdatalab/quail@{QUAIL_SHA}"
ROOT = Path(__file__).resolve().parent

app = modal.App("sae-eda")
results_volume = modal.Volume.from_name("quail-results", create_if_missing=True)
VOLUMES = {
    "/vol/results": results_volume,
    "/root/.cache/huggingface": modal.Volume.from_name(
        "quail-hf-cache", create_if_missing=True),
    "/root/.cache/kernels": modal.Volume.from_name(
        "quail-kernel-cache", create_if_missing=True),
}
RESULTS = Path("/vol/results/sae-eda")
INPUTS = Path("/root/inputs")

image = (_cuda_base()
         .uv_pip_install(QUAIL, "pandas==3.0.5", "hf-transfer==0.1.9",
                         uv_version=UV_VERSION)
         .add_local_dir(ROOT / "inputs", remote_path=str(INPUTS))
         .add_local_python_source("modal_common"))


def sample(docs, limit: int, seed: int = 0):
    """Return `limit` random rows of docs (all of them when limit is 0)."""
    import numpy as np

    if not limit or limit >= docs.num_rows:
        return docs
    rows = np.random.default_rng(seed).choice(docs.num_rows, limit,
                                              replace=False)
    return docs.take(np.sort(rows))
