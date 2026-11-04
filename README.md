# SAE EDA of reward-hacking traces

Can sparse autoencoder (SAE) features from [Quail](https://github.com/fsdatalab/quail)'s `AI.SAE_FEATURES` sort agent reward hacks into reliable groups, and pick out the steps where an agent hacks? Experiments on the [Terminal Wrench](https://github.com/few-sh/terminal-wrench) hack traces, with the Qwen-Scope layer-11 SAE on Qwen3.5-2B.

The write-up is `writeup/sae-reward-hacks.html` (open it through a local server, e.g. `python -m http.server` in `writeup/`, since it loads its JSON files).

## Layout

| Path | What it is |
| --- | --- |
| `prepare.py`, `eda_data.py` | Turn Terminal Wrench into `inputs/<dataset>/{trace,step}/docs.parquet` |
| `sae_extract_modal.py` | `AI.SAE_FEATURES` over traces or steps on a Modal H100 |
| `filter_modal.py` | `AI.FILTER`: tag each step as exploit or not |
| `quality.py`, `quality_modal.py` | LLM-judge trials (forced choice, Qwen3-32B through `AI.CLASSIFY`) |
| `explore/trace_clusters.py` | Part 1: K-Means on binarized trace vectors (`--pool max\|last\|exploit`) |
| `explore/feature_topics.py` | Each SAE feature as a topic, ranked by lift |
| `explore/topic_clusters.py` | Potential topics of high judge-accuracy clusters |
| `explore/step_classify.py` | Part 2: rank features by fire-rate difference, top-K classifier |
| `RUNS.md`, `runs/` | Run log (Modal call IDs, volume paths, takeaways) and launch logs |
| `writeup/` | The page and its data files |

Older grid code (`run_grid.py`, `cluster.py`, `topics.py`, `outliers.py`, `top_k.py`, ...) is from the first pass.

## Running

```bash
uv sync                      # installs Quail from the pinned upstream commit
PYTHONPATH=. uv run python explore/trace_clusters.py prep
uv run modal run --detach quality_modal.py --trials-file pairs_traces.parquet
PYTHONPATH=. uv run python explore/trace_clusters.py eval results/terminal_wrench/<quality run>
```

Each script's docstring gives its exact commands. `inputs/` and `results/` are not committed: they are rebuilt by the scripts, and SAE outputs are on the Modal volume `quail-results`. Some scripts read the Terminal Wrench labels from a sibling folder, `../terminal_wrench/inputs/labels.parquet`.

## Data

Terminal Wrench (Bercovich et al., 2026, [arXiv:2604.17596](https://arxiv.org/abs/2604.17596)) is released under the Apache License 2.0. The write-up's JSON files include excerpts of its traces.
