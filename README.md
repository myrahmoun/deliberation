# Deliberation Experiments

Group formation and recommendation tools for [Frankly](https://app.frankly.org), a large-scale deliberation platform.

## What's here

**`anticlustering/`** — forms groups of 8 from ~10k participants by maximizing viewpoint diversity. Uses a pairwise exchange algorithm (stochastic hill-climbing) with a quadratic quality function `a·D² + b·D + c` over embedding-space dispersion.

**`embeddings-for-preferences/`** — Carter's embedding pipeline (fine-tuned BGE-large with LoRA). Submodule; used by `embed.py` to produce participant embeddings.

## Quickstart

### 1. Embed participants (once per dataset)

```bash
python -m anticlustering.embed data/remesh_export.csv --output-dir output/
```

The model is set at the top of `embed.py` (`MODEL_PATH`). Saves `output/<topic>_embeddings.npy` and `output/<topic>_participant_ids.csv`.

### 2. Configure and run

Edit the `CONFIG` block at the top of `anticlustering/main.py`:

```python
DATA_PATH              = "data/remesh_export.csv"
PRECOMPUTED_EMBEDDINGS = "output/topic_embeddings.npy"
PRECOMPUTED_IDS        = "output/topic_participant_ids.csv"
GROUP_SIZE             = 8
VISUALIZE              = True
```

Then:

```bash
python -m anticlustering.main
```

Saves group assignments to `output/assignments.csv`.

### 3. Benchmark

```bash
python -m anticlustering.benchmark
```

Runs on synthetic data across several sizes — should complete 10k participants in ≤30s.

## File structure

```
anticlustering/
  main.py       — config + orchestration (start here)
  algorithm.py  — form_groups(): pairwise exchange anti-clustering
  helpers.py    — load data, embed, dispersion, quality, visualize
  embed.py      — one-time script to pre-compute and save embeddings
  benchmark.py  — runtime scaling benchmark on synthetic data
  proof.md      — correctness proof for the algorithm
```

## Extending

To change the **quality objective**, pass a different `quality_fn` in `main.py`:

```python
form_groups(embeddings, group_size=8, quality_fn=lambda d: d)
```

To swap the **algorithm** entirely, replace `form_groups` with any function taking `(embeddings, group_size, ...)` and returning an `(n,)` int array of group indices.

## Dependencies

Install from `environment.yml` (conda) or `pyproject.toml` (pip).

## Data

Remesh export CSVs: https://github.com/akonya/polarized-issues-data
