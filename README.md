# deliberation

Group formation and recommendation tools for [Frankly](https://app.frankly.org), a large-scale deliberation platform.

## What's here

**`anticlustering/`** — forms groups of 8 from ~10k participants by maximizing viewpoint diversity. Uses a pairwise exchange algorithm with a quadratic quality function `a·D² + b·D + c` over embedding-space dispersion. See `anticlustering.md` for the full design.

**`src/`** — Carter's embedding pipeline (fine-tuned BGE model with LoRA), evaluation harness, and opinion generation utilities.

## Quickstart

### 1. Embed participants (once per dataset)

Embedding is slow — run this once and reuse the output:

```bash
python -m anticlustering.embed data/remesh_export.csv path/to/model --output-dir output/
```

Saves `output/embeddings.npy` and `output/participant_ids.csv`.

### 2. Configure and run

Edit the `CONFIG` block at the top of `anticlustering/main.py`:

```python
DATA_PATH    = "data/remesh_export.csv"
MODEL_PATH   = "path/to/model"
GROUP_SIZE   = 8
VISUALIZE    = True   # set False to skip plots

# Point to pre-computed embeddings to skip re-embedding:
PRECOMPUTED_EMBEDDINGS = "output/campus_protests_embeddings.npy"
PRECOMPUTED_IDS        = "output/s/campus_protests_participant_ids.csv"
```

Then run:

```bash
python -m anticlustering.main
```

Saves group assignments to `output/assignments.csv`.

### 3. Benchmark

```bash
python -m anticlustering.benchmark
```

Runs on synthetic data across several sizes — should hit 10k participants in ≤30s.

## File structure

```
anticlustering/
  main.py       — config + orchestration (start here)
  algorithm.py  — form_groups(): pairwise exchange anti-clustering
  helpers.py    — utilities grouped by role: load data, embedding,
                  dispersion, quality, visualize
  embed.py      — one-time script to pre-compute and save embeddings
  benchmark.py  — runtime scaling benchmark on synthetic data
```

## Swapping the algorithm or objective

To change the **quality objective**, pass a different `quality_fn` in `main.py`:

```python
# e.g. maximize raw dispersion instead of quadratic
form_groups(embeddings, group_size=8, quality_fn=lambda d: d)
```

To change the **algorithm** entirely, replace the `form_groups` call in `main.py` with any function that accepts `(embeddings, group_size, ...)` and returns an `(n,)` int array of group indices.

## Dependencies

Install from `environment.yml` (conda) or `pyproject.toml` (pip).

## Link to Remesh data
https://github.com/akonya/polarized-issues-data
