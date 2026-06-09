# deliberation

Group formation and recommendation tools for [Frankly](https://app.frankly.org), a large-scale deliberation platform.

## What's here

**`anticlustering/`** — forms groups of 8 from ~10k participants by maximizing viewpoint diversity. Uses a pairwise exchange algorithm with a quadratic quality function `a·D² + b·D + c` over embedding-space dispersion. See `anticlustering.md` for the full design.

**`src/`** — Carter's embedding pipeline (fine-tuned BGE model with LoRA), evaluation harness, and opinion generation utilities.

## Quickstart

```bash
# benchmark anti-clustering on synthetic data (should hit 10k participants in ≤30s)
python -m anticlustering.benchmark

# run on real data
python -m anticlustering.pipeline path/to/data.csv path/to/model
```

## Dependencies

Install from `environment.yml` (conda) or `pyproject.toml` (pip).

## Link to Remesh data
https://github.com/akonya/polarized-issues-data
