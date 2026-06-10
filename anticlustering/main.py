"""Anti-clustering pipeline. Edit the CONFIG section below, then run:

    python -m anticlustering.main

To skip re-embedding every run, first generate embeddings once with:

    python -m anticlustering.embed <data_csv> <model> --output-dir <dir>

Then set PRECOMPUTED_EMBEDDINGS and PRECOMPUTED_IDS below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .algorithm import form_groups
from .helpers import (
    embed_participants,
    load_precomputed_embeddings,
    make_quality_fn,
    summarize,
    visualize,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_PATH = "data/remesh_export.csv"           # Remesh verbatim_map CSV
MODEL_PATH = "embeddings-for-preferences/..."  # Carter's model path or HuggingFace name
DEVICE = None                                  # e.g. "cuda", "mps", or None for auto

GROUP_SIZE = 8
RANDOM_SEED = 42
N_ITER = None                                  # None → n * 50 (default)

# Quality function parameters (a must be negative)
QUALITY_A = -1.0
QUALITY_B = 2.0
QUALITY_C = 0.0

# Pre-computed embeddings (set both to skip re-embedding; leave None to re-embed)
PRECOMPUTED_EMBEDDINGS = "output/campus_protests_embeddings.npy"            # e.g. "output/embeddings.npy"
PRECOMPUTED_IDS        = "output/campus_protests_participant_ids.csv"

OUTPUT_PATH = "output/campus_protests_assignments.csv"

VISUALIZE = False                                     # set False to skip plotting
VISUALIZE_OUTPUT = "output/campus_protests_plot.png"  # set None to show interactively

# ── RUN ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path("output").mkdir(exist_ok=True)

    # Load embeddings (from cache or by running the model)
    if PRECOMPUTED_EMBEDDINGS and PRECOMPUTED_IDS:
        participant_ids, embeddings = load_precomputed_embeddings(
            PRECOMPUTED_EMBEDDINGS, PRECOMPUTED_IDS
        )
    else:
        participant_ids, embeddings = embed_participants(DATA_PATH, MODEL_PATH, device=DEVICE)

    # Trim to largest multiple of GROUP_SIZE (drops at most GROUP_SIZE-1 participants)
    n_valid = (len(participant_ids) // GROUP_SIZE) * GROUP_SIZE
    if n_valid < len(participant_ids):
        print(f"Trimming {len(participant_ids) - n_valid} participants to fit group_size={GROUP_SIZE}")
        participant_ids = participant_ids[:n_valid]
        embeddings = embeddings[:n_valid]

    # Form anti-clustered groups
    assignments = form_groups(
        embeddings,
        group_size=GROUP_SIZE,
        quality_fn=make_quality_fn(QUALITY_A, QUALITY_B, QUALITY_C),
        n_iter=N_ITER,
        random_seed=RANDOM_SEED,
    )

    # Save results
    results = pd.DataFrame({"Participant ID": participant_ids, "group": assignments})
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved assignments to {OUTPUT_PATH}")

    # Summarize
    summarize(results, embeddings, a=QUALITY_A, b=QUALITY_B, c=QUALITY_C)

    # Visualize
    if VISUALIZE:
        visualize(
            assignments=results,
            embeddings=embeddings,
            a=QUALITY_A,
            b=QUALITY_B,
            c=QUALITY_C,
            output_path=VISUALIZE_OUTPUT,
            title=Path(DATA_PATH).stem.replace("_", " ").title(),
        )
