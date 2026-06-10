"""One-time script to embed participants and save results for reuse.

Run once per dataset:

    python -m anticlustering.embed data/topic_verbatim_map.csv path/to/model --output-dir output/

Produces (named after the CSV stem):
    output/topic_embeddings.npy      — (n, d) float32 participant embeddings
    output/topic_participant_ids.csv — single-column CSV mapping row index → Participant ID

Then in main.py set:
    PRECOMPUTED_EMBEDDINGS = "output/topic_embeddings.npy"
    PRECOMPUTED_IDS        = "output/topic_participant_ids.csv"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .helpers import embed_participants


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed Remesh participants and save for reuse")
    parser.add_argument("data_path", help="Path to Remesh verbatim_map CSV")
    parser.add_argument("model", help="Carter's embedding model path or HuggingFace name")
    parser.add_argument("--output-dir", default="output", help="Directory to save outputs (default: output/)")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or mps")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_ids, embeddings = embed_participants(args.data_path, args.model, device=args.device)

    topic = Path(args.data_path).stem.replace("_verbatim_map", "")
    embeddings_path = output_dir / f"{topic}_embeddings.npy"
    ids_path = output_dir / f"{topic}_participant_ids.csv"

    np.save(embeddings_path, embeddings)
    pd.DataFrame({"Participant ID": participant_ids}).to_csv(ids_path, index=False)

    print(f"Saved embeddings      → {embeddings_path}")
    print(f"Saved participant IDs → {ids_path}")
    print("Set PRECOMPUTED_EMBEDDINGS and PRECOMPUTED_IDS in main.py to use these.")


if __name__ == "__main__":
    main()
