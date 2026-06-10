"""Task 3: full pipeline from participant texts → anti-clustered groups via Carter's embeddings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# embeddings-for-preferences submodule has hyphens so can't be imported directly
_SUBMODULE = Path(__file__).parent.parent / "embeddings-for-preferences"


def embed_texts(
    texts: list[str],
    model_name_or_path: str | Path,
    batch_size: int = 64,
    device: str | None = None,
) -> np.ndarray:
    """Encode texts using Carter's embedding model. Returns (n, d) float32 array."""
    if str(_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(_SUBMODULE))
    from src.embedding.model import load_model

    model = load_model(model_name_or_path, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)


def load_remesh_texts(data_path: str | Path) -> list[str]:
    """Load participant response texts from a Remesh export.

    Accepts:
      - Remesh verbatim_map CSV (skips metadata header, reads 'Thought Text' column)
      - JSON file containing a list of strings
    """
    import pandas as pd

    data_path = Path(data_path)
    if data_path.suffix == ".csv":
        # Remesh verbatim_map CSVs have ~8 rows of metadata before the real header
        raw = pd.read_csv(data_path, header=None)
        header_row = raw[raw[0] == "Question ID"].index[0]
        df = pd.read_csv(data_path, skiprows=header_row)
        return df["Thought Text"].dropna().tolist()
    elif data_path.suffix == ".json":
        with open(data_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(x) for x in data]
        raise ValueError("JSON file must contain a list of strings")
    else:
        raise ValueError(f"Unsupported file type: {data_path.suffix} (expected .csv or .json)")


def run_pipeline(
    data_path: str | Path,
    model_name_or_path: str | Path,
    group_size: int = 8,
    a: float = -1.0,
    b: float = 2.0,
    c: float = 0.0,
    n_iter: int | None = None,
    random_seed: int | None = None,
    device: str | None = None,
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Embed Remesh participants and partition them into anti-clustered groups.

    Args:
        data_path: Path to Remesh data (.csv with 'text' column, or .json list).
        model_name_or_path: Carter's embedding model (HuggingFace name or local path).
        group_size: Target group size (default 8).
        a, b, c: Quadratic quality parameters (a < 0).
        n_iter: Swap proposals; defaults to n * 50 inside form_groups.
        random_seed: For reproducibility.
        device: Torch device override (None = auto).
        output_path: If given, save assignments as a .npy file.

    Returns:
        assignments: (n,) int array mapping each participant to a group index.
    """
    from .algorithm import form_groups

    texts = load_remesh_texts(data_path)
    print(f"Loaded {len(texts)} participant texts from {data_path}")

    embeddings = embed_texts(texts, model_name_or_path, device=device)
    print(f"Embeddings shape: {embeddings.shape}")

    assignments = form_groups(
        embeddings,
        group_size=group_size,
        a=a, b=b, c=c,
        n_iter=n_iter,
        random_seed=random_seed,
    )

    if output_path is not None:
        np.save(output_path, assignments)
        print(f"Saved assignments to {output_path}")

    return assignments


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run anti-clustering pipeline on Remesh data")
    parser.add_argument("data_path", help="Path to Remesh data (.csv or .json)")
    parser.add_argument("model", help="Carter's embedding model path or HuggingFace name")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--a", type=float, default=-1.0)
    parser.add_argument("--b", type=float, default=2.0)
    parser.add_argument("--c", type=float, default=0.0)
    parser.add_argument("--n-iter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Save assignments to .npy file")
    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        model_name_or_path=args.model,
        group_size=args.group_size,
        a=args.a, b=args.b, c=args.c,
        n_iter=args.n_iter,
        random_seed=args.seed,
        device=args.device,
        output_path=args.output,
    )
