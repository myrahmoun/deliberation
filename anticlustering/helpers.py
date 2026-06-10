"""Helper functions for anti-clustering, grouped by role."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# embeddings-for-preferences submodule has hyphens so can't be imported directly
_SUBMODULE = Path(__file__).parent.parent / "embeddings-for-preferences"


# ── Load data ─────────────────────────────────────────────────────────────────

def load_participants(data_path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Load a Remesh verbatim_map CSV and group each participant's opinions together.

    Returns:
        participant_ids: list of unique Participant IDs
        thought_lists: thought_lists[i] contains all Thought Texts for participant i
    """
    data_path = Path(data_path)
    # Remesh verbatim_map CSVs have ~8 rows of metadata before the real header
    with open(data_path, newline="", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if "Question ID" in line:
                header_row = i
                break
        else:
            raise ValueError("Could not find 'Question ID' header row in CSV")

    df = pd.read_csv(data_path, skiprows=header_row)
    df = df[["Participant ID", "Thought Text"]].dropna(subset=["Thought Text"])

    grouped = df.groupby("Participant ID", sort=False)["Thought Text"].apply(list)
    return list(grouped.index), list(grouped.values)


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_participants(
    data_path: str | Path,
    model_name_or_path: str | Path,
    device: str | None = None,
) -> tuple[list[str], np.ndarray]:
    """Load a Remesh CSV and return per-participant embeddings.

    Each participant is represented by the mean of their per-thought embeddings.

    Returns:
        participant_ids: list of Participant ID strings
        embeddings: (n, d) float32 array, one row per participant
    """
    participant_ids, thought_lists = load_participants(data_path)
    n = len(participant_ids)
    print(f"Loaded {n} participants from {data_path}")

    if str(_SUBMODULE) not in sys.path:
        sys.path.insert(0, str(_SUBMODULE))
    from src.embedding.model import load_model

    all_thoughts = [t for thoughts in thought_lists for t in thoughts]
    model = load_model(model_name_or_path, device=device)
    thought_embeddings = model.encode(
        all_thoughts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    participant_embeddings = np.zeros((n, thought_embeddings.shape[1]), dtype=np.float32)
    idx = 0
    for i, thoughts in enumerate(thought_lists):
        k = len(thoughts)
        participant_embeddings[i] = thought_embeddings[idx:idx + k].mean(axis=0)
        idx += k

    print(f"Participant embeddings shape: {participant_embeddings.shape}")
    return participant_ids, participant_embeddings


def load_precomputed_embeddings(
    embeddings_path: str | Path,
    ids_path: str | Path,
) -> tuple[list[str], np.ndarray]:
    """Load embeddings saved by embed.py instead of re-running the model.

    Returns:
        participant_ids: list of Participant ID strings
        embeddings: (n, d) float32 array
    """
    embeddings = np.load(embeddings_path).astype(np.float32)
    ids_df = pd.read_csv(ids_path)
    participant_ids = ids_df["Participant ID"].tolist()
    print(f"Loaded {len(participant_ids)} precomputed embeddings from {embeddings_path}")
    return participant_ids, embeddings


# ── Dispersion ────────────────────────────────────────────────────────────────

def group_dispersion(embeddings: np.ndarray) -> float:
    """Mean Euclidean distance of points to their group centroid."""
    centroid = embeddings.mean(axis=0)
    return float(np.linalg.norm(embeddings - centroid, axis=1).mean())


def all_group_dispersions(
    embeddings: np.ndarray,
    group_members: list[np.ndarray],
) -> np.ndarray:
    return np.array([group_dispersion(embeddings[members]) for members in group_members])


def updated_centroid(
    old_centroid: np.ndarray,
    removed: np.ndarray,
    added: np.ndarray,
    group_size: int,
) -> np.ndarray:
    return old_centroid + (added - removed) / group_size


def dispersion_after_swap(
    group_embeddings: np.ndarray,
    swap_index: int,
    new_point: np.ndarray,
    new_centroid: np.ndarray,
) -> float:
    """Dispersion of a group after replacing group_embeddings[swap_index] with new_point."""
    dists = np.linalg.norm(group_embeddings - new_centroid, axis=1)
    dists[swap_index] = np.linalg.norm(new_point - new_centroid)
    return float(dists.mean())


# ── Quality ───────────────────────────────────────────────────────────────────

def make_quality_fn(a: float = -1.0, b: float = 2.0, c: float = 0.0) -> Callable[[float], float]:
    """Return a quadratic quality function f(dispersion) = a·D² + b·D + c.

    a must be negative to produce an inverted-U shape (penalizes both extremes).
    """
    if a >= 0:
        raise ValueError("a must be negative for an inverted-U quality function")
    return lambda d: a * d**2 + b * d + c


def group_quality(dispersion: float, a: float, b: float, c: float) -> float:
    """Quadratic quality for one group: a·D² + b·D + c."""
    return a * dispersion**2 + b * dispersion + c


def total_quality(dispersions: np.ndarray, a: float, b: float, c: float) -> float:
    """Mean quality across all groups."""
    return float(np.mean(a * dispersions**2 + b * dispersions + c))


def optimal_dispersion(a: float, b: float) -> float:
    """Dispersion value that maximizes quality (vertex of parabola). Requires a < 0."""
    if a >= 0:
        raise ValueError("a must be negative for an inverted-U quality function")
    return -b / (2 * a)


# ── Visualize ─────────────────────────────────────────────────────────────────

def plot_group_sizes(assignments: pd.DataFrame, ax) -> None:
    import matplotlib.pyplot as plt
    sizes = assignments["group"].value_counts().sort_index()
    ax.bar(sizes.index, sizes.values, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axhline(sizes.mean(), color="red", linestyle="--", linewidth=1.2, label=f"mean = {sizes.mean():.1f}")
    ax.set_xlabel("Group")
    ax.set_ylabel("Participants")
    ax.set_title("Group sizes")
    ax.legend()


def plot_dispersion_chart(embeddings: np.ndarray, assignments: pd.DataFrame, ax) -> None:
    groups = sorted(assignments["group"].unique())
    group_members = [np.where(assignments["group"].values == g)[0] for g in groups]
    dispersions = all_group_dispersions(embeddings, group_members)

    ax.bar(groups, dispersions, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axhline(dispersions.mean(), color="red", linestyle="--", linewidth=1.2,
               label=f"mean = {dispersions.mean():.3f}")
    ax.set_xlabel("Group")
    ax.set_ylabel("Dispersion D(g)")
    ax.set_title("Per-group dispersion")
    ax.legend()


def plot_umap(embeddings: np.ndarray, assignments: pd.DataFrame, ax) -> None:
    try:
        from umap import UMAP
        reducer = UMAP(n_components=2, random_state=42)
        method = "UMAP"
    except ImportError:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        method = "PCA"

    coords = reducer.fit_transform(embeddings)
    groups = assignments["group"].values
    n_groups = len(np.unique(groups))

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20" if n_groups <= 20 else "hsv")
    colors = [cmap(g / n_groups) for g in groups]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=18, alpha=0.7, linewidths=0)
    ax.set_title(f"Participant embeddings ({method}), colored by group")
    ax.set_xticks([])
    ax.set_yticks([])


def visualize(
    assignments: pd.DataFrame,
    embeddings: np.ndarray | None = None,
    output_path: str | Path | None = None,
    title: str = "Anti-clustering results",
) -> None:
    """Plot group sizes, per-group dispersion, and UMAP projection.

    Args:
        assignments: DataFrame with a 'group' column (and optionally 'Participant ID').
        embeddings: (n, d) array. If None, only the group-size bar chart is shown.
        output_path: Save figure to this path instead of displaying interactively.
        title: Figure title.
    """

    has_embeddings = embeddings is not None
    n_plots = 3 if has_embeddings else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    plot_group_sizes(assignments, axes[0])
    if has_embeddings:
        plot_dispersion_chart(embeddings, assignments, axes[1])
        plot_umap(embeddings, assignments, axes[2])

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()
