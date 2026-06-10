import numpy as np


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
    """Dispersion of a group after replacing group_embeddings[swap_index] with new_point.

    group_embeddings is a (group_size, d) array (can be a fancy-index copy).
    Does not modify any input array.
    """
    dists = np.linalg.norm(group_embeddings - new_centroid, axis=1)
    dists[swap_index] = np.linalg.norm(new_point - new_centroid)
    return float(dists.mean())
