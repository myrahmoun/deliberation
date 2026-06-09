"""Select diverse issues via farthest-point sampling on embeddings."""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def farthest_point_sampling(embeddings: np.ndarray, n: int, seed: int = 42) -> List[int]:
    """Select n indices from embeddings that maximize diversity.

    Greedy farthest-point sampling: start with a random point, then
    iteratively add the point farthest from any already-selected point.
    """
    rng = np.random.RandomState(seed)
    n_total = len(embeddings)
    if n >= n_total:
        return list(range(n_total))

    # Normalize for cosine distance
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normed = embeddings / norms

    selected = [rng.randint(n_total)]
    min_distances = np.ones(n_total) * np.inf

    for _ in range(n - 1):
        last = normed[selected[-1]]
        # Cosine distance = 1 - cosine_similarity
        distances = 1.0 - normed @ last
        min_distances = np.minimum(min_distances, distances)
        # Exclude already selected
        min_distances[selected] = -1.0
        selected.append(int(np.argmax(min_distances)))

    return selected


def select_diverse_issues(
    issues: List[dict],
    n: int = 500,
    embedding_model: str = "all-MiniLM-L6-v2",
    seed: int = 42,
) -> List[dict]:
    """Select n maximally diverse issues from the pool.

    Args:
        issues: List of issue dicts with 'issue_text' key.
        n: Number to select.
        embedding_model: Model for computing issue embeddings.
        seed: Random seed.

    Returns:
        Selected subset of issues.
    """
    logger.info(f"Embedding {len(issues)} issues with {embedding_model}...")
    model = SentenceTransformer(embedding_model)
    texts = [issue["issue_text"] for issue in issues]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    logger.info(f"Selecting {n} diverse issues via farthest-point sampling...")
    indices = farthest_point_sampling(embeddings, n, seed=seed)

    selected = [issues[i] for i in indices]
    logger.info(f"Selected {len(selected)} issues")
    return selected
