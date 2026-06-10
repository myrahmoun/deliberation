"""Pairwise exchange algorithm for anti-clustering.

Maximizes mean group quality by iteratively proposing random swaps between groups
and accepting those that improve total quality.

To swap the objective, pass a different `quality_fn` to `form_groups`.
To swap the algorithm entirely, replace `form_groups` with any function sharing
the same signature: (embeddings, group_size, quality_fn, ...) -> assignments.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .helpers import all_group_dispersions, updated_centroid, dispersion_after_swap, make_quality_fn


def form_groups(
    embeddings: np.ndarray,
    group_size: int,
    quality_fn: Callable[[float], float] | None = None,
    n_iter: int | None = None,
    random_seed: int | None = None,
) -> np.ndarray:
    """Partition embeddings into groups of `group_size` to maximize mean group quality.

    Uses a greedy pairwise exchange algorithm: start from a random partition,
    repeatedly propose random swaps between groups, accept if quality improves.

    Args:
        embeddings: (n, d) float array of point embeddings.
        group_size: Each group will have exactly this many points. n must be divisible.
        quality_fn: Maps dispersion (float) → quality score (float). Should be
            maximized; defaults to the quadratic a=-1, b=2, c=0. Swap this to
            change the objective without touching the algorithm.
        n_iter: Total swap proposals. Defaults to n * 50.
        random_seed: Seed for reproducibility.

    Returns:
        assignments: (n,) int array where assignments[i] is the group index of point i.
    """
    n, _ = embeddings.shape
    if n % group_size != 0:
        raise ValueError(f"n={n} is not divisible by group_size={group_size}")

    if quality_fn is None:
        quality_fn = make_quality_fn(a=-1.0, b=2.0, c=0.0)

    rng = np.random.default_rng(random_seed)
    n_groups = n // group_size
    if n_iter is None:
        n_iter = n * 50

    # Random initial partition
    perm = rng.permutation(n)
    assignments = np.empty(n, dtype=np.intp)
    group_members: list[np.ndarray] = []
    for g in range(n_groups):
        members = perm[g * group_size:(g + 1) * group_size].copy()
        group_members.append(members)
        assignments[members] = g

    centroids = np.stack([embeddings[group_members[g]].mean(axis=0) for g in range(n_groups)])
    dispersions = all_group_dispersions(embeddings, group_members)

    for _ in range(n_iter):
        g1, g2 = rng.choice(n_groups, size=2, replace=False)
        i = int(rng.integers(group_size))
        j = int(rng.integers(group_size))

        p1 = group_members[g1][i]
        p2 = group_members[g2][j]
        x = embeddings[p1]
        y = embeddings[p2]

        new_c1 = updated_centroid(centroids[g1], removed=x, added=y, group_size=group_size)
        new_c2 = updated_centroid(centroids[g2], removed=y, added=x, group_size=group_size)

        new_d1 = dispersion_after_swap(embeddings[group_members[g1]], i, y, new_c1)
        new_d2 = dispersion_after_swap(embeddings[group_members[g2]], j, x, new_c2)

        delta_q = (
            quality_fn(new_d1) + quality_fn(new_d2)
            - quality_fn(dispersions[g1]) - quality_fn(dispersions[g2])
        )

        if delta_q > 0:
            group_members[g1][i] = p2
            group_members[g2][j] = p1
            assignments[p1] = g2
            assignments[p2] = g1
            centroids[g1] = new_c1
            centroids[g2] = new_c2
            dispersions[g1] = new_d1
            dispersions[g2] = new_d2

    return assignments
