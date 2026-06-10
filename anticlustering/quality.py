import numpy as np


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
