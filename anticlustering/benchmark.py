"""Task 2: benchmark anti-clustering on synthetic data and plot runtime vs. n."""

from __future__ import annotations

import time
import numpy as np

from .algorithm import form_groups
from .helpers import make_quality_fn, total_quality, optimal_dispersion, all_group_dispersions


DEFAULT_SIZES = [500, 1000, 2000, 5000, 10000]


def run_benchmark(
    sizes: list[int] = DEFAULT_SIZES,
    group_size: int = 8,
    embedding_dim: int = 1024,
    a: float = -1.0,
    b: float = 2.0,
    c: float = 0.0,
    n_iter_per_point: int = 50,
    random_seed: int = 42,
) -> dict[int, dict]:
    """Run anti-clustering on synthetic Gaussian data at each size in `sizes`.

    Returns a dict mapping n → {"runtime": float, "quality": float, "mean_dispersion": float}.
    """
    rng = np.random.default_rng(random_seed)
    results: dict[int, dict] = {}

    print(f"{'n':>8}  {'groups':>7}  {'runtime':>10}  {'quality':>10}  {'D*':>6}")
    print("-" * 50)

    for n in sizes:
        n = (n // group_size) * group_size  # round to valid size
        embeddings = rng.standard_normal((n, embedding_dim)).astype(np.float32)

        t0 = time.perf_counter()
        assignments = form_groups(
            embeddings,
            group_size=group_size,
            quality_fn=make_quality_fn(a, b, c),
            n_iter=n * n_iter_per_point,
            random_seed=random_seed,
        )
        elapsed = time.perf_counter() - t0

        n_groups = n // group_size
        group_members = [np.where(assignments == g)[0] for g in range(n_groups)]
        dispersions = all_group_dispersions(embeddings, group_members)
        q = total_quality(dispersions, a, b, c)
        d_star = optimal_dispersion(a, b)

        results[n] = {
            "runtime": elapsed,
            "quality": q,
            "mean_dispersion": float(dispersions.mean()),
        }
        status = "OK" if elapsed <= 30 else "SLOW"
        print(f"{n:>8}  {n_groups:>7}  {elapsed:>9.2f}s  {q:>10.4f}  {d_star:>6.2f}  {status}")

    return results


def plot_scaling(results: dict[int, dict], output_path: str | None = None) -> None:
    import matplotlib.pyplot as plt

    sizes = sorted(results.keys())
    times = [results[n]["runtime"] for n in sizes]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes, times, marker="o", linewidth=2)
    ax.axhline(30, color="red", linestyle="--", linewidth=1.5, label="30s target")
    ax.set_xlabel("Number of participants (n)")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("Anti-clustering runtime vs. dataset size")
    ax.legend()
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved scaling plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark anti-clustering on synthetic data")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n-iter-per-point", type=int, default=50)
    parser.add_argument("--plot", type=str, default=None, help="Path to save scaling plot")
    args = parser.parse_args()

    results = run_benchmark(
        sizes=args.sizes,
        group_size=args.group_size,
        embedding_dim=args.dim,
        n_iter_per_point=args.n_iter_per_point,
    )

    if args.plot:
        plot_scaling(results, output_path=args.plot)
