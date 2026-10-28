"""Second moments and inertia maps of simplices in arbitrary dimensions using geometric algebra.

The same square-root-mass-weighted points integrate both quadratic constructions.
The exact rule has total mass one; multiply either map by the simplex's mass,
or by its volume for a geometric moment.

Calculates inertia tensors of simplices via:
1. Exact vertex-lumped barycentric masses (analytic)
2. Uniform grid sampling (brute-force)
3. Uniform Monte Carlo sampling
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from numga import Extensor


@lru_cache(maxsize=None)
def simplex_inertia_weights(n_vertices: int) -> np.ndarray:
    """The barycentric weights mapping a uniform density simplex to its equivalent vertex-lumped-mass simplex."""
    n = n_vertices
    d = np.eye(n) - 1.0 / n
    f = np.sqrt(1.0 / (1.0 + n))
    b = 1.0 / n + d * f
    return b / n


def simplex_inertia(weights: np.ndarray, corners: Extensor) -> Extensor:
    """The inertia map of the barycentric samples of corner points.

    Parameters
    ----------
    weights : np.ndarray
        Array of shape `[n_samples, n_corners]` of barycentric weights.
        Each row `weights[j]` specifies the point position and mass `sum(weights[j])`.
    corners : Extensor
        Antivector points of shape `[n_corners]`.
    """
    samples = barycentric_samples(weights, corners)
    Bivector = corners.algebra.gatype.bivector()
    return samples.regressive(samples.commutator(Bivector)).sum(axis=-1)


def simplex_second_moment(weights: np.ndarray, corners: Extensor) -> Extensor:
    """Point <- Plane second moment from weighted barycentric samples.

    Corners have shape `[..., corners]`, and weights `[samples, corners]`.
    Each row's sum is its sample mass. The two appearances of the sampled point
    supply that mass once, so the map contains total mass and first moments too.
    """
    samples = barycentric_samples(weights, corners)
    Plane = corners.algebra.gatype.vector()
    return (samples * (Plane & samples)).sum(axis=-1)


def barycentric_samples(weights: np.ndarray, corners: Extensor) -> Extensor:
    """Square-root-mass-weighted points `[..., samples]` from corners `[..., corners]`."""
    points = (corners[..., None, :] * weights).sum(axis=-1)
    return points / np.sqrt(weights.sum(axis=-1))


def simplex_inertia_lumped(corners: Extensor) -> Extensor:
    """Inertia map of uniform density simplex via equivalent lumped masses."""
    weights = simplex_inertia_weights(corners.shape[-1])
    return simplex_inertia(weights, corners)


def simplex_second_moment_lumped(corners: Extensor) -> Extensor:
    """Exact second moment of a uniform simplex of total mass one."""
    weights = simplex_inertia_weights(corners.shape[-1])
    return simplex_second_moment(weights, corners)


def simplex_inertia_brute(corners: Extensor, n_steps: int) -> Extensor:
    """Inertia map via uniform grid sampling of simplex, `n_steps` cells along each edge."""
    k = corners.shape[0]
    s = np.linspace(0, 1, n_steps + 1)
    s = (s[1:] + s[:-1]) / 2.0
    grid = np.array(np.meshgrid(*[s] * (k - 1), indexing="ij"))
    b = grid.T.reshape(-1, k - 1)
    b = b[b.sum(axis=1) < 1.0]
    b = np.concatenate([b, 1.0 - b.sum(axis=1, keepdims=True)], axis=1)
    return simplex_inertia(b / len(b), corners)


def simplex_inertia_random(corners: Extensor, n_samples: int, rng: np.random.Generator) -> Extensor:
    """Inertia map via uniform Dirichlet / exponential random sampling."""
    b = rng.exponential(size=(n_samples, corners.shape[0]))
    weights = b / b.sum(axis=1, keepdims=True)
    return simplex_inertia(weights / n_samples, corners)


def run_simplex_demo() -> None:
    """Print a simplex's inertia by three quadratures, its action on a rate, the rate solved back, and the energy."""
    from numga import NumpyContext
    from numga.algebras import PGA3D

    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector

    # The corners of a tetrahedron; the grid's cells along an edge, and the random samples.
    corners = mv.point([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0], [0.0, 0.0, 1.414]])   # [corners] Point-
    n_steps, n_samples = 100, 100000

    # 1. The inertia maps, Antibivector- <- Bivector+, by lumped masses, a grid and random samples.
    Bivector = PGA3D.gatype.bivector()
    samples = barycentric_samples(simplex_inertia_weights(corners.shape[0]), corners)
    lumped_inertia = samples.regressive(samples.commutator(Bivector)).sum(axis=0)
    brute_inertia = simplex_inertia_brute(corners, n_steps)
    random_inertia = simplex_inertia_random(corners, n_samples, np.random.default_rng(0))

    # Both factors contain the square root of each sample's mass.
    Plane = PGA3D.gatype.vector()
    moment = (samples * (Plane & samples)).sum(axis=0)
    print(f"Second moment (Point <- Plane):\n{moment.coefficient_table()}")

    print("--- 3D Tetrahedron Inertia Tensor Comparison ---")
    with np.printoptions(precision=3):
        print(f"Exact Lumped Inertia:\n{lumped_inertia.coefficient_table()}")
        print(f"Grid-Sampled Brute Inertia:\n{brute_inertia.coefficient_table()}")
        print(f"Monte Carlo Random Inertia:\n{random_inertia.coefficient_table()}")

    # 2. The inertia applied to a rate gives its momentum.
    rate = mv.bivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    momentum = lumped_inertia(rate)
    print("\nAngular Rate bivector:", rate.kernel)
    print("Resulting Momentum antibivector:", np.around(momentum.kernel, 4))

    # 3. The rate back from the momentum.
    recovered_rate = lumped_inertia.solve(momentum)
    print("Recovered Rate via I.solve():", np.around(recovered_rate.kernel, 4))

    # 4. The kinetic energy, the rate paired with its momentum.
    kinetic_energy = 0.5 * rate.regressive(momentum)
    print("Kinetic Energy:", float(kinetic_energy.kernel.item()))


if __name__ == "__main__":
    run_simplex_demo()
