"""Exact and sampled simplex quadratures evaluate second-moment and inertia maps in any dimension.

Square-root-mass-weighted sample points feed both quadratic maps with identical scaling. The exact lumped quadrature has unit total mass, so multiplying either map by the simplex's mass or volume yields the physical inertia or geometric second moment directly.

Three quadrature rules cover exact and sampled evaluation:
1. Exact off-centre barycentric mass lumping (`simplex_inertia_lumped`, `simplex_second_moment_lumped`)
2. Midpoint grid quadrature (`simplex_inertia_brute`)
3. Uniform Dirichlet Monte Carlo sampling (`simplex_inertia_random`)
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np

from numga import Extensor


@lru_cache(maxsize=None)
def simplex_inertia_weights(n_vertices: int) -> np.ndarray:
    """Exact quadrature weights (`[n_vertices, n_vertices]`) map a unit-mass uniform simplex to `n_vertices` weighted sample points."""
    offsets = np.eye(n_vertices) - 1 / n_vertices                                    # [n_vertices, n_vertices]
    spread = np.sqrt(1 / (1 + n_vertices))
    barycentric_coords = 1 / n_vertices + offsets * spread                           # [n_vertices, n_vertices]
    return barycentric_coords / n_vertices                                           # [n_vertices, n_vertices]


def simplex_inertia(weights: np.ndarray, corners: Extensor) -> Extensor:
    """Weighted barycentric sample points sum to the simplex's `Antibivector- <- Bivector+` inertia map.

    Array `weights` has shape `[n_samples, n_corners]` whose row sums carry the sample masses, and `corners` holds `[..., n_corners] Point-`.
    """
    sample_points = barycentric_samples(weights, corners)                            # [..., n_samples] Point-
    Rate = corners.algebra.gatype.bivector()
    return sample_points.regressive(sample_points.commutator(Rate)).sum(axis=-1)     # [...] Antibivector- <- Bivector+


def simplex_second_moment(weights: np.ndarray, corners: Extensor) -> Extensor:
    """Weighted barycentric sample points sum to the simplex's `Point- <- Plane+` second-moment map.

    Each row of `weights` (`[n_samples, n_corners]`) carries its sample mass as its row sum, while `corners` holds `[..., n_corners] Point-`. Because each `barycentric_samples` point carries the square root of its row mass, the quadratic product includes total mass and first moments alongside spatial spread.
    """
    sample_points = barycentric_samples(weights, corners)                            # [..., n_samples] Point-
    Plane = corners.algebra.gatype.vector()
    return (sample_points * (Plane & sample_points)).sum(axis=-1)                    # [...] Point- <- Plane+


def barycentric_samples(weights: np.ndarray, corners: Extensor) -> Extensor:
    """Dividing weighted corner combinations by the square root of each row's sum produces mass-scaled points (`[..., n_samples] Point-`)."""
    weighted_sums = (corners[..., None, :] * weights).sum(axis=-1)                   # [..., n_samples] Point-
    return weighted_sums / np.sqrt(weights.sum(axis=-1))                             # [..., n_samples] Point-


def simplex_inertia_lumped(corners: Extensor) -> Extensor:
    """Exact off-centre barycentric weights give the unit-mass `Antibivector- <- Bivector+` inertia map of `corners`."""
    weights = simplex_inertia_weights(corners.shape[-1])
    return simplex_inertia(weights, corners)                                         # [...] Antibivector- <- Bivector+


def simplex_second_moment_lumped(corners: Extensor) -> Extensor:
    """Exact off-centre barycentric weights give the unit-mass `Point- <- Plane+` second-moment map of `corners`."""
    weights = simplex_inertia_weights(corners.shape[-1])
    return simplex_second_moment(weights, corners)                                   # [...] Point- <- Plane+


def simplex_inertia_brute(corners: Extensor, n_steps: int) -> Extensor:
    """A uniform midpoint grid with `n_steps` intervals per edge approximates the unit-mass `Antibivector- <- Bivector+` inertia map."""
    n_corners = corners.shape[0]
    edge_breaks = np.linspace(0, 1, n_steps + 1)                                     # [n_steps + 1]
    midpoints = (edge_breaks[1:] + edge_breaks[:-1]) / 2                             # [n_steps]
    coordinate_grid = np.array(np.meshgrid(*[midpoints] * (n_corners - 1), indexing="ij"))
    interior_coords = coordinate_grid.T.reshape(-1, n_corners - 1)
    interior_coords = interior_coords[interior_coords.sum(axis=1) < 1]
    barycentric_coords = np.concatenate(
        [interior_coords, 1 - interior_coords.sum(axis=1, keepdims=True)], axis=1,
    )
    n_grid_points = len(barycentric_coords)
    return simplex_inertia(barycentric_coords / n_grid_points, corners)              # [] Antibivector- <- Bivector+


def simplex_inertia_random(corners: Extensor, n_samples: int, rng: np.random.Generator) -> Extensor:
    """Normalized exponential variates draw `n_samples` uniform Dirichlet points to estimate the `Antibivector- <- Bivector+` inertia map."""
    n_corners = corners.shape[0]
    exponential_draws = rng.exponential(size=(n_samples, n_corners))                 # [n_samples, n_corners]
    barycentric_coords = exponential_draws / exponential_draws.sum(axis=1, keepdims=True)  # [n_samples, n_corners]
    return simplex_inertia(barycentric_coords / n_samples, corners)                  # [] Antibivector- <- Bivector+


def run_simplex_demo() -> None:
    """A tetrahedron's exact, grid, and Monte Carlo inertia maps agree on momentum, rate recovery, and kinetic energy."""
    from numga import NumpyContext
    from numga.algebras import PGA3D

    context = NumpyContext(PGA3D)
    mv = context.multivector

    # Tetrahedron corners, grid subdivisions per edge, and Monte Carlo sample count.
    n_corners = 4
    corners = mv.point([[1.0, 0.0, 0.0], [-0.5, 0.866, 0.0], [-0.5, -0.866, 0.0], [0.0, 0.0, 1.414]])  # [n_corners] Point-
    n_steps, n_samples = 100, 100000

    # 1. Exact lumped, midpoint-grid, and Monte Carlo inertia maps (`Antibivector- <- Bivector+`).
    Rate = PGA3D.gatype.bivector()
    sample_points = barycentric_samples(simplex_inertia_weights(n_corners), corners)  # [n_corners] Point-
    lumped_inertia = sample_points.regressive(sample_points.commutator(Rate)).sum(axis=0)  # [] Antibivector- <- Bivector+
    brute_inertia = simplex_inertia_brute(corners, n_steps)                          # [] Antibivector- <- Bivector+
    random_inertia = simplex_inertia_random(corners, n_samples, np.random.default_rng(0))  # [] Antibivector- <- Bivector+

    # Both factors carry the square root of each sample's mass.
    Plane = PGA3D.gatype.vector()
    moment = (sample_points * (Plane & sample_points)).sum(axis=0)                   # [] Point- <- Plane+
    print(f"Second moment (Point- <- Plane+):\n{moment.coefficient_table()}")

    print("--- 3D Tetrahedron Inertia Map Comparison ---")
    with np.printoptions(precision=3):
        print(f"Exact Lumped Inertia:\n{lumped_inertia.coefficient_table()}")
        print(f"Grid-Sampled Brute Inertia:\n{brute_inertia.coefficient_table()}")
        print(f"Monte Carlo Random Inertia:\n{random_inertia.coefficient_table()}")

    # 2. Applying the inertia map to a rate produces the body's momentum antibivector.
    rate = mv.bivector([1, 0, 0, 0, 0, 0])                                           # [] Bivector+
    momentum = lumped_inertia(rate)                                                  # [] Antibivector-
    print(f"\nRate bivector:\n{rate.coefficient_table()}")
    print(f"Momentum antibivector:\n{momentum.coefficient_table()}")

    # 3. Solving the linear inertia map recovers the original rate from the momentum.
    recovered_rate = lumped_inertia.solve(momentum)                                  # [] Bivector+
    print(f"Recovered rate:\n{recovered_rate.coefficient_table()}")

    # 4. Pairing the rate with its momentum gives twice the kinetic energy.
    kinetic_energy = 0.5 * (rate & momentum)                                         # [] Scalar+
    print(f"Kinetic energy:\n{kinetic_energy.coefficient_table()}")

    # checks
    rate_blades = mv.bivector(np.eye(6, dtype=int))                                  # [6] Bivector+
    forque_blades = rate_blades.dual()                                               # [6] Antibivector-
    assert ((forque_blades & (recovered_rate - rate)).abs() < 1e-12).all()
    assert ((rate_blades & (brute_inertia(rate) - momentum)).abs() < 1e-3).all()
    assert ((rate_blades & (random_inertia(rate) - momentum)).abs() < 1e-2).all()


if __name__ == "__main__":
    run_simplex_demo()
