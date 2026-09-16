"""Inertia maps of simplices in arbitrary dimensions using geometric algebra.

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
    """Compute barycentric weighting matrix mapping a uniform density simplex
    to its equivalent vertex-lumped-mass simplex.
    """
    n = n_vertices
    d = np.eye(n) - 1.0 / n
    f = np.sqrt(1.0 / (1.0 + n))
    b = 1.0 / n + d * f
    return b / n


def simplex_inertia(weights: np.ndarray, corners: Extensor) -> Extensor:
    """Compute inertia map given barycentric weights and corner points.

    Parameters
    ----------
    weights : np.ndarray
        Array of shape `[n_samples, n_corners]` of barycentric weights.
        Each row `weights[j]` specifies the point position and mass `sum(weights[j])`.
    corners : Extensor
        Antivector points of shape `[n_corners]`.
    """
    context = corners.context
    spaces = context.algebra.subspace
    bivector = spaces.bivector()

    # Form sample points: [n_samples, coordinate_dim]
    sample_coords = np.einsum("Ni,ik->Nk", weights, corners.kernel)
    # The homogeneous point mass is the sum of barycentric weights
    masses = weights.sum(axis=-1, keepdims=True)
    # Because inertia is quadratic in point positions (inertia ~ mass * x^2),
    # scale coordinates by 1 / sqrt(mass) so the resulting inertia is linear in mass.
    scaled_coords = sample_coords / np.sqrt(masses)
    samples = context.multivector(corners.subspace, scaled_coords)

    # In PGA, inertia = sum_p p & (p x Rate)
    per_point = samples.regressive(samples.commutator(bivector))
    return per_point.sum(axis=0)


def simplex_inertia_lumped(corners: Extensor) -> Extensor:
    """Inertia map of uniform density simplex via equivalent lumped masses."""
    weights = simplex_inertia_weights(corners.shape[0])
    return simplex_inertia(weights, corners)


def simplex_inertia_brute(corners: Extensor, n: int = 100) -> Extensor:
    """Inertia map via uniform grid sampling of simplex."""
    k = corners.shape[0]
    s = np.linspace(0, 1, n + 1)
    s = (s[1:] + s[:-1]) / 2.0
    grid = np.array(np.meshgrid(*[s] * (k - 1), indexing="ij"))
    b = grid.T.reshape(-1, k - 1)
    b = b[b.sum(axis=1) < 1.0]
    b = np.concatenate([b, 1.0 - b.sum(axis=1, keepdims=True)], axis=1)
    return simplex_inertia(b / len(b), corners)


def simplex_inertia_random(
    corners: Extensor,
    n_samples: int = 50000,
    rng: np.random.Generator = np.random.default_rng(0),
) -> Extensor:
    """Inertia map via uniform Dirichlet / exponential random sampling."""
    b = rng.exponential(size=(n_samples, corners.shape[0]))
    weights = b / b.sum(axis=1, keepdims=True)
    return simplex_inertia(weights / n_samples, corners)


def run_simplex_demo() -> None:
    """Demonstrate simplex inertia operator construction, linear action, and inverse."""
    from numga import NumpyContext
    from numga.algebras import PGA3D

    ctx = NumpyContext(PGA3D)
    mv = ctx.multivector

    # 3D Tetrahedron corners: 4 antivector points in PGA3D (x, y, z, 1)
    corners_coords = [
        [1.0, 0.0, 0.0, 1.0],
        [-0.5, 0.866, 0.0, 1.0],
        [-0.5, -0.866, 0.0, 1.0],
        [0.0, 0.0, 1.414, 1.0],
    ]
    corners = mv.antivector(corners_coords)

    # 1. Compute inertia operators (arity 1: bivector -> antibivector)
    I_lumped = simplex_inertia_lumped(corners)
    I_brute = simplex_inertia_brute(corners, n=100)
    I_random = simplex_inertia_random(corners, n_samples=100000)

    print("--- 3D Tetrahedron Inertia Tensor Comparison ---")
    print("Exact Lumped Inertia Matrix (6x6):\n", np.around(I_lumped.kernel, 3))
    print("Grid-Sampled Brute Inertia Matrix (6x6):\n", np.around(I_brute.kernel, 3))
    print("Monte Carlo Random Inertia Matrix (6x6):\n", np.around(I_random.kernel, 3))

    # 2. Linear map action: momentum = I(rate)
    rate = mv.bivector([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    momentum = I_lumped(rate)
    print("\nAngular Rate bivector:", rate.kernel)
    print("Resulting Momentum antibivector:", np.around(momentum.kernel, 4))

    # 3. Exact operator inverse: rate = I_inv(momentum)
    I_inv = I_lumped.inverse()
    recovered_rate = I_inv(momentum)
    print("Recovered Rate via I.inverse():", np.around(recovered_rate.kernel, 4))

    # 4. Kinetic energy: 0.5 * (rate & momentum)
    ke = 0.5 * rate.regressive(momentum)
    print("Kinetic Energy:", float(ke.kernel.item()))


if __name__ == "__main__":
    run_simplex_demo()

