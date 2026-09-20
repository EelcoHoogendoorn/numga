"""Sampling, coefficient-layout boundaries and readout for the inertia example."""

import numpy as np

from numga import Extensor, NumpyContext


def cloud(context: NumpyContext, seed: int) -> tuple[Extensor, np.ndarray]:
    """Normalize random points and place the cloud with a random motor."""
    ga, mv = context.algebra, context.multivector
    rng = np.random.default_rng(seed)
    points = mv.antivector(rng.normal(size=(80, ga.dimension))).normalized()
    placement = mv.bivector(rng.normal(size=len(ga.subspace.bivector())) * 0.3).exp()
    return placement >> points, rng.uniform(0.5, 1.5, size=80)


def report(cloud_energy: Extensor, cloud_diagonal: Extensor, reference: Extensor,
           moved: Extensor, recovered: Extensor) -> None:
    """Read out both diagonalizations and the controlled round trip."""
    for name, before, after in (("Point cloud", cloud_energy, cloud_diagonal),
                               ("Diagonal round trip", moved, recovered)):
        original, aligned = before.kernel[0], after.kernel[0]
        scale = np.linalg.norm(aligned)
        print(f"\n{name}: recovered kinetic-energy form:\n",
              np.round(np.real_if_close(aligned), 8))
        print(f"Relative off-diagonal energy before: {np.linalg.norm(original - np.diag(np.diag(original))) / scale:.3g}")
        print(f"Relative off-diagonal energy after:  {np.linalg.norm(aligned - np.diag(np.diag(aligned))) / scale:.3g}")

    expected = np.sort(np.diag(reference.kernel[0]).real)
    actual = np.sort(np.diag(recovered.kernel[0]).real)
    print(f"Relative diagonal-spectrum recovery error: {np.linalg.norm(actual - expected) / np.linalg.norm(expected):.3g}")
