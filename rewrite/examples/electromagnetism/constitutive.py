"""The constitutive extensor of a medium in Spacetime Algebra: birefringence and Fresnel drag.

In a medium Maxwell's equations split into the field bivector F and the excitation bivector
G, related by a linear map G = χ(F). That map is a Bivector-to-Bivector extensor, and every
medium in this file is built from it: an observer's electric and magnetic projectors give
the isotropic dielectric, permittivity and permeability quadrics lifted through the
observer give a crystal and a ferrite, the pseudoscalar gives the axion term, and
conjugating by a boost gives a moving medium. Plane waves exist where the wave map
a -> k · χ(k ∧ a) loses rank, which yields the phase speeds and polarisations, the
birefringence of the crystal, the invisibility of the axion term, and the Fresnel drag of
the moving medium.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples import PLOT_DIR
from examples.electromagnetism.constitutive_plumbing import (
    B,
    Spatial,
    minimum_speeds,
    draw_media,
    Constitutive,
    Permittivity,
    V,
    mv,
    t,
    x,
    y,
    z,
)


def main(plot_path: str = str(PLOT_DIR / "constitutive.png")) -> plt.Figure:
    """Build constitutive maps for media at rest and in motion, and read off their wave speeds."""
    eps, mu = 2.25, 1.0
    beta = 0.3
    axion_coupling = mv.scalar([0.4])

    # -----------------------------------------------------------------------
    # 1. Every medium is one bivector map, built from the observer
    # -----------------------------------------------------------------------
    # For observer t the electric part of F is (F . t) ^ t; with the bivector slot open that
    # is a projector, and the magnetic projector is its complement, a bare subspace being
    # the identity on it. Glass scales the two parts. A crystal lifts a permittivity quadric
    # on spatial vectors through the observer; a ferrite does the same for the inverse
    # permeability on the dual field. The axion term is the dual itself. A moving medium is
    # the rest map with fields pulled back and excitations pushed forward through the boost.
    electric: Constitutive = B.commutator(t).wedge(t)
    magnetic: Constitutive = B - electric
    glass: Constitutive = eps * electric + (1.0 / mu) * magnetic

    permittivity: Permittivity = -(2.25 * x * (x | V) + 1.5 * y * (y | V) + 1.5 * z * (z | V))
    crystal: Constitutive = permittivity(B.commutator(t)).wedge(t) + (1.0 / mu) * magnetic

    permeability_inv: Permittivity = -(1.0 * x * (x | V) + 0.5 * y * (y | V) + 1.0 * z * (z | V))
    ferrite: Constitutive = eps * electric + permeability_inv(B.dual().commutator(t)).wedge(t).dual_inverse()

    axion: Constitutive = glass + axion_coupling * B.dual()

    boost = (mv.zt * (np.arctanh(beta) / 2.0)).exp()
    moving: Constitutive = boost >> glass(boost << B)

    # 2. A wave exists where k . chi(k ^ a) loses rank. Leave polarization a open
    # and evaluate all media and trial speeds as one batch of vector maps.
    speeds = mv.scalar(np.linspace(0.05, 1.5, 6001)[:, None])
    media = stack((glass, axion, crystal, ferrite, moving, moving))
    directions = stack((z, z, z, z, z, -z))
    k = speeds * t + directions[:, None]
    wave = k.commutator(media[:, None](k.wedge(Spatial)))
    curves = wave.svdvals()[..., -1]
    allowed = [minimum_speeds(speeds, curve) for curve in curves]

    # Bind the crystal's two allowed speeds and take the right singular vectors
    # of the same wave map. Its null modes give the two polarizations.
    crystal_k = allowed[2] * t + z
    crystal_wave = crystal_k.commutator(crystal(crystal_k.wedge(Spatial)))
    _, _, vectors = crystal_wave.svd()
    polarizations = vectors[..., -1]

    # Relativistic velocity addition predicts the moving glass's phase speeds.
    n = np.sqrt(eps * mu)
    with_flow = (1.0 / n + beta) / (1.0 + beta / n)
    against_flow = (1.0 / n - beta) / (1.0 - beta / n)
    expected = [[1.0 / n], [], [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)],
                [1.0 / np.sqrt(2.25 * 2.0), 1.0 / np.sqrt(2.25)], [with_flow], [against_flow]]
    fig = draw_media(speeds, curves, expected, plot_path)

    # --- checks -------------------------------------------------------------
    np.testing.assert_allclose(electric(electric).kernel, electric.kernel, atol=1e-14)
    np.testing.assert_allclose((electric(mv.tx) - mv.tx).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(electric(mv.xy).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(permittivity(x).kernel, (2.25 * x).kernel, atol=1e-14)
    isotropic_inv = -(x * (x | V) + y * (y | V) + z * (z | V))
    np.testing.assert_allclose(isotropic_inv(B.dual().commutator(t)).wedge(t).dual_inverse().kernel, magnetic.kernel, atol=1e-14)
    for index in (0, 2, 3, 4, 5):
        np.testing.assert_allclose(allowed[index].kernel[..., 0], expected[index], atol=1e-3)
    np.testing.assert_allclose(allowed[1].kernel, allowed[0].kernel, atol=1e-12)
    np.testing.assert_allclose(polarizations[0].wedge(x).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose(polarizations[1].wedge(y).kernel, 0.0, atol=1e-6)
    return fig


if __name__ == "__main__":
    main()
    plt.show()
