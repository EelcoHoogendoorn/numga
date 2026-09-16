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

from examples import PLOT_DIR
from examples.electromagnetism.constitutive_plumbing import (
    B,
    Constitutive,
    Permittivity,
    V,
    mv,
    new_figure,
    phase_speeds,
    polarisation,
    dispersion,
    render_dispersion,
    t,
    x,
    y,
    z,
)


def main(plot_path: str = str(PLOT_DIR / "constitutive.png")) -> plt.Figure:
    """Build constitutive maps for media at rest and in motion, and read off their wave speeds."""
    eps, mu = 2.25, 1.0
    beta = 0.3

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

    axion: Constitutive = glass + 0.4 * B.dual()

    boost = (mv.zt * (np.arctanh(beta) / 2.0)).exp()
    moving: Constitutive = boost >> glass(boost << B)

    # -----------------------------------------------------------------------
    # 2. Checks on the maps themselves
    # -----------------------------------------------------------------------
    # The projector is idempotent and splits a field as expected. Spatial vectors square to
    # -1 in this signature, which is why the quadrics carry a minus sign: ε(x) = ε_x x. The
    # isotropic permeability lift reproduces the magnetic projector exactly.
    np.testing.assert_allclose(electric(electric).kernel, electric.kernel, atol=1e-14)
    np.testing.assert_allclose((electric(mv.tx) - mv.tx).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(electric(mv.xy).kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(permittivity(x).kernel, (2.25 * x).kernel, atol=1e-14)
    isotropic_inv = -(x * (x | V) + y * (y | V) + z * (z | V))
    np.testing.assert_allclose(isotropic_inv(B.dual().commutator(t)).wedge(t).dual_inverse().kernel, magnetic.kernel, atol=1e-14)

    # -----------------------------------------------------------------------
    # 3. Plane waves: the wave map and what it predicts
    # -----------------------------------------------------------------------
    # A plane wave with wave vector k and polarisation a has F = k ∧ a, so the source-free
    # equation k . G = 0 reads k . χ(k ∧ a) = 0. With a open that is a map on polarisations,
    # and a wave exists exactly where it loses rank. Scanning the phase speed then gives:
    # 1 / n for glass; two speeds for the crystal, polarised along the axes with ε = 2.25
    # and 1.5, the null vector of the wave map being the polarisation; two speeds for the
    # ferrite through μ instead of ε; nothing new for the axion term, which bulk waves
    # cannot see; and for the moving glass the exact relativistic Fresnel drag, with and
    # against the flow.
    speeds = np.linspace(0.05, 1.5, 6001)
    along_z = np.array([0.0, 0.0, 1.0])
    n = np.sqrt(eps * mu)
    with_flow = (1.0 / n + beta) / (1.0 + beta / n)
    against_flow = (1.0 / n - beta) / (1.0 - beta / n)

    np.testing.assert_allclose(phase_speeds(glass, along_z, speeds), [1.0 / n], atol=1e-3)
    slow, fast = phase_speeds(crystal, along_z, speeds)
    np.testing.assert_allclose([slow, fast], [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)], atol=1e-3)
    np.testing.assert_allclose(polarisation(crystal, along_z, slow).wedge(x).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose(polarisation(crystal, along_z, fast).wedge(y).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose(phase_speeds(ferrite, along_z, speeds), [1.0 / np.sqrt(2.25 * 2.0), 1.0 / np.sqrt(2.25)], atol=1e-3)
    np.testing.assert_allclose(phase_speeds(axion, along_z, speeds), phase_speeds(glass, along_z, speeds), atol=1e-12)
    np.testing.assert_allclose(phase_speeds(moving, along_z, speeds), [with_flow], atol=1e-3)
    np.testing.assert_allclose(phase_speeds(moving, -along_z, speeds), [against_flow], atol=1e-3)

    # -----------------------------------------------------------------------
    # 4. Draw
    # -----------------------------------------------------------------------
    fig, ax = new_figure()
    curves = {
        "glass at rest": dispersion(glass, along_z, speeds),
        "glass with axion term": dispersion(axion, along_z, speeds),
        "crystal along z": dispersion(crystal, along_z, speeds),
        "ferrite along z": dispersion(ferrite, along_z, speeds),
        "glass moving with the wave": dispersion(moving, along_z, speeds),
        "glass moving against the wave": dispersion(moving, -along_z, speeds),
    }
    expected = {
        "glass at rest": [1.0 / n],
        "glass with axion term": [],
        "crystal along z": [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)],
        "ferrite along z": [1.0 / np.sqrt(2.25 * 2.0), 1.0 / np.sqrt(2.25)],
        "glass moving with the wave": [with_flow],
        "glass moving against the wave": [against_flow],
    }
    render_dispersion(ax, speeds, curves, expected)
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
