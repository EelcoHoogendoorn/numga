"""Elastic waves in crystals: three cubic materials, their waves along chosen headings, and the focusing
of their energy."""

from __future__ import annotations

import numpy as np

from examples.mechanics.crystal_waves import core

# Cubic elastic constants c11, c12, c44 in GPa and density in g/cm^3: km/s for the speeds.
# Fused silica is isotropic, `c12 == c11 - 2 * c44`; silicon is moderately and beta-brass strongly
# anisotropic, measured by the ratio `2 * c44 / (c11 - c12)`: 1, 1.6 and 8.5.
NAMES = ["fused silica", "silicon", "beta-brass"]
C11 = np.array([78.5, 165.7, 129.1])
C12 = np.array([16.1, 63.9, 109.7])
C44 = np.array([31.2, 79.6, 82.4])
DENSITY = np.array([2.20, 2.33, 7.60])

MODES = ("slow shear", "fast shear", "compression")

# Headings sampled over the sphere for the focusing, and their seed.
N_HEADINGS = 400_000
SEED = 0
# Headings around the cube face for the wave fronts.
N_ANGLES = 4000


def crystals() -> tuple[core.Stiffness, np.ndarray]:
    """Every material's stiffness from its elastic constants, and its density, on the first axis."""
    return core.stiffness(C11, C12, C44), DENSITY                               # [materials] Stiffness


def focusing(n_headings: int, seed: int) -> tuple[core.Antivector, core.Vector, core.Scalar]:
    """Group velocities of the three waves in every material, for headings spread evenly over the
    sphere; then the headings, and the stiffness eigenvalues of the three waves along them."""
    stiffness, density = crystals()
    heading = core.mv.vector(np.random.default_rng(seed).normal(size=(n_headings, 3))).normalized()   # [headings] Vector+
    values, polarization = core.waves(stiffness[:, None], heading)                # [materials, headings, 3] each
    velocity = core.energy_flow(stiffness[:, None], heading, polarization, density[:, None])   # [materials, headings, 3] Antivector-
    return velocity, heading, values


def wave_fronts(n_angles: int) -> core.Vector:
    """Group velocities in every material for headings in the xy plane, a cube face: where each
    wave's energy is after unit time, a section through its wave surface.

    The waves are ordered by speed, so where the two shear speeds cross, the slower and the faster
    shear wave swap polarizations; each curve follows a speed rank, not one polarization.
    """
    stiffness, density = crystals()
    angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    # The x axis turned through every angle in the face z = 0.
    heading = (core.mv.xy * (-angles / 2)).exp() >> core.mv.x                    # [headings] Vector+
    _, polarization = core.waves(stiffness[:, None], heading)                     # [materials, headings, 3] Antivector-
    velocity = core.energy_flow(stiffness[:, None], heading, polarization, density[:, None])   # [materials, headings, 3] Antivector-
    return velocity


def main() -> None:
    from examples.util.animation import save_figure
    from examples.mechanics.crystal_waves import render

    velocity, heading, values = focusing(N_HEADINGS, SEED)
    save_figure(render.draw_focusing(velocity, NAMES), "crystal_waves_focusing")
    save_figure(render.draw_wave_fronts(wave_fronts(N_ANGLES), NAMES), "crystal_waves_fronts")

    # checks
    stiffness, density = crystals()
    # Read by its heading, the group velocity has the phase speed.
    phase = (values / density[:, None, None]).square_root()
    np.testing.assert_allclose((heading[:, None] & velocity).to_array(), phase.to_array(), rtol=1e-8)
    # The textbook speeds along a cube edge and a face diagonal.
    edge_and_diagonal = core.waves(stiffness[:, None], core.mv.vector([[1.0, 0, 0], [1, 1, 0]]).normalized())[0]   # [materials, 2, 3] Scalar+
    speeds = (edge_and_diagonal / density[:, None, None]).square_root().to_array()      # [materials, 2, 3]
    np.testing.assert_allclose(speeds[:, 0], np.sqrt(np.stack([C44, C44, C11], axis=-1) / density[:, None]), rtol=1e-8)
    expected = np.sort(np.stack([(C11 - C12) / 2, C44, (C11 + C12 + 2 * C44) / 2], axis=-1), axis=-1)
    np.testing.assert_allclose(speeds[:, 1], np.sqrt(expected / density[:, None]), rtol=1e-8)


if __name__ == "__main__":
    main()
