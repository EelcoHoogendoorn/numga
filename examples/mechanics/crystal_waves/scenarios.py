"""Scenes for elastic waves in cubic crystals.

One function per figure. Builds each crystal's stiffness, calls the mathematics in `core`, and
returns the geometry for `render`.
"""

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


def crystals() -> tuple[core.Stiffness, np.ndarray]:
    """Every material's stiffness from its elastic constants, and its density, on the first axis."""
    return core.stiffness(C11, C12, C44), DENSITY                               # [materials] Stiffness


def focusing(count: int, seed: int) -> core.Vector:
    """Group velocities of the three waves in every material, for headings spread evenly over the
    sphere."""
    stiffness, density = crystals()
    heading = core.mv.vector(np.random.default_rng(seed).normal(size=(count, 3))).normalized()   # [count] Vector
    values, polarization = core.waves(stiffness[:, None], heading)                # [materials, count, 3] each
    velocity = core.energy_flow(stiffness[:, None], heading, polarization, density[:, None])   # [materials, count, 3] Vector

    # --- checks
    # Along its heading, the group velocity has the phase speed.
    phase = (values / density[:, None, None]).square_root()
    np.testing.assert_allclose((velocity | heading[:, None]).to_array(), phase.to_array(), rtol=1e-8)
    return velocity


def wave_fronts(count: int) -> core.Vector:
    """Group velocities in every material for headings in the xy plane, a cube face: where each
    wave's energy is after unit time, a section through its wave surface.

    The waves are ordered by speed, so where the two shear speeds cross, the slower and the faster
    shear wave swap polarizations; each curve follows a speed rank, not one polarization.
    """
    stiffness, density = crystals()
    angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    # The x axis turned through every angle in the face z = 0.
    heading = (core.mv.xy * (-angle / 2)).exp() >> core.mv.x                    # [count] Vector
    _, polarization = core.waves(stiffness[:, None], heading)                     # [materials, count, 3] Vector
    velocity = core.energy_flow(stiffness[:, None], heading, polarization, density[:, None])   # [materials, count, 3] Vector

    # --- checks
    # The textbook speeds along a cube edge and a face diagonal.
    values = core.waves(stiffness[:, None], core.mv.vector([[1.0, 0, 0], [1, 1, 0]]).normalized())[0]   # [materials, 2, 3] Scalar
    speeds = (values / density[:, None, None]).square_root().to_array()      # [materials, 2, 3]
    np.testing.assert_allclose(speeds[:, 0], np.sqrt(np.stack([C44, C44, C11], axis=-1) / density[:, None]), rtol=1e-8)
    expected = np.sort(np.stack([(C11 - C12) / 2, C44, (C11 + C12 + 2 * C44) / 2], axis=-1), axis=-1)
    np.testing.assert_allclose(speeds[:, 1], np.sqrt(expected / density[:, None]), rtol=1e-8)
    return velocity


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.mechanics.crystal_waves import render

    save_figure(render.draw_focusing(focusing(400_000, 0), NAMES, 400), "crystal_waves_focusing")
    save_figure(render.draw_wave_fronts(wave_fronts(4000), NAMES), "crystal_waves_fronts")
