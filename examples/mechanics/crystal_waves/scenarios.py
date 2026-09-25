"""Scenes for elastic waves in cubic crystals.

One function per figure. Builds each crystal's stiffness, calls the mathematics in `core`, and
returns the geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from examples.mechanics.crystal_waves import core

# Cubic elastic constants C11, C12, C44 in GPa and density in g/cm^3: km/s for the speeds.
# Fused silica is isotropic, C12 = C11 - 2 C44; silicon is moderately and beta-brass strongly
# anisotropic, measured by the ratio 2 C44 / (C11 - C12): 1, 1.6 and 8.5.
MATERIALS = {
    "fused silica": (78.5, 16.1, 31.2, 2.20),
    "silicon": (165.7, 63.9, 79.6, 2.33),
    "beta-brass": (129.1, 109.7, 82.4, 7.60),
}
MODES = ("slow shear", "fast shear", "compression")


def crystal(name: str) -> tuple[core.Stiffness, float]:
    """A cubic crystal's stiffness from its elastic constants, and its density."""
    c11, c12, c44, density = MATERIALS[name]
    return core.stiffness(c11, c12, c44), density


def focusing(name: str, count: int, seed: int) -> core.Vector:
    """Group velocities of the three waves for headings spread evenly over the sphere."""
    stiffness, density = crystal(name)
    heading = core.mv.vector(np.random.default_rng(seed).normal(size=(count, 3))).normalized()   # [count] Vector
    values, polarization = core.waves(stiffness, heading)                          # [count, 3] each
    velocity = core.energy_flow(stiffness, heading, polarization, density)         # [count, 3] Vector

    # --- checks
    # Along its heading, the group velocity has the phase speed.
    phase = (values / density).square_root()
    np.testing.assert_allclose((velocity | heading[:, None]).to_array(), phase.to_array(), rtol=1e-8)
    return velocity


def wave_fronts(name: str, count: int) -> core.Vector:
    """Group velocities for headings around the cube face z = 0: where each wave's energy is after
    unit time, a section through its wave surface.

    The waves are ordered by speed, so where the two shear speeds cross, the slower and the faster
    shear wave swap polarizations; each curve follows a speed rank, not one polarization.
    """
    stiffness, density = crystal(name)
    angle = np.linspace(0.0, 2 * np.pi, count, endpoint=False)
    heading = core.mv.vector(np.stack([np.cos(angle), np.sin(angle), np.zeros(count)], axis=-1))
    _, polarization = core.waves(stiffness, heading)
    velocity = core.energy_flow(stiffness, heading, polarization, density)

    # --- checks
    # The textbook speeds along a cube edge and a face diagonal.
    c11, c12, c44, _ = MATERIALS[name]
    edge, diagonal = core.waves(stiffness, core.mv.vector(np.array([[1.0, 0, 0], [1, 1, 0]])).normalized())[0] / density
    np.testing.assert_allclose(edge.square_root().to_array(), np.sqrt(np.array([c44, c44, c11]) / density), rtol=1e-8)
    expected = np.sort([(c11 - c12) / 2, c44, (c11 + c12 + 2 * c44) / 2])
    np.testing.assert_allclose(diagonal.square_root().to_array(), np.sqrt(expected / density), rtol=1e-8)
    return velocity


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.mechanics.crystal_waves import render

    names = list(MATERIALS)
    save_figure(render.draw_focusing([focusing(name, 400_000, 0) for name in names], names, 400), "crystal_waves_focusing")
    save_figure(render.draw_wave_fronts([wave_fronts(name, 4000) for name in names], names), "crystal_waves_fronts")
