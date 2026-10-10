"""Loading and slowly turning the field applied to a polarizable elastic lattice."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.backend.jax import derivative
from examples.electromagnetism.electrostriction import core

RINGS = 2
SPACING = 1.0
POLARIZABILITY = 0.035 * SPACING**3
STIFFNESS = 1.0
TETHER = 0.15 * STIFFNESS
FIELD = 3.05
NEWTON_STEPS = 4
LOAD_SAMPLES = 25
FRAMES = 72
DURATION_MS = 70
DIPOLE_SCALE = 4.0
FIELD_SCALE = 0.65 / FIELD


# --- math -----------------------------------------------------------------------------
def lattice() -> core.Lattice:
    indices, bonds = core.hexagonal_indices(RINGS)
    # The two lattice directions meet at sixty degrees.
    across = (core.mv.xy * (-np.pi / 6)).exp() >> core.mv.x
    axes = stack((core.mv.x, across))                                         # [axes] Vector
    rest = (SPACING * (axes * indices).sum(axis=-1)).field()                  # Vector[particles]
    return core.Lattice(rest, bonds, POLARIZABILITY, STIFFNESS, TETHER)


def loading(model: core.Lattice) -> tuple[np.ndarray, core.Vector, core.Vector, core.Vector]:
    strengths = np.linspace(0, FIELD, LOAD_SAMPLES)
    applied = core.mv.x * strengths
    positions = model.equilibrium(applied, NEWTON_STEPS)
    dipoles = model.response(positions).solve(applied)
    return strengths, applied, positions, dipoles


def turning(model: core.Lattice) -> tuple[core.Vector, core.Vector, core.Vector]:
    phase = np.linspace(0, 2 * np.pi, FRAMES, endpoint=False)
    # A smooth rise and fall closes the loop at the unloaded lattice. Every frame is an
    # independent static equilibrium; the phase is a control setting rather than time.
    strength = FIELD * (1 - np.cos(phase)) / 2
    applied = ((core.mv.xy * (-phase / 2)).exp() >> core.mv.x) * strength
    positions = model.equilibrium(applied, NEWTON_STEPS)
    dipoles = model.response(positions).solve(applied)
    return applied, positions, dipoles


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.electrostriction import render

    model = lattice()
    strengths, applied, positions, dipoles = loading(model)
    fixed = model.response(model.rest).solve(applied[-1])
    save_figure(render.draw_lattice(model.rest, model.bonds, model.rest, fixed, applied[-1],
                                    DIPOLE_SCALE, FIELD_SCALE), "electrostriction_fixed")
    save_figure(render.draw_lattice(model.rest, model.bonds, positions[-1], dipoles[-1], applied[-1],
                                    DIPOLE_SCALE, FIELD_SCALE), "electrostriction_relaxed")
    parallel = core.strain(positions, model.rest, core.mv.x)
    transverse = core.strain(positions, model.rest, core.mv.y)
    save_figure(render.draw_loading(strengths, parallel, transverse), "electrostriction_loading")
    turned, shapes, polarizations = turning(model)
    frames = render.animate_lattice(model.rest, model.bonds, shapes, polarizations, turned,
                                     DIPOLE_SCALE, FIELD_SCALE)
    save_animation(frames, "electrostriction", DURATION_MS)

    # --- checks
    # The Newton relaxation has converged at the strongest field, the unloaded lattice rests at
    # its reference sites, and the lattice contracts along the field.
    gradient = derivative(lambda value: model.energy(value, applied[-1]))(positions[-1])
    np.testing.assert_allclose(gradient.kernel, 0, atol=1e-3)
    np.testing.assert_allclose((positions[0] - model.rest).kernel, 0, atol=1e-12)
    assert parallel[-1].to_array() < 0


if __name__ == "__main__":
    main()
