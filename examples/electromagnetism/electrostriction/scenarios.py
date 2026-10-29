"""Electrostriction: loading and slowly turning the field applied to a polarizable elastic lattice."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.backend.jax import derivative
from examples.electromagnetism.electrostriction import core

n_rings = 2
lattice_spacing = 1.0
polarizability = 0.035 * lattice_spacing**3
stiffness = 1.0
tether = 0.15 * stiffness
field_strength = 3.05
n_newton_steps = 4
n_load_samples = 25
n_frames = 126
duration_ms = 40
dipole_scale = 4.0
field_scale = 0.65 / field_strength


# --- math -----------------------------------------------------------------------------
def lattice() -> core.Lattice:
    indices, bonds = core.hexagonal_indices(n_rings)
    # The points one spacing along the two lattice directions, which meet at sixty degrees.
    first_axis = core.mv.x.dual()                                                  # [] Point-
    axes = stack((first_axis, (core.mv.xy * (-np.pi / 6)).exp() >> first_axis))  # [axes] Point-
    rest = (lattice_spacing * (axes * indices).sum(axis=-1)).field()               # Point-[particles]
    return core.Lattice(rest, bonds, polarizability, stiffness, tether)


def loading(model: core.Lattice) -> tuple[np.ndarray, core.Field, core.Point, core.Point]:
    strengths = np.linspace(0, field_strength, n_load_samples)
    applied = core.field_along(core.mv.x.dual()) * strengths
    positions = model.equilibrium(applied, n_newton_steps)
    dipoles = model.response(positions).solve(applied)
    return strengths, applied, positions, dipoles


def turning(model: core.Lattice) -> tuple[core.Field, core.Point, core.Point]:
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    # The field at full strength turns once around at an even rate, so the loop runs at an even pace. Every frame is
    # an independent static equilibrium; the angle is a control setting rather than time.
    turns = (core.mv.xy * (-angles / 2)).exp()                                     # [frames] Rotor+
    applied = core.field_along(turns >> core.mv.x.dual()) * field_strength          # [frames] Field+
    positions = model.equilibrium(applied, n_newton_steps)
    dipoles = model.response(positions).solve(applied)
    return applied, positions, dipoles


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.electrostriction import render

    model = lattice()
    strengths, applied, positions, dipoles = loading(model)
    rest_dipoles = model.response(model.rest).solve(applied[-1])
    save_figure(render.draw_lattice(model.rest, model.bonds, model.rest, rest_dipoles, applied[-1],
                                    dipole_scale, field_scale), "electrostriction_fixed")
    save_figure(render.draw_lattice(model.rest, model.bonds, positions[-1], dipoles[-1], applied[-1],
                                    dipole_scale, field_scale), "electrostriction_relaxed")
    parallel = core.strain(positions, model.rest, core.mv.x)
    transverse = core.strain(positions, model.rest, core.mv.y)
    save_figure(render.draw_loading(strengths, parallel, transverse), "electrostriction_loading")
    turning_fields, shapes, polarizations = turning(model)
    frames = render.animate_lattice(model.rest, model.bonds, shapes, polarizations, turning_fields,
                                     dipole_scale, field_scale)
    save_animation(frames, "electrostriction", duration_ms)

    # --- checks
    # The Newton relaxation has converged at the strongest field, the unloaded lattice rests at
    # its reference sites, and the lattice contracts along the field.
    gradient = derivative(lambda value: model.energy(value, applied[-1]))(positions[-1])
    np.testing.assert_allclose(gradient.kernel, 0, atol=1e-3)
    np.testing.assert_allclose((positions[0] - model.rest).kernel, 0, atol=1e-12)
    assert parallel[-1] < 0


if __name__ == "__main__":
    main()
