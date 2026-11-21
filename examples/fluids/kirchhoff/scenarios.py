"""Scenes for Kirchhoff's equations: the added-mass ratios of spheroids from a flat disc to a slender rod, and an elongated ellipsoid given an impulsive blow without the fluid, nearly end-on, and nearly broadside."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import dim, stack
from examples.fluids.kirchhoff import core

mv = core.mv
x, y, z = core.axes                                                           # [] Plane+, Plane+, Plane+
long_axis, across_axis = core.directions[0], core.directions[1]               # [] Direction-, Direction-
# Spheroid aspect ratios from a disc to a rod, and quadrature node count for the confocal pencil.
n_aspects = 41
aspects = np.geomspace(0.1, 10, n_aspects)
n_widenings = 48
# Gliding ellipsoid parameters across three cases: in vacuum, nearly end-on in fluid, and nearly broadside in fluid.
density_unit = dim.mass / dim.length**3
semi_axes = mv.scalar([[3.0], [1.0], [1.0]], unit=dim.length)                 # [3] Scalar+, length
density = mv.scalar(1.0, unit=density_unit)                                   # [] Scalar+, mass/length^3
fluid_densities = mv.scalar([[0.0], [1.0], [1.0]], unit=density_unit)         # [3] Scalar+, mass/length^3
n_cases = fluid_densities.shape[0]
blow = mv.scalar(12.0, unit=dim.mass * dim.length / dim.time)                 # [] Scalar+, mass*length/time
headings = np.array([0.1, 0.1, np.pi / 2 - 0.1])
# Time step, total step count, frame stride, outline sample count, and frame duration.
dt = mv.scalar(0.16, unit=dim.time)                                           # [] Scalar+, time
n_steps = 250
n_steps_per_frame = 2
n_outline = 48
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def energy_ratios() -> core.Scalar:
    """Fluid-to-solid kinetic energy ratios at equal density across spheroid aspect ratios for axial shift, transverse shift, and transverse turn."""
    spheroid_axes = mv.scalar(
        np.stack([aspects, np.ones_like(aspects), np.ones_like(aspects)], axis=-1)[..., None],
        unit=dim.length,
    )                                                                         # [n_aspects, 3] Scalar+, length
    stretches = core.ellipsoid_stretch(spheroid_axes)                         # [n_aspects] Direction- <- Direction-, length
    shapes = stretches(stretches)                                             # [n_aspects] Direction- <- Direction-, length^2
    unit_density = mv.scalar(1.0, unit=density_unit)
    zero_density = mv.scalar(0.0, unit=density_unit)
    solid_inertia = core.total_inertia(shapes, unit_density, zero_density, n_widenings)[:, None]   # [n_aspects, 1] Forque- <- Twist+
    fluid_inertia = core.total_inertia(shapes, zero_density, unit_density, n_widenings)[:, None]   # [n_aspects, 1] Forque- <- Twist+
    rates = stack([mv.wx * (dim.length / dim.time), mv.wy * (dim.length / dim.time), mv.zx / dim.time])  # [3] Twist+, 1/time | length/time
    return (rates & fluid_inertia(rates)) / (rates & solid_inertia(rates))    # [n_aspects, 3] Scalar+


def launch_setup() -> tuple[core.Inertia, core.Forque, core.Motor]:
    """Total inertia per case, impulsive blow along the first spatial axis, and initial motor placements."""
    body_stretch = core.ellipsoid_stretch(semi_axes)                          # [] Direction- <- Direction-
    inertias = core.total_inertia(body_stretch(body_stretch), density, fluid_densities, n_widenings)   # [n_cases] Forque- <- Twist+
    impulse = blow * (core.centre & long_axis)                                # [] Forque-
    initial_motors = (mv.xy * (-headings / 2)).exp()                          # [n_cases] Motor+
    return inertias, impulse, initial_motors


def body_geometry() -> tuple[core.Point, core.Point]:
    """Planar cross-section outline and nose point of the gliding ellipsoid in its own frame."""
    angles = np.linspace(0, 2 * np.pi, n_outline, endpoint=False)
    circle = (mv.xy * (-angles / 2)).exp() >> long_axis                       # [n_outline] Direction-
    body_stretch = core.ellipsoid_stretch(semi_axes)                          # [] Direction- <- Direction-
    return core.centre + body_stretch(circle), core.centre + body_stretch(long_axis)   # [n_outline] Point-, [] Point-


def gliding_frames(n_run_steps: int) -> Iterator[core.Motor]:
    """Yield subsampled motor placements across all gliding cases over the requested step count."""
    inertias, impulse, initial_motors = launch_setup()
    return islice(core.trajectory(initial_motors, impulse, inertias, dt, n_run_steps), 0, None, n_steps_per_frame)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.kirchhoff import render

    ratios = energy_ratios()                                                  # [n_aspects, 3] Scalar+
    save_figure(render.draw_coefficients(aspects, ratios), "kirchhoff_added_mass")

    inertias, impulse, _ = launch_setup()
    body_outline, nose = body_geometry()                                      # [n_outline] Point-, [] Point-
    recorded_frames: list[core.Motor] = []

    def recording() -> Iterator[core.Motor]:
        for state in gliding_frames(n_steps):
            recorded_frames.append(state)
            yield state

    save_animation(render.animate_gliding(impulse, body_outline, nose, recording()), "kirchhoff_gliding", frame_duration_ms)
    motors = stack(recorded_frames)                                           # [..., n_cases] Motor+

    # checks
    # A sphere's added mass equals half the displaced fluid mass in every direction, and pure rotation moves no fluid.
    unit_stretch = core.ellipsoid_stretch(mv.scalar([[1.0], [1.0], [1.0]], unit=dim.length))   # [] Direction- <- Direction-, length
    sphere_fluid = core.total_inertia(
        unit_stretch(unit_stretch), mv.scalar(0.0, unit=density_unit), mv.scalar(1.0, unit=density_unit), n_widenings,
    )                                                                         # [] Forque- <- Twist+
    rates = stack([mv.wx * (dim.length / dim.time), mv.wy * (dim.length / dim.time), mv.zx / dim.time])  # [3] Twist+, 1/time | length/time
    np.testing.assert_allclose((rates & sphere_fluid(rates)).to_array(), [2 * np.pi / 3, 2 * np.pi / 3, 0], atol=1e-12)
    # Total kinetic energy of body and fluid is conserved across the trajectory within midpoint-integrator accuracy.
    body_impulses = motors << impulse                                         # [..., n_cases] Forque-
    energies = (body_impulses & inertias.inverse()(body_impulses)).to_array()
    np.testing.assert_allclose(energies / energies[0], 1, rtol=1e-2)
    # Without fluid the heading stays constant; nearly broadside rocks about broadside; nearly end-on swings past broadside.
    long_axes = motors >> long_axis                                           # [..., n_cases] Direction-
    headings_over_time = np.arctan2((y & long_axes).to_array(), (x & long_axes).to_array())
    np.testing.assert_allclose(headings_over_time[:, 0], headings[0], atol=1e-12)
    assert np.abs(headings_over_time[:, 2] - np.pi / 2).max() < 2 * (np.pi / 2 - headings[2])
    assert headings_over_time[:, 1].max() > np.pi / 2


if __name__ == "__main__":
    main()
