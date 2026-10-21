"""Scenes for Kirchhoff's equations: the added mass of spheroids from a disc to a rod, and an
elongated body given a blow nearly end-on, nearly broadside, and the same body without the fluid."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.fluids.kirchhoff import core

mv = core.mv
x, y, z = core.axes                                        # the planes square to the body's axes
long_axis, across = core.directions[0], core.directions[1]
# Spheroids from a disc to a rod: the semi-axis along the axis of symmetry over the other two.
ASPECTS = np.geomspace(0.1, 10, 41)
# The samples of the confocal family.
N_WIDENINGS = 48
# The gliding body: its semi-axes, its density and the fluid's, the blow it is given along the first
# axis of space, and its long axis's angle from the blow at the start, in each case: without the fluid,
# nearly end-on and nearly broadside.
SEMI_AXES = np.array([3, 1, 1])
DENSITY = 1
FLUID_DENSITIES = np.array([0, 1, 1])
BLOW = 12
HEADINGS = np.array([0.1, 0.1, np.pi / 2 - 0.1])
# The time step, the steps, the steps per frame, the samples around the body's outline, and the
# duration of a frame.
DT = 0.2
N_STEPS = 200
N_STEPS_PER_FRAME = 2
N_OUTLINE = 48
DURATION_MS = 50


# --- math -----------------------------------------------------------------------------
def coefficients() -> core.Scalar:
    """The fluid's kinetic energy over the solid's, at equal densities, for spheroids of each aspect
    `[aspects, 3]`: shifting along the axis of symmetry, shifting across it, and turning it."""
    semi_axes = np.stack([ASPECTS, np.ones_like(ASPECTS), np.ones_like(ASPECTS)], axis=-1)   # [aspects, 3]
    stretches = core.stretch(semi_axes)                                       # [aspects] Stretch
    shapes = stretches(stretches)                                             # [aspects] Shape
    solid = core.inertia(shapes, 1, 0, N_WIDENINGS)[:, None]                  # [aspects, 1] Forque <- Twist
    fluid = core.inertia(shapes, 0, 1, N_WIDENINGS)[:, None]                  # [aspects, 1] Forque <- Twist
    rates = stack([mv.w ^ x, mv.w ^ y, z ^ x])                                # [3] Twist
    return (rates & fluid(rates)) / (rates & solid(rates))                    # [aspects, 3] Scalar


def launch() -> tuple[core.Inertia, core.Forque, core.Motor]:
    """Each case's inertia `[cases]`, the impulse of the blow along the first axis of space through the
    body's centre, and the body's placement at the start, its long axis turned from the blow."""
    stretch = core.stretch(SEMI_AXES)                                         # [] Stretch
    inertia = core.inertia(stretch(stretch), DENSITY, FLUID_DENSITIES, N_WIDENINGS)   # [cases] Forque <- Twist
    impulse = BLOW * (core.CENTRE & long_axis)                                # [] Forque
    starts = ((x ^ y) * (-HEADINGS / 2)).exp()                                # [cases] Motor
    return inertia, impulse, starts


def gliding() -> core.Motor:
    """Each case's placement over the time span `[steps + 1, cases]`."""
    inertia, impulse, starts = launch()
    return stack(list(core.glide(starts, impulse, inertia, DT, N_STEPS)))     # [steps + 1, cases] Motor


def outline() -> core.Point:
    """The body's outline in the plane of its first two axes, in its own frame `[outline + 1]`: the unit
    circle turned out by rotors, stretched onto the body."""
    angles = np.linspace(0, 2 * np.pi, N_OUTLINE + 1)
    circle = ((x ^ y) * (-angles / 2)).exp() >> long_axis                     # [outline + 1] Direction
    return core.CENTRE + core.stretch(SEMI_AXES)(circle)                      # [outline + 1] Point


def frames(motors: core.Motor) -> Iterator[tuple[core.Point, core.Point, core.Point]]:
    """Per frame, every case's outline `[cases, outline + 1]`, centre `[cases]` and nose, the tip of
    its long axis `[cases]`."""
    shape = outline()                                                         # [outline + 1] Point
    nose = core.CENTRE + core.stretch(SEMI_AXES)(long_axis)                   # [] Point
    for index in range(0, motors.shape[0], N_STEPS_PER_FRAME):
        placed = motors[index]                                                # [cases] Motor
        yield placed[:, None] >> shape, placed >> core.CENTRE, placed >> nose


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.fluids.kirchhoff import render

    ratios = coefficients()                                                   # [aspects, 3] Scalar
    save_figure(render.draw_coefficients(ASPECTS, ratios), "kirchhoff_added_mass")
    motors = gliding()                                                        # [steps + 1, cases] Motor
    inertia, impulse, _ = launch()
    save_animation(render.animate_gliding(impulse, frames(motors)), "kirchhoff_gliding", DURATION_MS)

    # checks
    # A sphere's added mass is half the mass of the fluid it displaces, whichever way it moves, and a turn
    # moves no fluid.
    ball = core.stretch(np.array([1, 1, 1]))                                  # [] Stretch
    sphere = core.inertia(ball(ball), 0, 1, N_WIDENINGS)                      # [] Forque <- Twist
    rates = stack([mv.w ^ x, mv.w ^ y, z ^ x])                                # [3] Twist
    np.testing.assert_allclose((rates & sphere(rates)).to_array(), [2 * np.pi / 3, 2 * np.pi / 3, 0], atol=1e-12)
    # The kinetic energy of body and fluid is kept to within the step's error.
    seen = motors << impulse                                                  # [steps + 1, cases] Forque
    energies = (seen & inertia.inverse()(seen)).to_array()                    # [steps + 1, cases]
    np.testing.assert_allclose(energies / energies[0], 1, rtol=1e-2)
    # Without the fluid the body keeps its heading; given the blow nearly broadside it stays so; nearly
    # end-on it turns past broadside.
    pointing = motors >> long_axis                                            # [steps + 1, cases] Direction
    headings = np.arctan2((y & pointing).to_array(), (x & pointing).to_array())
    np.testing.assert_allclose(headings[:, 0], HEADINGS[0], atol=1e-12)
    assert np.abs(headings[:, 2] - np.pi / 2).max() < 2 * (np.pi / 2 - HEADINGS[2])
    assert headings[:, 1].max() > np.pi / 2


if __name__ == "__main__":
    main()
