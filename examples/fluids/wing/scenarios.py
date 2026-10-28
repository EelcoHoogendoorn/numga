"""A wing pitched up and back in a steady stream: the pressure around it, its streamlines, and its lift;
and the flow past a cylinder deformed into the flow past the wing."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.fluids.wing import core

# The cylinder's centre, behind and above the origin before the map: the wing's thickness and camber.
THICKNESS = 0.1
CAMBER = 0.08
SCALE = 1.0
SPEED = 1.0
DENSITY = 1.0
LOWEST_ATTACK = np.deg2rad(-4.0)
HIGHEST_ATTACK = np.deg2rad(12.0)
# The rings and angles the checks sample; the animations draw from fewer, all a picture needs.
N_RINGS = 140
N_ANGLES = 360
N_DRAWN_RINGS = 35
N_DRAWN_ANGLES = 90
OUTER_DISTANCE = 6.0
# README picture: the sweep's frames and their duration make plots/wing.gif, 400 pixels across and under 1.5 MB,
# tuned as a visual for the README. Don't change them; to experiment, copy them.
N_FRAMES = 48
SWEEP_DURATION_MS = 100
N_MORPH_FRAMES = 60
MORPH_ATTACK = np.deg2rad(8.0)
DURATION_MS = 60
# The stream, along the wing's chord when level.
stream = SPEED * core.mv.x.dual()                                            # [] Point-


# --- math -----------------------------------------------------------------------------
def wing(attack: float) -> core.Wing:
    """The wing pitched nose up by the angle of attack: its cylinder's centre and its chord, level along the
    stream, turned by one rotor."""
    turn = (core.mv.xy * (attack / 2)).exp()                                 # [] Rotor+
    return core.Wing(turn >> (-THICKNESS * core.mv.x + CAMBER * core.mv.y).dual(), SCALE, turn >> core.mv.x.dual())


def trailing_edge(wing: core.Wing) -> core.Point:
    """The map's critical point that makes the wing's sharp trailing edge."""
    return wing.scale * wing.chord


def morph() -> Iterator[tuple[core.Point, core.Point, core.Bivector, core.Force, core.Point, core.Point]]:
    """The flow past the cylinder as the map's critical points move apart from its centre out to the
    trailing edge, deforming the cylinder into the wing, with the wing's circulation throughout; the
    critical points' images, where the map doubles angles; and the cylinder's centre before the map,
    whose offset from where the critical points start shapes the wing."""
    pitched = wing(MORPH_ATTACK)
    plane = core.rings(pitched, N_DRAWN_RINGS, N_DRAWN_ANGLES, OUTER_DISTANCE)   # [rings, angles] Point-
    # The way out alone; the animation plays it forward and back.
    spread = (1 - np.cos(np.linspace(0, np.pi, N_MORPH_FRAMES // 2 + 1))) / 2
    for fraction in spread:
        critical = fraction * trailing_edge(pitched)                         # [] Point-
        images = 2 * stack([critical, -critical])                            # [2] Point-
        points, velocity, _, flux = core.flow(pitched, stream, plane, critical)
        yield points, velocity, flux, core.lift(pitched, stream, DENSITY), images, pitched.centre


# README picture: this makes plots/wing.gif, tuned as a visual for the README. Don't change it; to
# experiment, copy it.
def sweep() -> Iterator[tuple[core.Point, core.Point, core.Bivector, core.Force]]:
    """The flow past the wing and its lift as the wing pitches up; the animation plays it forward and back."""
    phase = (1 - np.cos(np.linspace(0, np.pi, N_FRAMES // 2 + 1))) / 2
    for attack in LOWEST_ATTACK + (HIGHEST_ATTACK - LOWEST_ATTACK) * phase:
        pitched = wing(attack)
        plane = core.rings(pitched, N_DRAWN_RINGS, N_DRAWN_ANGLES, OUTER_DISTANCE)   # [rings, angles] Point-
        points, velocity, _, flux = core.flow(pitched, stream, plane, trailing_edge(pitched))
        yield points, velocity, flux, core.lift(pitched, stream, DENSITY)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.fluids.wing import render

    # README picture: this makes plots/wing.gif, tuned as a visual for the README. Don't change it; to
    # experiment, copy it.
    save_animation(render.animate(sweep(), SPEED), "wing", SWEEP_DURATION_MS, forward_then_back=True)
    save_animation(render.animate_marked(morph(), SPEED), "wing_morph", DURATION_MS, forward_then_back=True)

    # checks
    pitched = wing(HIGHEST_ATTACK)
    plane = core.rings(pitched, N_RINGS, N_ANGLES, OUTER_DISTANCE)           # [rings, angles] Point-
    points, velocity, _, _ = core.flow(pitched, stream, plane, trailing_edge(pitched))
    # The velocity's derivative vanishes everywhere past the cylinder.
    _, gradient, _ = core.cylinder(pitched, stream, plane)
    np.testing.assert_allclose(core.derivative(gradient).kernel, 0.0, atol=1e-12)
    # Around every ring about the wing the velocity times each step adds up to the circulation, with
    # no flux: the derivative between any two rings vanishes too.
    next_idx = (np.arange(N_ANGLES) + 1) % N_ANGLES
    loops = ((velocity + velocity[:, next_idx]) / 2 * (points[:, next_idx] - points)).sum(axis=-1)
    np.testing.assert_allclose(loops.kernel - loops.kernel[:1], 0.0, atol=1e-11)
    # Each ring's own oriented area sets the sense its sum is read in.
    areas = (points ^ points[:, next_idx]).sum(axis=-1)                      # [rings] Bivector+
    np.testing.assert_allclose((loops * (areas / areas.norm()) - core.circulation(pitched, stream)).kernel, 0.0, atol=1e-3)
    # The pressure on the surface, low where the flow is fast, pushes on every step: on a displacement of
    # the wing it does the pressure times the area the step sweeps, `steps & Point`, in the sense the surface
    # winds. Summed around the wing it is the Kutta–Joukowski lift, across the stream, with no drag.
    surface, surface_velocity, _, _ = core.flow(pitched, stream, core.rings(pitched, 1, N_ANGLES, 1.0)[0], trailing_edge(pitched))
    steps = surface[next_idx] - surface                                      # [angles] Point-
    speed_squared = surface_velocity.dot(surface_velocity)                   # [angles] Scalar+
    pressure = -0.5 * DENSITY * (speed_squared + speed_squared[next_idx]) / 2   # [angles] Scalar+
    winding = (surface & surface[next_idx]).sum(axis=-1)                     # [] Scalar-
    force = (pressure * (steps & core.Point)).sum(axis=-1) * (winding / winding.norm())   # [] Scalar+ <- Point-
    np.testing.assert_allclose((force - (core.lift(pitched, stream, DENSITY) & core.Point)).kernel, 0.0, atol=1e-3)


if __name__ == "__main__":
    main()
