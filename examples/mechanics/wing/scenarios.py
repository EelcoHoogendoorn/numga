"""A wing pitched up and back in a steady stream: the pressure around it, its streamlines, and its lift;
and the flow past a cylinder deformed into the flow past the wing."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack

from examples.mechanics.wing import core

# The cylinder's centre, behind and above the origin before the map: the wing's thickness and camber.
THICKNESS = 0.1
CAMBER = 0.08
SCALE = 1.0
SPEED = 1.0
DENSITY = 1.0
LOWEST_ATTACK = np.deg2rad(-4.0)
HIGHEST_ATTACK = np.deg2rad(12.0)
# The rings and angles the checks sample; the animations draw from fewer, all a picture needs.
RINGS = 140
ANGLES = 360
DRAWN_RINGS = 35
DRAWN_ANGLES = 90
REACH = 6.0
FRAMES = 80
MORPH_FRAMES = 60
MORPH_ATTACK = np.deg2rad(8.0)
DURATION_MS = 60


# --- math -----------------------------------------------------------------------------
def wing() -> core.Wing:
    """The wing level, its chord along the stream."""
    return core.Wing(-THICKNESS * core.mv.x + CAMBER * core.mv.y, SCALE, core.mv.x)


def stream() -> core.Vector:
    return SPEED * core.mv.x


def trailing_edge(wing: core.Wing) -> core.Vector:
    """The map's critical point that makes the wing's sharp trailing edge."""
    return wing.scale * wing.chord


def morph() -> Iterator[tuple[core.Flow, core.Vector, core.Vector, core.Vector]]:
    """The flow past the cylinder as the map's critical points move apart from its centre out to the
    trailing edge, deforming the cylinder into the wing, with the wing's circulation throughout; the
    critical points' images, where the map doubles angles; and the cylinder's centre before the map,
    whose offset from where the critical points start shapes the wing."""
    pitched = core.pitched(wing(), MORPH_ATTACK)
    plane = core.rings(pitched, DRAWN_RINGS, DRAWN_ANGLES, REACH)            # [rings, angles] Vector
    spread = (1 - np.cos(2 * np.pi * np.arange(MORPH_FRAMES) / MORPH_FRAMES)) / 2
    for fraction in spread:
        critical = fraction * trailing_edge(pitched)                         # [] Vector
        images = 2 * stack([critical, -critical])                            # [2] Vector
        yield core.flow(pitched, stream(), plane, critical), core.lift(pitched, stream(), DENSITY), images, pitched.centre


def sweep() -> Iterator[tuple[core.Flow, core.Vector]]:
    """The flow past the wing and its lift as the wing pitches up and back."""
    phase = (1 - np.cos(2 * np.pi * np.arange(FRAMES) / FRAMES)) / 2
    for attack in LOWEST_ATTACK + (HIGHEST_ATTACK - LOWEST_ATTACK) * phase:
        pitched = core.pitched(wing(), attack)
        plane = core.rings(pitched, DRAWN_RINGS, DRAWN_ANGLES, REACH)        # [rings, angles] Vector
        yield core.flow(pitched, stream(), plane, trailing_edge(pitched)), core.lift(pitched, stream(), DENSITY)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.wing import render

    states = list(sweep())
    save_animation(render.animate(states, SPEED), "wing", DURATION_MS)
    save_animation(render.animate_marked(list(morph()), SPEED), "wing_morph", DURATION_MS)

    # --- checks
    pitched = core.pitched(wing(), HIGHEST_ATTACK)
    plane = core.rings(pitched, RINGS, ANGLES, REACH)                        # [rings, angles] Vector
    flow = core.flow(pitched, stream(), plane, trailing_edge(pitched))
    # The velocity's derivative vanishes everywhere past the cylinder.
    _, gradient, _ = core.cylinder(pitched, stream(), plane)
    np.testing.assert_allclose(core.derivative(gradient).kernel, 0.0, atol=1e-12)
    # Around every ring about the wing the velocity times each step adds up to the circulation, with
    # no flux: the derivative between any two rings vanishes too.
    next_idx = (np.arange(ANGLES) + 1) % ANGLES
    loops = ((flow.velocity + flow.velocity[:, next_idx]) / 2 * (flow.points[:, next_idx] - flow.points)).sum(axis=-1)
    np.testing.assert_allclose(loops.kernel - loops.kernel[:1], 0.0, atol=1e-11)
    np.testing.assert_allclose((loops - core.circulation(pitched, stream()).dual()).kernel, 0.0, atol=1e-3)
    # The pressure on the surface, half the density times the squared speed along the inward
    # normal, adds up to the Kutta–Joukowski lift, across the stream, with no drag.
    surface = core.flow(pitched, stream(), core.rings(pitched, 1, ANGLES, 1.0)[0], trailing_edge(pitched))
    steps = surface.points[next_idx] - surface.points                        # [angles] Vector
    speed_squared = surface.velocity | surface.velocity                      # [angles] Scalar
    force = (0.5 * DENSITY * (speed_squared + speed_squared[next_idx]) / 2 * -steps.dual()).sum(axis=-1)
    np.testing.assert_allclose((force - core.lift(pitched, stream(), DENSITY)).kernel, 0.0, atol=1e-3)


if __name__ == "__main__":
    main()
