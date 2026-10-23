"""Scenes for hydrostatic stability: a Wigley hull with its load at several heights, its righting arm from
upright to upside down; a floating cube of every density, heeled through a quarter turn; and the hull
released from a heel with a low load and with a high one."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.fluids.floating import core

mv = core.mv
# Fresh water's weight per cubic metre, in newtons, and gravity.
FLUID_WEIGHT = 9810.0
GRAVITY = 9.81
# The hull's length, beam, draft and freeboard, in metres, and the cells its surface is cut into.
LENGTH, BEAM, DRAFT, FREEBOARD = 4.0, 1.2, 0.5, 0.4
N_CELLS = 8
# The heights of the hull's centre of gravity over its waterline, from low ballast to a high load; and the
# load and heel of the hull drawn whole.
CENTRE_HEIGHTS = [-0.2, -0.1, 0.0, 0.1, 0.2]
SHOWN_LOAD = 0
SHOWN_HEEL = 35.0
# The heel angles of the righting arm curves, and of the cube's map, in degrees: the map reaches past a
# quarter turn either side, so that its ends are rests like any other.
HEELS = np.linspace(0, 180, 91)
CUBE_HEELS = np.linspace(-10, 100, 56)
# The cube's densities relative to water, across the map and for the cubes drawn above it.
DENSITIES = np.linspace(0.02, 0.98, 49)
SHOWN_DENSITIES = [0.1, 0.23, 0.5, 0.77, 0.9]
# Newton steps for the floating height, from upright and from the previous step.
N_NEWTON_STEPS = 6
N_WARM_STEPS = 1
# The released hull: which of the loads it carries, its heel at release, its radius of gyration about the
# axis of heel as a share of the beam, the damping of its roll per second, the time step in seconds, the
# frames, and the steps per frame.
RELEASED = [1, 4]
RELEASE_HEEL = 30.0
GYRATION = 0.4
DAMPING = 0.4
STEP = 0.05
N_FRAMES = 80
N_STEPS_PER_FRAME = 3
DURATION_MS = 50


# --- math -----------------------------------------------------------------------------
def heeled(degrees: np.ndarray) -> core.Motor:
    """The turns heeling a body by each angle, about its axis of heel."""
    return (mv.yz * (-np.radians(degrees) / 2)).exp()


def ship() -> tuple[core.Point, core.Scalar, core.Point]:
    """The hull; its weight, the water it displaces upright at its waterline; and its centre of gravity at
    each height `[loads]`."""
    hull = core.wigley(LENGTH, BEAM, DRAFT, FREEBOARD, N_CELLS)                  # [triangles, 3] Point
    upright, _ = core.buoyancy(hull, core.water, FLUID_WEIGHT)               # [] Forque
    centres = mv.point([[0, 0, height] for height in CENTRE_HEIGHTS])         # [loads] Point
    return hull, upright & core.heave, centres


def shown_ship(hull: core.Point, weight: core.Scalar) -> tuple[core.Motor, core.Forque]:
    """The hull heeled by the shown angle and floating: its placement in the world, and its buoyancy."""
    turn = heeled(np.array([SHOWN_HEEL]))                                    # [1] Motor
    height = core.afloat(hull, turn, weight, mv.scalar([[0.0]]), FLUID_WEIGHT, N_NEWTON_STEPS)   # [1] Scalar
    push, _ = core.buoyancy(hull, turn << (core.water - height * mv.w), FLUID_WEIGHT)   # [1] Forque
    return core.placement(turn, height)[0], push[0]


def righting_curves(hull: core.Point, weight: core.Scalar, centres: core.Point) -> tuple[core.Forque, core.Stiffness]:
    """The righting couple of the hull with each load at each heel `[heels, loads]`, and its stiffness
    against a small tilt upright `[loads]`."""
    turns = heeled(HEELS)                                                    # [heels] Motor
    start = mv.scalar(np.zeros((len(HEELS), 1)))
    # Every load weighs the same, so the hull floats at the same height at each heel, whatever the load.
    heights = core.afloat(hull, turns, weight, start, FLUID_WEIGHT, N_NEWTON_STEPS)   # [heels] Scalar
    push, weighing = core.forces(hull, turns[:, None], heights[:, None], weight, centres, FLUID_WEIGHT)   # [heels, loads] Forque each
    # Upright, the hull floats at its waterline, its weight being the water it displaces there.
    return push + weighing, core.stiffness(hull, mv.scalar([0.0]), weight, centres, FLUID_WEIGHT)


def cubes() -> tuple[core.Point, core.Scalar, core.Forque]:
    """The unit cube, the weights of the densities `[densities, 1]`, and its righting couple at each density
    and heel `[densities, heels]`, its centre of gravity at its middle."""
    cube = core.box(np.ones(3))                                              # [triangles, 3] Point
    weights = mv.scalar((FLUID_WEIGHT * DENSITIES)[:, None, None])          # [densities, 1] Scalar
    turns = heeled(CUBE_HEELS)                                               # [heels] Motor
    start = mv.scalar(np.zeros((len(DENSITIES), len(CUBE_HEELS), 1)))
    heights = core.afloat(cube, turns, weights, start, FLUID_WEIGHT, N_NEWTON_STEPS)   # [densities, heels] Scalar
    push, weighing = core.forces(cube, turns, heights, weights, mv.w.dual(), FLUID_WEIGHT)   # [densities, heels] Forque each
    return cube, weights, push + weighing


def resting(cube: core.Point, densities: list[float], heels: np.ndarray) -> core.Motor:
    """The placements of cubes of the given densities, floating at the given heels."""
    weights = mv.scalar([[FLUID_WEIGHT * density] for density in densities])   # [shown] Scalar
    turns = heeled(heels)                                                    # [shown] Motor
    start = mv.scalar(np.zeros((len(densities), 1)))
    return core.placement(turns, core.afloat(cube, turns, weights, start, FLUID_WEIGHT, N_NEWTON_STEPS))


def released(hull: core.Point, weight: core.Scalar, centres: core.Point
             ) -> Iterator[tuple[core.Twist, core.Motor, core.Forque, core.Forque]]:
    """The hull released from a heel with each load `[loads]`: at every frame its roll, its placement in the
    world, its buoyancy and its weight. The hull turns about its axis of heel, floating at each moment at the
    height where its buoyancy carries its weight, its roll driven by the righting couple and damped."""
    inertia = weight / GRAVITY * (GYRATION * BEAM) ** 2                     # [] Scalar
    # The roll and its rate are turns about the axis of heel, the axis scaled by the angle and the rate.
    roll = mv.yz * np.full(centres.shape[0], np.radians(RELEASE_HEEL))      # [loads] Twist
    spin = mv.yz * np.zeros(centres.shape[0])                               # [loads] Twist
    turns = (roll * (-1 / 2)).exp()                                         # [loads] Motor
    heights = core.afloat(hull, turns, weight, mv.scalar(np.zeros((centres.shape[0], 1))), FLUID_WEIGHT, N_NEWTON_STEPS)
    push, weighing = core.forces(hull, turns, heights, weight, centres, FLUID_WEIGHT)   # [loads] Forque each
    for _ in range(N_FRAMES):
        yield roll, core.placement(turns, heights), push, weighing
        for _ in range(N_STEPS_PER_FRAME):
            # The righting couple about the axis of heel turns the roll, against its damping.
            spin = spin + (mv.yz * (((push + weighing) & core.heel_axis) / inertia) - DAMPING * spin) * STEP
            roll = roll + spin * STEP
            turns = (roll * (-1 / 2)).exp()
            heights = core.afloat(hull, turns, weight, heights, FLUID_WEIGHT, N_WARM_STEPS)
            push, weighing = core.forces(hull, turns, heights, weight, centres, FLUID_WEIGHT)


# --- the step outside the algebra ---------------------------------------------------------
def stable_heels(couples: core.Forque) -> np.ndarray:
    """For each row of righting couples over the heels of the cube's map, the smallest heel from upright at
    which the cube rests stably: where the couple turns from pushing the cube over to pushing it back,
    between the two heels around it in proportion to the couples there."""
    arms = -(couples & core.heel_axis).cast(core.ga.subspace.scalar()).kernel[..., 0]   # [rows, heels]
    below, above = arms[:, :-1], arms[:, 1:]
    share = -below / np.where(above == below, 1, above - below)
    rests = CUBE_HEELS[:-1] + share * np.diff(CUBE_HEELS)                   # [rows, heels - 1]
    stable = (below < 0) & (above >= 0) & (rests > -1e-6)
    return rests[np.arange(len(arms)), np.argmax(stable, axis=1)]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.floating import render

    hull, weight, centres = ship()
    couples, stiffness = righting_curves(hull, weight, centres)
    placed, push = shown_ship(hull, weight)
    save_figure(render.draw_ship(hull, placed, push, centres[SHOWN_LOAD]), "floating_ship")
    save_figure(render.draw_curves(HEELS, couples, weight, stiffness), "floating_righting")
    cube, weights, cube_couples = cubes()
    rests = stable_heels(cube_couples[np.searchsorted(DENSITIES, SHOWN_DENSITIES)])
    shown = resting(cube, SHOWN_DENSITIES, rests)
    save_figure(render.draw_cubes(DENSITIES, CUBE_HEELS, cube_couples, FLUID_WEIGHT, cube, shown, SHOWN_DENSITIES, rests),
                "floating_cubes")
    frames = list(released(hull, weight, centres[RELEASED]))
    save_animation(render.animate_release(hull, centres[RELEASED], frames, HEELS, couples[:, RELEASED], weight),
                   "floating_release", DURATION_MS)

    # checks
    arms = (-(couples & core.heel_axis) / weight).to_array()              # [heels, loads]
    heights = (stiffness(mv.yz, mv.yz) / weight).to_array()              # [loads]
    # The slope of each righting arm curve at upright is the metacentric height in roll.
    assert np.abs(arms[1] / np.sin(np.radians(HEELS[1])) - heights).max() < 1e-2
    # A cube floats face up only below a density of (3 - sqrt(3)) / 6 or above (3 + sqrt(3)) / 6.
    edges = (3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6
    face_up = np.abs(stable_heels(cube_couples)) < 1e-6
    assert np.array_equal(face_up, (DENSITIES < edges[0]) | (DENSITIES > edges[1]))
    # The hull with the low load comes back upright; with the high one it capsizes and floats upside down.
    low, high = render.degrees(frames[-1][0])                               # [loads]
    assert abs(low) < 5 and abs(high - 180) < 10


if __name__ == "__main__":
    main()
