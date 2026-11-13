"""Scenes for hydrostatic stability: a Wigley hull with its centre of gravity at five heights, its righting
arm from upright to upside down; a floating cube across relative densities heeled through a quarter turn; and
the hull released from a heel with a low load and a high load."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np
from numga import where

from examples.fluids.floating import core

mv = core.mv
# Fresh water's weight per cubic metre, in newtons, and gravity.
fluid_weight = 9810.0
gravity = 9.81
# Hull length, beam, draft and freeboard, in metres, and panel count below the waterline.
length, beam, draft, freeboard = 4.0, 1.2, 0.5, 0.4
n_cells = 8
# Centre-of-gravity heights over the waterline, and the load index and heel of the 3D hull figure.
centre_heights = [-0.2, -0.1, 0.0, 0.1, 0.2]
n_loads = len(centre_heights)
shown_load = 0
shown_heel = 35.0
# Heel angles in degrees for the righting arm curves and the cube stability map.
n_heels = 91
heels = np.linspace(0, 180, n_heels)
n_cube_heels = 56
cube_heels = np.linspace(-10, 100, n_cube_heels)
# Cube densities relative to water across the map and for the cubes drawn above it.
n_densities = 49
densities = np.linspace(0.02, 0.98, n_densities)
shown_densities = [0.1, 0.23, 0.5, 0.77, 0.9]
n_shown_cubes = len(shown_densities)
# Newton steps for equilibrium draft from zero and from the previous time step.
n_newton_steps = 6
n_warm_steps = 1
# Released-hull load indices, initial heel, roll gyration ratio, damping rate, step size and frame timing.
released_loads = [1, 4]
release_heel = 30.0
gyration = 0.4
damping = 0.4
step_dt = 0.03
n_frames = 100
n_steps_per_frame = 4
frame_duration_ms = 40
n_colours = 64


# --- math -----------------------------------------------------------------------------
def heel_motors(angles: np.ndarray | core.Scalar) -> core.Motor:
    """Motors heeling a body about the longitudinal heel axis by angles given in degrees."""
    return (core.heel_axis * (-angles * (np.pi / 180) / 2)).exp()


def ship_scene(n_hull_cells: int) -> tuple[core.Topology, core.Point, core.Scalar, core.Point]:
    """The triangulated Wigley hull, its upright displaced weight, and its centres of gravity across loads."""
    topology, vertices = core.wigley_hull(length, beam, draft, freeboard, n_hull_cells)  # Point-[V]
    hull = topology.incidence[2, 0] * vertices                          # [3] Point-[F]
    upright_push, _ = core.buoyancy(hull, core.water, fluid_weight)     # [] Forque-
    centres = mv.point([[0, 0, height] for height in centre_heights])   # [n_loads] Point-
    return topology, vertices, upright_push & core.heave, centres


def heeled_ship(hull: core.Point, weight: core.Scalar) -> tuple[core.Motor, core.Forque]:
    """World placement and buoyancy forque of the hull floating at `shown_heel`."""
    turn = heel_motors(np.array([shown_heel]))                          # [1] Motor+
    height = core.equilibrium_heights(hull, turn, weight, core.unit * np.zeros(1), fluid_weight, n_newton_steps)  # [1] Scalar+
    push, _ = core.buoyancy(hull, turn << (core.water - height * mv.w), fluid_weight)  # [1] Forque-
    return core.world_placement(turn, height)[0], push[0]


def righting_curves(hull: core.Point, weight: core.Scalar, centres: core.Point) -> tuple[core.Forque, core.Stiffness]:
    """Righting couples across heels and loads, together with the upright tilt stiffness of each load."""
    turns = heel_motors(heels)                                          # [n_heels] Motor+
    initial_heights = core.unit * np.zeros(n_heels)                     # [n_heels] Scalar+
    heights = core.equilibrium_heights(hull, turns, weight, initial_heights, fluid_weight, n_newton_steps)  # [n_heels] Scalar+
    push, weighing = core.body_forces(hull, turns[:, None], heights[:, None], weight, centres, fluid_weight)  # [n_heels, n_loads] Forque-, Forque-
    return push + weighing, core.tilt_stiffness(hull, 0 * weight, weight, centres, fluid_weight)


def cube_stability_map() -> tuple[core.Topology, core.Point, core.Scalar, core.Forque]:
    """The unit cube surface, its weights across densities, and its righting couples over density and heel."""
    topology, vertices = core.box_surface(np.ones(3))                   # Point-[V]
    cube = topology.incidence[2, 0] * vertices                          # [3] Point-[F]
    weights = core.unit * (fluid_weight * densities)[:, None]           # [n_densities, 1] Scalar+
    turns = heel_motors(cube_heels)                                     # [n_cube_heels] Motor+
    initial_heights = core.unit * np.zeros((n_densities, n_cube_heels))  # [n_densities, n_cube_heels] Scalar+
    heights = core.equilibrium_heights(cube, turns, weights, initial_heights, fluid_weight, n_newton_steps)  # [n_densities, n_cube_heels] Scalar+
    push, weighing = core.body_forces(cube, turns, heights, weights, core.origin, fluid_weight)  # [n_densities, n_cube_heels] Forque-, Forque-
    return topology, vertices, weights, push + weighing


def stable_heels(couples: core.Forque, heel_grid: np.ndarray) -> core.Scalar:
    """Smallest non-negative heel angle in degrees at which each row of righting couples crosses upward through zero."""
    arms = -(couples & core.heel_axis)                                  # [..., n_cube_heels] Scalar+
    below, above = arms[..., :-1], arms[..., 1:]
    span = above - below
    share = -below / where(span.abs() > 0, span, core.unit)
    rests = heel_grid[:-1] + share * np.diff(heel_grid)                 # [..., n_cube_heels - 1] Scalar+
    stable = (below < 0) & (above >= 0) & (rests > -1e-6)
    first_index = np.argmax(stable, axis=-1)
    chosen = np.arange(rests.shape[-1]) == first_index[..., None]
    return (rests * chosen).sum(axis=-1)                                # [...] Scalar+


def resting_cubes(cube: core.Point, cube_densities: list[float], rest_heels: core.Scalar) -> core.Motor:
    """World placements of cubes of the given relative densities floating at their equilibrium heels."""
    weights = core.unit * (fluid_weight * np.asarray(cube_densities))   # [n_shown_cubes] Scalar+
    turns = heel_motors(rest_heels)                                     # [n_shown_cubes] Motor+
    initial_heights = core.unit * np.zeros(len(cube_densities))         # [n_shown_cubes] Scalar+
    heights = core.equilibrium_heights(cube, turns, weights, initial_heights, fluid_weight, n_newton_steps)  # [n_shown_cubes] Scalar+
    return core.world_placement(turns, heights)                         # [n_shown_cubes] Motor+


def release_states(hull: core.Point, weight: core.Scalar, centres: core.Point
                   ) -> Iterator[tuple[core.Twist, core.Motor, core.Forque, core.Forque]]:
    """Yield roll twist, world placement, buoyancy and weight as the heeled hull rolls under hydrostatic torque."""
    n_cases = centres.shape[0]
    inertia = weight / gravity * (gyration * beam) ** 2                 # [] Scalar+
    roll = core.heel_axis * np.full(n_cases, np.radians(release_heel))  # [n_cases] Twist+
    roll_rate = core.heel_axis * np.zeros(n_cases)                      # [n_cases] Twist+
    turns = (roll * (-1 / 2)).exp()                                     # [n_cases] Motor+
    heights = core.equilibrium_heights(hull, turns, weight, core.unit * np.zeros(n_cases), fluid_weight, n_newton_steps)  # [n_cases] Scalar+
    push, weighing = core.body_forces(hull, turns, heights, weight, centres, fluid_weight)  # [n_cases] Forque-, Forque-
    for _ in range(n_frames):
        yield roll, core.world_placement(turns, heights), push, weighing
        for _ in range(n_steps_per_frame):
            # Roll acceleration comes from the righting couple paired with the heel axis, damped linearly.
            roll_rate = roll_rate + (core.heel_axis * (((push + weighing) & core.heel_axis) / inertia) - damping * roll_rate) * step_dt
            roll = roll + roll_rate * step_dt
            turns = (roll * (-1 / 2)).exp()
            heights = core.equilibrium_heights(hull, turns, weight, heights, fluid_weight, n_warm_steps)
            push, weighing = core.body_forces(hull, turns, heights, weight, centres, fluid_weight)


# --- plumbing -------------------------------------------------------------------------
def save_states(states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments,
                name: str, duration_ms: int, colors: int) -> list:
    """Save an animation from `states` in a single pass and return the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing_states() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing_states()), name, duration_ms, colors)
    return kept


def main() -> None:
    from examples.fluids.floating import render
    from examples.util.animation import save_figure

    topology, vertices, weight, centres = ship_scene(n_cells)
    hull = topology.incidence[2, 0] * vertices                          # [3] Point-[F]
    couples, stiffnesses = righting_curves(hull, weight, centres)
    ship_placement, ship_push = heeled_ship(hull, weight)
    save_figure(render.draw_ship(topology, vertices, ship_placement, ship_push, centres[shown_load]), "floating_ship")
    save_figure(render.draw_curves(heels, couples, weight, stiffnesses), "floating_righting")

    cube_topology, cube_vertices, _, cube_couples = cube_stability_map()
    rest_heels = stable_heels(cube_couples[np.searchsorted(densities, shown_densities)], cube_heels)
    cube_placements = resting_cubes(cube_topology.incidence[2, 0] * cube_vertices, shown_densities, rest_heels)
    save_figure(
        render.draw_cubes(densities, cube_heels, cube_couples, fluid_weight, cube_topology, cube_vertices,
                          cube_placements, shown_densities, rest_heels),
        "floating_cubes",
    )

    roll_history = save_states(
        release_states(hull, weight, centres[released_loads]),
        render.animate_release,
        topology,
        vertices,
        centres[released_loads],
        heels,
        couples[:, released_loads],
        weight,
        name="floating_release",
        duration_ms=frame_duration_ms,
        colors=n_colours,
    )

    # checks
    arms = -(couples & core.heel_axis) / weight                         # [n_heels, n_loads] Scalar+
    metacentric_heights = stiffnesses(core.heel_axis, core.heel_axis) / weight  # [n_loads] Scalar+
    assert ((arms[1] / np.sin(np.radians(heels[1])) - metacentric_heights).abs() < 1e-2).all()
    face_up_bounds = (3 - np.sqrt(3)) / 6, (3 + np.sqrt(3)) / 6
    face_up = stable_heels(cube_couples, cube_heels).abs() < 1e-6
    assert np.array_equal(face_up, (densities < face_up_bounds[0]) | (densities > face_up_bounds[1]))
    final_rolls = (roll_history[-1][0] & core.heel_axis.dual()) * (180 / np.pi)  # [2] Scalar+
    assert (final_rolls[0].abs() < 5).all() and ((final_rolls[1] - 180).abs() < 10).all()


if __name__ == "__main__":
    main()
