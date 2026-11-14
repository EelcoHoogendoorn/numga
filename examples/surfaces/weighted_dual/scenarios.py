"""Scenario parameters and streaming generators run the radial breathing polyhedron and the weight-rescued triangulation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.surfaces.weighted_dual import core, numerical
from numga.topology import Topology

n_points = 32
jitter = 0.03
direction_seed, phase_seed = 0, 3
amplitude = 0.06
n_frames = 96
duration_ms = 40
radius_spread, radius_seed = 0.06, 57
weight_bounds = (0.5, 2.0)
margin_share = 0.3
n_rescue_frames = 96

directions = core.spiral_directions(n_points, jitter, direction_seed)          # Point-[V]


def closing_schedule(n_steps: int) -> np.ndarray:
    """Interpolation progress (`[n_steps]`) advances linearly for three quarters of the run and settles exponentially over the final quarter."""
    closing_start, e_folds = 3 / 4, 3
    rate = e_folds / (1 - closing_start)
    # Match the linear and exponential slopes at closing_start so velocity stays continuous.
    tail_share = 1 / (1 + closing_start * rate)
    progress = np.linspace(0, 1, n_steps)                                      # [n_steps]
    return np.where(
        progress < closing_start,
        progress * tail_share * rate,
        1 - tail_share * np.exp(-(progress - closing_start) * rate),
    )


def breath() -> Iterator[tuple[Topology, core.Point, core.Point]]:
    """Yields `(topology, points, duals)` frames as vertices oscillate radially on the fixed unit-sphere convex hull."""
    topology = numerical.convex_hull(directions)
    phases = np.random.default_rng(phase_seed).uniform(0, 2 * np.pi, n_points)  # [n_points]
    yield from core.breath_states(topology, directions, amplitude, phases, n_frames)


def rescue() -> Iterator[tuple[Topology, core.Point, core.Point]]:
    """Yields `(topology, points, duals)` frames along the weight line from a tangled unit-sphere complex to its nearest positive-dual rescue."""
    random_radii = core.mv.scalar(
        np.exp(radius_spread * np.random.default_rng(radius_seed).normal(size=(n_points, 1)))
    ).field()                                                                  # Scalar+[V]
    topology = numerical.convex_hull(directions * random_radii)
    even_weights = core.mv.scalar(np.ones((n_points, 1))).field()              # Scalar+[V]
    unit_weights = core.mv.scalar(np.eye(n_points)[..., None]).field()         # [n_points] Scalar+[V]
    fold_columns = core.edge_folds(topology, directions, unit_weights)         # [n_points] Scalar+[E_interior]
    inside_columns = core.pole_insides(topology, directions, unit_weights)     # [n_points, 3] Scalar+[F]
    _, max_margin = numerical.widest_margin(fold_columns, inside_columns, weight_bounds)  # Scalar+[V], Scalar+
    rescued_weights = numerical.nearest_weights(
        fold_columns, inside_columns, margin_share * max_margin, even_weights, weight_bounds
    )                                                                          # Scalar+[V]
    schedule = closing_schedule(n_rescue_frames)                               # [n_rescue_frames]
    yield from core.rescue_states(topology, directions, even_weights, rescued_weights, schedule)


def main() -> None:
    """Saves the static complex figure and both radial-weight animations before verifying incidence and untangling."""
    from examples.surfaces.weighted_dual import render
    from examples.util.animation import save_animation, save_figure

    breath_history = []
    save_animation(
        render.complex_frames(breath_history.append(state) or state for state in breath()),
        "weighted_dual_breath",
        duration_ms,
    )
    save_figure(render.complex_figure(*breath_history[0]), "weighted_dual")

    rescued_history = []
    save_animation(
        render.complex_frames(rescued_history.append(state) or state for state in rescue()),
        "weighted_dual_rescue",
        duration_ms,
    )

    # checks
    for topology, points, duals in breath_history:
        # Every face plane is the inverse dual of its pole and pairs to one with all three corners.
        assert (((duals.dual_inverse() & (topology.incidence[2, 0] * points)) - 1).abs() < 1e-10).all()

    tangled_topology = rescued_history[0][0]
    even_weights = core.mv.scalar(np.ones((n_points, 1))).field()              # Scalar+[V]
    crossed_counts = [(core.dual_lengths(tangled_topology, points) < 0).sum() for _, points, _ in rescued_history]
    assert np.array_equal(
        core.dual_lengths(tangled_topology, rescued_history[0][1]) < 0,
        core.edge_folds(tangled_topology, directions, even_weights) < 0,
    )
    assert crossed_counts[0] > 0 and crossed_counts[-1] == 0 and (np.diff(crossed_counts) <= 0).all()
    assert (core.pole_sides(*rescued_history[-1]) > 0).all()


if __name__ == "__main__":
    main()
