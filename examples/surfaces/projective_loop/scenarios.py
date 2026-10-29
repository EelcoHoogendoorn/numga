"""Scenes for projective Loop subdivision refine a weighted octahedron, sweep its top vertex weight, and compare refinement before and after a perspective collineation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.surfaces.projective_loop import core

# The octahedron's six vertices, with TOP_VERTEX at the +z apex, and its eight counter-clockwise faces.
OCTAHEDRON = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
TOP_VERTEX = 4
FACES = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
# Number of Loop subdivision steps.
N_LOOP_STEPS = 3
# Vertex weights per case: uniform, heavy top vertex, and light top vertex.
WEIGHTS = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 4.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.25, 1.0]])
# Stretch and offset placing the octahedron in front of a central collineation with centre at the origin,
# axis plane z = 0, and unit strength.
STRETCH = np.array([1.0, 1.0, 1.5])
OFFSET = np.array([0.0, 0.0, 1.6])
centre, axis, STRENGTH = core.mv.w.dual(), core.mv.z, 1.0
# Bounds, frame count, and frame duration for sweeping the top vertex weight from light to heavy and back.
WEIGHT_BOUNDS = (0.25, 4.0)
N_SWEEP_FRAMES = 72
DURATION_MS = 40

# The octahedron's mesh, and three Loop steps of it, none of its vertices pinned, composed into one map from its
# vertex sites to those of the refined mesh.
CONTROL_MESH = core.Mesh(len(OCTAHEDRON), FACES)
REFINED_MESH = CONTROL_MESH.loop_subdivided(np.zeros(len(OCTAHEDRON), dtype=bool), N_LOOP_STEPS)
subdivision = REFINED_MESH.subdivision


# --- math -----------------------------------------------------------------------------
def weighted() -> tuple[core.Point, core.Point, core.Mesh]:
    """The octahedron's control meshes, subdivided surfaces, and the refined mesh across the three vertex-weight cases."""
    controls = (core.mv.point(OCTAHEDRON) * WEIGHTS).field()                   # [cases] Point-[vertices]
    surfaces = subdivision * controls                                          # [cases] Point-[refined]
    return controls, surfaces, REFINED_MESH


def sweep() -> tuple[Iterator[tuple[core.Point, core.Point]], core.Mesh]:
    """Each frame's control mesh and subdivided surface as the top vertex weight sweeps from light to heavy along a log-cosine profile."""
    # Sweep from light to heavy over half the cycle; the animation plays forward and back.
    phase = np.linspace(0, np.pi, N_SWEEP_FRAMES // 2 + 1)
    top_weights = np.exp(np.log(WEIGHT_BOUNDS).mean() - np.log(WEIGHT_BOUNDS[1] / WEIGHT_BOUNDS[0]) / 2 * np.cos(phase))  # [frames]
    weights = np.where(np.arange(len(OCTAHEDRON)) == TOP_VERTEX, top_weights[:, None], 1.0)  # [frames, vertices]
    controls = ((core.mv.point(OCTAHEDRON) * weight).field() for weight in weights)
    return ((control, subdivision * control) for control in controls), REFINED_MESH


def perspective() -> tuple[core.Point, core.Point, core.Point, core.Mesh]:
    """The octahedron refined before and after a central collineation, for weighted projective points and for normalised Euclidean positions."""
    rest_control = core.mv.point(OCTAHEDRON * STRETCH + OFFSET).field()        # [] Point-[vertices]
    # The central collineation shifts each point along its line through the centre in proportion to its
    # signed distance from the axis plane.
    collineation = core.Point + centre * (axis & core.Point) * STRENGTH        # [] Point- <- Point-
    perspective_control = collineation(rest_control)                           # [] Point-[vertices]
    weighted_orders = stack([subdivision * perspective_control, collineation(subdivision * rest_control)])  # [orders] Point-[refined]
    rest_positions = rest_control.normalized()                                  # [] Point-[vertices]
    perspective_positions = perspective_control.normalized()                    # [] Point-[vertices]
    positional_orders = stack([subdivision * perspective_positions, collineation(subdivision * rest_positions)])  # [orders] Point-[refined]
    return perspective_control, weighted_orders, positional_orders, REFINED_MESH


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.projective_loop import render

    controls, surfaces, refined_mesh = weighted()
    save_figure(render.draw_weights(controls, CONTROL_MESH, surfaces, refined_mesh), "projective_loop_weights")
    frames, refined_mesh = sweep()
    save_animation(render.animate_weights(frames, CONTROL_MESH, refined_mesh), "projective_loop_sweep", DURATION_MS,
                   forward_then_back=True)
    control, weighted_orders, positional_orders, refined_mesh = perspective()
    save_figure(render.draw_orders(control, CONTROL_MESH, weighted_orders, positional_orders, refined_mesh),
                "projective_loop_perspective")

    # checks
    # Refining points with their weights commutes with the perspective; refining positions does not.
    weighted_unit = weighted_orders.normalized()
    positional_unit = positional_orders.normalized()
    assert np.all((weighted_unit[0] & weighted_unit[1]).norm() < 1e-12)
    assert np.any((positional_unit[0] & positional_unit[1]).norm() > 1e-2)
    # With all weights alike the two schemes agree.
    assert np.all((surfaces[0] & surfaces[0].normalized()).norm() < 1e-12)


if __name__ == "__main__":
    main()
