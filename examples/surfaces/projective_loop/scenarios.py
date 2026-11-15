"""Scenes for projective Loop subdivision refine a weighted octahedron, sweep its top vertex weight, and compare refinement before and after a perspective collineation."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.surfaces.projective_loop import core

# The octahedron's six vertices, with top_vertex at the +z apex, and its eight counter-clockwise faces.
octahedron = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
top_vertex = 4
octahedron_faces = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
# Number of Loop subdivision steps.
n_loop_steps = 3
# Vertex weights per case: uniform, heavy top vertex, and light top vertex.
case_weights = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 4.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.25, 1.0]])
# Stretch and offset placing the octahedron in front of a central collineation with centre at the origin,
# axis plane z = 0, and unit strength.
stretch = np.array([1.0, 1.0, 1.5])
offset = np.array([0.0, 0.0, 1.6])
centre = core.mv.w.dual()                                                   # Antivector-
axis = core.mv.z                                                            # Vector+
strength = 1.0
# Bounds, frame count, and frame duration for sweeping the top vertex weight from light to heavy and back.
weight_bounds = (0.25, 4.0)
n_sweep_frames = 72
frame_duration_ms = 40

# The octahedron's topology, and three Loop steps of it with no vertices pinned, composed into one map from its
# vertex sites to those of the refined topology.
control_topology = core.Topology(len(octahedron), octahedron_faces)
refined_topology = control_topology.loop_subdivided(np.zeros(len(octahedron), dtype=bool), n_loop_steps)
subdivision = refined_topology.subdivision                                  # Scalar+[V_refined <- V]


# --- math -----------------------------------------------------------------------------
def weighted_cases() -> tuple[core.Point, core.Point, core.Topology]:
    """Subdivides the octahedron across the three static vertex-weight cases (`[3] Point-[V]`, `[3] Point-[V_refined]`)."""
    controls = (core.mv.point(octahedron) * case_weights).field()           # [3] Point-[V]
    surfaces = subdivision * controls                                       # [3] Point-[V_refined]
    return controls, surfaces, refined_topology


def weight_sweep() -> tuple[Iterator[tuple[core.Point, core.Point]], core.Topology]:
    """Yields each frame's control cage (`Point-[V]`) and subdivided surface (`Point-[V_refined]`) as the top vertex weight sweeps along a log-cosine profile."""
    V, E, F = control_topology.n_elements
    n_half_frames = n_sweep_frames // 2 + 1
    # Sweep from light to heavy over half the cycle; the animation plays forward and back.
    phase = np.linspace(0, np.pi, n_half_frames)                            # [n_half_frames]
    top_weights = np.exp(np.log(weight_bounds).mean() - np.log(weight_bounds[1] / weight_bounds[0]) / 2 * np.cos(phase))  # [n_half_frames]
    weights = np.where(np.arange(V) == top_vertex, top_weights[:, None], 1.0)  # [n_half_frames, V]
    controls = ((core.mv.point(octahedron) * weight).field() for weight in weights)
    return ((control, subdivision * control) for control in controls), refined_topology


def perspective_orders() -> tuple[core.Point, core.Point, core.Point, core.Topology]:
    """Subdivides the octahedron before and after a central collineation (`[2] Point-[V_refined]`) for weighted projective points and for normalised Euclidean positions."""
    rest_control = core.mv.point(octahedron * stretch + offset).field()     # [] Point-[V]
    # The central collineation shifts each point along its line through the centre in proportion to its
    # signed distance from the axis plane.
    collineation = core.Point + centre * (axis & core.Point) * strength     # [] Point- <- Point-
    perspective_control = collineation(rest_control)                        # [] Point-[V]
    weighted_orders = stack([subdivision * perspective_control, collineation(subdivision * rest_control)])  # [2] Point-[V_refined]
    rest_positions = rest_control.normalized()                              # [] Point-[V]
    perspective_positions = perspective_control.normalized()                # [] Point-[V]
    positional_orders = stack([subdivision * perspective_positions, collineation(subdivision * rest_positions)])  # [2] Point-[V_refined]
    return perspective_control, weighted_orders, positional_orders, refined_topology


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.projective_loop import render

    controls, surfaces, refined = weighted_cases()
    save_figure(render.weights_figure(controls, control_topology, surfaces, refined), "projective_loop_weights")
    frames, refined = weight_sweep()
    save_animation(render.weights_animation(frames, control_topology, refined), "projective_loop_sweep", frame_duration_ms,
                   forward_then_back=True)
    control, weighted_orders, positional_orders, refined = perspective_orders()
    save_figure(render.orders_figure(control, control_topology, weighted_orders, positional_orders, refined),
                "projective_loop_perspective")

    # checks
    # Refining points with their weights commutes with the perspective; refining positions does not.
    weighted_unit = weighted_orders.normalized()                            # [2] Point-[V_refined]
    positional_unit = positional_orders.normalized()                        # [2] Point-[V_refined]
    assert ((weighted_unit[0] & weighted_unit[1]).norm() < 1e-12).all()
    assert ((positional_unit[0] & positional_unit[1]).norm() > 1e-2).any()
    # With all weights alike the two schemes agree.
    assert ((surfaces[0] & surfaces[0].normalized()).norm() < 1e-12).all()


if __name__ == "__main__":
    main()
