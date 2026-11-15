"""Paraboloid lift and harmonic height oscillation on a jittered Delaunay triangulation."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.topology import Topology
from examples.surfaces.reciprocal import core
from examples.util import shapes

# Grid resolution across [-1, 1]^2, interior vertex jitter scale, and deterministic RNG seed.
n_side = 5
jitter = 0.08
grid_seed = 11
# Harmonic height amplitude around the paraboloid, phase seed, frame count, and frame duration.
amplitude = 0.02
phase_seed = 3
n_frames = 96
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def paraboloid_grid() -> tuple[Topology, core.Point, core.Scalar]:
    """A jittered Delaunay triangulation over `[-1, 1]^2` pairs its planar vertices (`Point-[V]`) with paraboloid heights (`Scalar+[V]`) equal to half the squared radius."""
    grid_axis = np.linspace(-1, 1, n_side)                                      # [n_side]
    coordinates = np.stack(np.meshgrid(grid_axis, grid_axis, indexing="ij"), axis=-1).reshape(-1, 2)  # [n_side * n_side, 2]
    is_interior = (np.abs(coordinates) < 1).all(axis=-1)                        # [n_side * n_side]
    coordinates[is_interior] += jitter * np.random.default_rng(grid_seed).normal(size=(is_interior.sum(), 2))
    topology, coordinates = shapes.delaunay(coordinates)
    V, E, F = topology.n_elements
    points = core.mv.vector(np.concatenate([coordinates, np.zeros((V, 1))], axis=-1)).dual().field()  # Point-[V]
    return topology, points, points.scalar_norm_squared() / 2                   # Scalar+[V]


def vertex_phases(topology: Topology) -> np.ndarray:
    """Deterministic uniform phases (`[V]`) give each vertex of `topology` an independent oscillation offset."""
    V, E, F = topology.n_elements
    return np.random.default_rng(phase_seed).uniform(0, 2 * np.pi, V)           # [V]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.surfaces.reciprocal import render
    from examples.util.animation import save_animation, save_figure

    topology, points, paraboloid = paraboloid_grid()
    lifted_stresses = core.bar_stresses(topology, points, paraboloid)           # Scalar+[E_interior]
    lifted_duals = (core.face_planes(topology, points, paraboloid) ^ core.mv.z) * core.mv.z  # Plane+[F]
    figure = render.reciprocal_figure(topology, points, paraboloid, lifted_stresses, lifted_duals)
    save_figure(figure, "reciprocal")
    import matplotlib.pyplot as plt
    plt.close(figure)

    phases = vertex_phases(topology)                                            # [V]
    recorded: list[tuple[core.Scalar, core.Scalar, core.Plane]] = []

    def streaming():
        for state in core.breathing_states(topology, points, paraboloid, phases, amplitude, n_frames):
            recorded.append(state)
            yield state

    save_animation(render.breath_animation(topology, points, streaming()), "reciprocal_breath", frame_duration_ms)

    # checks
    breaths = stack([state[0] for state in recorded])                           # [n_frames] Scalar+[V]
    breath_stresses = stack([state[1] for state in recorded])                   # [n_frames] Scalar+[E_interior]
    breath_duals = stack([state[2] for state in recorded])                      # [n_frames] Plane+[F]
    folds = core.edge_folds(topology, points, breaths)                          # [n_frames] Scalar+[E_interior]
    # Every bar carries tension exactly where the terrain crease bends convex, while concave creases enter compression.
    assert np.array_equal(breath_stresses > 0, folds > 0) and (breath_stresses < 0).any()
    # Every reciprocal edge is orthogonal to its primal bar because adjacent face planes agree along their shared edge.
    interior = topology.reading(topology.interior[1], 1)                        # Scalar+[E_interior <- E]
    bars = interior * (topology.d[0] * points)                                  # Point-[E_interior]
    assert (((interior * (topology.d[1].adjugate() * breath_duals)) & bars).abs() < 1e-10).all()
    # Summing axial bar forces onto vertices leaves every interior truss joint in static equilibrium.
    pulls = bars * breath_stresses                                              # [n_frames] Point-[E_interior]
    bar_ends = interior * topology.d[0]                                         # Scalar+[E_interior <- V]
    net_forces = -(bar_ends.adjugate() * pulls)                                 # [n_frames] Point-[V]
    assert ((topology.reading(topology.interior[0], 0) * net_forces).antinorm() < 1e-10).all()
    # On the paraboloid lift, bar stresses equal the cotangent weights and reciprocal vertices lie at the triangle circumcentres.
    directed_sides = topology.boundary[2, 1] * (topology.d[0] * points)         # [3] Point-[F]
    first_sides, second_sides = topology.corners[2, 1]                          # [3], [3]
    after, before = directed_sides[first_sides], directed_sides[second_sides]   # [3] Point-[F], [3] Point-[F]
    cotangents = (after | before) / (after & before).antinorm()                 # [3] Scalar+[F]
    cotangent_weights = (topology.incidence[2, 1].adjugate() * (cotangents / 2)).sum(axis=0)  # Scalar+[E]
    assert ((lifted_stresses - interior * cotangent_weights).abs() < 1e-10).all()
    corners = topology.incidence[2, 0] * points                                 # [3] Point-[F]
    squared_distances = (lifted_duals.dual() - corners).scalar_norm_squared()   # [3] Scalar+[F]
    assert ((squared_distances[0] - squared_distances[1]).abs() < 1e-10).all()
    assert ((squared_distances[0] - squared_distances[2]).abs() < 1e-10).all()


if __name__ == "__main__":
    main()
