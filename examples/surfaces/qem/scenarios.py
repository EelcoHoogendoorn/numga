"""Scenes for quadric error metrics contract a ridge edge on a triangulated patch and progressively simplify the Stanford bunny."""

from __future__ import annotations

import numpy as np

from examples import MODEL_DIR
from examples.util import shapes
from examples.surfaces.qem.core import Point, Topology, collapse_position, mv, simplify

# A triangulated patch with a sharp ridge from corner `a` to gentler end `b`, listing `a` and `b` first and the two flanking triangles first.
ridge_vertices = [
    [ 0.35,  0.0,   0.25],  # 0: vertex a (sharp corner apex)
    [-0.35,  0.0,   0.25],  # 1: vertex b (crease continuation)
    [ 0.0,   0.55, -0.15],  # 2: left base vertex
    [ 0.0,  -0.55, -0.15],  # 3: right base vertex
    [ 0.75,  0.0,  -0.15],  # 4: front tip (steep corner at a)
    [-0.85,  0.0,  -0.15],  # 5: back tip (gentle slope at b)
]
ridge_faces = np.array([
    [0, 1, 2],  # 0: left flank (shared by edge a-b)
    [1, 0, 3],  # 1: right flank (shared by edge a-b)
    [0, 2, 4],  # 2: corner front-left (incident to a only)
    [0, 4, 3],  # 3: corner front-right (incident to a only)
    [1, 5, 2],  # 4: slope back-left (incident to b only)
    [1, 3, 5],  # 5: slope back-right (incident to b only)
])
ridge_edge = np.array([0, 1])
incident_a = np.array([0, 1, 2, 3])
incident_b = np.array([0, 1, 4, 5])
surviving_faces = np.array([2, 3, 4, 5])
face_targets = np.array([3500, 1800, 900, 450, 250])
surface_size, frame_duration_ms = 2, 600


# --- math -----------------------------------------------------------------------------
def qem():
    """The ridge edge `(a, b)` of a six-triangle patch contracts to its quadric-optimal vertex."""
    ridge = Topology(len(ridge_vertices), ridge_faces)
    vertices = mv.point(ridge_vertices).field()                                 # Point-[V]
    # Join each triangle's three corners into its unit supporting plane.
    corners = ridge.incidence[2, 0] * vertices                                  # [3] Point-[F]
    face_planes = (corners[0] & corners[1] & corners[2]).normalized()           # Plane-[F]
    # Sum each triangle's squared-distance plane dyad onto its incident vertices.
    dyads = face_planes * (face_planes & Point)                                 # (Plane+ <- Point-)[F]
    quadrics = ridge.incidence[2, 0].sum(axis=0).adjugate() * dyads             # (Plane+ <- Point-)[V]
    ends = ridge.reading(ridge_edge, 0)                                         # Scalar+[2 <- V]
    quadric_a, quadric_b = (ends * quadrics).batch()                            # [] Plane+ <- Point-, Plane+ <- Point-
    # Collapsing the edge adds the endpoint quadrics and places the replacement vertex at their unit-weight minimum.
    edge_quadric = quadric_a + quadric_b                                        # [] Plane+ <- Point-
    optimum = collapse_position(edge_quadric, (ends * vertices).batch().mean(axis=0))   # [] Point-
    collapsed_vertices = vertices + ends.adjugate() * (optimum - ends * vertices)       # Point-[V]
    return ridge, vertices, incident_a, incident_b, quadric_a, quadric_b, edge_quadric, optimum, collapsed_vertices, surviving_faces


# --- plumbing -------------------------------------------------------------------------
def bunny() -> tuple[Topology, Point]:
    """The bundled Stanford bunny triangulated surface stands centred and upright (`Point-[V]`)."""
    topology, coordinates = shapes.surface(MODEL_DIR / "bunny.obj", surface_size)
    upright = (mv.yz * (-np.pi / 4)).exp()                                      # [] Rotor+
    return topology, upright >> mv.point(coordinates).field()


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.qem import render

    scene = qem()
    save_figure(render.draw_qem(*scene), "qem_mesh_simplification")
    topology, vertices = bunny()
    bunny_states = []

    def recorded():
        for state in simplify(topology, vertices, face_targets):
            bunny_states.append(state)
            yield state

    save_animation(render.animate_surface(vertices, recorded()), "qem_bunny", frame_duration_ms, stepped=True)

    # checks
    # At the optimum, the spatial part of the polar plane vanishes and the joint error undercuts both endpoints.
    ridge, vertices, _, _, _, _, edge_quadric, optimum, _, _ = scene
    assert edge_quadric(optimum).norm() < 1e-9
    endpoints = (ridge.reading(ridge_edge, 0) * vertices).batch()               # [2] Point-
    assert ((edge_quadric(optimum) & optimum) < (edge_quadric(endpoints) & endpoints)).all()
    V, E, F = topology.n_elements
    assert [len(step_topology.elements[2]) for step_topology, _ in bunny_states] == [F, *face_targets]
    assert bunny_states[-1][0].is_closed and bunny_states[-1][0].euler_characteristic == 2


if __name__ == "__main__":
    main()
