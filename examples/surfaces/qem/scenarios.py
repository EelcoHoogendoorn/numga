"""Scenes for quadric error metrics contract a ridge edge on a triangle patch and progressively simplify the Stanford bunny."""

from __future__ import annotations

import numpy as np

from examples import MODEL_DIR
from examples.surfaces.qem.core import Mesh, Point, collapse_position, mv, simplify

# A surface patch with a sharp ridge from a corner a to a gentler end b: its vertices, a and b first, and its
# triangles, the two flanking the ridge edge first.
RIDGE_VERTICES = [
    [ 0.35,  0.0,   0.25],  # 0: vertex a (sharp corner apex)
    [-0.35,  0.0,   0.25],  # 1: vertex b (crease continuation)
    [ 0.0,   0.55, -0.15],  # 2: left base vertex
    [ 0.0,  -0.55, -0.15],  # 3: right base vertex
    [ 0.75,  0.0,  -0.15],  # 4: front tip (steep corner at a)
    [-0.85,  0.0,  -0.15],  # 5: back tip (gentle ramp at b)
]
RIDGE_FACES = np.array([
    [0, 1, 2],  # 0: left flank (shared by edge a-b)
    [1, 0, 3],  # 1: right flank (shared by edge a-b)
    [0, 2, 4],  # 2: corner front-left (incident to a only)
    [0, 4, 3],  # 3: corner front-right (incident to a only)
    [1, 5, 2],  # 4: ramp back-left (incident to b only)
    [1, 3, 5],  # 5: ramp back-right (incident to b only)
])
# The ends of the ridge edge, a and b.
RIDGE_EDGE = np.array([0, 1])
# The faces drawn at each endpoint of the ridge edge (a, b), those at it: the two flanking the edge appear in both.
INCIDENT_A = np.array([0, 1, 2, 3])
INCIDENT_B = np.array([0, 1, 4, 5])
# The faces that survive the collapse; the two flanking faces degenerate and vanish.
SURVIVING_FACES = np.array([2, 3, 4, 5])
FACE_TARGETS = np.array([3500, 1800, 900, 450, 250])
MESH_SIZE, DURATION_MS = 2, 600


# --- math -----------------------------------------------------------------------------
def qem():
    """The ridge edge `(a, b)` of a triangle patch collapses to its quadric-optimal vertex."""
    ridge = Mesh(len(RIDGE_VERTICES), RIDGE_FACES)
    vertices = mv.point(RIDGE_VERTICES).field()                                 # Point-[vertices]
    # Each face's supporting plane, the join of its three corners.
    corners = ridge.incidence[2, 0] * vertices                                  # [corners] Point-[faces]
    face_planes = (corners[0] & corners[1] & corners[2]).normalized()           # Plane-[faces]
    # A unit plane measures a point's signed distance from it as `plane & point`; its dyad, a map from points to
    # planes, gives the squared distance. Each vertex's error quadric sums the dyads of its triangles' planes.
    dyads = face_planes * (face_planes & Point)                                 # (Plane+ <- Point-)[faces]
    quadrics = ~ridge.incidence[2, 0].sum(axis=0) * dyads                       # (Plane+ <- Point-)[vertices]
    ends = ridge.reading(RIDGE_EDGE)                                            # Scalar+[ends <- vertices]
    quadric_a, quadric_b = (ends * quadrics).batch()                            # [] Plane+ <- Point-, Plane+ <- Point-
    # Contracting the edge adds the endpoints' quadrics; the best vertex is its unit-weight minimum.
    edge_quadric = quadric_a + quadric_b                                        # [] Plane+ <- Point-
    optimum = collapse_position(edge_quadric, (ends * vertices).batch().mean(axis=0))   # [] Point-
    # The collapsed mesh: the surviving faces contract their a and b vertices to the optimum.
    collapsed_vertices = vertices + ~ends * (optimum - ends * vertices)         # Point-[vertices]
    return ridge, vertices, INCIDENT_A, INCIDENT_B, quadric_a, quadric_b, edge_quadric, optimum, collapsed_vertices, SURVIVING_FACES


# --- plumbing -------------------------------------------------------------------------
def bunny() -> tuple[Mesh, Point]:
    """The bundled Stanford bunny's triangles, and its vertices, centred and stood upright."""
    mesh, vertices = Mesh.read(MODEL_DIR / "bunny.obj", MESH_SIZE)
    upright = (mv.yz * (-np.pi / 4)).exp()                                      # [] Rotor+
    return mesh, upright >> vertices


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.qem import render

    scene = qem()
    save_figure(render.draw_qem(*scene), "qem_mesh_simplification")
    mesh, vertices = bunny()
    save_animation(render.animate_mesh(simplify(mesh, vertices, FACE_TARGETS), vertices),
                   "qem_bunny", DURATION_MS, stepped=True)

    # checks
    # The optimum's polar plane is the plane at infinity: it has no Euclidean part; and its joint error
    # undercuts that of both endpoints.
    ridge, vertices, _, _, _, _, edge_quadric, optimum, _, _ = scene
    assert edge_quadric(optimum).norm() < 1e-9
    endpoints = (ridge.reading(RIDGE_EDGE) * vertices).batch()                  # [ends] Point-
    assert np.all((edge_quadric(optimum) & optimum) < (edge_quadric(endpoints) & endpoints))


if __name__ == "__main__":
    main()
