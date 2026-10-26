"""A ridge-edge contraction and progressive simplification of the Stanford bunny."""

from __future__ import annotations

import numpy as np

from numga import concatenate
from examples import MODEL_DIR
from examples.surfaces.qem.core import Point, collapse_position, mv, simplify
from examples.surfaces.qem.plumbing import read_mesh

# A surface patch with a sharp ridge from a corner a to a gentler end b: its vertices, a and b first, and its
# triangles, the two flanking the ridge edge first.
RIDGE = np.array([
    [ 0.35,  0.0,   0.25],  # 0: vertex a (sharp corner apex)
    [-0.35,  0.0,   0.25],  # 1: vertex b (crease continuation)
    [ 0.0,   0.55, -0.15],  # 2: left base vertex
    [ 0.0,  -0.55, -0.15],  # 3: right base vertex
    [ 0.75,  0.0,  -0.15],  # 4: front tip (steep corner at a)
    [-0.85,  0.0,  -0.15],  # 5: back tip (gentle ramp at b)
])
RIDGE_FACES = np.array([
    [0, 1, 2],  # 0: left flank (shared by edge a-b)
    [1, 0, 3],  # 1: right flank (shared by edge a-b)
    [0, 2, 4],  # 2: corner front-left (incident to a only)
    [0, 4, 3],  # 3: corner front-right (incident to a only)
    [1, 5, 2],  # 4: ramp back-left (incident to b only)
    [1, 3, 5],  # 5: ramp back-right (incident to b only)
])
# The faces incident to each endpoint of the ridge edge (a, b): the two flanking the edge appear in both.
INCIDENT_A = np.array([0, 1, 2, 3])
INCIDENT_B = np.array([0, 1, 4, 5])
# The faces that survive the collapse; the two flanking faces degenerate and vanish.
SURVIVING = np.array([2, 3, 4, 5])
FACE_TARGETS = np.array([3500, 1800, 900, 450, 250])
MESH_SIZE, DURATION_MS = 2, 600


# --- math -----------------------------------------------------------------------------
def qem():
    """Collapse the ridge edge (a, b) of a triangle patch to its quadric-optimal vertex."""
    vertices, faces = mv.point(RIDGE), RIDGE_FACES                             # [vertices] Point, [faces, 3]
    # Each face's supporting plane, the join of its three vertices.
    face_planes = (vertices[faces[:, 0]] & vertices[faces[:, 1]] & vertices[faces[:, 2]]).normalized()   # [faces] Plane
    # A unit plane measures a point's signed distance from it as `plane & point`; its dyad, a map from points to
    # planes, gives the squared distance. Each endpoint's error quadric sums the dyads of its triangles' planes.
    dyads = face_planes * (face_planes & Point)                                 # [faces] Quadric
    quadric_a, quadric_b = dyads[INCIDENT_A].sum(axis=0), dyads[INCIDENT_B].sum(axis=0)   # Quadric each
    # Contracting the edge adds the endpoints' quadrics; the best vertex is its unit-weight minimum.
    edge_quadric = quadric_a + quadric_b                                        # Quadric
    optimum = collapse_position(edge_quadric, (vertices[0] + vertices[1]) / 2)  # Point
    # The collapsed mesh: the surviving faces contract their a and b vertices to the optimum.
    collapsed = concatenate([optimum.broadcast_to((2,)), vertices[2:]])        # [vertices] Point
    return vertices, faces, INCIDENT_A, INCIDENT_B, quadric_a, quadric_b, edge_quadric, optimum, collapsed, SURVIVING


# --- plumbing -------------------------------------------------------------------------
def bunny() -> tuple[Point, np.ndarray]:
    """The bundled bunny centred and stood upright, with its original triangle connectivity."""
    vertices, faces = read_mesh(MODEL_DIR / "bunny.obj", MESH_SIZE)
    upright = (mv.yz * (-np.pi / 4)).exp()
    return upright >> vertices, faces


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.qem import render

    scene = qem()
    save_figure(render.draw_qem(*scene), "qem_mesh_simplification")
    vertices, faces = bunny()
    save_animation(render.animate_mesh(simplify(vertices, faces, FACE_TARGETS), vertices),
                   "qem_bunny", DURATION_MS)

    # checks
    # The optimum's polar plane is the plane at infinity: it has no Euclidean part; and its joint error
    # undercuts that of both endpoints.
    vertices, _, _, _, _, _, edge_quadric, optimum, _, _ = scene
    assert edge_quadric(optimum).norm().to_array() < 1e-9
    endpoints = vertices[:2]
    assert np.all((edge_quadric(optimum) & optimum).to_array() < (edge_quadric(endpoints) & endpoints).to_array())


if __name__ == "__main__":
    main()
