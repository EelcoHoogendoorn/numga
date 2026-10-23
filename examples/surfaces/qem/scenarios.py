"""Scenes for the Garland–Heckbert QEM example: one ridge edge of a triangle patch collapsed to its
quadric-optimal vertex."""

from __future__ import annotations

import numpy as np

from numga import concatenate
from examples.surfaces.qem.core import edge_collapse, ridge_patch

# The faces incident to each endpoint of the ridge edge (a, b): the two flanking the edge appear in both.
INCIDENT_A = np.array([0, 1, 2, 3])
INCIDENT_B = np.array([0, 1, 4, 5])
# The faces that survive the collapse; the two flanking faces degenerate and vanish.
SURVIVING = np.array([2, 3, 4, 5])


# --- math -----------------------------------------------------------------------------
def qem():
    """Collapse the ridge edge (a, b) of a triangle patch to its quadric-optimal vertex."""
    vertices, faces = ridge_patch()
    # Each face's supporting plane, the join of its three vertices.
    face_planes = (vertices[faces[:, 0]] & vertices[faces[:, 1]] & vertices[faces[:, 2]]).normalized()   # [faces] Plane
    quadric_a, quadric_b, edge_quadric, optimum = edge_collapse(face_planes[INCIDENT_A], face_planes[INCIDENT_B])
    # The collapsed mesh: the surviving faces contract their a and b vertices to the optimum.
    collapsed = concatenate([optimum.broadcast_to((2,)), vertices[2:]])        # [vertices] Point
    return vertices, faces, INCIDENT_A, INCIDENT_B, quadric_a, quadric_b, edge_quadric, optimum, collapsed, SURVIVING


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.surfaces.qem import render

    scene = qem()
    save_figure(render.draw_qem(*scene), "qem_mesh_simplification")

    # checks
    # The optimum's polar plane is the plane at infinity: it has no Euclidean part; and its joint error
    # undercuts that of both endpoints.
    vertices, _, _, _, _, _, edge_quadric, optimum, _, _ = scene
    assert edge_quadric(optimum).norm().to_array() < 1e-9
    endpoints = vertices[:2]
    assert np.all((edge_quadric(optimum) & optimum).to_array() < (edge_quadric(endpoints) & endpoints).to_array())


if __name__ == "__main__":
    main()
