"""Scenes for the Garland–Heckbert QEM example.

One function per figure. Each builds the concrete scene, hands it to the mathematics
in `core`, and returns the resulting geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from numga import concatenate
from examples.geometry.qem.core import edge_collapse, ridge_patch


def qem():
    """Collapse the ridge edge (a, b) of a triangle patch to its quadric-optimal vertex."""
    vertices, faces = ridge_patch()

    # Supporting planes for all faces via GA regressive product (join of 3 vertices):
    face_planes = (vertices[faces[:, 0]] & vertices[faces[:, 1]] & vertices[faces[:, 2]]).normalized()

    # Incident face plane sets for the two endpoints:
    # Notice faces 0 and 1 (flanking the edge) appear in BOTH sets!
    incident_a = np.array([0, 1, 2, 3])
    incident_b = np.array([0, 1, 4, 5])
    qa, qb, q_edge, v_edge = edge_collapse(face_planes[incident_a], face_planes[incident_b])

    # Collapsed mesh: the 2 flanking faces (0 and 1) degenerate and vanish.
    # The remaining 4 faces contract their a and b vertices to v_edge:
    collapsed = concatenate([v_edge.broadcast_to((2,)), vertices[2:]])
    surviving = np.array([2, 3, 4, 5])

    # --- checks
    # The optimum's polar plane is the plane at infinity: it has no Euclidean part.
    assert q_edge(v_edge).norm().to_array() < 1e-9
    # And its joint error undercuts that of both endpoints.
    endpoints = vertices[:2]
    assert np.all((q_edge(v_edge) & v_edge).to_array() < (q_edge(endpoints) & endpoints).to_array())

    return vertices, faces, incident_a, incident_b, qa, qb, q_edge, v_edge, collapsed, surviving


if __name__ == "__main__":
    from examples.animation import save_figure
    from examples.geometry.qem import render

    save_figure(render.draw_qem(*qem()), "qem_mesh_simplification")
