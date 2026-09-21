"""Scenes and entry points for the Garland–Heckbert QEM example.

One function per figure. Each builds the concrete scene, hands it to the
mathematics in `core`, and hands the resulting geometry to `render`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D
from examples import PLOT_DIR
from examples.geometry.qem import core, render

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()


def point(coords: np.ndarray) -> Point:
    """Construct finite points from xyz coordinates in an explicitly named basis."""
    return mv("yzw zxw xyw", coords) + mv.zyx


def plane(normal: np.ndarray, point_on_plane: np.ndarray) -> Plane:
    """Construct normalized planes from normal vectors and points on the planes."""
    n = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    d = -np.sum(n * point_on_plane, axis=-1)
    return mv("x y z", n) + mv.w * d


def qem_figure(plot_path: Path) -> plt.Figure:
    """Demonstrate Garland–Heckbert QEM across an authentic triangle mesh edge."""
    # Vertices of a 3D surface patch with a sharp ridge terminating at an apex:
    coords = np.array([
        [ 0.35,  0.0,   0.25],  # 0: vertex a (sharp corner apex)
        [-0.35,  0.0,   0.25],  # 1: vertex b (crease continuation)
        [ 0.0,   0.55, -0.15],  # 2: left base vertex
        [ 0.0,  -0.55, -0.15],  # 3: right base vertex
        [ 0.75,  0.0,  -0.15],  # 4: front tip (steep corner at a)
        [-0.85,  0.0,  -0.15],  # 5: back tip (gentle ramp at b)
    ])
    verts = point(coords)

    # 6 triangular faces forming the 2-manifold surface patch:
    faces = np.array([
        [0, 1, 2],  # 0: left flank (shared by edge a-b)
        [1, 0, 3],  # 1: right flank (shared by edge a-b)
        [0, 2, 4],  # 2: corner front-left (incident to a only)
        [0, 4, 3],  # 3: corner front-right (incident to a only)
        [1, 5, 2],  # 4: ramp back-left (incident to b only)
        [1, 3, 5],  # 5: ramp back-right (incident to b only)
    ])

    # Supporting planes for all faces via GA regressive product (join of 3 vertices):
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_planes = (v0 & v1 & v2).normalized()

    # Incident face plane sets for the two endpoints:
    # Notice faces 0 and 1 (flanking the edge) appear in BOTH sets!
    planes_a = face_planes[[0, 1, 2, 3]]
    planes_b = face_planes[[0, 1, 4, 5]]

    # Compute QEM for both vertices and their collapsed edge in one coherent scope:
    qa, qb, q_edge, v_edge = core.edge_collapse(planes_a, planes_b)

    opt_xyz = render.euclidean(v_edge)
    edge_coords = coords[[0, 1]]

    # Collapsed mesh: the 2 flanking faces (0 and 1) degenerate and vanish.
    # The remaining 4 faces contract their a and b vertices to v_edge:
    coords_collapsed = coords.copy()
    coords_collapsed[0] = opt_xyz
    coords_collapsed[1] = opt_xyz
    collapsed_faces = faces[2:]

    # Consistent facet colors:
    # Blue: shared flanking ridge triangles (faces 0, 1)
    # Red/Rose: steep corner triangles at vertex a (faces 2, 3)
    # Purple/Violet: gentle ramp triangles at vertex b (faces 4, 5)
    mesh_colors = ["#0ea5e9", "#0284c7", "#f43f5e", "#e11d48", "#8b5cf6", "#7c3aed"]

    panel_a = (
        coords[faces[[0, 1, 2, 3]]],
        coords[0],
        qa,
        "1. Vertex a Neighborhood (Sharp Corner)\nFlanking ridge faces (blue) + end-cap faces (red)",
        [mesh_colors[i] for i in [0, 1, 2, 3]],
        "#f43f5e",
    )
    panel_b = (
        coords[faces[[0, 1, 4, 5]]],
        coords[1],
        qb,
        "2. Vertex b Neighborhood (Crease Transition)\nShared ridge faces (blue) + transition ramp (purple)",
        [mesh_colors[i] for i in [0, 1, 4, 5]],
        "#8b5cf6",
    )
    panel_edge = (
        coords[faces],
        coords_collapsed[collapsed_faces],
        edge_coords,
        q_edge,
        v_edge,
        "3. Simplified Mesh (Edge Contraction)\nFlanking faces vanish; optimal vertex minimizes Q_edge",
        mesh_colors,
        [mesh_colors[i] for i in [2, 3, 4, 5]],
    )

    return render.draw_qem_figure(panel_a, panel_b, panel_edge, plot_path)


def main(plot_path: Path) -> plt.Figure:
    """Render the Garland–Heckbert QEM mesh simplification figure."""
    return qem_figure(plot_path)


if __name__ == "__main__":
    out_file = PLOT_DIR / "qem_mesh_simplification.png"
    main(out_file)
