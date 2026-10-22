"""Drawing for the skinning example: each skin as a striped quad mesh."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.surfaces.skinning.core import Point, mv
from examples.wgsl import meshes, show, solids

# The view, elevation and azimuth in radians, azimuth zero looking along +y, the margin around the skins as a
# share of their larger side, and the pixels across a picture; the stripes' colour map around the cylinder.
elevation, azimuth = np.radians([30.0, 30.0])
margin = 0.08
n_pixels = 900
stripe_map = "viridis"
background = np.ones(3)
# The quads' edges: their width in pixels and how dark they are inked.
edge_width = 0.9
edge_darkness = 0.7
# Each face's colour, darkened toward the edges of its quad that lie within `edge_width` pixels: the ink of the
# edge opposite each corner in its face values, by how near the pixel lies to that edge.
inked = f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {{
    let edges = clamp(1 - corner / ({edge_width} * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    return vec4<f32>(values.rgb * (1 - max(edges.x, max(edges.y, edges.z))), 1.0);
}}
"""


# --- plumbing -------------------------------------------------------------------------
def xyz(points: Point) -> np.ndarray:
    """Coordinates of points: their pairings with the planes x, y and z, over their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def quads(n_rings: int, n_around: int) -> tuple[np.ndarray, np.ndarray]:
    """The cylinder's triangles, two to each quad between neighbouring rings and neighbouring vertices around,
    closed around, `[rings - 1, around, 2, 3]` indices into the vertices; and each triangle's ink at the edge
    opposite each corner, the quad's own edges inked and its diagonal not."""
    # Closed around: each ring's first vertex again after its last.
    index = np.pad(np.arange(n_rings * n_around).reshape(n_rings, n_around), ((0, 0), (0, 1)), mode="wrap")
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    faces = corners[..., [[0, 1, 2], [0, 2, 3]]]
    return faces, np.broadcast_to([[1, 0, 1], [1, 1, 0]], faces.shape) * edge_darkness


def surface(points: Point, n_rings: int, n_around: int, view: solids.View) -> np.ndarray:
    """The skinned cylinder as a smoothly lit quad mesh, striped around, its quads' edges inked."""
    places = solids.mv.point(xyz(points))
    faces, ink = quads(n_rings, n_around)
    # Each stripe its colour, at every corner of the triangles that run along it.
    stripes = plt.get_cmap(stripe_map)(np.linspace(0.0, 1.0, n_around + 1)[:n_around])[:, :3]   # [around, 3]
    colours = np.broadcast_to(stripes[None, :, None, None], faces.shape + (3,))                 # [rings - 1, around, 2, 3, 3]
    faces = faces.reshape(-1, 3)
    return meshes.draw(places, faces, colours.reshape(-1, 3, 3), meshes.smooth(places, faces), view, background,
                       face_values=ink.reshape(-1, 3), colour=inked)


def draw_skinning(motor_skin: Point, slerp_skin: Point, matrix_skin: Point, n_rings: int, n_around: int) -> plt.Figure:
    """The three skins side by side, to one scale: blended motors normalised, slerped, and blended maps."""
    skins = (motor_skin, slerp_skin, matrix_skin)
    view = solids.view([solids.mv.point(xyz(skin)) for skin in skins], 0.0, elevation, azimuth, margin, n_pixels)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    for ax, skin in zip(axes, skins):
        show(ax, surface(skin, n_rings, n_around, view), view)
    return fig
