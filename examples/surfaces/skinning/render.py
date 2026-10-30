"""Render the skinned cylinder meshes for the skinning example as smoothly lit, striped quad surfaces."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.skinning.core import Point
from examples.util.wgsl import fitted, meshes, show, solids

# The view's elevation and azimuth in radians, with azimuth zero looking along +y, the margin around the skins
# as a share of their larger side, the pixel width before fitting to the axes, and the colour map for the stripes.
elevation, azimuth = np.radians([30.0, 30.0])
margin = 0.08
n_pixels = 900
stripe_map = "viridis"
background = np.ones(3)
# The width of the quad edges in pixels and how darkly they are inked.
edge_width = 0.4
edge_darkness = 0.7
# Darken each face toward the quad edges that lie within `edge_width` pixels, using the ink weight of the edge
# opposite each corner stored in the face values.
inked = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let edges = clamp(1 - corner / (edge_width * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    return vec4<f32>(values.rgb * (1 - max(edges.x, max(edges.y, edges.z))), 1.0);
}
""", dict(edge_width=edge_width))


# --- plumbing -------------------------------------------------------------------------
def quads(n_rings: int, n_around: int) -> tuple[np.ndarray, np.ndarray]:
    """The cylinder's `[rings - 1, around, 2, 3]` triangle indices, two per quad between adjacent rings and
    azimuth steps, and the per-edge ink weights that darken quad boundaries while leaving diagonals uninked."""
    # Wrap each ring around by repeating its first vertex after its last.
    index = np.pad(np.arange(n_rings * n_around).reshape(n_rings, n_around), ((0, 0), (0, 1)), mode="wrap")
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    faces = corners[..., [[0, 1, 2], [0, 2, 3]]]
    return faces, np.broadcast_to([[1, 0, 1], [1, 1, 0]], faces.shape) * edge_darkness


def surface(points: Point, n_rings: int, n_around: int, view: solids.View) -> np.ndarray:
    """Render a skinned cylinder as a smoothly lit quad mesh with azimuth stripes and inked quad edges."""
    faces, ink = quads(n_rings, n_around)
    # Assign each azimuth stripe its colour at every corner of the triangles along that stripe.
    stripes = plt.get_cmap(stripe_map)(np.linspace(0.0, 1.0, n_around + 1)[:n_around])[:, :3]   # [around, 3]
    colours = np.broadcast_to(stripes[None, :, None, None], faces.shape + (3,))                 # [rings - 1, around, 2, 3, 3]
    faces = faces.reshape(-1, 3)
    topology = meshes.Mesh(len(points), faces)
    return meshes.draw(points, faces, colours.reshape(-1, 3, 3), meshes.smooth(points, topology), view, background,
                       face_values=ink.reshape(-1, 3), colour=inked, gloss=solids.gloss)


def draw_skinning(motor_skin: Point, slerp_skin: Point, matrix_skin: Point, n_rings: int, n_around: int) -> plt.Figure:
    """Draw the three skinned cylinders side by side to a common scale for normalised motor blending, motor slerp, and point-map blending."""
    skins = (motor_skin, slerp_skin, matrix_skin)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=120)
    frame = solids.view(list(skins), 0.0, elevation, azimuth, margin, n_pixels)
    view = fitted(frame, axes[0])
    for ax, skin in zip(axes, skins):
        show(ax, surface(skin, n_rings, n_around, view), view)
    return fig
