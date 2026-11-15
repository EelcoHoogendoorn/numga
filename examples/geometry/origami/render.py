"""GPU rendering for origami draws depth-tested, two-sided paper layers with an orange front, a cream back, and inked creases.

Each paper face is a convex polygon drawn as a fan of triangles. Coplanar layers are separated in display space by scaling their topological layer offsets by a fraction of a pixel, which the depth test resolves, while polygon perimeter edges are inked directly in the fragment shader so buried creases stay hidden underneath overlying flaps.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.origami import core
from examples.util.panels import fitted, show
from numga.render import meshes, solids
from numga.sparse import Sparse

front_colour = np.array([0.90, 0.46, 0.30])
back_colour = np.array([0.98, 0.88, 0.65])
edge_colour = np.array([0.37, 0.24, 0.17])
background = np.ones(3)
n_pixels = 300
elevation, azimuth = np.radians([35.0, 200.0])
margin = 0.136
view_room = 1.0
layer_spacing = 1e-4
edge_width = 0.9


colour = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    // Ink where the pixel lies within the crease width of an edge of its polygon, by how many pixels away it lies.
    let edges = clamp(1 - corner / (edge_width * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    let ink = max(edges.x, max(edges.y, edges.z));
    let paper = select(back_colour, front_colour, front);
    return vec4<f32>(mix(paper, edge_colour, ink), 1.0);
}
""", dict(edge_width=edge_width, front_colour=front_colour, back_colour=back_colour, edge_colour=edge_colour))


# --- plumbing -------------------------------------------------------------------------
def polygon_fans(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate each polygon as a corner fan `[F, 3]` and mark which edge opposite each triangle corner lies on the polygon perimeter `[F, 3]`."""
    triangles, perimeter = [], []
    for face in core.polygon_faces(counts):
        steps = np.arange(1, len(face) - 1)
        triangles.append(np.stack([np.full_like(steps, face[0]), face[steps], face[steps + 1]], axis=-1))
        perimeter.append(np.stack([np.ones_like(steps), steps == len(face) - 2, steps == 1], axis=-1))
    return np.concatenate(triangles), np.concatenate(perimeter).astype(float)


def sheet_viewport(points: core.Point) -> solids.View:
    """Frame a batch of points in a square orthographic view."""
    return solids.view([points], 0.0, elevation, azimuth, margin, n_pixels).squared()


def state_picture(state: core.PaperState, view: solids.View) -> np.ndarray:
    """Render a paper state from `view` as an `[n_pixels, n_pixels, 3]` RGB image."""
    points, counts, offsets, _ = state
    F = len(counts)
    corner_offsets = Sparse.selection(np.repeat(np.arange(F), counts), F) * offsets  # Direction-[V]
    corners = (points + corner_offsets * layer_spacing).batch()                 # [V] Point-
    triangles, perimeter = polygon_fans(counts)
    return meshes.draw(corners, triangles, np.zeros(3), meshes.flat(corners, triangles), view, background,
                       face_values=perimeter, colour=colour, gloss=0.0)


def paper_figure(state: core.PaperState) -> plt.Figure:
    """Render a single paper state in a tight square figure."""
    points = state[0].batch()                                                   # [V] Point-
    figure, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    ax.set_axis_off()
    view = fitted(sheet_viewport(points), ax)
    show(ax, state_picture(state, view), view)
    return figure


def folding_animation(states: Iterator[core.PaperState]) -> Iterator[np.ndarray]:
    """Render each state of the folding sequence from a fixed view framing the initial flat sheet."""
    states = iter(states)
    initial_state = next(states)
    initial_view = sheet_viewport(initial_state[0].batch())
    view = initial_view._replace(half=initial_view.half * view_room)
    for state in chain([initial_state], states):
        yield state_picture(state, view)
