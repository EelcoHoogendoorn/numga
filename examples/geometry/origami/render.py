"""GPU rendering for origami draws depth-tested, two-sided paper layers with an orange front, a cream back, and inked creases.

Each paper face is a convex polygon drawn as a fan of triangles. Layers that lie in one plane are held apart by their offsets, a few hundredths of a pixel each, which the depth test resolves; the creases are inked by the faces themselves where a pixel lies near an edge of its polygon, so a crease buried under a flap stays hidden.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.geometry.origami import core
from examples.util.wgsl import fitted, meshes, show, solids

front_colour = np.array([0.90, 0.46, 0.30])
back_colour = np.array([0.98, 0.88, 0.65])
edge_colour = np.array([0.37, 0.24, 0.17])
background = np.ones(3)
n_pixels = 300
# The view, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the paper as a
# share of its larger side.
elevation, azimuth = np.radians([35.0, 200.0])
margin = 0.136
# How many times over the animation's view frames the paper as it starts: once, as the flat sheet spans the most.
view_room = 1.0
# The separation of one layer from the next, and the creases' width in pixels.
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
def coordinates(points: core.Point) -> np.ndarray:
    """Read out the Euclidean coordinates of finite points."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & points.normalized()[..., None]).to_array()


def fans(paper: core.Paper) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate each polygon as a fan from its first corner `[triangles, 3]`, and mark which edge opposite each corner lies on the polygon's perimeter `[triangles, 3]`."""
    triangles, perimeter = [], []
    for face in paper.faces():
        steps = np.arange(1, len(face) - 1)
        triangles.append(np.stack([np.full_like(steps, face[0]), face[steps], face[steps + 1]], axis=-1))
        perimeter.append(np.stack([np.ones_like(steps), steps == len(face) - 2, steps == 1], axis=-1))
    return np.concatenate(triangles), np.concatenate(perimeter).astype(float)


def viewport(points: core.Point) -> solids.View:
    """Frame a batch of points in a square view."""
    return solids.view([points], 0.0, elevation, azimuth, margin, n_pixels).squared()


def picture(paper: core.Paper, view: solids.View) -> np.ndarray:
    """Render the paper from a view as an `[n_pixels, n_pixels, 3]` image."""
    # Each corner held off its layer by its facet's offset, carried with the facet through every fold.
    corners = paper.points + paper.offsets[np.repeat(np.arange(len(paper.counts)), paper.counts)] * layer_spacing  # [corners] Point-
    triangles, perimeter = fans(paper)
    return meshes.draw(corners, triangles, np.zeros(3), meshes.flat(corners, triangles), view, background,
                       face_values=perimeter, colour=colour, gloss=0.0)


def draw(paper: core.Paper) -> plt.Figure:
    """Render a single paper state in a figure."""
    figure, ax = plt.subplots(figsize=(3, 3), layout="constrained")
    # The axes as bare as the picture leaves them, for the layout to give them their room.
    ax.set_axis_off()
    view = fitted(viewport(paper.points), ax)
    show(ax, picture(paper, view), view)
    return figure


def animate(states: Iterator[core.Paper], room: float = view_room) -> Iterator[np.ndarray]:
    """Render each state of the folding sequence from a single view framing the initial sheet `room` times over."""
    states = iter(states)
    initial_paper = next(states)
    initial_view = viewport(initial_paper.points)
    view = initial_view._replace(half=initial_view.half * room)
    for state in chain([initial_paper], states):
        yield picture(state, view)
