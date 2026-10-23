"""Depth-tested paper layers on the GPU, two-sided: an orange front and a cream back, the creases inked.

Each paper face is a convex polygon, drawn as a fan of triangles. Layers that lie in one plane are held apart by
their offsets, a few hundredths of a pixel each, which the depth test resolves; the creases are inked by the faces
themselves, where a pixel lies near an edge of its polygon, so a crease buried under a flap stays hidden.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.geometry.origami import core
from examples.util.wgsl import meshes, show, solids, vec3

front_colour = np.array([0.90, 0.46, 0.30])
back_colour = np.array([0.98, 0.88, 0.65])
edge_colour = np.array([0.37, 0.24, 0.17])
background = np.ones(3)
pixels = 600
# The view, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the paper as a
# share of its larger side.
elevation, azimuth = np.radians([35.0, 200.0])
margin = 0.136
# How many times over the animation's view frames the paper as it starts: once, as the flat sheet spans the most.
view_room = 1.0
# The separation of one layer from the next, and the creases' width in pixels.
layer_spacing = 1e-4
edge_width = 1.8


colour = f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {{
    // Ink where the pixel lies within the crease width of an edge of its polygon, by how many pixels away it lies.
    let edges = clamp(1 - corner / ({edge_width} * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    let ink = max(edges.x, max(edges.y, edges.z));
    let paper = select({vec3(back_colour)}, {vec3(front_colour)}, front);
    return vec4<f32>(mix(paper, {vec3(edge_colour)}, ink), 1.0);
}}
"""


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    values = points.cast(core.ga.subspace("wzy wxz wyx xyz")).kernel
    return values[..., :3] / values[..., 3:]


def fans(paper: core.Paper) -> tuple[np.ndarray, np.ndarray]:
    """Each polygon as a fan of triangles from its first corner `[triangles, 3]`, and which edge of each triangle,
    the one opposite each corner, is an edge of the polygon `[triangles, 3]`."""
    triangles, perimeter = [], []
    for face in paper.faces():
        steps = np.arange(1, len(face) - 1)
        triangles.append(np.stack([np.full_like(steps, face[0]), face[steps], face[steps + 1]], axis=-1))
        perimeter.append(np.stack([np.ones_like(steps), steps == len(face) - 2, steps == 1], axis=-1))
    return np.concatenate(triangles), np.concatenate(perimeter).astype(float)


def viewport(points: core.Point) -> solids.View:
    """A view framing points in a square image."""
    return solids.view([points], 0.0, elevation, azimuth, margin, pixels).squared()


def picture(paper: core.Paper, view: solids.View) -> np.ndarray:
    """The paper as the view sees it, `[pixels, pixels, 3]` bytes."""
    # Each corner held off its layer by its facet's offset, carried with the facet through every fold.
    corners = paper.points + paper.offsets[np.repeat(np.arange(len(paper.counts)), paper.counts)] * layer_spacing
    triangles, perimeter = fans(paper)
    return meshes.draw(corners, triangles, np.zeros(3), meshes.flat(corners, triangles), view, background,
                       face_values=perimeter, colour=colour, gloss=0.0)


def draw(paper: core.Paper) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    view = viewport(paper.points)
    show(ax, picture(paper, view), view)
    return figure


def animate(states: Iterator[core.Paper], room: float = view_room) -> Iterator[np.ndarray]:
    """The paper through its folds, all from one view framing the paper as it starts, `room` times over."""
    states = iter(states)
    first = next(states)
    framed = viewport(first.points)
    view = framed._replace(half=framed.half * room)
    for state in chain([first], states):
        yield picture(state, view)
