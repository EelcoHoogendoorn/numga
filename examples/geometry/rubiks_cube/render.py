"""GPU mesh rendering for the Rubik's cube draws each sticker as a coloured square with darkened edges and shades the inner cut beneath the turning layer."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate
from examples.geometry.rubiks_cube import core
from examples.util.wgsl import meshes, solids

# The six face colours by side and axis (+x red, +y blue, +z white, -x orange, -y green, -z yellow),
# together with the inner cut colour and the background colour.
colours_by_face = np.array([to_rgb(colour) for colour in ("#c41e3a", "#0051ba", "#ffffff", "#ff5800", "#009e60", "#ffd500")])
cut_colour = np.array(to_rgb("#111111"))
background = np.ones(3)
# The camera elevation and azimuth in radians (azimuth zero looking along +y), the viewport margin,
# and the image size in pixels.
elevation, azimuth = np.radians([28.0, 32.0])
margin = 0.08
n_pixels = 300
# The factor by which the viewport widens around the initial cube so turning corners stay in frame.
view_room = 1.25
# The width in pixels and darkness of the border inked along each sticker square's outer edges.
edge_width = 1.0
edge_darkness = 0.9
# Each square splits into two outward-facing triangles of its corners, which ink only the square's perimeter edges,
# not its diagonal.
square_triangles = np.array([[0, 2, 1], [0, 3, 2]])
perimeter = np.array([[1, 1, 0], [1, 0, 1]])
# Darken each sticker toward its outer square edges within `edge_width` pixels using the per-triangle edge mask in `face_values`.
edge_shader = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let edges = clamp(1 - corner / (edge_width * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    return vec4<f32>(values.rgb * (1 - max(edges.x, max(edges.y, edges.z))), 1.0);
}
""", dict(edge_width=edge_width))


# --- plumbing -------------------------------------------------------------------------
def picture(corners: core.Point, colours: np.ndarray, cut: core.Point, view: solids.View) -> np.ndarray:
    """Render the cube's stickers as coloured squares with dark edges together with the dark inner cut beneath the turning layer."""
    squares = concatenate([corners, cut.reshape(-1, cut.shape[-1])])                                  # [squares, corners] Point-
    fills = np.concatenate([colours_by_face[colours], np.broadcast_to(cut_colour, (cut.shape[0] * cut.shape[1], 3))])  # [squares, 3]
    places = solids.origin + solids.context.include_from_algebra(squares.reshape(-1))                           # [squares * corners] Point-
    # The places of each square's corners, and its two triangles through them.
    square_corners = np.arange(len(places)).reshape(squares.shape)                                    # [squares, corners]
    faces = square_corners[:, square_triangles].reshape(-1, 3)                                        # [squares * 2, 3]
    ink = np.tile(perimeter, (len(squares), 1)) * edge_darkness                                       # [faces, 3]
    values = np.repeat(fills, 2, axis=0)[:, None]                                                     # [faces, 1, 3]
    normals = meshes.flat(places, faces)
    return meshes.draw(places, faces, values, normals, view, background, face_values=ink, colour=edge_shader,
                       gloss=solids.gloss)


def animate(states: Iterable[tuple[core.Point, core.Point]], colours: np.ndarray,
            room: float = view_room) -> Iterator[np.ndarray]:
    """Render each frame of the cube's sticker corners and inner cut from a fixed camera framed on the initial cube."""
    states = iter(states)
    first_state = next(states)
    corners, _ = first_state
    initial_view = solids.view([solids.origin + solids.context.include_from_algebra(corners)], 0.0, elevation, azimuth, margin,
                               n_pixels)
    view = initial_view._replace(half=initial_view.half * room)
    for corners, cut in chain([first_state], states):
        yield picture(corners, colours, cut, view)
