"""Drawing for the cube: each sticker a square, coloured by the face it belongs to."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.colors import to_rgb

from examples.geometry.rubiks_cube import core
from examples.wgsl import meshes, solids

# The faces' colours, by side and axis: +x red, +y blue, +z white, -x orange, -y green, -z yellow; the colour
# of the cut, and of the background.
COLOURS = np.array([to_rgb(colour) for colour in ("#c41e3a", "#0051ba", "#ffffff", "#ff5800", "#009e60", "#ffd500")])
CUT = np.array(to_rgb("#111111"))
BACKGROUND = np.ones(3)
# The view, elevation and azimuth in radians, azimuth zero looking along +y, the margin around the cube as a
# share of its larger side, and the pixels across a frame.
ELEVATION, AZIMUTH = np.radians([28.0, 32.0])
MARGIN = 0.08
N_PIXELS = 420
# The squares' edges: the width each square inks along its side of them, in pixels, and how dark.
EDGE_WIDTH = 1.0
EDGE_DARKNESS = 0.9
# Each face's colour, darkened toward the edges of its square that lie within `EDGE_WIDTH` pixels: the ink of
# the edge opposite each corner in its face values, by how near the pixel lies to that edge.
INKED = f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32> {{
    let edges = clamp(1 - corner / ({EDGE_WIDTH} * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    return values.rgb * (1 - max(edges.x, max(edges.y, edges.z)));
}}
"""


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def picture(corners: core.Point, colours: np.ndarray, cut: core.Point, view: solids.View) -> np.ndarray:
    """The cube's stickers as squares coloured by their faces, with dark edges, and the dark cut under the
    turning layer on both sides."""
    squares = np.concatenate([coordinates(corners), coordinates(cut).reshape(-1, 4, 3)])   # [squares, 4, 3]
    fills = np.concatenate([COLOURS[colours], np.broadcast_to(CUT, (cut.shape[0] * cut.shape[1], 3))])   # [squares, 3]
    places = solids.mv.point(squares.reshape(-1, 3))
    # Two triangles to each square, wound against its corners, which run clockwise seen from outside, so that
    # their normals point out; each inks the square's own edges, not the diagonal between them.
    faces = (4 * np.arange(len(squares))[:, None, None] + np.array([[0, 2, 1], [0, 3, 2]])).reshape(-1, 3)
    ink = np.tile([[1, 1, 0], [1, 0, 1]], (len(squares), 1)) * EDGE_DARKNESS                       # [faces, 3]
    values = np.repeat(fills, 2, axis=0)[:, None]                                                    # [faces, 1, 3]
    normals = meshes.flat(places, faces)
    return meshes.draw(places, faces, values, normals, view, BACKGROUND, face_values=ink, colour=INKED)


def animate(states: Iterable[tuple[core.Point, core.Point]], colours: np.ndarray) -> list[np.ndarray]:
    """A frame of the cube for each state of its stickers' corners and its turning cut, all seen by one camera
    framing every state."""
    states = list(states)
    view = solids.view([solids.mv.point(coordinates(corners).reshape(-1, 3)) for corners, _ in states], 0.0,
                       ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    return [picture(corners, colours, cut, view) for corners, cut in states]
