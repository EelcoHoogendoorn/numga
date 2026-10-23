"""The cube's faces, oriented patches and volume: the deformed cube traced on the GPU, each face a translucent plate
of its own colour so that every face shows through the others, its edges rods, and each patch an arrow from its face's
centre, as long as its area; beside it, the areas and the volume as bars."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb

from numga import stack
from examples.util.animation import capture
from examples.mechanics.area_transport import core
from examples.util.wgsl import show, solids

FACES = np.array([(4, 5, 7, 6), (2, 3, 7, 6), (1, 3, 7, 5),
                  (0, 2, 3, 1), (0, 1, 5, 4), (0, 4, 6, 2)])
# The cube's twelve edges: the corners, numbered by their bits along x, y and z, joined across each bit.
EDGES = np.array([(corner, corner | bit) for bit in (4, 2, 1) for corner in range(8) if not corner & bit])
COLOURS = ("#2563eb", "#0d9488", "#f97316")
EDGE_COLOUR = "#334155"
ARROW_SCALE = 0.36
background = np.ones(3)
# The view, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the cubes as a share
# of the larger side, and the pixels across a picture.
elevation, azimuth = np.radians([23.0, -57.0 + 90.0])
margin = 0.16
n_pixels = 520
# How much of what lies behind each of a face's two sides hides; the edges' and the arrows' shafts' radii in pixels; an arrow's head, its length a share of the
# arrow's, and its radius a share of its own length.
face_opacity = 0.1
edge_radius = 1.5
shaft_radius = 2.5
head_share = 0.25
head_width = 0.4


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors, `[..., 3]`."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def arrow_ends(centres: core.Point, patches: core.Patch) -> np.ndarray:
    """Where each patch's arrow ends, `[..., 3]`: its face's centre, moved along the patch's normal as far as its
    area."""
    return coordinates(centres) + patches.cast(core.ga.subspace("x y z")).kernel * ARROW_SCALE


def view(vertices: core.Point, centres: core.Point, patches: core.Patch) -> solids.View:
    """A view framing every cube and arrow given."""
    points = [solids.mv.point(coordinates(vertices)), solids.mv.point(arrow_ends(centres, patches))]
    return solids.view(points, 0.0, elevation, azimuth, margin, n_pixels)


def picture(vertices: core.Point, centres: core.Point, patches: core.Patch, view: solids.View) -> np.ndarray:
    """The cube as the view sees it: its faces translucent, its edges and arrows opaque."""
    corners = coordinates(vertices)                                            # [vertices, 3]
    places = solids.mv.point(corners)
    starts, ends = coordinates(centres), arrow_ends(centres, patches)           # [faces, 3]
    # A crushed face has no area: nothing to draw, and no arrow.
    lengths = np.linalg.norm(ends - starts, axis=-1)
    shown = lengths > view.pixel
    colours = np.array([to_rgb(colour) for colour in COLOURS * 2])            # [faces, 3]
    shafts, heads = solids.arrows(solids.mv.point(starts[shown]), solids.mv.point(ends[shown]),
                                  shaft_radius * view.pixel, head_share * lengths[shown], head_width * head_share * lengths[shown])
    edges, joints = solids.edges(places[EDGES[:, 0]], places[EDGES[:, 1]], edge_radius * view.pixel)
    edge_colour = np.array(to_rgb(EDGE_COLOUR))
    opaque = [(shafts, colours[shown]), (heads, colours[shown]), (edges, edge_colour), (joints, edge_colour)]
    # Each face a plate a pixel thick, in its own colour.
    faces = solids.plates(corners[FACES[shown]], view.pixel)                 # [faces, 6] Quadric
    return solids.draw(opaque, view, background, translucent=[(faces, colours[shown], face_opacity)])


def measures(ax: Axes, patches: core.Patch, volume: core.Scalar) -> None:
    areas = patches[:3].norm().kernel[..., 0]
    values = np.concatenate([areas, volume.kernel])
    ax.bar(np.arange(len(values)), values, color=(*COLOURS, EDGE_COLOUR), width=0.58)
    ax.axhline(1, color="#94a3b8", linestyle=":", linewidth=1)
    # The unit cube's measures, the one value read off; each column named by what it measures.
    ax.set(xticks=np.arange(len(values)), xticklabels=["face $x$", "face $y$", "face $z$", "volume"], yticks=[1],
           ylim=(0, 1.35))
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right"]].set_visible(False)


def comparison(vertices: core.Point, centres: core.Point, patches: core.Patch, volume: core.Scalar) -> plt.Figure:
    """One column per deformation: the cube with its patches above, and below, the areas of the faces
    normal to x, y and z and the volume, against the unit cube's."""
    figure = plt.figure(figsize=(12, 6.5), layout="constrained")
    count = vertices.shape[0]
    seen = view(vertices, centres, patches)
    for i in range(count):
        show(figure.add_subplot(2, count, i + 1), picture(vertices[i], centres[i], patches[i], seen), seen)
        measures(figure.add_subplot(2, count, count + i + 1), patches[i], volume[i])
    return figure


def frame(image: np.ndarray, view: solids.View, patches: core.Patch, volume: core.Scalar) -> np.ndarray:
    figure = plt.figure(figsize=(9, 4.3), dpi=85, layout="constrained")
    show(figure.add_subplot(1, 2, 1), image, view)
    measures(figure.add_subplot(1, 2, 2), patches, volume)
    pixels = capture(figure)
    plt.close(figure)
    return pixels


def animate(frames: Iterable[tuple[core.Point, core.Point, core.Patch, core.Scalar]]) -> list[np.ndarray]:
    """The frames, all from one camera framing every frame's cube."""
    frames = list(frames)
    vertices, centres, patches, _ = (stack(list(parts)) for parts in zip(*frames))   # [frames, ...]
    seen = view(vertices, centres, patches)
    return [frame(picture(vertices, centres, patches, seen), seen, patches, volume)
            for vertices, centres, patches, volume in frames]
