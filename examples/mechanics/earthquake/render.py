"""Drawing for the earthquake focal mechanisms: beachballs in the equal-area projection of the lower
hemisphere, and the fault in space with the lobes of its P waves, on the GPU: the lobes a lit mesh, the fault a
translucent disk, and the slip arrows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import to_rgb
from matplotlib.contour import ContourSet

from numga import stack
from examples.util.animation import capture
from examples.mechanics.earthquake import core
from examples.util.wgsl import compose, fitted, meshes, show, solids

# Pushes shaded, pulls left white; the nodal lines, the pressure and tension axes, the slip.
PUSH, PULL = "#1e293b", "#f8fafc"
NODAL, PRESSURE, TENSION, SLIP = "#0f172a", "#0ea5e9", "#e11d48", "#f59e0b"
# The lobes in space, pushing and pulling; the fault patch and its rim.
PUSH_LOBE, PULL_LOBE = "#475569", "#e2e8f0"
FAULT, RIM = "#fde68a", "#b45309"
# The radius of the fault patch, the opacity of each of its faces, front and back alike, and its thickness in pixels;
# the slip arrows' offset from the fault, their reach behind its middle and their length.
PATCH = 0.55
PATCH_OPACITY, PATCH_THICKNESS = 0.16, 1.0
OFFSET, BEHIND, SLIP_LENGTH = 0.15, 0.45, 0.9
# The view in space, elevation and azimuth in radians, azimuth zero looking along +y, and the margin as a share of
# the larger side.
ELEVATION, AZIMUTH = np.radians([24.0, -60.0 + 90.0])
MARGIN = 0.12
# The width of the rim and the radius of the slip arrows' shafts, in pixels; an arrow's head, its length a share of
# the arrow's, and its radius a share of its own length.
RIM_WIDTH, SHAFT_RADIUS = 2.2, 2.0
HEAD_SHARE = 0.25
HEAD_WIDTH = 0.4


def coordinates(directions: core.Point) -> np.ndarray:
    """The north, east and down coordinates of directions."""
    return directions.cast(core.ga.subspace("yz zx xy")).kernel


def lambert(directions: core.Point) -> np.ndarray:
    """Directions in the equal-area projection of the lower hemisphere, east to the right and north
    up: a direction on the upper hemisphere is drawn as its opposite."""
    ned = coordinates(directions)
    ned = np.where(ned[..., 2:] < 0, -ned, ned)
    return ned[..., [1, 0]] / np.sqrt(1 + ned[..., 2:])


def upright(directions: core.Point) -> np.ndarray:
    """Directions in space for drawing: east, north and up."""
    ned = coordinates(directions)
    return np.stack([ned[..., 1], ned[..., 0], -ned[..., 2]], axis=-1)


def shade_ball(ax, directions: core.Point, push: core.Scalar) -> list[ContourSet]:
    """The lower hemisphere `[rings, spokes]` with the pushes shaded, and its nodal lines, for an animation to
    replace."""
    grid, push = lambert(directions), push.to_array()
    bound = 2 * np.abs(push).max()
    return [ax.contourf(*np.moveaxis(grid, -1, 0), push, levels=[-bound, 0, bound], colors=[PULL, PUSH]),
            ax.contour(*np.moveaxis(grid, -1, 0), push, levels=[0], colors=NODAL, linewidths=1.4)]


def draw_ball(ax, directions: core.Point, push: core.Scalar,
              axes: core.Point) -> tuple[list[ContourSet], list[PathCollection]]:
    """The beachball of one fault: the lower hemisphere `[rings, spokes]` with the pushes shaded, its
    nodal lines, and its pressure and tension axes, north up. The shading and the axes' marks, for an animation to
    replace and move."""
    shading = shade_ball(ax, directions, push)
    angle = np.linspace(0, 2 * np.pi, 200)
    # The rim over the nodal lines, also over those an animation draws after it.
    ax.plot(np.cos(angle), np.sin(angle), color=NODAL, linewidth=1.6, zorder=2.5)
    principal = lambert(axes)
    marks = [ax.scatter(*point, s=110, color=colour, edgecolor="white", linewidth=1.2, zorder=5)
             for point, colour in [(principal[0], PRESSURE), (principal[2], TENSION)]]
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return shading, marks


def draw_mechanisms(directions: core.Point, push: core.Scalar, axes: core.Point) -> plt.Figure:
    """The beachball of each fault `[faults]`, side by side."""
    count = push.shape[0]
    figure, panels = plt.subplots(1, count, figsize=(3.2 * count, 3.2), dpi=55)
    for index, ax in enumerate(panels):
        draw_ball(ax, directions, push[index], axes[index])
    figure.tight_layout()
    return figure


def lobes(directions: core.Point, push: core.Scalar) -> np.ndarray:
    """The lobes of the P waves in space `[rings, spokes, 3]`: each direction drawn as far as its first motion."""
    return upright(directions) * np.abs(push.to_array())[..., None]


def slip_arrows(normal: core.Point, slip: core.Point) -> tuple[np.ndarray, np.ndarray]:
    """Where the slip of each side of the fault is drawn from and to `[sides, 3]`: just off the fault on the side's
    face, along the side's slip."""
    sides = np.array([1.0, -1.0])[:, None]                                   # [sides, 1]
    starts = sides * (OFFSET * upright(normal) - BEHIND * upright(slip))
    return starts, starts + sides * SLIP_LENGTH * upright(slip)


def view(surfaces: np.ndarray, ends: np.ndarray) -> solids.View:
    """A view framing lobes `[..., 3]` and arrows' ends `[..., 3]`."""
    return solids.view([solids.mv.point(surfaces), solids.mv.point(ends)], 0.0, ELEVATION, AZIMUTH, MARGIN, 1)


def draw_fault(directions: core.Point, push: core.Scalar, normal: core.Point, slip: core.Point,
               view: solids.View) -> np.ndarray:
    """One fault in space: the patch of the fault, the slip of each side across it, and the lobes of
    the P waves `[rings, spokes]`, each direction drawn as far as its first motion, pushes dark and pulls pale."""
    surface = lobes(directions, push)                                         # [rings, spokes, 3]
    # Each cell of the grid two triangles, by the indices of its corners.
    index = np.arange(surface[..., 0].size).reshape(surface.shape[:-1])       # [rings, spokes]
    cells = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)   # [rings - 1, spokes - 1, 4]
    # The colour of each cell by the sign at its middle: the nodal planes run along the grid lines.
    values = push.to_array()
    middle = values[1:, 1:] + values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:]
    colours = np.where(middle[..., None] > 0, to_rgb(PUSH_LOBE), to_rgb(PULL_LOBE))   # [rings - 1, spokes - 1, 3]
    triangles = cells[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)             # [cells * 2, 3]
    colours = np.repeat(colours.reshape(-1, 1, 3), 2, axis=0)                 # [cells * 2, 1, 3]
    # At the grid's poles a ring of directions gathers into one point: the halves of the cells there have no
    # area and no normal, and are left out.
    corners = surface.reshape(-1, 3)[triangles]                               # [cells * 2, 3, 3]
    spanned = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=-1) > 0
    places = solids.mv.point(surface.reshape(-1, 3))
    lobe_samples = meshes.samples(places, triangles[spanned], colours[spanned], meshes.smooth(places, triangles[spanned]),
                                  view, face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    # The patch: a thin disc about the source, square to the normal, reaching half the rim's width past the patch so
    # that its rim, inked where the cylinder of the patch's radius about the normal meets it, lies whole on it.
    across = PATCH_THICKNESS / 2 * view.pixel * upright(normal)              # [3]
    faces = solids.mv.point(np.stack([-across, across]))                     # [2] Point-
    disc = solids.rods(faces[0], faces[1], PATCH + RIM_WIDTH / 2 * view.pixel)[None]   # [1, 3] Plane+ <- Point-
    rim = solids.Ink(0, solids.rods(faces[0], faces[1], PATCH)[:1], solids.everywhere[None], np.array(to_rgb(RIM)),
                     RIM_WIDTH, 1.0)
    fault = solids.layer([], view, [rim], translucent=[(disc, np.array(to_rgb(FAULT)), PATCH_OPACITY)],
                         meshes=[(lobe_samples, 1.0)])
    starts, ends = slip_arrows(normal, slip)
    shafts, heads = solids.arrows(solids.mv.point(starts), solids.mv.point(ends), SHAFT_RADIUS * view.pixel,
                                  HEAD_SHARE * SLIP_LENGTH, HEAD_WIDTH * HEAD_SHARE * SLIP_LENGTH)
    arrow_colour, arrow_depth = solids.layer([(shafts, np.array(to_rgb(SLIP))), (heads, np.array(to_rgb(SLIP)))], view,
                                             inks=[], translucent=[], meshes=[])
    # The slip arrows lie inside the lobes, which meet the fault at the source: drawn over the fault and its lobes,
    # in front.
    return compose([(arrow_colour, np.full_like(arrow_depth, -np.inf), 1.0), (*fault, 1.0)], np.ones(3))


def animate_sweep(hemisphere: core.Point, frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Scalar,
                                                                     core.Point, core.Scalar]]) -> Iterator[np.ndarray]:
    """Each frame of the rake's turn: the fault in space with its P lobes, and its beachball over the
    lower hemisphere; all from one camera framing every frame's lobes and arrows."""
    frames = list(frames)
    normals, slips, _, _, spheres, pushes = (stack(list(parts)) for parts in zip(*frames))   # [frames, ...] each
    # The figure and the beachball's rim set up once; the fault's picture, the shading and the axes' marks set for
    # each frame.
    figure = plt.figure(figsize=(8.4, 4.2), dpi=50)
    fault_ax, ax = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    shading, marks = draw_ball(ax, hemisphere, frames[0][3], frames[0][2])
    # The figure laid out with the fault's axes bare, as its picture shows them, and the view fitted to them.
    fault_ax.set_axis_off()
    figure.tight_layout()
    seen = fitted(view(lobes(spheres, pushes), np.concatenate(slip_arrows(normals[:, None], slips[:, None]))), fault_ax)
    image = show(fault_ax, np.ones((*seen.shape, 3)), seen)
    for normal, slip, axes, ball, sphere, push in frames:
        image.set_data(draw_fault(sphere, push, normal, slip, seen))
        for contours in shading:
            contours.remove()
        shading = shade_ball(ax, hemisphere, ball)
        principal = lambert(axes)
        for mark, point in zip(marks, principal[[0, 2]]):
            mark.set_offsets(point[None])
        yield capture(figure)
    plt.close(figure)
