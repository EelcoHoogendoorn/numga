"""Drawing for the earthquake focal mechanisms: beachballs in the equal-area projection of the lower
hemisphere, and the fault in space with the lobes of its P waves, on the GPU: the lobes a lit mesh, the fault a
translucent disk, and the slip arrows."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb, to_rgba
from matplotlib.image import AxesImage

from numga import concatenate, stack, where
from examples.util.animation import capture
from examples.mechanics.earthquake import core
from examples.util.wgsl import compose, contours, fitted, meshes, show, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel

# Pushes shaded, pulls left white; the nodal lines, the pressure and tension axes, the slip.
PUSH, PULL = "#1e293b", "#f8fafc"
NODAL, PRESSURE, TENSION, SLIP = "#0f172a", "#0ea5e9", "#e11d48", "#f59e0b"
# In pixels: the width of the beachball's nodal lines and of its rim, and the radius of its axes' marks, with their
# white rims' width.
nodal_width, ball_rim_width = 1.1, 1.2
axis_radius, axis_rim = 4.0, 0.9
# The lobes in space, pushing and pulling; the fault patch and its rim.
PUSH_LOBE, PULL_LOBE = "#475569", "#e2e8f0"
FAULT, RIM = "#fde68a", "#b45309"
# The radius of the fault patch, the opacity of each of its faces, front and back alike, and its thickness in pixels;
# the slip arrows' offset from the fault, their reach behind its middle and their length.
patch = 0.55
patch_opacity, patch_thickness = 0.16, 1.0
offset, behind, slip_length = 0.15, 0.45, 0.9
# The view in space, elevation and azimuth in radians, azimuth zero looking along +y, and the margin as a share of
# the larger side.
elevation, azimuth = np.radians([24.0, -60.0 + 90.0])
margin = 0.12
# The width of the rim, and the slip arrows' width with the length of the strokes of the chevrons at their tips, in
# pixels; the samples around the beachball's rim.
rim_width, arrow_width, chevron_length = 2.2, 4.0, 16.0
n_rim = 199
# Space is drawn east, north and up: the half turn about the level line halfway between north and east swaps the two
# and turns down up.
drawing = (core.down.commutator(core.north - core.east).normalized() * (np.pi / 2)).exp()   # [] Rotor+


# --- plumbing -------------------------------------------------------------------------
def lambert(directions: core.Point) -> core.Point:
    """Directions in the equal-area projection of the lower hemisphere onto the level ground: a direction on the
    upper hemisphere is drawn as its opposite, and each keeps its level part, shrunk by the square root of one plus
    how far down it points."""
    # The level ground faces up, so its reading of a direction is how far up the direction points.
    lower = where((core.ground & directions) > 0, -directions, directions)    # [...] Point-
    depth = -(core.ground & lower)                                            # [...] Scalar+
    return (lower - core.down * depth) / (1 + depth).square_root()            # [...] Point-


def ball_grid(directions: core.Point) -> tuple[solids.Point, contours.Topology]:
    """The lower hemisphere's directions `[rings, spokes]` on the level ground of the drawing's frame, numbered ring by
    ring, and the grid between them."""
    places = drawing >> lambert(directions).reshape(-1)                      # [rings * spokes] Point-
    return places, contours.Topology.grid(*directions.shape)


def shade_ball(places: core.Point, grid: contours.Topology, push: core.Scalar, view: Camera) -> np.ndarray:
    """The lower hemisphere at its places on the grid `ball_grid` gives, with the pushes `[rings, spokes]` shaded and
    its nodal lines, where the pushes change sign, seen by the view of its panel: `[rows, columns, 4]`, transparent
    beyond the rim."""
    values = push.to_array().reshape(-1)
    bound = 2 * np.abs(values).max()
    # Below zero the pulls' colour, from zero on the pushes'.
    sides = contours.Bands(-bound, bound, np.array([to_rgba(PULL), to_rgba(PUSH)]))
    nodal = contours.Level(0.0, np.array(to_rgba(NODAL)), nodal_width)
    return contours.picture(places, grid, values, np.array(to_rgb(PULL)), view, levels=(nodal,), bands=(sides,))


def frame_ball(ax) -> None:
    """The panel of one fault's beachball, north up."""
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect("equal")
    ax.set_axis_off()


def ball_pictures(places: core.Point, grid: contours.Topology, push: core.Scalar, axes: core.Point,
                  view: Camera) -> list[np.ndarray]:
    """One fault's beachball, shaded as `shade_ball` shades it; the rim of the lower hemisphere over its nodal lines;
    and its pressure and tension axes."""
    # The rim: the level directions, which the projection leaves where they are, turned from north toward east.
    level = (core.horizontal * (-np.linspace(0, 2 * np.pi, n_rim, endpoint=False) / 2)).exp() >> core.north   # [rim] Point-
    principal = drawing >> lambert(axes)[[0, 2]]                              # [2] Point-
    return [shade_ball(places, grid, push, view),
            simplices.strokes(drawing >> level, simplices.Topology.loops([n_rim]), np.array(to_rgb(NODAL)),
                              ball_rim_width, view),
            simplices.markers(principal, np.ones(3), axis_radius + axis_rim, view),
            simplices.markers(principal, np.array([to_rgb(PRESSURE), to_rgb(TENSION)]), axis_radius, view)]


def pin_ball(ax, places: core.Point, grid: contours.Topology, push: core.Scalar,
             axes: core.Point) -> tuple[AxesImage, Camera]:
    """The beachball pinned in its panel, once the figure is laid out; the image and its camera, for an animation to
    set its data."""
    view = flat_panel(ax, core.ga)
    return simplices.overlay(ax, ball_pictures(places, grid, push, axes, view)), view


def draw_mechanisms(directions: core.Point, push: core.Scalar, axes: core.Point) -> plt.Figure:
    """The beachball of each fault `[faults]`, side by side."""
    count = push.shape[0]
    figure, panels = plt.subplots(1, count, figsize=(3.2 * count, 3.2), dpi=55)
    for ax in panels:
        frame_ball(ax)
    figure.tight_layout()
    places, grid = ball_grid(directions)
    for index, ax in enumerate(panels):
        pin_ball(ax, places, grid, push[index], axes[index])
    return figure


def lobes(directions: core.Point, push: core.Scalar) -> core.Point:
    """The lobes of the P waves in the drawing's frame `[rings, spokes]`: each direction drawn as far as its first
    motion."""
    return drawing >> (directions * push.abs())


def slip_arrows(normal: core.Point, slip: core.Point) -> tuple[core.Point, core.Point]:
    """Where the slip of each side of the fault is drawn from and to in the drawing's frame `[..., sides]`: just off
    the fault on the side's face, along the side's slip."""
    sides = np.array([1.0, -1.0])                                             # [sides]
    normal, slip = drawing >> normal, drawing >> slip                         # [...] Point-, Point-
    starts = (normal * offset - slip * behind) * sides                        # [..., sides] Point-
    return starts, starts + slip * (slip_length * sides)


def view(surfaces: core.Point, ends: core.Point) -> solids.View:
    """A view framing lobes and arrows' ends, in the drawing's frame."""
    # The projective embedding: places of space, offsets from the source, as the renderer's points about its origin.
    framed = [solids.origin + solids.context.include_from_algebra(tips) for tips in (surfaces, ends)]
    return solids.view(framed, 0.0, elevation, azimuth, margin, 1)


def draw_fault(directions: core.Point, push: core.Scalar, normal: core.Point, slip: core.Point,
               view: solids.View) -> np.ndarray:
    """One fault in space: the patch of the fault, the slip of each side across it, and the lobes of
    the P waves `[rings, spokes]`, each direction drawn as far as its first motion, pushes dark and pulls pale."""
    surface = lobes(directions, push).reshape(-1)                            # [rings * spokes] Point-
    # Each cell of the grid two triangles, by the indices of its corners.
    index = np.arange(np.prod(push.shape)).reshape(push.shape)               # [rings, spokes]
    cells = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)   # [rings - 1, spokes - 1, 4]
    # The colour of each cell by the sign at its middle: the nodal planes run along the grid lines.
    values = push.to_array()
    middle = values[1:, 1:] + values[:-1, :-1] + values[1:, :-1] + values[:-1, 1:]
    colours = np.where(middle[..., None] > 0, to_rgb(PUSH_LOBE), to_rgb(PULL_LOBE))   # [rings - 1, spokes - 1, 3]
    triangles = cells[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)             # [cells * 2, 3]
    colours = np.repeat(colours.reshape(-1, 1, 3), 2, axis=0)                 # [cells * 2, 1, 3]
    # The projective embedding: places of space, offsets from the source, as the renderer's points about its origin.
    places = solids.origin + solids.context.include_from_algebra(surface)    # [rings * spokes] Point-
    # At the grid's poles a ring of directions gathers into one point, and along the nodal planes the lobes shrink
    # into the source: the halves of the cells there have no area, the join of their corners no weight, and are
    # left out.
    corners = places[triangles]                                               # [cells * 2, 3] Point-
    spanned = (corners[:, 0] & corners[:, 1] & corners[:, 2]).norm() > 0
    lobe_samples = meshes.samples(places, triangles[spanned], colours[spanned], meshes.smooth(places, meshes.Topology(len(places), triangles[spanned])),
                                  view, face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    # The patch: a thin disc about the source, square to the normal, reaching half the rim's width past the patch so
    # that its rim, inked where the cylinder of the patch's radius about the normal meets it, lies whole on it.
    across = solids.context.include_from_algebra(drawing >> normal)                     # [] Point-
    faces = solids.origin + stack([-across, across]) * (patch_thickness / 2 * view.pixel)  # [2] Point-
    disc = solids.rods(faces[0], faces[1], patch + rim_width / 2 * view.pixel)[None]   # [1, 3] Plane+ <- Point-
    rim = solids.Ink(0, solids.rods(faces[0], faces[1], patch)[:1], solids.everywhere[None], np.array(to_rgb(RIM)),
                     rim_width, 1.0)
    fault = solids.layer([], view, [rim], translucent=[(disc, np.array(to_rgb(FAULT)), patch_opacity)],
                         meshes=[(lobe_samples, 1.0)])
    # Each slip arrow a tube ending in a chevron.
    starts, ends = (solids.origin + solids.context.include_from_algebra(points) for points in slip_arrows(normal, slip))
    n_sides = starts.shape[0]
    arrows = simplices.tubes(stack((starts, ends), axis=-1).reshape(-1), simplices.Topology.chains([2] * n_sides),
                             np.array(to_rgb(SLIP)), arrow_width, view, gloss=solids.gloss, chevrons=np.ones(n_sides),
                             chevron=chevron_length)
    arrow_colour, arrow_depth = solids.layer([], view, inks=[], translucent=[], meshes=[(arrows, 1.0)])
    # The slip arrows lie inside the lobes, which meet the fault at the source: drawn over the fault and its lobes,
    # in front.
    return compose([(arrow_colour, np.full_like(arrow_depth, -np.inf), 1.0), (*fault, 1.0)], np.ones(3))


def animate_sweep(hemisphere: core.Point, frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Scalar,
                                                                     core.Point, core.Scalar]]) -> Iterator[np.ndarray]:
    """Each frame of the rake's turn: the fault in space with its P lobes, and its beachball over the
    lower hemisphere; all from one camera framing every frame's lobes and arrows."""
    frames = list(frames)
    normals, slips, _, _, spheres, pushes = (stack(list(parts)) for parts in zip(*frames))   # [frames, ...] each
    # The figure set up once; the fault's picture and the beachball set for each frame.
    figure = plt.figure(figsize=(8.4, 4.2), dpi=50)
    fault_ax, ax = figure.add_subplot(1, 2, 1), figure.add_subplot(1, 2, 2)
    frame_ball(ax)
    # The figure laid out with the fault's axes bare, as its picture shows them, and the view fitted to them.
    fault_ax.set_axis_off()
    figure.tight_layout()
    seen = fitted(view(lobes(spheres, pushes), concatenate(slip_arrows(normals[:, None], slips[:, None]))), fault_ax)
    image = show(fault_ax, np.ones((*seen.shape, 3)), seen)
    places, grid = ball_grid(hemisphere)
    shading, ball_view = pin_ball(ax, places, grid, frames[0][3], frames[0][2])
    for normal, slip, axes, ball, sphere, push in frames:
        image.set_data(draw_fault(sphere, push, normal, slip, seen))
        shading.set_data(simplices.composited(ball_pictures(places, grid, ball, axes, ball_view)))
        yield capture(figure)
    plt.close(figure)
