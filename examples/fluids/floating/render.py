"""Drawing for hydrostatic stability: the heeled hull in space with its wet surface by depth and its two
forces; the righting arm curves; the floating cube's map over density and heel, with cubes at rest; and the
released hull rolling back upright or capsizing, beside its righting arm curve."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from examples.fluids.floating import core
from examples.util.wgsl import image_figure, meshes, solids

ga, mv = core.ga, core.mv
# The colours of the hull, of its wet surface by depth, of the water, of the weight and of the buoyancy;
# the colour maps of the loads and of the cube's map.
HULL, DECK, WATER, WEIGHT, BUOYANCY = "#e7e2d6", "#c9bfa8", "#5aa9c9", "#d06c36", "#1f6f99"
WET_MAP, LOAD_MAP, ARM_MAP = "Blues", "viridis", "RdBu"
# The depth at which the hull's wet surface is darkest, the reach of the water drawn around it, the drawn
# length of the forces, and the heels over which each curve's tangent is drawn.
DEEPEST = 0.6
SEA = (3.0, 1.8)
FORCE = 0.9
TANGENT = 25.0
# The view in space, elevation and azimuth in radians, azimuth zero looking along +y, its margin as a share of
# its larger side, the pixels across its picture, and how much of what lies behind it each face of the water hides,
# so the wet hull shows through it.
ELEVATION, AZIMUTH = np.radians([18.0, 35.0])
MARGIN = 0.04
N_PIXELS = 1400
WATER_OPACITY = 0.13
# The glint on the hull, faint, so its facets do not glitter.
HULL_GLOSS = 0.1
BACKGROUND = np.ones(3)
# The windows of the end views of the rolling hull and of the hull drawn whole, across and up, and the size
# of the cube drawn above the map.
END = ((-1.6, 1.6), (-1.0, 1.6))
SHIP_END = ((-1.1, 1.1), (-0.8, 1.0))
CUBE_VIEW = 0.95
# The end views: seen along the axis of heel from its +x side, y across and z up, as an azimuth; the pixels across
# the rolling hull's views and the cubes'; how much of what lies behind it the water's near face hides; and in pixels,
# the length of the force lines' dashes, the radii of the force lines, the arrows' shafts and heads, the arm, the
# waterline, the centre of gravity and the marker of each load, and the arrows' heads' length.
END_ON = np.radians(90.0)
N_END_PIXELS, N_CUBE_PIXELS = 400, 260
END_WATER_OPACITY = 0.3
DASH = 6.0
LINE_RADIUS, SHAFT_RADIUS, HEAD_RADIUS, ARM_RADIUS, WATERLINE_RADIUS, CENTRE_RADIUS, MARKER_RADIUS = (
    0.6, 1.6, 6.0, 1.6, 0.7, 4.0, 5.0)
HEAD_LENGTH = 14.0
# The resolution of the plots beside the end views, in dots per inch.
DPI = 90


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of points, on the last axis."""
    kernel = points.cast(ga.subspace("ywz wxz xwy xyz")).kernel
    return kernel[..., :3] / kernel[..., 3:]


def degrees(rolls: core.Twist) -> np.ndarray:
    """The angle of turns about the axis of heel, in degrees."""
    return np.degrees(rolls.cast(ga.subspace("yz")).kernel[..., 0])


def arms(couples: core.Forque, weight: core.Scalar) -> np.ndarray:
    """The righting arm of each couple: its turn back about the axis of heel, over the weight."""
    return scalars(-(couples & core.heel_axis) / weight)


def end_view(window: tuple[tuple[float, float], tuple[float, float]], pixels: int) -> solids.View:
    """The view along the axis of heel of a window across and up, `pixels` across."""
    (left, right), (bottom, top) = window
    middle = solids.mv.point([0.0, (left + right) / 2, (bottom + top) / 2])
    return solids.View(solids.aim(middle, 0.0, END_ON), np.array([right - left, top - bottom]) / 2, pixels)


def middle(view: solids.View) -> np.ndarray:
    """The middle of an end view, across and up `[2]`: where its camera stands."""
    return coordinates(view.camera >> solids.origin)[1:]


def ahead(triangles: np.ndarray, view: solids.View) -> float:
    """How far along the axis of heel to draw in front of a body `[triangles, 3, 3]` and of the water around it."""
    return np.abs(triangles[..., 0]).max() + 2 * view.pixel


def on_end(places: np.ndarray, front: float) -> solids.Point:
    """Points at places across and up an end view `[..., 2]`, `front` along the axis of heel."""
    return solids.mv.point(np.concatenate([np.full(places.shape[:-1] + (1,), front), places], axis=-1))


def forces_end_on(view: solids.View, placed: core.Motor, push: core.Forque, centre: core.Point,
                  front: float) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The weight from the centre of gravity and the buoyancy from the water surface, as arrows on their dashed
    vertical lines across an end view, and the righting arm between them at the centre's height; `front` along the
    axis of heel, in front of the body."""
    gravity = coordinates(placed >> centre)[1:]
    buoyant = coordinates((placed >> push) ^ core.water)[1:]
    bottom, top = middle(view)[1] + np.array([-1, 1]) * view.half[1]
    # Each force line's dashes, up the whole view.
    marks = np.arange(bottom, top, 2 * DASH * view.pixel)                     # [dashes]
    parts = []
    for start, colour, sense in ((buoyant, BUOYANCY, 1), (gravity, WEIGHT, -1)):
        dashes = np.stack([np.full_like(marks, start[0]), marks], axis=-1)    # [dashes, 2]
        lines = solids.rods(on_end(dashes, front), on_end(dashes + [0.0, DASH * view.pixel], front), LINE_RADIUS * view.pixel)
        shaft, head = solids.arrows(on_end(start, front), on_end(start + [0.0, sense * FORCE * 0.6], front),
                                    SHAFT_RADIUS * view.pixel, HEAD_LENGTH * view.pixel, HEAD_RADIUS * view.pixel)
        parts += [(lines, to_rgb(colour)), (shaft[None], to_rgb(colour)), (head[None], to_rgb(colour))]
    arm = solids.rods(on_end(gravity, front), on_end(np.array([buoyant[0], gravity[1]]), front), ARM_RADIUS * view.pixel)
    centre_ball = solids.ball(on_end(gravity, front), CENTRE_RADIUS * view.pixel)
    return parts + [(arm[None], (0.1, 0.1, 0.1)), (centre_ball[None, None], to_rgb(WEIGHT))]


def end_on(triangles: np.ndarray, view: solids.View, parts: list[tuple[solids.Quadric, np.ndarray]]) -> np.ndarray:
    """A body seen along its axis of heel: its triangles `[triangles, 3, 3]`, the water over its lower part with its
    waterline, and parts in front of it, `[rows, columns, 3]` bytes."""
    corners = solids.mv.point(triangles.reshape(-1, 3))
    faces = np.arange(len(corners)).reshape(-1, 3)
    body = meshes.samples(corners, faces, np.array(to_rgb(HULL)), meshes.flat(corners, faces), view, gloss=HULL_GLOSS)
    # The water: a box up to the surface, past the view's sides and bottom, and as long along the axis as the body.
    across, up = view.half + view.pixel
    centre = middle(view)                                                       # [2]
    depth = up - centre[1]
    reach = np.abs(triangles[..., 0]).max() + view.pixel
    water = solids.boxes(solids.translator(on_end(np.array([centre[0], -depth / 2]), 0.0) - solids.origin),
                         np.array([reach, across, depth / 2]))
    front = ahead(triangles, view)
    waterline = solids.rods(on_end(np.array([centre[0] - across, 0.0]), front),
                            on_end(np.array([centre[0] + across, 0.0]), front), WATERLINE_RADIUS * view.pixel)
    return solids.draw(parts + [(waterline[None], to_rgb(WATER))], view, BACKGROUND,
                       translucent=[(water[None], np.array(to_rgb(WATER)), END_WATER_OPACITY)], meshes=[(body, 1.0)])


def side_by_side(images: list[np.ndarray]) -> np.ndarray:
    """Pictures `[rows, columns, 3]` bytes in a row, the shorter ones padded below with the background."""
    rows = max(image.shape[0] for image in images)
    return np.concatenate([np.pad(image, ((0, rows - image.shape[0]), (0, 0), (0, 0)), constant_values=255)
                           for image in images], axis=1)


def in_water(triangles: np.ndarray) -> np.ndarray:
    """The hull's triangles `[triangles, 3, 3]` in space, flat, the dry ones pale and the wet ones darker with
    depth, under a translucent plate of the water's surface, framed on the plate."""
    depth = np.clip(-triangles[..., 2].mean(axis=-1) / DEEPEST, 0, 1)        # [triangles]
    wet = plt.get_cmap(WET_MAP)(0.35 + 0.6 * depth)[:, :3]
    colours = np.where((depth > 0)[:, None], wet, to_rgb(HULL))              # [triangles, 3]
    reach, breadth = SEA
    sea = np.array([[-reach, -breadth, 0], [reach, -breadth, 0], [reach, breadth, 0], [-reach, breadth, 0]])
    corners = solids.mv.point(triangles.reshape(-1, 3))
    faces = np.arange(len(corners)).reshape(-1, 3)
    view = solids.view([solids.mv.point(sea), corners], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    normals = meshes.flat(corners, faces)
    hull = meshes.samples(corners, faces, np.zeros(3), normals, view, face_values=colours, colour=meshes.by_face,
                          gloss=HULL_GLOSS)
    # The water's surface, a plate a pixel thick.
    water = solids.plates(sea[None], view.pixel)                              # [1, 6] Quadric
    return solids.draw([], view, BACKGROUND, translucent=[(water, np.array(to_rgb(WATER)), WATER_OPACITY)],
                       meshes=[(hull, 1.0)])


def draw_ship(hull: core.Point, placed: core.Motor, push: core.Forque, centre: core.Point) -> plt.Figure:
    """The heeled hull: in space, its dry surface pale and its wet surface darker with depth, in the water;
    and seen along its axis of heel, with its weight, its buoyancy and the righting arm between them."""
    triangles = coordinates(placed >> hull)                                  # [triangles, 3, 3]
    in_space = in_water(triangles)
    (left, right), (bottom, top) = SHIP_END
    view = end_view(SHIP_END, round(in_space.shape[0] * (right - left) / (top - bottom)))
    forces = forces_end_on(view, placed, push, centre, ahead(triangles, view))
    return image_figure(side_by_side([in_space, end_on(triangles, view, forces)]))


def draw_curves(heels: np.ndarray, couples: core.Forque, weight: core.Scalar, stiffness: core.Stiffness) -> plt.Figure:
    """The righting arm of each load over the heels `[heels, loads]`, with the tangent at upright that the
    stiffness gives, and the line of no righting."""
    righting = arms(couples, weight)                                         # [heels, loads]
    slopes = scalars(stiffness(mv.yz, mv.yz) / weight)                       # [loads]
    colours = plt.get_cmap(LOAD_MAP)(np.linspace(0.05, 0.85, righting.shape[1]))
    figure, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    ax.axhline(0, color="0.6", lw=0.8)
    near = heels[heels <= TANGENT]
    for curve, slope, colour in zip(righting.T, slopes, colours):
        ax.plot(heels, curve, color=colour, lw=2)
        ax.plot(near, slope * np.radians(near), color=colour, lw=1, linestyle=(0, (4, 3)))
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(heels[0], heels[-1])
    ax.set(xlabel="heel (degrees)", ylabel="righting arm (m)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def draw_cubes(densities: np.ndarray, heels: np.ndarray, couples: core.Forque, fluid_weight: float, cube: core.Point,
               shown: core.Motor, shown_densities: list[float], rests: np.ndarray) -> plt.Figure:
    """Above, cubes of a few densities at rest; below, the cube's righting couple over density and heel, blue
    turning it back upright and red over, its rests where it turns from over to back drawn thick, its
    balances the other way thin, and the cubes above marked on them."""
    moments = scalars(-(couples & core.heel_axis) / fluid_weight)            # [densities, heels]
    # A rest is stable where the couple turns from pushing over to pushing back as the heel grows.
    rising = np.gradient(moments, heels, axis=1) > 0
    cube_view = end_view(((-CUBE_VIEW, CUBE_VIEW), (-CUBE_VIEW, CUBE_VIEW)), N_CUBE_PIXELS)
    cubes = side_by_side([end_on(coordinates(placed >> cube), cube_view, []) for placed in shown])
    figure, ax = plt.subplots(figsize=(cubes.shape[1] / DPI, 0.6 * cubes.shape[1] / DPI), dpi=DPI)
    largest = np.abs(moments).max()
    ax.imshow(moments.T, origin="lower", aspect="auto", cmap=ARM_MAP, vmin=-largest, vmax=largest, interpolation="bilinear",
              extent=(densities[0], densities[-1], heels[0], heels[-1]))
    for stable, width in ((True, 3), (False, 0.8)):
        ax.contour(densities, heels, np.where(rising == stable, moments, np.nan).T, levels=[0], colors="0.1",
                   linewidths=width)
    ax.plot(shown_densities, rests, "o", color="white", markeredgecolor="0.1", markersize=7)
    ax.set(xlim=(densities[0], densities[-1]), ylim=(-3, 93), yticks=[0, 45, 90])
    ax.set(xlabel="density over the water's", ylabel="heel (degrees)")
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)
    plot = capture(figure)
    plt.close(figure)
    return image_figure(np.concatenate([cubes, plot[:, :cubes.shape[1]]], axis=0))


def animate_release(hull: core.Point, centres: core.Point, frames: Iterable[tuple[core.Twist, core.Motor, core.Forque, core.Forque]],
                    heels: np.ndarray, couples: core.Forque, weight: core.Scalar) -> Iterator[np.ndarray]:
    """Each frame: the released hull with each load seen along its axis of heel, its weight from its centre of
    gravity and its buoyancy from the water surface, as arrows on their lines, beside the righting arm curves
    of the loads, each with the hull's heel so far."""
    curves = arms(couples, weight)                                           # [heels, loads]
    # Past half a turn the righting arm repeats reversed: heeled the other way round, it pushes the other way.
    beyond = 360 - heels[::-1][1:]
    extended, extended_curves = np.concatenate([heels, beyond]), np.concatenate([curves, -curves[::-1][1:]])
    colours = plt.get_cmap(LOAD_MAP)(np.array([0.3, 0.85]))
    view = end_view(END, N_END_PIXELS)
    rows = view.shape[0]
    marker = np.array([0.9 * END[0][0], 0.85 * END[1][1]])
    # The curves set up once; each load's heel so far and its heel now move.
    figure, ax = plt.subplots(figsize=(1.4 * rows / DPI, rows / DPI), dpi=DPI)
    ax.axhline(0, color="0.6", lw=0.8)
    so_far, now = [], []
    for case, colour in enumerate(colours):
        ax.plot(extended, extended_curves[:, case], color=colour, lw=1.6, alpha=0.5)
        so_far += ax.plot([], [], color=colour, lw=2)
        now += ax.plot([], [], "o", color=colour, markersize=7)
    ax.set_xticks([0, 90, 180])
    ax.set_xlim(-30, 215)
    ax.set(xlabel="heel (degrees)", ylabel="righting arm (m)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.97)
    states = []
    for roll, placed, push, weighing in frames:
        states.append([degrees(roll), arms(push + weighing, weight)])
        trail = np.array(states)                                             # [frames, 2, loads]
        views = []
        for case, colour in enumerate(colours):
            triangles = coordinates(placed[case] >> hull)                        # [triangles, 3, 3]
            front = ahead(triangles, view)
            # Which load this is, a dot in the corner in its curve's colour.
            dot = solids.ball(on_end(marker, front), MARKER_RADIUS * view.pixel)
            views.append(end_on(triangles, view, forces_end_on(view, placed[case], push[case], centres[case], front)
                                + [(dot[None, None], colour[:3])]))
            so_far[case].set_data(*trail[:, :, case].T)
            now[case].set_data(*trail[-1, :, case, None])
        yield side_by_side(views + [capture(figure)])
    plt.close(figure)
