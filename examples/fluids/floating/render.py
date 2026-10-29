"""Drawing for hydrostatic stability: the heeled hull in space with its wet surface by depth and its two
forces; the righting arm curves; the floating cube's map over density and heel, with cubes at rest; and the
released hull rolling back upright or capsizing, beside its righting arm curve."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util import charts
from examples.util.animation import capture
from examples.fluids.floating import core
from examples.util.wgsl import image_figure, meshes, solids

mv = core.mv
# The colours of the hull, of its wet surface by depth, of the water, of the weight and of the buoyancy;
# the colour maps of the loads and of the cube's map.
hull_colour, deck_colour, water_colour, weight_colour, buoyancy_colour = (
    "#e7e2d6", "#c9bfa8", "#5aa9c9", "#d06c36", "#1f6f99")
wet_map, load_map, arm_map = "Blues", "viridis", "RdBu"
# The depth at which the hull's wet surface is darkest, the half length and half breadth of the water drawn around
# it, the drawn length of the forces, and the heels over which each curve's tangent is drawn.
deepest = 0.6
sea = (3.0, 1.8)
force = 0.9
tangent = 25.0
# The view in space, elevation and azimuth in radians, azimuth zero looking along +y, its margin as a share of
# its larger side, the pixels across its picture, and how much of what lies behind it each face of the water hides,
# so the wet hull shows through it.
elevation, azimuth = np.radians([18.0, 35.0])
margin = 0.04
n_pixels = 700
water_opacity = 0.13
# The glint on the hull, faint, so its facets do not glitter.
hull_gloss = 0.1
background = np.ones(3)
# The windows of the end views of the rolling hull and of the hull drawn whole, across and up, and the size
# of the cube drawn above the map.
end_window = ((-1.6, 1.6), (-1.0, 1.6))
ship_window = ((-1.1, 1.1), (-0.8, 1.0))
cube_window = 0.95
# The end views: seen along the axis of heel from its +x side, y across and z up, as an azimuth; the pixels across
# the rolling hull's views and the cubes'; how much of what lies behind it the water's near face hides; and in pixels,
# the length of the force lines' dashes, the radii of the force lines, the arrows' shafts and heads, the arm, the
# waterline, the centre of gravity and the marker of each load, and the arrows' heads' length.
end_on_azimuth = np.radians(90.0)
n_end_pixels, n_cube_pixels = 200, 130
end_water_opacity = 0.3
dash = 3.0
line_radius, shaft_radius, head_radius, arm_radius, waterline_radius, centre_radius, marker_radius = (
    0.4, 0.8, 3.0, 0.8, 0.4, 2.0, 2.5)
head_length = 7.0
# The direction along the axis of heel, which an end view looks along, a unit apart across the plane x = 0; and a
# projection of points onto points.
along_heel = mv.x.dual()                                                    # [] Point-
Projection = core.ga.gatype((core.Point, core.Point))


# --- plumbing -------------------------------------------------------------------------
def degrees(rolls: core.Twist) -> np.ndarray:
    """The angle of turns about the axis of heel, in degrees: their pairing with the axis's dual."""
    return np.degrees((rolls & core.heel_axis.dual()).to_array())


def arms(couples: core.Forque, weight: core.Scalar) -> np.ndarray:
    """The righting arm of each couple: its turn back about the axis of heel, over the weight."""
    return (-(couples & core.heel_axis) / weight).to_array()


def end_view(window: tuple[tuple[float, float], tuple[float, float]], pixels: int) -> solids.View:
    """The view along the axis of heel of a window across and up, `pixels` across."""
    (left, right), (bottom, top) = window
    middle = mv.point([0.0, (left + right) / 2, (bottom + top) / 2])
    return solids.View(solids.aim(middle, 0.0, end_on_azimuth), np.array([right - left, top - bottom]) / 2, pixels)


def front_projection(points: core.Point, view: solids.View) -> Projection:
    """The projection along the axis of heel onto a plane square to it, in front of a body, its points of unit weight,
    and of the water around it: a point moved along the axis by how far it lies behind that plane."""
    reach = np.abs((mv.x & points).to_array()).max() + 2 * view.pixel
    front = mv.x - mv.w * reach                                                # [] Plane+
    return core.Point - along_heel * (front & core.Point)                     # [] Point- <- Point-


def forces_end_on(view: solids.View, placed: core.Motor, push: core.Forque, centre: core.Point,
                  onto_front: Projection) -> list[tuple[solids.Quadric, np.ndarray]]:
    """The weight from the centre of gravity and the buoyancy from the water surface, as arrows on their dashed
    vertical lines across an end view, and the righting arm between them at the centre's height; in front of the
    body, as `onto_front` places them."""
    gravity = onto_front(placed >> centre)                                    # [] Point-
    # The buoyancy's line meets the water surface in a point of positive weight.
    buoyant = onto_front(((placed >> push) ^ core.water).normalized())       # [] Point-
    # The heights of the view's bottom and top, about the middle where its camera stands.
    bottom, top = (core.water & (view.camera >> solids.origin)).to_array() + np.array([-1, 1]) * view.half[1]
    # Each force line's dashes, up the whole view.
    marks = np.arange(bottom, top, 2 * dash * view.pixel)                     # [dashes]
    parts = []
    for start, colour, sense in ((buoyant, buoyancy_colour, 1), (gravity, weight_colour, -1)):
        # The start dropped onto the water surface, and raised to each mark.
        dashes = start - core.up * (core.water & start) + core.up * marks     # [dashes] Point-
        lines = solids.rods(dashes, dashes + core.up * (dash * view.pixel), line_radius * view.pixel)
        shaft, head = solids.arrows(start, start + core.up * (sense * force * 0.6),
                                    shaft_radius * view.pixel, head_length * view.pixel, head_radius * view.pixel)
        parts += [(lines, to_rgb(colour)), (shaft[None], to_rgb(colour)), (head[None], to_rgb(colour))]
    # The arm reaches from the centre of gravity across to the buoyancy's vertical, at the centre's height.
    level = core.water - mv.w * (core.water & gravity)                        # [] Plane+
    arm = solids.rods(gravity, (buoyant & core.up) ^ level, arm_radius * view.pixel)
    centre_ball = solids.ball(gravity, centre_radius * view.pixel)
    return parts + [(arm[None], (0.1, 0.1, 0.1)), (centre_ball[None, None], to_rgb(weight_colour))]


def end_on(mesh: core.Mesh, points: core.Point, view: solids.View,
           parts: list[tuple[solids.Quadric, np.ndarray]]) -> np.ndarray:
    """A body seen along its axis of heel: the triangles of its points of unit weight, the water over its lower part
    with its waterline, and parts in front of it, `[rows, columns, 3]` bytes."""
    body = meshes.samples(points, mesh.faces, np.array(to_rgb(hull_colour)), meshes.flat(points, mesh.faces), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=hull_gloss)
    # The water: a box up to the surface, past the view's sides and bottom, and as long along the axis as the body,
    # centred below the view's middle, where its camera stands.
    across, up = view.half + view.pixel
    middle = view.camera >> solids.origin                                     # [] Point-
    height = (core.water & middle).to_array()
    depth = up - height
    half_length = np.abs((mv.x & points).to_array()).max() + view.pixel
    water = solids.boxes(solids.translator(middle - core.up * (height + depth / 2) - solids.origin),
                         np.array([half_length, across, depth / 2]))
    # The waterline runs across the view, along the camera's own x, through the middle dropped onto the surface.
    surface_middle = middle - core.up * height                                # [] Point-
    sideways = view.camera >> mv.x.dual()                                     # [] Point-
    onto_front = front_projection(points, view)                               # [] Point- <- Point-
    waterline = solids.rods(onto_front(surface_middle - sideways * across), onto_front(surface_middle + sideways * across),
                            waterline_radius * view.pixel)
    return solids.draw(parts + [(waterline[None], to_rgb(water_colour))], view, background,
                       inks=[], translucent=[(water[None], np.array(to_rgb(water_colour)), end_water_opacity)],
                       meshes=[(body, 1.0)])


def side_by_side(images: list[np.ndarray]) -> np.ndarray:
    """Pictures `[rows, columns, 3]` bytes in a row, the shorter ones padded below with the background."""
    rows = max(image.shape[0] for image in images)
    return np.concatenate([np.pad(image, ((0, rows - image.shape[0]), (0, 0), (0, 0)), constant_values=255)
                           for image in images], axis=1)


def in_water(mesh: core.Mesh, points: core.Point) -> np.ndarray:
    """The hull's triangles in space, of its points of unit weight, flat, the dry ones pale and the wet ones darker
    with depth, under a translucent plate of the water's surface, framed on the plate."""
    # Each triangle's depth, at its centroid, below the water surface.
    centroids = mesh.incidence[2, 0].mean(axis=0) * points.field()            # Point-[triangles]
    depth = np.clip(-(core.water & centroids).batch().to_array() / deepest, 0, 1)   # [triangles]
    wet = plt.get_cmap(wet_map)(0.35 + 0.6 * depth)[:, :3]
    colours = np.where((depth > 0)[:, None], wet, to_rgb(hull_colour))       # [triangles, 3]
    half_length, half_breadth = sea
    surface = mv.point([[-half_length, -half_breadth, 0], [half_length, -half_breadth, 0],
                        [half_length, half_breadth, 0], [-half_length, half_breadth, 0]])   # [corners] Point-
    view = solids.view([surface, points], 0.0, elevation, azimuth, margin, n_pixels)
    normals = meshes.flat(points, mesh.faces)
    hull = meshes.samples(points, mesh.faces, np.zeros(3), normals, view, face_values=colours, colour=meshes.by_face,
                          gloss=hull_gloss)
    # The water's surface, a plate a pixel thick.
    water = solids.plates(surface[None], view.pixel)                          # [1, 6] Plane+
    return solids.draw([], view, background, inks=[], translucent=[(water, np.array(to_rgb(water_colour)), water_opacity)],
                       meshes=[(hull, 1.0)])


def draw_ship(mesh: core.Mesh, vertices: core.Point, placed: core.Motor, push: core.Forque,
              centre: core.Point) -> plt.Figure:
    """The heeled hull: in space, its dry surface pale and its wet surface darker with depth, in the water;
    and seen along its axis of heel, with its weight, its buoyancy and the righting arm between them."""
    points = placed >> vertices.batch()                                      # [vertices] Point-
    in_space = in_water(mesh, points)
    (left, right), (bottom, top) = ship_window
    view = end_view(ship_window, round(in_space.shape[0] * (right - left) / (top - bottom)))
    forces = forces_end_on(view, placed, push, centre, front_projection(points, view))
    return image_figure(side_by_side([in_space, end_on(mesh, points, view, forces)]))


@charts.style()
def draw_curves(heels: np.ndarray, couples: core.Forque, weight: core.Scalar, stiffness: core.Stiffness) -> plt.Figure:
    """The righting arm of each load over the heels `[heels, loads]`, with the tangent at upright that the
    stiffness gives, and the line of no righting."""
    righting = arms(couples, weight)                                         # [heels, loads]
    slopes = (stiffness(mv.yz, mv.yz) / weight).to_array()                       # [loads]
    colours = plt.get_cmap(load_map)(np.linspace(0.05, 0.85, righting.shape[1]))
    # Half the column: a few smooth curves and their tangents at upright.
    figure, ax = charts.subplots(420, 260)
    ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
    near = heels[heels <= tangent]
    for curve, slope, colour in zip(righting.T, slopes, colours):
        ax.plot(heels, curve, color=colour)
        ax.plot(near, slope * np.radians(near), color=colour, linestyle=(0, (4, 3)))
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(heels[0], heels[-1])
    ax.set(xlabel="heel (degrees)", ylabel="righting arm (m)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


@charts.style()
def draw_cubes(densities: np.ndarray, heels: np.ndarray, couples: core.Forque, fluid_weight: float, mesh: core.Mesh,
               vertices: core.Point, shown: core.Motor, shown_densities: list[float], rests: np.ndarray) -> plt.Figure:
    """Above, cubes of a few densities at rest; below, the cube's righting couple over density and heel, blue
    turning it back upright and red over, its rests where it turns from over to back drawn thick, its
    balances the other way thin, and the cubes above marked on them."""
    moments = (-(couples & core.heel_axis) / fluid_weight).to_array()            # [densities, heels]
    # A rest is stable where the couple turns from pushing over to pushing back as the heel grows.
    rising = np.gradient(moments, heels, axis=1) > 0
    cube_view = end_view(((-cube_window, cube_window), (-cube_window, cube_window)), n_cube_pixels)
    cubes = side_by_side([end_on(mesh, placed >> vertices.batch(), cube_view, [])
                          for placed in shown])
    # The map as wide as the cubes above it.
    figure, ax = charts.subplots(cubes.shape[1], round(0.6 * cubes.shape[1]))
    largest = np.abs(moments).max()
    ax.contourf(densities, heels, moments.T, levels=np.linspace(-largest, largest, 41), cmap=arm_map)
    # The stable rests twice as wide as a curve, the balances the other way as thin as the frame.
    for stable, width in ((True, charts.points(2 * charts.LINE_PX)), (False, charts.FRAME_WIDTH)):
        ax.contour(densities, heels, np.where(rising == stable, moments, np.nan).T, levels=[0], colors="0.1",
                   linewidths=width)
    ax.plot(shown_densities, rests, "o", color="white", markeredgecolor="0.1")
    ax.set(xlim=(densities[0], densities[-1]), ylim=(-3, 93), yticks=[0, 45, 90])
    ax.set(xlabel="density over the water's", ylabel="heel (degrees)")
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)
    plot = capture(figure)
    plt.close(figure)
    return image_figure(np.concatenate([cubes, plot[:, :cubes.shape[1]]], axis=0))


def animate_release(mesh: core.Mesh, vertices: core.Point, centres: core.Point,
                    frames: Iterable[tuple[core.Twist, core.Motor, core.Forque, core.Forque]], heels: np.ndarray,
                    couples: core.Forque, weight: core.Scalar) -> Iterator[np.ndarray]:
    """Each frame: the released hull with each load seen along its axis of heel, its weight from its centre of
    gravity and its buoyancy from the water surface, as arrows on their lines, beside the righting arm curves
    of the loads, each with the hull's heel so far."""
    curves = arms(couples, weight)                                           # [heels, loads]
    # Past half a turn the righting arm repeats reversed: heeled the other way round, it pushes the other way.
    beyond = 360 - heels[::-1][1:]
    extended, extended_curves = np.concatenate([heels, beyond]), np.concatenate([curves, -curves[::-1][1:]])
    colours = plt.get_cmap(load_map)(np.array([0.3, 0.85]))
    view = end_view(end_window, n_end_pixels)
    rows = view.shape[0]
    # Where each load's marker sits in the corner of its view.
    marker = mv.point([0.0, 0.9 * end_window[0][0], 0.85 * end_window[1][1]])   # [] Point-
    # The curves set up once, as tall as the end views beside them, under the charts' style; each load's heel so far
    # and its heel now move.
    with charts.style():
        figure, ax = charts.subplots(round(1.6 * rows), rows, layout="constrained")
        ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
        so_far, now = [], []
        for case, colour in enumerate(colours):
            ax.plot(extended, extended_curves[:, case], color=colour, alpha=0.5)
            so_far += ax.plot([], [], color=colour)
            now += ax.plot([], [], "o", color=colour)
        ax.set_xticks([0, 90, 180])
        ax.set_xlim(-30, 215)
        ax.set_xlabel("heel (degrees)")
        ax.set_ylabel("righting arm (m)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # Laid out once; only the heels move between frames.
        figure.canvas.draw()
        figure.set_layout_engine("none")
    states = []
    for roll, placed, push, weighing in frames:
        states.append([degrees(roll), arms(push + weighing, weight)])
        trail = np.array(states)                                             # [frames, 2, loads]
        views = []
        for case, colour in enumerate(colours):
            points = placed[case] >> vertices.batch()                            # [vertices] Point-
            onto_front = front_projection(points, view)                          # [] Point- <- Point-
            # Which load this is, a dot in the corner in its curve's colour.
            dot = solids.ball(onto_front(marker), marker_radius * view.pixel)
            views.append(end_on(mesh, points, view,
                                forces_end_on(view, placed[case], push[case], centres[case], onto_front)
                                + [(dot[None, None], colour[:3])]))
            so_far[case].set_data(*trail[:, :, case].T)
            now[case].set_data(*trail[-1, :, case, None])
        yield side_by_side(views + [capture(figure)])
    plt.close(figure)
