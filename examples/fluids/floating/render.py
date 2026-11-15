"""Drawing for hydrostatic stability: the heeled hull in space with its wet surface coloured by depth and its
force lines; the righting arm curves; the floating cube stability map over density and heel with resting cubes
above; and the released hull rolling upright or capsizing beside its righting arm curve."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from numga import stack

from examples.fluids.floating import core
from examples.util import charts
from examples.util.animation import capture
from numga.topology import Topology
from numga.render.drawing import Draw
from numga.render import meshes, simplices, solids
from examples.util.panels import image_figure

scene_mv = solids.mv
include = solids.context.include_from_algebra
Projection = solids.ga.gatype((solids.Point, solids.Point))             # Point- <- Point-

# Colours of the hull, water, weight and buoyancy, and colour maps of depth, load height and righting couple.
hull_colour, deck_colour, water_colour, weight_colour, buoyancy_colour = (
    "#e7e2d6", "#c9bfa8", "#5aa9c9", "#d06c36", "#1f6f99",
)
wet_map, load_map, arm_map = "Blues", "viridis", "RdBu"
max_shade_depth = 0.6
sea_half_extents = (3.0, 1.8)
force_arrow_scale = 0.9
tangent_span_degrees = 25.0
elevation, azimuth = np.radians([18.0, 35.0])
margin = 0.04
n_pixels = 700
water_opacity = 0.13
hull_gloss = 0.1
background = np.ones(3)
end_window = ((-1.6, 1.6), (-1.0, 1.6))
ship_window = ((-1.1, 1.1), (-0.8, 1.0))
cube_window = 0.95
end_on_azimuth = np.radians(90.0)
n_end_pixels, n_cube_pixels = 200, 130
end_water_opacity = 0.3
dash_pixels = 3.0
line_radius, arm_radius, waterline_radius, centre_radius, marker_radius = 0.4, 0.8, 0.4, 2.0, 2.5
arrow_width, arrow_chevron = 1.6, 6.4
heel_direction = scene_mv.x.dual()                                      # [] Point-
scene_water = include(core.water)                                       # [] Plane+
scene_up = include(core.up)                                             # [] Point-
scene_unit = scene_water | scene_water                                  # [] Scalar+


# --- plumbing -------------------------------------------------------------------------
def heel_degrees(rolls: core.Twist) -> np.ndarray:
    """Heel angles in degrees read by pairing roll twists with the dual of the heel axis."""
    return np.degrees((rolls & core.heel_axis.dual()).to_array())


def righting_arms(couples: core.Forque, weight: core.Scalar) -> np.ndarray:
    """Righting arms in metres read from righting couples about the heel axis divided by weight."""
    return (-(couples & core.heel_axis) / weight).to_array()


def end_view(window: tuple[tuple[float, float], tuple[float, float]], n_view_pixels: int) -> solids.View:
    """Camera view along the heel axis framing a horizontal and vertical window."""
    (left, right), (bottom, top) = window
    middle = scene_mv.point([0.0, (left + right) / 2, (bottom + top) / 2])
    return solids.View(solids.aim(middle, 0.0, end_on_azimuth), np.array([right - left, top - bottom]) / 2, n_view_pixels)


def front_projection(points: solids.Point, view: solids.View) -> Projection:
    """Orthographic projection along the heel axis onto a plane in front of the hull and water box."""
    depths = (scene_mv.x & points).abs().reshape(-1)
    reach = depths.max() + 2 * view.pixel
    front = scene_mv.x - scene_mv.w * reach                             # [] Plane+
    return solids.Point - heel_direction * (front & solids.Point)       # [] Point- <- Point-


def forces_end_on(view: solids.View, placement: core.Motor, push: core.Forque, centre: core.Point,
                  onto_front: Projection) -> tuple[list[tuple[solids.Quadric, np.ndarray]], Draw]:
    """Dashed action lines, upward buoyancy and downward weight arrows, centre of gravity marker, and horizontal
    righting arm projected in front of an end view."""
    placed = include(placement)                                         # [] Motor+
    gravity_point = onto_front(placed >> include(centre))               # [] Point-
    buoyancy_point = onto_front(((placed >> include(push)) ^ scene_water).normalized())  # [] Point-
    bottom = (scene_water & (view.camera >> solids.origin)) - view.half[1]
    n_dashes = int(np.ceil(view.half[1] / (dash_pixels * view.pixel)))
    marks = bottom + np.arange(n_dashes) * (2 * dash_pixels * view.pixel)  # [n_dashes] Scalar+
    parts = []
    for start, colour in ((buoyancy_point, buoyancy_colour), (gravity_point, weight_colour)):
        dashes = start - scene_up * (scene_water & start) + scene_up * marks  # [n_dashes] Point-
        lines = solids.rods(dashes, dashes + scene_up * (dash_pixels * view.pixel), line_radius * view.pixel)
        parts += [(lines, to_rgb(colour))]
    starts = stack((buoyancy_point, gravity_point))                     # [2] Point-
    tips = starts + scene_up * (np.array([1.0, -1.0]) * force_arrow_scale * 0.6)  # [2] Point-
    arrows = simplices.tubes(
        stack((starts, tips), axis=-1).reshape(-1),
        Topology.chains([2, 2]),
        np.array([to_rgb(buoyancy_colour), to_rgb(weight_colour)]),
        arrow_width,
        view,
        gloss=solids.gloss,
        chevrons=np.ones(2),
        chevron=arrow_chevron,
    )
    level = scene_water - scene_mv.w * (scene_water & gravity_point)    # [] Plane+
    arm = solids.rods(gravity_point, (buoyancy_point & scene_up) ^ level, arm_radius * view.pixel)
    centre_ball = solids.ball(gravity_point, centre_radius * view.pixel)
    return parts + [(arm[None], (0.1, 0.1, 0.1)), (centre_ball[None, None], to_rgb(weight_colour))], arrows


def end_on(topology: core.Topology, points: solids.Point, view: solids.View,
           parts: list[tuple[solids.Quadric, np.ndarray]],
           arrows: list[tuple[Draw, float]]) -> np.ndarray:
    """End-on RGB image of a triangulated body, translucent water box, waterline rod, and overlaid force geometry."""
    triangles = topology.elements[2]                                    # [F, 3]
    body = meshes.samples(
        points, triangles, np.array(to_rgb(hull_colour)), meshes.flat(points, triangles), view,
        face_values=np.zeros(4), colour=meshes.plain, gloss=hull_gloss,
    )
    across, up_span = view.half + view.pixel
    middle = view.camera >> solids.origin                               # [] Point-
    height = scene_water & middle
    depth = up_span - height
    depths = (scene_mv.x & points).abs().reshape(-1)
    half_length = depths.max() + view.pixel
    water_box = solids.boxes(
        solids.translator(middle - scene_up * (height + depth / 2) - solids.origin),
        stack((half_length, scene_unit * across, depth / 2)),
    )
    surface_middle = middle - scene_up * height                         # [] Point-
    sideways = view.camera >> scene_mv.x.dual()                         # [] Point-
    onto_front = front_projection(points, view)                         # [] Point- <- Point-
    waterline = solids.rods(
        onto_front(surface_middle - sideways * across),
        onto_front(surface_middle + sideways * across),
        waterline_radius * view.pixel,
    )
    return solids.draw(
        parts + [(waterline[None], to_rgb(water_colour))],
        view,
        background,
        inks=[],
        translucent=[(water_box[None], np.array(to_rgb(water_colour)), end_water_opacity)],
        meshes=[(body, 1.0)] + arrows,
    )


def side_by_side(images: list[np.ndarray]) -> np.ndarray:
    """Horizontal concatenation of RGB images, bottom-padded with white where heights differ."""
    rows = max(image.shape[0] for image in images)
    return np.concatenate(
        [np.pad(image, ((0, rows - image.shape[0]), (0, 0), (0, 0)), constant_values=255) for image in images],
        axis=1,
    )


def in_water(topology: core.Topology, points: solids.Point) -> np.ndarray:
    """Perspective RGB view of the hull in space with submerged triangles shaded by depth under a translucent water plate."""
    centroids = topology.incidence[2, 0].mean(axis=0) * points.field()  # Point-[F]
    depth = np.clip(-(scene_water & centroids).batch().to_array() / max_shade_depth, 0, 1)
    wet_colours = plt.get_cmap(wet_map)(0.35 + 0.6 * depth)[:, :3]
    colours = np.where((depth > 0)[:, None], wet_colours, to_rgb(hull_colour))
    half_length, half_breadth = sea_half_extents
    surface = scene_mv.point([
        [-half_length, -half_breadth, 0],
        [half_length, -half_breadth, 0],
        [half_length, half_breadth, 0],
        [-half_length, half_breadth, 0],
    ])                                                                  # [4] Point-
    view = solids.view([surface, points], 0.0, elevation, azimuth, margin, n_pixels)
    triangles = topology.elements[2]                                    # [F, 3]
    normals = meshes.flat(points, triangles)
    hull = meshes.samples(points, triangles, np.zeros(3), normals, view, face_values=colours, colour=meshes.by_face, gloss=hull_gloss)
    water_plate = solids.plates(surface[None], view.pixel)              # [1, 6] Plane+
    return solids.draw([], view, background, inks=[], translucent=[(water_plate, np.array(to_rgb(water_colour)), water_opacity)], meshes=[(hull, 1.0)])


def draw_ship(topology: core.Topology, vertices: core.Point, placement: core.Motor, push: core.Forque,
              centre: core.Point) -> plt.Figure:
    """Side-by-side 3D perspective and end-on cross-section of the heeled hull with its hydrostatic force lines."""
    points = include(placement >> vertices.batch())                     # [V] Point-
    in_space = in_water(topology, points)
    (left, right), (bottom, top) = ship_window
    view = end_view(ship_window, round(in_space.shape[0] * (right - left) / (top - bottom)))
    force_solids, arrows = forces_end_on(view, placement, push, centre, front_projection(points, view))
    return image_figure(side_by_side([in_space, end_on(topology, points, view, force_solids, [(arrows, 1.0)])]))


@charts.style()
def draw_curves(heels: np.ndarray, couples: core.Forque, weight: core.Scalar, stiffness: core.Stiffness) -> plt.Figure:
    """Righting arm curves across heel angles for each load height, with dashed upright metacentric tangents."""
    righting = righting_arms(couples, weight)
    slopes = (stiffness(core.heel_axis, core.heel_axis) / weight).to_array()
    colours = plt.get_cmap(load_map)(np.linspace(0.05, 0.85, righting.shape[1]))
    figure, ax = charts.subplots(420, 260)
    ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
    near = heels[heels <= tangent_span_degrees]
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
def draw_cubes(densities: np.ndarray, heels: np.ndarray, couples: core.Forque, fluid_weight: float,
               topology: core.Topology, vertices: core.Point, placements: core.Motor,
               shown_densities: list[float], rest_heels: core.Scalar) -> plt.Figure:
    """Equilibrium floating cubes above a contour map of the cube's righting couple over relative density and heel."""
    moments = (-(couples & core.heel_axis) / fluid_weight).to_array()
    rising = np.gradient(moments, heels, axis=1) > 0
    cube_view = end_view(((-cube_window, cube_window), (-cube_window, cube_window)), n_cube_pixels)
    cubes = side_by_side([
        end_on(topology, include(placement >> vertices.batch()), cube_view, [], [])
        for placement in placements
    ])
    figure, ax = charts.subplots(cubes.shape[1], round(0.6 * cubes.shape[1]))
    largest = np.abs(moments).max()
    ax.contourf(densities, heels, moments.T, levels=np.linspace(-largest, largest, 41), cmap=arm_map)
    for stable, width in ((True, charts.points(2 * charts.LINE_PX)), (False, charts.FRAME_WIDTH)):
        ax.contour(densities, heels, np.where(rising == stable, moments, np.nan).T, levels=[0], colors="0.1", linewidths=width)
    ax.plot(shown_densities, rest_heels.to_array(), "o", color="white", markeredgecolor="0.1")
    ax.set(xlim=(densities[0], densities[-1]), ylim=(-3, 93), yticks=[0, 45, 90])
    ax.set(xlabel="density over the water's", ylabel="heel (degrees)")
    figure.subplots_adjust(left=0.1, right=0.98, bottom=0.12, top=0.98)
    plot = capture(figure)
    plt.close(figure)
    return image_figure(np.concatenate([cubes, plot[:, :cubes.shape[1]]], axis=0))


def animate_release(topology: core.Topology, vertices: core.Point, centres: core.Point, heels: np.ndarray,
                    couples: core.Forque, weight: core.Scalar,
                    frames: Iterable[tuple[core.Twist, core.Motor, core.Forque, core.Forque]]) -> Iterator[np.ndarray]:
    """Yield side-by-side end views of the rolling hulls and their trajectories along the righting arm curves."""
    curves = righting_arms(couples, weight)
    beyond = 360 - heels[::-1][1:]
    extended, extended_curves = np.concatenate([heels, beyond]), np.concatenate([curves, -curves[::-1][1:]])
    colours = plt.get_cmap(load_map)(np.array([0.3, 0.85]))
    view = end_view(end_window, n_end_pixels)
    rows = view.shape[0]
    marker = scene_mv.point([0.0, 0.9 * end_window[0][0], 0.85 * end_window[1][1]])  # [] Point-
    with charts.style():
        figure, ax = charts.subplots(round(1.6 * rows), rows, layout="constrained")
        ax.axhline(0, color="0.6", linewidth=charts.FRAME_WIDTH)
        trail_lines, current_dots = [], []
        for case, colour in enumerate(colours):
            ax.plot(extended, extended_curves[:, case], color=colour, alpha=0.5)
            trail_lines += ax.plot([], [], color=colour)
            current_dots += ax.plot([], [], "o", color=colour)
        ax.set_xticks([0, 90, 180])
        ax.set_xlim(-30, 215)
        ax.set_xlabel("heel (degrees)")
        ax.set_ylabel("righting arm (m)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        figure.canvas.draw()
        figure.set_layout_engine("none")
    history = []
    for roll, placement, push, weighing in frames:
        history.append([heel_degrees(roll), righting_arms(push + weighing, weight)])
        trail = np.array(history)
        views = []
        for case, colour in enumerate(colours):
            points = include(placement[case] >> vertices.batch())       # [V] Point-
            onto_front = front_projection(points, view)                 # [] Point- <- Point-
            dot = solids.ball(onto_front(marker), marker_radius * view.pixel)
            force_solids, arrows = forces_end_on(view, placement[case], push[case], centres[case], onto_front)
            views.append(end_on(topology, points, view, force_solids + [(dot[None, None], colour[:3])], [(arrows, 1.0)]))
            trail_lines[case].set_data(*trail[:, :, case].T)
            current_dots[case].set_data(*trail[-1, :, case, None])
        yield side_by_side(views + [capture(figure)])
    plt.close(figure)
