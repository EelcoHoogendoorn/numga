"""Ray-traced shadow scene and flat stereo screen panels for the projective camera example.

The shadow scene ray-traces cube edges as rods, the corner shadow trail as a ring of beads, the point light as a ball, and the sun's direction as a chevroned arrow; the two camera views draw flat screen strokes and corner markers read directly against each rig's coordinate planes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.geometry.projection.core import Line, Motor, Point, mv
from numga.topology import Topology
from numga.render import simplices, solids
from examples.util.panels import fitted, overlay, panel, show

image_margin = 0.08
subject_colour, epipolar_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#f43f5e"))
subject_width, epipolar_width, corner_radius = 3.0, 1.3, 3.5
epipolar_opacity = 0.8

body_colour, light_colour, sun_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#fbbf24", "#a855f7"))
sun_arrow_start = mv.point([-2, 2, 4])                                         # [] Point-
background = np.ones(3)

elevation, azimuth = np.radians([30.0, -60.0 + 90.0])
scene_margin = 0.1

body_radius, shadow_radius, bead_radius, light_radius = 1.6, 1.2, 2.4, 6.5
arrow_width, arrow_chevron = 4.8, 19.2


def render_shadow_scene(
    ax,
    body: Point,
    edges: np.ndarray,
    cast: tuple,
    point_light: Point,
    sun: Point,
) -> None:
    """Draws the lit cube, its spotlight and sunlight shadows, the lights, and the corner's shadow trail onto `ax`."""
    light_corners, sun_corners, trail = cast                                   # [...] Point-, [...] Point-, [...] Point-
    ends = np.asarray(edges)                                                   # [..., 2]
    # A direction's Euclidean norm equals the norm of its dual plane through the origin.
    arrow_end = sun_arrow_start + sun / sun.dual().norm()                      # [] Point-
    view = fitted(solids.view([body, light_corners, sun_corners, trail, point_light, sun_arrow_start, arrow_end],
                              0.0, elevation, azimuth, scene_margin, 1), ax)
    pixel = view.pixel
    arrow = simplices.tubes(stack((sun_arrow_start, arrow_end)), Topology.chains([2]), sun_colour, arrow_width,
                            view, gloss=solids.gloss, chevrons=np.ones(1), chevron=arrow_chevron)
    rods, joints = solids.edges(body[ends[:, 0]], body[ends[:, 1]], body_radius * pixel)
    parts = [(rods, body_colour), (joints, body_colour),
             (solids.rods(light_corners[ends[:, 0]], light_corners[ends[:, 1]], shadow_radius * pixel), light_colour),
             (solids.rods(sun_corners[ends[:, 0]], sun_corners[ends[:, 1]], shadow_radius * pixel), sun_colour),
             (solids.ball(trail, bead_radius * pixel)[:, None], light_colour),
             (solids.ball(point_light, light_radius * pixel)[None, None], light_colour)]
    show(ax, solids.draw(parts, view, background, inks=[], translucent=[], meshes=[(arrow, 1.0)]), view)


def screen_line_ends(rig: Motor, lines: Line, half_width: float) -> Point:
    """Meeting `lines` on `rig`'s screen with its left and right border planes at `±half_width` gives segment endpoints (`[..., 2] Point-`)."""
    return stack([lines ^ (rig >> (mv.x - mv.w * x)) for x in (-half_width, half_width)], axis=-1)


def render_stereo_scene(
    ax_1, ax_2,
    edges: np.ndarray,
    views: tuple,
    rig_1: Motor, rig_2: Motor,
    margin: float,
) -> None:
    """Draws the two camera views of the subject onto `ax_1` and `ax_2`, overlaying the second view's epipolar lines and corners."""
    image_1, image_2, _epipole_2, epipolar_lines_2, _correspondence = views
    both = concatenate(((rig_1 << image_1).normalized(), (rig_2 << image_2).normalized()))
    readings = stack((mv.x, mv.y))[:, None] & both
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    spans = high - low
    padding = margin * spans.max()
    low, high = low - padding, high + padding
    widths = stack((low[0], high[0])).abs()
    limits = stack((low, high), axis=-1).to_array()
    for ax in (ax_1, ax_2):
        ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
        ax.set_axis_off()
    subject = Topology(len(image_1), np.asarray(edges))
    # Each rig's panel camera pairs image points directly with the rig's own x and y planes over their weight.
    camera_1 = panel(ax_1, rig_1 >> mv.x, rig_1 >> mv.y, mv.w)
    camera_2 = panel(ax_2, rig_2 >> mv.x, rig_2 >> mv.y, mv.w)
    overlay(ax_1, [simplices.strokes(image_1, subject, subject_colour, subject_width, camera_1)])
    epipolar_ends = screen_line_ends(rig_2, epipolar_lines_2, widths.max())   # [..., 2] Point-
    overlay(ax_2, [
        simplices.strokes(image_2, subject, subject_colour, subject_width, camera_2),
        simplices.strokes(epipolar_ends.reshape(-1), Topology.chains([2] * len(image_2)), epipolar_colour,
                          epipolar_width, camera_2, opacity=epipolar_opacity),
        simplices.markers(image_2, epipolar_colour, corner_radius, camera_2)])


def projection_figure(
    body: Point,
    edges: np.ndarray,
    cast: tuple,
    point_light: Point,
    sun: Point,
    views: tuple,
    rig_1: Motor,
    rig_2: Motor,
) -> plt.Figure:
    """Three-panel figure combining the 3D shadow scene with both stereo camera views."""
    fig = plt.figure(figsize=(16, 5), dpi=120)
    scene_axes, camera_1_axes, camera_2_axes = (fig.add_subplot(1, 3, i + 1) for i in range(3))
    for ax in (scene_axes, camera_1_axes, camera_2_axes):
        ax.set_axis_off()
    fig.tight_layout()
    render_stereo_scene(camera_1_axes, camera_2_axes, edges, views, rig_1, rig_2, image_margin)
    render_shadow_scene(scene_axes, body, edges, cast, point_light, sun)
    return fig
