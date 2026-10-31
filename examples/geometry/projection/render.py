"""Drawing for the projective camera example.

This module consumes the geometry that `core` produces and turns it into a figure.
It constructs viewport geometry only, never scene geometry, and the mathematics
never imports it. The shadow scene is ray traced on the GPU: edges as rods, the
shadow trail as a row of beads, the point light a ball and the sun's direction an
arrow, a rod with a ball at its tip; the two camera images are drawn flat, as strokes and markers read by each
rig's own planes.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.geometry.projection.core import Line, Motor, Point, mv
from examples.util.wgsl import fitted, show, simplices, solids
from examples.util.wgsl.camera import panel

# The image panels' margin around both images, as a fraction of their larger side; the colours of the subject's edges
# and of the epipolar lines and corners; the widths of the edges and of the epipolar lines, and the corners' radius,
# in pixels; and how much the epipolar lines cover.
image_margin = 0.08
subject_colour, epipolar_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#f43f5e"))
subject_width, epipolar_width, corner_radius = 3.0, 1.3, 3.5
epipolar_opacity = 0.8
# The shadow scene: the colours of the body, of its shadows under the point light and the sun, and of the sun's
# direction, drawn as an arrow of unit length from a place above the scene.
body_colour, light_colour, sun_colour = (np.array(to_rgb(colour)) for colour in ("#38bdf8", "#fbbf24", "#a855f7"))
sun_arrow_start = mv.point([-2, 2, 4])                                         # [] Point-
background = np.ones(3)
# The shadow scene's view, elevation and azimuth in radians, azimuth zero looking along +y, and its margin as a share
# of the larger side.
elevation, azimuth = np.radians([30.0, -60.0 + 90.0])
scene_margin = 0.1
# Radii in pixels: of the body's edges, of the shadows' edges, of the beads along the trail, of the point light, and
# of the arrow's shaft and the ball at its tip.
body_radius, shadow_radius, bead_radius, light_radius = 1.6, 1.2, 2.4, 6.5
shaft_radius, tip_radius = 2.4, 6.5


def render_shadow_scene(
    ax,
    body: Point,
    edges,
    shadows: tuple,
    point_light: Point,
    sun: Point,
) -> None:
    """Render the lit cube, its shadows under the point light and the sun, the lights, and the corner's shadow trail."""
    light_corners, sun_corners, trail = shadows                               # [corners] Point-, [corners] Point-, [trail] Point-
    ends = np.array(edges)                                                     # [edges, 2]
    # The arrow along the sun's direction at unit length: a direction's length is the size of its dual, the plane
    # through the origin square to it.
    arrow_end = sun_arrow_start + sun / sun.dual().norm()                      # [] Point-
    # Framed by everything drawn, as many pixels across as the axes' room holds.
    view = fitted(solids.view([body, light_corners, sun_corners, trail, point_light, sun_arrow_start, arrow_end],
                              0.0, elevation, azimuth, scene_margin, 1), ax)
    pixel = view.pixel
    shaft = solids.rods(sun_arrow_start, arrow_end, shaft_radius * pixel)    # [3] Plane+ <- Point-
    tip = solids.ball(arrow_end, tip_radius * pixel)                          # [] Plane+ <- Point-
    rods, joints = solids.edges(body[ends[:, 0]], body[ends[:, 1]], body_radius * pixel)
    parts = [(rods, body_colour), (joints, body_colour),
             (solids.rods(light_corners[ends[:, 0]], light_corners[ends[:, 1]], shadow_radius * pixel), light_colour),
             (solids.rods(sun_corners[ends[:, 0]], sun_corners[ends[:, 1]], shadow_radius * pixel), sun_colour),
             (solids.ball(trail, bead_radius * pixel)[:, None], light_colour),
             (solids.ball(point_light, light_radius * pixel)[None, None], light_colour),
             (shaft[None], sun_colour), (tip[None, None], sun_colour)]
    show(ax, solids.draw(parts, view, background, inks=[], translucent=[], meshes=[]), view)


def screen_line_ends(rig: Motor, lines: Line, half_width: float) -> Point:
    """Where lines on a rig's screen meet its left and right edges at x = ±half_width, `[lines, ends]`."""
    return stack([lines ^ (rig >> (mv.x - mv.w * x)) for x in (-half_width, half_width)], axis=-1)


def render_stereo_scene(
    ax_1, ax_2,
    edges,
    stereo: tuple,
    rig_1: Motor, rig_2: Motor,
    margin: float,
) -> None:
    """Draw the two camera views of the subject, with camera 2's corners on the epipolar lines of camera 1's corners."""
    image_1, image_2, _epipole_2, epipolar_lines_2, _correspondence = stereo
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
    subject = simplices.Mesh(len(image_1), np.array(edges))
    # Each rig's camera reads image points by the rig's own planes x and y, so the screen is seen as the rig sees it.
    camera_1 = panel(ax_1, rig_1 >> mv.x, rig_1 >> mv.y)
    camera_2 = panel(ax_2, rig_2 >> mv.x, rig_2 >> mv.y)
    simplices.overlay(ax_1, [simplices.strokes(image_1, subject, subject_colour, subject_width, camera_1)])
    epipolar_ends = screen_line_ends(rig_2, epipolar_lines_2, widths.max())   # [corners, ends] Point-
    simplices.overlay(ax_2, [
        simplices.strokes(image_2, subject, subject_colour, subject_width, camera_2),
        simplices.strokes(epipolar_ends.reshape(-1), simplices.Mesh.chains([2] * len(image_2)), epipolar_colour,
                          epipolar_width, camera_2, opacity=epipolar_opacity),
        simplices.markers(image_2, epipolar_colour, corner_radius, camera_2)])


def draw_projection(
    body: Point,
    edges,
    shadows: tuple,
    point_light: Point,
    sun: Point,
    stereo: tuple,
    rig_1: Motor,
    rig_2: Motor,
) -> plt.Figure:
    """Lay out the shadow scene and both camera views."""
    fig = plt.figure(figsize=(16, 5), dpi=120)
    scene_axes, camera_1_axes, camera_2_axes = (fig.add_subplot(1, 3, i + 1) for i in range(3))
    # The figure laid out, the axes as bare as their pictures leave them, before the pictures are drawn at the size of
    # their room.
    for ax in (scene_axes, camera_1_axes, camera_2_axes):
        ax.set_axis_off()
    fig.tight_layout()
    render_stereo_scene(camera_1_axes, camera_2_axes, edges, stereo, rig_1, rig_2, image_margin)
    render_shadow_scene(scene_axes, body, edges, shadows, point_light, sun)
    return fig
