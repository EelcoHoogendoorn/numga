"""Ordered points and the chosen sides of a circle's two cutting surfaces. The circle, its plane and its sphere
are drawn on the GPU: the plane a translucent sheet, the sphere a translucent ball with the circles of its sampled
wireframe inked on it, the points balls and the sides arrows; the circle's motions are drawn by matplotlib."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from numga import concatenate, stack
from examples.orientation.conformal_circle.core import Point, Sphere, mv
from examples.util.wgsl import show, solids, strokes

sm = solids.mv
plane_ink, sphere_ink, curve_ink = "#218c9d", "#ce713f", "#3d354a"
marker_inks = ("#9861ba", "#dd9650", "#4f9b85")
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the scene as a share
# of the larger side, and the pixels across each panel.
elevation, azimuth = np.radians(26.0), np.radians(32.0)
margin, n_pixels = 0.25, 300
# The plane's sheet, a square about the origin, in the renderer's points; the opacities of each face of the sheet,
# its rim, each face of the ball and the sphere's wireframe, front and back alike.
sheet_corners = sm.point(1.23 * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]))   # [corners] Point-
sheet_opacity, rim_opacity, ball_opacity, wireframe_opacity = 0.073, 0.45, 0.062, 0.26
# The widths of the circle, the sheet's rim, the wireframe and the chord arrows, the plane's and the sphere's
# arrows, in pixels, the arrowheads' lengths and widths, and the points' radius, in pixels; the sheet's thickness,
# in pixels.
circle_width, rim_width, wireframe_width, sheet_thickness = 2.4, 1.8, 1.8, 0.6
chord_width, plane_arrow_width, sphere_arrow_width = 2.1, 3.0, 2.7
chord_head, arrow_head, head_width = 13.0, 13.0, 9.6
dot_radius = 4.8
# The lengths of the plane's and the sphere's arrows per unit of their normals.
plane_arrow_length, sphere_arrow_length = 0.85, 0.45
# How small against the largest the plane a sampled circle's points span may be and still count as one: a circle
# gathered into one point, as a parallel at a pole, spans none.
spanned = 1e-12
# The pixels across each panel of the motions, and the dots per inch their line widths are given at.
motion_pixels, motion_dpi = 150, 45
# The point at infinity, the origin, and the sphere at infinity.
infinity = mv.e.dual() + mv.w.dual()                                            # [] Point-
origin = (mv.e.dual() - mv.w.dual()) / 2                                        # [] Point-
horizon = mv.e + mv.w                                                           # [] Sphere+


# --- plumbing -------------------------------------------------------------------------


def coordinates(points: Point) -> np.ndarray:
    """Read conformal points for numerical diagnostics, dividing spatial readings by the infinity weight."""
    weights = -horizon & points
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / weights[..., None]).to_array()


def gpu_points(points: Point) -> solids.Point:
    """Forget the conformal infinity part and place the weighted spatial part in the projective scene."""
    weights = -horizon & points
    readings = (stack((mv.x, mv.y, mv.z)) & points[..., None]) / weights[..., None]
    return solids.origin + (solids.axes.dual() * solids.context.include_from_algebra(readings)).sum(axis=-1)


def plane_coordinates(points: Point) -> np.ndarray:
    """x and y of points, on the last axis, for matplotlib: their pairings with the planes x and y, over their
    weights."""
    weights = -horizon & points
    return ((stack((mv.x, mv.y)) & points[..., None]) / weights[..., None]).to_array()


def downhill(surface: Sphere, anchor: Point, length: float) -> Point:
    """The anchor carried down the surface's pairing with points, by the length times the pairing's rate of change.
    The plane tangent there to the level set of the pairing is the combination of the surface with the anchor as a
    sphere of radius zero that passes through infinity; its normal, wedged with the sphere at infinity, generates the
    translation along it."""
    point_sphere = anchor.dual()                                                # [...] Sphere+
    tangent = surface - (surface & infinity) / (point_sphere & infinity) * point_sphere   # [...] Sphere+
    return ((tangent ^ horizon) * (length / 2)).exp() >> anchor


def arrows(starts: Point, ends: Point, width: float, head: float, view: solids.View
           ) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows between points, their shafts `width` and their heads `head` pixels long."""
    return solids.arrows(gpu_points(starts), gpu_points(ends), width / 2 * view.pixel, head * view.pixel,
                         head_width / 2 * view.pixel)


def circle_planes(first: Point, second: Point, third: Point) -> solids.Plane:
    """The planes of circles through three of their sampled points each: the join of the three, made extrinsic by
    the intrinsic scalar one."""
    return (gpu_points(first) & gpu_points(second) & gpu_points(third)) * solids.flip


def panel(vertices: Point, circle: Point, globe: Point, plane: Sphere, sphere: Sphere,
          view: solids.View) -> np.ndarray:
    """One side of the mirror: the circle through its ordered points, on the plane's sheet and in the sphere's ball,
    with arrows down the plane's and the sphere's pairings with points: into their negative sides."""
    plane_colour, sphere_colour, curve_colour = (np.array(to_rgb(ink)) for ink in (plane_ink, sphere_ink, curve_ink))
    sheet = solids.plates(sheet_corners[None], sheet_thickness * view.pixel)  # [1, corners + 2] Plane+
    # The sheet's rim: its edges, where the half-spaces along them pair to zero; only the inner half of each line lies
    # on the sheet.
    edges = solids.half_space(sheet[0, 2:])                                     # [corners] Plane+ <- Point-
    # The ball the sphere's parallels and meridians lie on, about the centre a sphere's reflection sends infinity to;
    # the circles inked on it where their planes cut it.
    radius = (sphere.squared() / (sphere & infinity).squared()).square_root()   # [] Scalar+
    ball = solids.ball(gpu_points(sphere >> infinity), solids.context.include_from_algebra(radius))[None, None]
    n_parallels, n_meridians = globe.shape
    parallels = circle_planes(globe[:, 0], globe[:, n_meridians // 3], globe[:, 2 * n_meridians // 3])
    meridians = circle_planes(globe[0], globe[n_parallels // 3], globe[2 * n_parallels // 3])
    planes = concatenate([parallels, meridians])                                # [circles] Plane+
    spans = planes.scalar_norm_squared()
    planes = planes[spans > spanned * spans.max()]
    curve = strokes.polylines([(gpu_points(circle), curve_colour)], circle_width, view)

    # Short arrows follow the point order around the circle in both views.
    count = len(circle) - 1
    starts = np.arange(3) * count // 3 + count // 12
    chords = arrows(circle[starts], circle[starts + count // 18], chord_width, chord_head, view)
    # The arrows point down the surfaces' pairings with points: below the plane and into the sphere.
    plane_arrow = arrows(origin[None], downhill(plane, origin, plane_arrow_length)[None], plane_arrow_width,
                         arrow_head, view)
    sphere_anchor = circle[None, count // 3]
    sphere_arrow = arrows(sphere_anchor, downhill(sphere, sphere_anchor, sphere_arrow_length), sphere_arrow_width,
                          arrow_head, view)
    marks = [(chords[0], curve_colour), (chords[1], curve_colour),
             (plane_arrow[0], plane_colour), (plane_arrow[1], plane_colour),
             (sphere_arrow[0], sphere_colour), (sphere_arrow[1], sphere_colour),
             (solids.ball(gpu_points(vertices), dot_radius * view.pixel)[:, None],
              np.array([to_rgb(colour) for colour in marker_inks]))]
    # The sheet and the ball are the first and the second translucent parts, after the opaque marks.
    rim = solids.Ink(len(marks), edges, solids.everywhere[None].broadcast_to(edges.shape),
                     plane_colour, 2 * rim_width, rim_opacity)
    wireframe = solids.Ink(len(marks) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                           sphere_colour, wireframe_width, wireframe_opacity)
    return solids.draw(marks, view, background, [rim, wireframe],
                       translucent=[(sheet, plane_colour, sheet_opacity), (ball, sphere_colour, ball_opacity)],
                       meshes=[(curve, 1.0)])


def draw_circle(points: Point, curve: Point, surface: Point, planes: Sphere,
                spheres: Sphere) -> plt.Figure:
    """Both sides of the mirror, side by side, from the points, the circle, the sphere's samples, the planes and the
    spheres of each side."""
    # One view for both sides, framing the sheet and the sphere.
    view = solids.view([sheet_corners, gpu_points(surface)], 0.0, elevation, azimuth, margin, n_pixels)
    figure = plt.figure(figsize=(2 * n_pixels / 100, view.shape[0] / 100), dpi=100)
    for index in range(len(points)):
        image = panel(points[index], curve[index], surface[index], planes[index], spheres[index], view)
        show(figure.add_axes((index / 2, 0.0, 0.5, 1.0)), image, view)
    return figure


def animate_circle_motions(frames: Iterable[Point], fixed_points: Point) -> Iterator[np.ndarray]:
    """A coloured disk turns rigidly beside its reflected, conformally distorted image, from frames of the two disks
    with radius and angle axes."""
    frames = iter(frames)
    first = plane_coordinates(next(frames))                                     # [sides, radii, angles, 2]
    fixed = plane_coordinates(fixed_points)
    ring_indices = np.linspace(0, first.shape[1] - 1, 6, dtype=int)[1:-1]
    spoke_indices = np.linspace(0, first.shape[2] - 1, 13, dtype=int)[:-1]
    colours = plt.get_cmap("twilight_shifted")(np.linspace(0, 1, len(spoke_indices), endpoint=False))

    figure, axes = plt.subplots(1, 2, figsize=(2 * motion_pixels / motion_dpi, motion_pixels / motion_dpi),
                                dpi=motion_dpi)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, wspace=0.06)
    artists = []
    for ax, disk in zip(axes, first):
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
        ax.set_axis_off()
        ax.fill(*disk[-1].T, color="#f5f6f8", zorder=0)
        ax.plot(*disk[-1].T, color=curve_ink, linewidth=1.5, zorder=2)
        rings = LineCollection(disk[ring_indices], colors="#b8c1cc", linewidths=0.85)
        spokes = LineCollection(disk[:, spoke_indices].transpose(1, 0, 2),
                                colors=colours, linewidths=1.15, alpha=0.85)
        ax.add_collection(rings)
        ax.add_collection(spokes)
        dots = ax.scatter(*disk[-1, spoke_indices].T, c=colours, s=33,
                          edgecolors="white", linewidths=0.85, zorder=4)
        artists.append((rings, spokes, dots))
    axes[1].scatter(*fixed.T, s=94, facecolors="none", edgecolors=curve_ink,
                    linewidths=1.35, zorder=5)

    for pair in chain([first], (plane_coordinates(frame) for frame in frames)):
        for disk, (rings, spokes, dots) in zip(pair, artists):
            rings.set_segments(disk[ring_indices])
            spokes.set_segments(disk[:, spoke_indices].transpose(1, 0, 2))
            dots.set_offsets(disk[-1, spoke_indices])
        yield capture(figure)
    plt.close(figure)
