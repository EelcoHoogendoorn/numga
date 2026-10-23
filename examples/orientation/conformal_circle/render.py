"""Ordered points and the chosen sides of a circle's two cutting surfaces. The circle, its plane and its sphere
are drawn on the GPU: the plane a translucent sheet, the sphere a translucent ball with the circles of its sampled
wireframe inked on it, the points balls and the sides arrows; the circle's motions are drawn by matplotlib."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from numga import concatenate
from examples.orientation.conformal_circle.core import Point, Sphere
from examples.util.wgsl import meshes, show, solids, strokes

sm = solids.mv
PLANE = "#218c9d"
SPHERE = "#ce713f"
CURVE = "#3d354a"
MARKERS = ("#9861ba", "#dd9650", "#4f9b85")
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the scene as a share
# of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(26.0), np.radians(32.0)
MARGIN, N_PIXELS = 0.25, 500
# The plane's sheet, a square about the origin; the opacities of each face of the sheet, its rim, each face of the
# ball and the sphere's wireframe, front and back alike.
SHEET = 1.23 * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
SHEET_OPACITY, RIM_OPACITY, BALL_OPACITY, WIREFRAME_OPACITY = 0.073, 0.45, 0.062, 0.26
# The widths of the circle, the sheet's rim, the wireframe and the chord arrows, the plane's and the sphere's
# arrows, in pixels, the arrowheads' lengths and widths, and the points' radius, in pixels; the sheet's thickness,
# in pixels.
CIRCLE_WIDTH, RIM_WIDTH, WIREFRAME_WIDTH, SHEET_THICKNESS = 4.0, 3.0, 3.0, 1.0
CHORD_WIDTH, PLANE_ARROW_WIDTH, SPHERE_ARROW_WIDTH = 3.5, 5.0, 4.5
CHORD_HEAD, ARROW_HEAD, HEAD_WIDTH = 22.0, 22.0, 16.0
DOT_RADIUS = 8.0
# The lengths of the plane's and the sphere's arrows per unit of their normals.
PLANE_ARROW_LENGTH, SPHERE_ARROW_LENGTH = 0.85, 0.45
# How small against the largest the plane a sampled circle's points span may be and still count as one: a circle
# gathered into one point, as a parallel at a pole, spans none.
SPANNED = 1e-12


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    values = points.dual().cast(points.algebra.subspace("x y z w e")).kernel
    return values[..., :3] / (values[..., 4:] - values[..., 3:4])


def arrows(starts: np.ndarray, ends: np.ndarray, width: float, head: float, view: solids.View
           ) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows between places `[..., 3]`, their shafts `width` and their heads `head` pixels long."""
    return solids.arrows(sm.point(starts), sm.point(ends), width / 2 * view.pixel, head * view.pixel,
                         HEAD_WIDTH / 2 * view.pixel)


def circle_planes(lines: np.ndarray) -> solids.Plane:
    """The planes of circles sampled along lines `[lines, n, 3]`, each through its first sample and those a third and
    two thirds along it: the join of the three, made extrinsic by the intrinsic unit."""
    count = lines.shape[1]
    first, second, third = (sm.point(lines[:, index]) for index in (0, count // 3, 2 * count // 3))
    return (first & second & third) * meshes.unit


def panel(vertices: np.ndarray, circle: np.ndarray, globe: np.ndarray, plane: np.ndarray, sphere: np.ndarray,
          view: solids.View) -> np.ndarray:
    """One side of the mirror: the circle through its ordered points, on the plane's sheet and in the sphere,
    with arrows into the positive sides of the plane and the sphere."""
    plane_colour, sphere_colour, curve_colour = (np.array(to_rgb(colour)) for colour in (PLANE, SPHERE, CURVE))
    sheet = solids.plates(SHEET[None], SHEET_THICKNESS * view.pixel)        # [1, corners + 2] Quadric
    # The sheet's rim: its edges, where the half-spaces along them pair to zero; only the inner half of each line lies
    # on the sheet.
    edges = sheet[0, 2:]                                                      # [corners] Quadric
    # The ball the sphere's parallels and meridians lie on: its centre and radius from the sphere's coefficients, as
    # the centre's coordinates and squared radius over its weight; the circles inked on it where their planes cut it.
    weight = sphere[4] - sphere[3]
    centre = sphere[:3] / weight
    ball = solids.ball(sm.point(centre), np.sqrt(centre @ centre - (sphere[3] + sphere[4]) / weight))[None, None]
    planes = concatenate([circle_planes(globe), circle_planes(globe.transpose(1, 0, 2))])   # [circles] Plane
    spans = (planes | planes).to_array()
    planes = planes[spans > SPANNED * spans.max()]
    curve = strokes.polylines([(circle, curve_colour)], CIRCLE_WIDTH, view)

    # Short arrows follow the point order around the circle in both views.
    count = len(circle) - 1
    starts = np.arange(3) * count // 3 + count // 12
    chords = arrows(circle[starts], circle[starts + count // 18], CHORD_WIDTH, CHORD_HEAD, view)
    # The arrows point into the positive sides of the incidence functions.
    plane_arrow = arrows(np.zeros((1, 3)), -plane[None, :3] * PLANE_ARROW_LENGTH, PLANE_ARROW_WIDTH, ARROW_HEAD, view)
    sphere_anchor = circle[None, count // 3]
    sphere_direction = -sphere[:3] - (sphere[3] - sphere[4]) * sphere_anchor
    sphere_arrow = arrows(sphere_anchor, sphere_anchor + sphere_direction * SPHERE_ARROW_LENGTH, SPHERE_ARROW_WIDTH,
                          ARROW_HEAD, view)
    marks = [(chords[0], curve_colour), (chords[1], curve_colour),
             (plane_arrow[0], plane_colour), (plane_arrow[1], plane_colour),
             (sphere_arrow[0], sphere_colour), (sphere_arrow[1], sphere_colour),
             (solids.ball(sm.point(vertices), DOT_RADIUS * view.pixel)[:, None],
              np.array([to_rgb(colour) for colour in MARKERS]))]
    # The sheet and the ball are the first and the second translucent parts, after the opaque marks.
    rim = solids.Ink(len(marks), edges, solids.everywhere[None].broadcast_to(edges.shape),
                     plane_colour, 2 * RIM_WIDTH, RIM_OPACITY)
    wireframe = solids.Ink(len(marks) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                           sphere_colour, WIREFRAME_WIDTH, WIREFRAME_OPACITY)
    return solids.draw(marks, view, background, [rim, wireframe],
                       translucent=[(sheet, plane_colour, SHEET_OPACITY), (ball, sphere_colour, BALL_OPACITY)],
                       meshes=[(curve, 1.0)])


def draw_circle(points: Point, curve: Point, surface: Point, planes: Sphere,
                spheres: Sphere) -> plt.Figure:
    vertices = coordinates(points)
    curves = coordinates(curve)
    surfaces = coordinates(surface)
    plane_values = planes.cast(planes.algebra.subspace("x y z w e")).kernel
    sphere_values = spheres.cast(spheres.algebra.subspace("x y z w e")).kernel

    # One view for both sides, framing the sheet and the sphere.
    view = solids.view([sm.point(SHEET), sm.point(surfaces)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    figure = plt.figure(figsize=(2 * N_PIXELS / 100, view.shape[0] / 100), dpi=100)
    for index in range(len(vertices)):
        image = panel(vertices[index], curves[index], surfaces[index], plane_values[index], sphere_values[index], view)
        show(figure.add_axes((index / 2, 0.0, 0.5, 1.0)), image, view)
    return figure


def animate_circle_motions(grid: Point, fixed_points: Point) -> Iterator[np.ndarray]:
    """A coloured disk turns rigidly beside its reflected, conformally distorted image."""
    positions = coordinates(grid)[..., :2]
    fixed = coordinates(fixed_points)[..., :2]
    ring_indices = np.linspace(0, positions.shape[2] - 1, 6, dtype=int)[1:-1]
    spoke_indices = np.linspace(0, positions.shape[3] - 1, 13, dtype=int)[:-1]
    colours = plt.get_cmap("twilight_shifted")(np.linspace(0, 1, len(spoke_indices), endpoint=False))

    figure, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=90)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, wspace=0.06)
    artists = []
    for ax, disk in zip(axes, positions[0]):
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
        ax.set_axis_off()
        ax.fill(*disk[-1].T, color="#f5f6f8", zorder=0)
        ax.plot(*disk[-1].T, color=CURVE, linewidth=1.8, zorder=2)
        rings = LineCollection(disk[ring_indices], colors="#b8c1cc", linewidths=1)
        spokes = LineCollection(disk[:, spoke_indices].transpose(1, 0, 2),
                                colors=colours, linewidths=1.4, alpha=0.85)
        ax.add_collection(rings)
        ax.add_collection(spokes)
        dots = ax.scatter(*disk[-1, spoke_indices].T, c=colours, s=48,
                          edgecolors="white", linewidths=1, zorder=4)
        artists.append((rings, spokes, dots))
    axes[1].scatter(*fixed.T, s=135, facecolors="none", edgecolors=CURVE,
                    linewidths=1.6, zorder=5)

    for pair in positions:
        for disk, (rings, spokes, dots) in zip(pair, artists):
            rings.set_segments(disk[ring_indices])
            spokes.set_segments(disk[:, spoke_indices].transpose(1, 0, 2))
            dots.set_offsets(disk[-1, spoke_indices])
        yield capture(figure)
    plt.close(figure)
