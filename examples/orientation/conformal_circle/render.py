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
from examples.orientation.conformal_circle.core import Point, Sphere
from examples.util.wgsl import show, solids, strokes

sm = solids.mv
PLANE = "#218c9d"
SPHERE = "#ce713f"
CURVE = "#3d354a"
MARKERS = ("#9861ba", "#dd9650", "#4f9b85")
background = np.ones(3)
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the scene as a share
# of the larger side, and the pixels across each panel.
ELEVATION, AZIMUTH = np.radians(26.0), np.radians(32.0)
MARGIN, N_PIXELS = 0.25, 300
# The plane's sheet, a square about the origin; the opacities of each face of the sheet, its rim, each face of the
# ball and the sphere's wireframe, front and back alike.
SHEET = 1.23 * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
SHEET_OPACITY, RIM_OPACITY, BALL_OPACITY, WIREFRAME_OPACITY = 0.073, 0.45, 0.062, 0.26
# The widths of the circle, the sheet's rim, the wireframe and the chord arrows, the plane's and the sphere's
# arrows, in pixels, the arrowheads' lengths and widths, and the points' radius, in pixels; the sheet's thickness,
# in pixels.
CIRCLE_WIDTH, RIM_WIDTH, WIREFRAME_WIDTH, SHEET_THICKNESS = 2.4, 1.8, 1.8, 0.6
CHORD_WIDTH, PLANE_ARROW_WIDTH, SPHERE_ARROW_WIDTH = 2.1, 3.0, 2.7
CHORD_HEAD, ARROW_HEAD, HEAD_WIDTH = 13.0, 13.0, 9.6
DOT_RADIUS = 4.8
# The lengths of the plane's and the sphere's arrows per unit of their normals.
PLANE_ARROW_LENGTH, SPHERE_ARROW_LENGTH = 0.85, 0.45
# How small against the largest the plane a sampled circle's points span may be and still count as one: a circle
# gathered into one point, as a parallel at a pole, spans none.
SPANNED = 1e-12
# The pixels across each panel of the motions, and the dots per inch their line widths are given at.
MOTION_PIXELS, MOTION_DPI = 300, 90


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: Point) -> np.ndarray:
    """x, y and z of points, on the last axis: their pairings with the planes x, y and z, over their weights, which
    the sphere at infinity reads."""
    mv = points.context.multivector
    weights = -(mv.e + mv.w) & points
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / weights[..., None]).to_array()


def arrows(starts: np.ndarray, ends: np.ndarray, width: float, head: float, view: solids.View
           ) -> tuple[solids.Quadric, solids.Quadric]:
    """Arrows between places `[..., 3]`, their shafts `width` and their heads `head` pixels long."""
    return solids.arrows(sm.point(starts), sm.point(ends), width / 2 * view.pixel, head * view.pixel,
                         HEAD_WIDTH / 2 * view.pixel)


def circle_planes(lines: np.ndarray) -> solids.Plane:
    """The planes of circles sampled along lines `[lines, n, 3]`, each through its first sample and those a third and
    two thirds along it: the join of the three, made extrinsic by the intrinsic scalar one."""
    count = lines.shape[1]
    first, second, third = (sm.point(lines[:, index]) for index in (0, count // 3, 2 * count // 3))
    return (first & second & third) * solids.flip


def panel(vertices: np.ndarray, circle: np.ndarray, globe: np.ndarray, plane_normal: np.ndarray,
          sphere_normal: np.ndarray, sphere_weight: float, centre: np.ndarray, radius: float,
          view: solids.View) -> np.ndarray:
    """One side of the mirror: the circle through its ordered points, on the plane's sheet and in the sphere's ball
    `centre` and `radius`, with arrows into the negative sides of the plane and the sphere, whose pairings with the
    point at a place grow along `normal + weight * place`."""
    plane_colour, sphere_colour, curve_colour = (np.array(to_rgb(colour)) for colour in (PLANE, SPHERE, CURVE))
    sheet = solids.plates(sm.point(SHEET)[None], SHEET_THICKNESS * view.pixel)  # [1, corners + 2] Plane+
    # The sheet's rim: its edges, where the half-spaces along them pair to zero; only the inner half of each line lies
    # on the sheet.
    edges = solids.half_space(sheet[0, 2:])                                     # [corners] Plane+ <- Point-
    # The ball the sphere's parallels and meridians lie on; the circles inked on it where their planes cut it.
    ball = solids.ball(sm.point(centre), radius)[None, None]
    planes = concatenate([circle_planes(globe), circle_planes(globe.transpose(1, 0, 2))])   # [circles] Plane+
    spans = (planes | planes).to_array()
    planes = planes[spans > SPANNED * spans.max()]
    curve = strokes.polylines([(sm.point(circle), curve_colour)], CIRCLE_WIDTH, view)

    # Short arrows follow the point order around the circle in both views.
    count = len(circle) - 1
    starts = np.arange(3) * count // 3 + count // 12
    chords = arrows(circle[starts], circle[starts + count // 18], CHORD_WIDTH, CHORD_HEAD, view)
    # The arrows point down the surfaces' pairings with places: below the plane and into the sphere.
    plane_arrow = arrows(np.zeros((1, 3)), -plane_normal[None] * PLANE_ARROW_LENGTH, PLANE_ARROW_WIDTH, ARROW_HEAD,
                         view)
    sphere_anchor = circle[None, count // 3]
    sphere_direction = -(sphere_normal + sphere_weight * sphere_anchor)
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
    """Both sides of the mirror, side by side, from the points, the circle, the sphere's samples, the planes and the
    spheres of each side."""
    mv = points.context.multivector
    infinity = mv.e.dual() + mv.w.dual()
    directions = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    vertices = coordinates(points)
    curves = coordinates(curve)
    surfaces = coordinates(surface)
    # A surface's pairing with the point at a place grows along its normal, its pairing with the directions, plus its
    # weight, its pairing with infinity, times the place.
    plane_normals = (directions & planes[..., None]).to_array()                 # [sides, 3]
    sphere_normals = (directions & spheres[..., None]).to_array()               # [sides, 3]
    sphere_weights = (spheres & infinity).to_array()                            # [sides]
    # A sphere's reflection sends infinity to its centre.
    centres = coordinates(spheres >> infinity)                                  # [sides, 3]
    radii = np.sqrt((spheres.squared() / (spheres & infinity).squared()).to_array())   # [sides]

    # One view for both sides, framing the sheet and the sphere.
    view = solids.view([sm.point(SHEET), sm.point(surfaces)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS)
    figure = plt.figure(figsize=(2 * N_PIXELS / 100, view.shape[0] / 100), dpi=100)
    for index in range(len(vertices)):
        image = panel(vertices[index], curves[index], surfaces[index], plane_normals[index], sphere_normals[index],
                      sphere_weights[index], centres[index], radii[index], view)
        show(figure.add_axes((index / 2, 0.0, 0.5, 1.0)), image, view)
    return figure


def animate_circle_motions(frames: Iterable[Point], fixed_points: Point) -> Iterator[np.ndarray]:
    """A coloured disk turns rigidly beside its reflected, conformally distorted image, from frames of the two disks
    with radius and angle axes."""
    frames = iter(frames)
    first = coordinates(next(frames))[..., :2]                                  # [sides, radii, angles, 2]
    fixed = coordinates(fixed_points)[..., :2]
    ring_indices = np.linspace(0, first.shape[1] - 1, 6, dtype=int)[1:-1]
    spoke_indices = np.linspace(0, first.shape[2] - 1, 13, dtype=int)[:-1]
    colours = plt.get_cmap("twilight_shifted")(np.linspace(0, 1, len(spoke_indices), endpoint=False))

    figure, axes = plt.subplots(1, 2, figsize=(2 * MOTION_PIXELS / MOTION_DPI, MOTION_PIXELS / MOTION_DPI),
                                dpi=MOTION_DPI)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, wspace=0.06)
    artists = []
    for ax, disk in zip(axes, first):
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
        ax.set_axis_off()
        ax.fill(*disk[-1].T, color="#f5f6f8", zorder=0)
        ax.plot(*disk[-1].T, color=CURVE, linewidth=1.5, zorder=2)
        rings = LineCollection(disk[ring_indices], colors="#b8c1cc", linewidths=0.85)
        spokes = LineCollection(disk[:, spoke_indices].transpose(1, 0, 2),
                                colors=colours, linewidths=1.15, alpha=0.85)
        ax.add_collection(rings)
        ax.add_collection(spokes)
        dots = ax.scatter(*disk[-1, spoke_indices].T, c=colours, s=33,
                          edgecolors="white", linewidths=0.85, zorder=4)
        artists.append((rings, spokes, dots))
    axes[1].scatter(*fixed.T, s=94, facecolors="none", edgecolors=CURVE,
                    linewidths=1.35, zorder=5)

    for pair in chain([first], (coordinates(frame)[..., :2] for frame in frames)):
        for disk, (rings, spokes, dots) in zip(pair, artists):
            rings.set_segments(disk[ring_indices])
            spokes.set_segments(disk[:, spoke_indices].transpose(1, 0, 2))
            dots.set_offsets(disk[-1, spoke_indices])
        yield capture(figure)
    plt.close(figure)
