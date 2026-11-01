"""Ordered points and the chosen sides of a circle's two cutting surfaces. The circle, its plane and its sphere
are drawn on the GPU: the plane a translucent sheet, the sphere a translucent ball with the circles of its sampled
wireframe inked on it, the points balls and the sides arrows; the circle's motions are drawn by matplotlib."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from numga import concatenate, stack
from examples.orientation.conformal_circle.core import Point, Sphere, mv
from examples.util.wgsl import contours, show, simplices, solids
from examples.util.wgsl.camera import Camera, panel as panel_camera
from examples.util.wgsl.meshes import Samples

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
# The widths of the circle, the sheet's rim and the wireframe, the plane's and the sphere's arrows, the length of the
# strokes of the chevrons on the circle and at the plane's and the sphere's arrows' tips, and the points' radius, in
# pixels; the sheet's thickness, in pixels.
circle_width, rim_width, wireframe_width, sheet_thickness = 2.4, 1.8, 1.8, 0.6
plane_arrow_width, sphere_arrow_width = 3.0, 2.7
chevron_length = 10.0
plane_chevron, sphere_chevron = 12.0, 10.8
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
    """The projective embedding of conformal points: each point's spatial part, its offset from the origin times its
    weight, over that weight, placed at the projective scene's origin."""
    # The line joining the origin and infinity; its meet with a point, wedged with the plane the two span, keeps the
    # point's spatial part, an antivector of space alone.
    origin_to_infinity = origin & infinity                                      # [] Antibivector-
    offsets = (points & origin_to_infinity) ^ origin_to_infinity.dual()         # [...] Antivector-
    return solids.origin + solids.context.include_from_algebra(offsets / (-horizon & points))


def downhill(surface: Sphere, anchor: Point, length: float) -> Point:
    """The anchor carried down the surface's pairing with points, by the length times the pairing's rate of change.
    The plane tangent there to the level set of the pairing is the combination of the surface with the anchor as a
    sphere of radius zero that passes through infinity; its normal, wedged with the sphere at infinity, generates the
    translation along it."""
    point_sphere = anchor.dual()                                                # [...] Sphere+
    tangent = surface - (surface & infinity) / (point_sphere & infinity) * point_sphere  # [...] Sphere+
    return ((tangent ^ horizon) * (length / 2)).exp() >> anchor


def arrows(starts: Point, ends: Point, colour: np.ndarray, width: float, chevron: float,
           view: solids.View) -> Samples:
    """Arrows between points `[arrows]`, tubes `width` pixels across ending in chevrons of strokes `chevron` pixels
    long, for `solids.draw` to compose among the solids."""
    n_arrows = starts.shape[0]
    return simplices.tubes(gpu_points(stack((starts, ends), axis=-1).reshape(-1)),
                           simplices.Topology.chains([2] * n_arrows), colour, width, view, gloss=solids.gloss,
                           chevrons=np.ones(n_arrows), chevron=chevron)


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
    # Chevrons on the circle follow the point order around it in both views: three of its edges, a third of the way
    # round from each other, end in one.
    count = len(circle)
    chevrons = np.zeros(count)
    chevrons[np.arange(3) * count // 3 + count // 12 + count // 18 - 1] = 1
    curve = simplices.tubes(gpu_points(circle), simplices.Topology.loops([count]), curve_colour, circle_width, view,
                            gloss=solids.gloss, chevrons=chevrons, chevron=chevron_length)

    # The arrows point down the surfaces' pairings with points: below the plane and into the sphere.
    plane_arrow = arrows(origin[None], downhill(plane, origin, plane_arrow_length)[None], plane_colour,
                         plane_arrow_width, plane_chevron, view)
    sphere_anchor = circle[None, count // 3]
    sphere_arrow = arrows(sphere_anchor, downhill(sphere, sphere_anchor, sphere_arrow_length), sphere_colour,
                          sphere_arrow_width, sphere_chevron, view)
    marks = [(solids.ball(gpu_points(vertices), dot_radius * view.pixel)[:, None],
              np.array([to_rgb(colour) for colour in marker_inks]))]
    # The sheet and the ball are the first and the second translucent parts, after the opaque marks.
    rim = solids.Ink(len(marks), edges, solids.everywhere[None].broadcast_to(edges.shape),
                     plane_colour, 2 * rim_width, rim_opacity)
    wireframe = solids.Ink(len(marks) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                           sphere_colour, wireframe_width, wireframe_opacity)
    return solids.draw(marks, view, background, [rim, wireframe],
                       translucent=[(sheet, plane_colour, sheet_opacity), (ball, sphere_colour, ball_opacity)],
                       meshes=[(curve, 1.0), (plane_arrow, 1.0), (sphere_arrow, 1.0)])


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
    with radius and angle axes, drawn on the GPU in the conformal algebra itself: a camera reading its points by the
    planes x and y over their weights, their pairings with the point at infinity."""
    frames = iter(frames)
    first = next(frames)                                                        # [sides, radii, angles] Point-
    n_radii, n_angles = first.shape[1:]
    ring_indices = np.linspace(0, n_radii - 1, 6, dtype=int)[1:-1]
    spoke_indices = np.linspace(0, n_angles, 12, endpoint=False, dtype=int)
    colours = plt.get_cmap("twilight_shifted")(np.linspace(0, 1, len(spoke_indices), endpoint=False))[:, :3]
    # The disk's faces: a closed loop round the angles, swept out along the radii.
    topology = simplices.Topology.loops([n_angles]).product(simplices.Topology.chains([n_radii]))
    rim_loop = simplices.Topology.loops([n_angles])
    ring_loops = simplices.Topology.loops([n_angles] * len(ring_indices))
    spoke_chains = simplices.Topology.chains([n_radii] * len(spoke_indices))

    figure, axes = plt.subplots(1, 2, figsize=(2 * motion_pixels / motion_dpi, motion_pixels / motion_dpi),
                                dpi=motion_dpi)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, wspace=0.06)
    for ax in axes:
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
        ax.set_axis_off()
    cameras = [panel_camera(ax, mv.x, mv.y)._replace(weight=-horizon, unit=0.0) for ax in axes]

    def pictures(disk: Point, camera: Camera, fixed: list[np.ndarray]) -> list[np.ndarray]:
        """The disk `[radii, angles]` filled, its rings and its spokes, each spoke in its colour, its rim, the
        pictures `fixed`, and the spokes' ends on it."""
        # Each spoke's places from the middle out, a chain of them.
        spokes = disk[np.arange(n_radii), spoke_indices[:, None]].reshape(-1)  # [spokes * radii] Point-
        ends = disk[-1, spoke_indices]                                           # [spokes] Point-
        return [contours.picture(disk.reshape(-1), topology, np.zeros(n_radii * n_angles), to_rgb("#f5f6f8"), camera),
                simplices.strokes(disk[ring_indices].reshape(-1), ring_loops, to_rgb("#b8c1cc"), 1.0, camera),
                simplices.strokes(spokes, spoke_chains, np.repeat(colours, n_radii - 1, axis=0), 1.2, camera,
                                  opacity=0.85),
                simplices.strokes(disk[-1], rim_loop, to_rgb(curve_ink), 1.5, camera),
                *fixed,
                simplices.markers(ends, background, 2.3, camera),
                simplices.markers(ends, colours, 1.8, camera)]

    # The reflection's fixed points, ringed on the image's side: a white dot over a larger one, under the dots of the
    # spokes' ends.
    marks = ([], [simplices.markers(fixed_points, to_rgb(curve_ink), 3.0, cameras[1]),
                  simplices.markers(fixed_points, background, 2.0, cameras[1])])
    images = [simplices.overlay(ax, pictures(disk, camera, fixed))
              for ax, disk, camera, fixed in zip(axes, first, cameras, marks)]
    for pair in chain([first], frames):
        for image, disk, camera, fixed in zip(images, pair, cameras, marks):
            image.set_data(simplices.composited(pictures(disk, camera, fixed)))
        yield capture(figure)
    plt.close(figure)
