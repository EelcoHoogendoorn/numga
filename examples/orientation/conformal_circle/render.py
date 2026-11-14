"""GPU 3D surface panels and conformal 2D disk animations display circle orientations and circle-preserving motions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.util.animation import capture
from numga import concatenate, stack
from examples.orientation.conformal_circle.core import Point, Sphere, mv
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices, solids
from examples.util.panels import overlay, panel as panel_camera, show
from numga.render.drawing import Draw

sm = solids.mv
plane_ink, sphere_ink, curve_ink = "#218c9d", "#ce713f", "#3d354a"
marker_inks = ("#9861ba", "#dd9650", "#4f9b85")
background = np.ones(3)
# Camera elevation and azimuth in radians, viewport margin fraction, and pixel width per panel.
elevation, azimuth = np.radians(26.0), np.radians(32.0)
margin, n_pixels = 0.25, 300
# Square plane sheet corners in projective 3D space and surface/wireframe opacities.
sheet_corners = sm.point(1.23 * np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]]))  # [4] Point-
sheet_opacity, rim_opacity, ball_opacity, wireframe_opacity = 0.073, 0.45, 0.062, 0.26
# Stroke widths, chevron stroke lengths, marker radius, and sheet thickness in pixels.
circle_width, rim_width, wireframe_width, sheet_thickness = 2.4, 1.8, 1.8, 0.6
plane_arrow_width, sphere_arrow_width = 3.0, 2.7
chevron_length = 10.0
plane_chevron, sphere_chevron = 12.0, 10.8
dot_radius = 4.8
# Drawn lengths of plane and sphere normal arrows per unit surface normal.
plane_arrow_length, sphere_arrow_length = 0.85, 0.45
# Relative squared-norm threshold filtering degenerate pole circles from the sphere wireframe.
spanned = 1e-12
# Pixel width per motion panel and reference DPI for stroke widths.
motion_pixels, motion_dpi = 150, 45
# Null point at infinity, conformal origin, and unit-weight horizon sphere.
infinity = mv.e.dual() + mv.w.dual()                                            # [] Point-
origin = (mv.e.dual() - mv.w.dual()) / 2                                        # [] Point-
horizon = mv.e + mv.w                                                           # [] Sphere+


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: Point) -> solids.Point:
    """Projecting normalized conformal `points` (`[...] Point-`) along `origin & infinity` embeds them in `solids`."""
    origin_to_infinity = origin & infinity                                      # [] Antibivector-
    offsets = (points & origin_to_infinity) ^ origin_to_infinity.dual()         # [...] Point-
    return solids.origin + solids.context.include_from_algebra(offsets / (-horizon & points))


def downhill_point(surface: Sphere, anchor: Point, length: float) -> Point:
    """Translating `anchor` (`[...] Point-`) by `length` into the negative side of `surface` (`[...] Sphere+`)
    places the tip (`[...] Point-`) of its orientation arrow."""
    point_sphere = anchor.dual()                                                # [...] Sphere+
    tangent = surface - (surface & infinity) / (point_sphere & infinity) * point_sphere  # [...] Sphere+
    shift = ((tangent ^ horizon) * (length / 2)).exp()                          # [...] Rotor+
    return shift >> anchor                                                      # [...] Point-


def surface_arrows(starts: Point, ends: Point, colour: np.ndarray, width: float, chevron: float,
                   view: solids.View) -> Draw:
    """Tube arrows between `starts` and `ends` (`[n_arrows] Point-`) ending in chevrons for `solids.draw`."""
    n_arrows = starts.shape[0]
    return simplices.tubes(gpu_points(stack((starts, ends), axis=-1).reshape(-1)),
                           Topology.chains([2] * n_arrows), colour, width, view, gloss=solids.gloss,
                           chevrons=np.ones(n_arrows), chevron=chevron)


def circle_planes(first: Point, second: Point, third: Point) -> solids.Plane:
    """Extrinsic planes (`[...] Plane+`) spanned by triples of sampled circle points in `solids`."""
    return (gpu_points(first) & gpu_points(second) & gpu_points(third)) * solids.flip


def side_panel(vertices: Point, circle: Point, globe: Point, plane: Sphere, sphere: Sphere,
               view: solids.View) -> np.ndarray:
    """Rendered RGB image of one side of the mirror showing the ordered circle, plane sheet, sphere, and normal arrows."""
    plane_colour, sphere_colour, curve_colour = (np.array(to_rgb(ink)) for ink in (plane_ink, sphere_ink, curve_ink))
    sheet = solids.plates(sheet_corners[None], sheet_thickness * view.pixel)    # [1, 6] Plane+
    edges = solids.half_space(sheet[0, 2:])                                     # [4] Plane+ <- Point-
    radius = (sphere.squared() / (sphere & infinity).squared()).square_root()   # [] Scalar+
    ball = solids.ball(gpu_points(sphere >> infinity), solids.context.include_from_algebra(radius))[None, None]
    n_parallels, n_meridians = globe.shape
    parallels = circle_planes(globe[:, 0], globe[:, n_meridians // 3], globe[:, 2 * n_meridians // 3])
    meridians = circle_planes(globe[0], globe[n_parallels // 3], globe[2 * n_parallels // 3])
    planes = concatenate([parallels, meridians])                                # [n_parallels + n_meridians] Plane+
    spans = planes.scalar_norm_squared()                                        # [n_parallels + n_meridians] Scalar+
    planes = planes[spans > spanned * spans.max()]

    n_curve = len(circle)
    chevrons = np.zeros(n_curve)
    chevrons[np.arange(3) * n_curve // 3 + n_curve // 12 + n_curve // 18 - 1] = 1
    curve = simplices.tubes(gpu_points(circle), Topology.loops([n_curve]), curve_colour, circle_width, view,
                            gloss=solids.gloss, chevrons=chevrons, chevron=chevron_length)

    plane_arrow = surface_arrows(origin[None], downhill_point(plane, origin, plane_arrow_length)[None], plane_colour,
                                 plane_arrow_width, plane_chevron, view)
    sphere_anchor = circle[None, n_curve // 3]                                  # [1] Point-
    sphere_arrow = surface_arrows(sphere_anchor, downhill_point(sphere, sphere_anchor, sphere_arrow_length),
                                  sphere_colour, sphere_arrow_width, sphere_chevron, view)
    marks = [(solids.ball(gpu_points(vertices), dot_radius * view.pixel)[:, None],
              np.array([to_rgb(colour) for colour in marker_inks]))]
    rim = solids.Ink(len(marks), edges, solids.everywhere[None].broadcast_to(edges.shape),
                     plane_colour, 2 * rim_width, rim_opacity)
    wireframe = solids.Ink(len(marks) + 1, solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape),
                           sphere_colour, wireframe_width, wireframe_opacity)
    return solids.draw(marks, view, background, [rim, wireframe],
                       translucent=[(sheet, plane_colour, sheet_opacity), (ball, sphere_colour, ball_opacity)],
                       meshes=[(curve, 1.0), (plane_arrow, 1.0), (sphere_arrow, 1.0)])


def circle_figure(points: Point, curves: Point, surfaces: Point, planes: Sphere,
                  spheres: Sphere) -> plt.Figure:
    """Side-by-side 3D panels comparing the conformal circle construction (`[2, ...]`) before and after reflection."""
    view = solids.view([sheet_corners, gpu_points(surfaces)], 0.0, elevation, azimuth, margin, n_pixels)
    figure = plt.figure(figsize=(2 * n_pixels / 100, view.shape[0] / 100), dpi=100)
    for index in range(len(points)):
        image = side_panel(points[index], curves[index], surfaces[index], planes[index], spheres[index], view)
        show(figure.add_axes((index / 2, 0.0, 0.5, 1.0)), image, view)
    return figure


def circle_motions_animation(fixed_points: Point, states: Iterable[Point]) -> Iterator[np.ndarray]:
    """Yields side-by-side `[rows, columns, 3]` RGB frames of rigid disk rotation and hyperbolic glide reflection
    from `states` (`[2, n_radii, n_angles] Point-`), marking `fixed_points` (`[2] Point-`) on the glide panel."""
    iterator = iter(states)
    first = next(iterator)                                                      # [2, n_radii, n_angles] Point-
    n_radii, n_angles = first.shape[1:]
    ring_indices = np.linspace(0, n_radii - 1, 6, dtype=int)[1:-1]
    spoke_indices = np.linspace(0, n_angles, 12, endpoint=False, dtype=int)
    n_spokes = len(spoke_indices)
    colours = plt.get_cmap("twilight_shifted")(np.linspace(0, 1, n_spokes, endpoint=False))[:, :3]
    topology = Topology.loops([n_angles]).product(Topology.chains([n_radii]))
    rim_loop = Topology.loops([n_angles])
    ring_loops = Topology.loops([n_angles] * len(ring_indices))
    spoke_chains = Topology.chains([n_radii] * n_spokes)

    figure, axes = plt.subplots(1, 2, figsize=(2 * motion_pixels / motion_dpi, motion_pixels / motion_dpi),
                                dpi=motion_dpi)
    figure.subplots_adjust(left=0.015, right=0.985, bottom=0.025, top=0.975, wspace=0.06)
    for ax in axes:
        ax.set(xlim=(-1.12, 1.12), ylim=(-1.12, 1.12), aspect="equal")
        ax.set_axis_off()
    cameras = [panel_camera(ax, mv.x, mv.y, -horizon) for ax in axes]

    def disk_layers(disk: Point, camera: Camera, fixed: list[np.ndarray]) -> list[np.ndarray]:
        """Composited fill, ring, spoke, rim, and marker layers for `disk` (`[n_radii, n_angles] Point-`)."""
        spokes = disk[np.arange(n_radii), spoke_indices[:, None]].reshape(-1)   # [n_spokes * n_radii] Point-
        spoke_ends = disk[-1, spoke_indices]                                    # [n_spokes] Point-
        return [contours.picture(disk.reshape(-1), topology, np.zeros(n_radii * n_angles), to_rgb("#f5f6f8"), camera),
                simplices.strokes(disk[ring_indices].reshape(-1), ring_loops, to_rgb("#b8c1cc"), 1.0, camera),
                simplices.strokes(spokes, spoke_chains, np.repeat(colours, n_radii - 1, axis=0), 1.2, camera,
                                  opacity=0.85),
                simplices.strokes(disk[-1], rim_loop, to_rgb(curve_ink), 1.5, camera),
                *fixed,
                simplices.markers(spoke_ends, background, 2.3, camera),
                simplices.markers(spoke_ends, colours, 1.8, camera)]

    marks = ([], [simplices.markers(fixed_points, to_rgb(curve_ink), 3.0, cameras[1]),
                  simplices.markers(fixed_points, background, 2.0, cameras[1])])
    images = [overlay(ax, disk_layers(disk, camera, fixed))
              for ax, disk, camera, fixed in zip(axes, first, cameras, marks)]
    for pair in chain([first], iterator):
        for image, disk, camera, fixed in zip(images, pair, cameras, marks):
            image.set_data(drawing.composited(disk_layers(disk, camera, fixed)))
        yield capture(figure)
    plt.close(figure)
