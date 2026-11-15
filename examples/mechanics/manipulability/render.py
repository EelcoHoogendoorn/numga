"""GPU ray-traced views of the articulated arm's opaque link boxes under translucent manipulability ellipsoids.

Each link box is the unit cube carried by its body placement map. Because a carried plane pairs with each
carried point just as the unit cube's face plane paired with the uncarried point, the world bounding planes
of the five link boxes come from `bodies.adjugate().solve(cube_planes)`.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.manipulability import core
from numga.render import solids
from examples.util.panels import room, show_pixels

mv, w = core.mv, core.w
box_colours = np.array([
    [0.45, 0.47, 0.52],
    [0.62, 0.64, 0.70],
    [0.78, 0.60, 0.30],
    [0.30, 0.55, 0.62],
    [0.55, 0.40, 0.60],
])
velocity_colour = np.array([0.25, 0.45, 0.85])
force_colour = np.array([0.92, 0.55, 0.20])
quadric_opacity = 0.33
background = np.ones(3)
# Outward-facing planes and corners of the unit cube centred on the origin.
cube_planes = solids.boxes(mv.rotor(), np.full(3, 0.5))                     # [6] Plane+
cube_corners = mv.point(list(product((-0.5, 0.5), repeat=3)))               # [8] Point-
panel_margin = 0.3


# --- plumbing -------------------------------------------------------------------------
def camera_angles(view_degrees: tuple[float, float]) -> tuple[float, float]:
    """Elevation and azimuth in radians, with azimuth converted from `+x` degrees to `solids`'s `+y` convention."""
    return np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90)


def framed_view(
    bodies: core.PointMap,
    surface: core.Quadric,
    view_degrees: tuple[float, float],
    n_pixels: int,
) -> solids.View:
    """Square camera view framing both the arm's link boxes and the translucent quadric `surface` across `n_pixels`."""
    elevation, azimuth = camera_angles(view_degrees)
    box_vertices = bodies[:, None](cube_corners)                            # [5, 8] Point-
    # Screen-right and screen-up planes shifted by `w * shifts` touch the quadric where `plane & dual_surface(plane) == 0`.
    aim_motor = solids.aim(solids.origin, elevation, azimuth)               # [] Motor+
    screen_planes = aim_motor >> stack((mv.x, mv.y))                        # [2] Plane+
    dual_surface = surface.inverse()                                        # [] Point- <- Plane+
    quadratic_a = w & dual_surface(w)                                       # [] Scalar+
    quadratic_b = screen_planes & dual_surface(w)                           # [2] Scalar+
    quadratic_c = screen_planes & dual_surface(screen_planes)               # [2] Scalar+
    discriminant_root = (quadratic_b.squared() - quadratic_a * quadratic_c).square_root()   # [2] Scalar+
    plane_shifts = stack([
        (-quadratic_b - discriminant_root) / quadratic_a,
        (-quadratic_b + discriminant_root) / quadratic_a,
    ])                                                                      # [2, 2] Scalar+
    tangent_points = dual_surface(screen_planes + w * plane_shifts)         # [2, 2] Point-
    return solids.view([box_vertices, tangent_points], 0.0, elevation, azimuth, panel_margin, n_pixels).squared()


def panel_image(
    bodies: core.PointMap,
    surface: core.Quadric,
    colour: np.ndarray,
    camera_view: solids.View,
) -> np.ndarray:
    """RGB byte image (`[n_rows, n_columns, 3]`) of the opaque link boxes under one translucent quadric `surface`."""
    world_boxes = bodies[:, None].adjugate().solve(cube_planes)             # [5, 6] Plane+
    return solids.draw(
        [(world_boxes, box_colours)],
        camera_view,
        background,
        inks=[],
        translucent=[(surface[None, None], colour, quadric_opacity)],
        meshes=[],
    )


def paired_panels(velocity_image: np.ndarray, force_image: np.ndarray) -> np.ndarray:
    """Horizontal pair placing `velocity_image` beside `force_image` separated by a twentieth-panel white gap."""
    gap = np.full((velocity_image.shape[0], velocity_image.shape[1] // 20, 3), 255, dtype=np.uint8)
    return np.concatenate([velocity_image, gap, force_image], axis=1)


def pose_frame(
    pose: tuple[core.PointMap, core.Point, core.Quadric, core.Quadric],
    view_degrees: tuple[float, float],
    workspace_centre: core.Point,
    workspace_extent: float,
    n_pixels: int,
) -> np.ndarray:
    """Single RGB animation frame showing the velocity and force ellipsoids side by side in a fixed workspace view."""
    bodies, _, velocity, force = pose
    camera_view = solids.View(
        solids.aim(workspace_centre, *camera_angles(view_degrees)),
        np.full(2, workspace_extent),
        n_pixels,
    )
    return paired_panels(
        panel_image(bodies, velocity, velocity_colour, camera_view),
        panel_image(bodies, force, force_colour, camera_view),
    )


def animate_poses(
    view_degrees: tuple[float, float],
    workspace_centre: core.Point,
    workspace_extent: float,
    n_pixels: int,
    poses: Iterable[tuple[core.PointMap, core.Point, core.Quadric, core.Quadric]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames across a stream of arm poses in a fixed workspace view."""
    for pose in poses:
        yield pose_frame(pose, view_degrees, workspace_centre, workspace_extent, n_pixels)


def ellipsoids_figure(
    poses: list[tuple[core.PointMap, core.Point, core.Quadric, core.Quadric]],
    view_degrees: tuple[float, float],
) -> plt.Figure:
    """Multi-row figure with one pose per row and individually framed velocity and force ellipsoid panels."""
    figure, axes = plt.subplots(len(poses), 1, figsize=(5.5, 2.75 * len(poses)))
    axes = np.atleast_1d(axes)
    columns, rows = room(axes[0])
    n_pixels = min(rows, columns * 20 // 41)
    for ax, (bodies, _, velocity, force) in zip(axes, poses):
        velocity_view = framed_view(bodies, velocity, view_degrees, n_pixels)
        force_view = framed_view(bodies, force, view_degrees, n_pixels)
        show_pixels(
            ax,
            paired_panels(
                panel_image(bodies, velocity, velocity_colour, velocity_view),
                panel_image(bodies, force, force_colour, force_view),
            ),
        )
    return figure
