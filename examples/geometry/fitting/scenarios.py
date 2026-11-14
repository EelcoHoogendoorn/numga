"""Scenes for projective primitive fitting sample a point cloud, a line segment, a planar patch, and a line bundle in a shared pose and fit the corresponding primitive to each."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.fitting.core import Line, Plane, Point, fitted_primitive, mv

# The pose every scene is placed by: a shift, after turns in the yz and xy planes.
shift = np.array([0.4, -0.3, 0.6])
tilt, spin = 0.3, 0.5
pose_shift = ((stack((mv.xw, mv.yw, mv.zw)) * shift).sum(axis=0)).exp()
pose = pose_shift * (mv.yz * tilt).exp() * (mv.xy * spin).exp()                                                     # [] Motor+
origin = mv.w.dual()                                                                                               # Antivector-

# The seed of every scene's samples, and the size of the noise that moves each sample point.
seed = 0
noise_sigma = 0.05
# A cloud of points about the pose's origin.
n_cloud, cloud_spread = 200, 0.5
# Points along a segment of the pose's y axis, and the half length the lines are drawn to.
n_segment, half_length, drawn_half_length = 120, 2.0, 2.5
# Points on a square of the pose's z == 0 plane.
n_patch, half_width = 200, 2.0
# Lines through points near the pose's origin, and the half length they are drawn to.
n_rays, ray_spread, ray_half_length = 30, 0.05, 2.0


# --- plumbing -------------------------------------------------------------------------
def point_cloud(n_points: int, spread: float, rng: np.random.Generator) -> Point:
    """Normal offsets (`[n_points] Point-`) scatter a point cloud around the origin."""
    return mv.point(rng.normal(scale=spread, size=(n_points, 3)))


def line_segment(n_points: int, span_half_length: float) -> Point:
    """Evenly spaced points (`[n_points] Point-`) sample a segment along the y axis."""
    return origin + mv.direction([0, 1, 0]) * np.linspace(-span_half_length, span_half_length, n_points)


def planar_patch(n_points: int, span_half_width: float, rng: np.random.Generator) -> Point:
    """Uniform offsets (`[n_points] Point-`) sample a square patch in the z == 0 plane."""
    in_plane = mv.direction([[1, 0, 0], [0, 1, 0]])                                     # [2] Point-
    offsets = rng.uniform(-span_half_width, span_half_width, size=(n_points, 2))
    return origin + (in_plane * offsets).sum(axis=-1)


def noisy_points(points: Point, sigma: float, rng: np.random.Generator) -> Point:
    """Independent Gaussian translations (`[...] Point-`) perturb each point in `points`."""
    noise = rng.normal(scale=sigma, size=(*points.shape, 3))
    translations = ((stack((mv.xw, mv.yw, mv.zw)) * noise).sum(axis=-1) / 2).exp()     # [...] Motor+
    return translations >> points


def ray_bundle(n_lines: int, spread: float, rng: np.random.Generator) -> Line:
    """Isotropic lines (`[n_lines] Line-`) pass within `spread` of the origin with unit directions."""
    directions = mv.direction(rng.normal(size=(n_lines, 3)))                            # [n_lines] Point-
    feet = mv.point(rng.normal(scale=spread, size=(n_lines, 3)))                        # [n_lines] Point-
    # A direction has no weight; its length is the norm of its complement.
    return feet & (directions / directions.dual().norm())


def segment_end_planes(span_half_length: float) -> Plane:
    """Two cap planes (`[2] Plane+`) clip a y-axis line at plus and minus `span_half_length`."""
    return mv.y - mv.w * np.array([span_half_length, -span_half_length])


def square_patch_edges(span_half_width: float) -> Line:
    """Four boundary lines (`[4] Line-`) frame a square of half-width `span_half_width` in the z == 0 plane."""
    x_planes = mv.x - mv.w * (np.array([1, 1, -1, -1]) * span_half_width)              # [4] Plane+
    y_planes = mv.y - mv.w * (np.array([1, -1, -1, 1]) * span_half_width)              # [4] Plane+
    return x_planes ^ y_planes


# --- math -----------------------------------------------------------------------------
def point_to_points() -> tuple[Point, Point, Point]:
    """A point fitted to a noisy cloud recovers its centroid alongside the posed cloud and true centre."""
    rng = np.random.default_rng(seed)
    raw_cloud = point_cloud(n_cloud, cloud_spread, rng)                                 # [n_cloud] Point-
    base_cloud = pose >> raw_cloud                                                      # [n_cloud] Point-
    cloud_points = noisy_points(base_cloud, noise_sigma, rng)                           # [n_cloud] Point-
    centroid = fitted_primitive(Point, cloud_points)                                    # [] Point-
    true_centre = pose >> origin                                                        # [] Point-
    return cloud_points, true_centre, centroid


def line_to_points() -> tuple[Point, Point, Point]:
    """A line fitted to a noisy segment recovers its principal axis, clipped with the true line by two end planes."""
    rng = np.random.default_rng(seed)
    raw_segment = line_segment(n_segment, half_length)                                  # [n_segment] Point-
    base_segment = pose >> raw_segment                                                  # [n_segment] Point-
    segment_points = noisy_points(base_segment, noise_sigma, rng)                       # [n_segment] Point-
    line = fitted_primitive(Line, segment_points)                                       # [] Line-
    y_axis_line = origin & mv.direction([0, 1, 0])                                      # [] Line-
    true_line = pose >> y_axis_line                                                     # [] Line-
    local_caps = segment_end_planes(drawn_half_length)                                  # [2] Plane+
    end_caps = pose >> local_caps                                                       # [2] Plane+
    return segment_points, true_line ^ end_caps, line ^ end_caps


def plane_to_points() -> tuple[Point, Point, Point]:
    """A plane fitted to a noisy patch recovers the patch plane, cut with the true plane into a quad by four edge lines."""
    rng = np.random.default_rng(seed)
    raw_patch = planar_patch(n_patch, half_width, rng)                                  # [n_patch] Point-
    base_patch = pose >> raw_patch                                                      # [n_patch] Point-
    patch_points = noisy_points(base_patch, noise_sigma, rng)                           # [n_patch] Point-
    plane = fitted_primitive(Plane, patch_points)                                       # [] Plane+
    true_plane = pose >> mv.z                                                           # [] Plane+
    local_edges = square_patch_edges(half_width)                                        # [4] Line-
    frame_edges = pose >> local_edges                                                   # [4] Line-
    return patch_points, true_plane ^ frame_edges, plane ^ frame_edges


def point_to_lines() -> tuple[Point, Point, Point]:
    """A point fitted to a bundle of lines triangulates their point of closest approach."""
    rng = np.random.default_rng(seed)
    local_rays = ray_bundle(n_rays, ray_spread, rng)                                    # [n_rays] Line-
    rays = pose >> local_rays                                                           # [n_rays] Line-
    intersection = fitted_primitive(Point, rays).normalized()                           # [] Point-

    # Each line's point at infinity is its direction. An ideal point has no weight to
    # normalize; its length is the norm of its dual, a Euclidean vector.
    directions = rays ^ mv.w                                                            # [n_rays] Point-
    unit_directions = directions / directions.dual().norm()                             # [n_rays] Point-
    ray_ends = intersection + unit_directions * np.array([[-ray_half_length], [ray_half_length]])   # [2, n_rays] Point-
    true_centre = pose >> origin                                                        # [] Point-
    return ray_ends, true_centre, intersection


def main() -> None:
    from examples.geometry.fitting import render
    from examples.util.animation import save_figure

    cloud_fit, line_fit, plane_fit, bundle_fit = (
        point_to_points(),
        line_to_points(),
        plane_to_points(),
        point_to_lines(),
    )
    save_figure(render.point_fit_figure(*cloud_fit), "fitting_point_to_points")
    save_figure(render.line_fit_figure(*line_fit), "fitting_line_to_points")
    save_figure(render.plane_fit_figure(*plane_fit), "fitting_plane_to_points")
    save_figure(render.bundle_fit_figure(*bundle_fit), "fitting_point_to_lines")

    # checks
    # Finite sample counts limit each fit: the centroid lies within 10 cm of the true centre,
    # while the fitted line ends, fitted plane corners, and triangulated bundle point lie within 5 cm.
    _, true_centre, centroid = cloud_fit
    assert (centroid.normalized() & true_centre.normalized()).norm() < 0.1
    _, _, fit_ends = line_fit
    true_line = pose >> (origin & mv.direction([0, 1, 0]))
    assert ((fit_ends.normalized() & true_line.normalized()).norm() < 0.05).all()
    _, _, fit_quad = plane_fit
    true_plane = pose >> mv.z
    assert ((true_plane.normalized() & fit_quad.normalized()).norm() < 0.05).all()
    _, true_centre, intersection = bundle_fit
    assert (intersection.normalized() & true_centre.normalized()).norm() < 0.05


if __name__ == "__main__":
    main()
