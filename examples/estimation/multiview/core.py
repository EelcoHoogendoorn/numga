"""Scene reconstruction and camera alignment recover the positions of observed landmarks and the poses of the cameras that view them from image pixels alone. Several cameras photograph the same landmarks from different vantage points, with each image recording only the direction from its pinhole toward each landmark. Where sight lines from multiple cameras cross fixes each landmark in space, and when a camera pose is misplaced its sight lines miss those of the other views, driving the misplaced camera back into alignment.

A pinhole camera is modelled as the linear map `(pinhole & Point) ^ sensor`, and a quadric is represented by its polarity map taking each point to its polar plane with quadratic value `quadric(point) & point`. Feeding the camera into a pixel's sensor quadric `ball` and pulling polar planes back through `camera.adjugate()` lifts pixel uncertainty into a sight cone `camera.adjugate()(ball(camera))`, which moves between frames as `pose >> cone(pose << Point)`. Summing a landmark's cones across views forms its fused splat, whose spatial polar vanishes at the landmark so `(splats + w * (w & Point)).solve(w)` recovers its centre in closed form, while Gauss-Newton alignment evaluates cone curvature and slope directly on `-Twist.commutator(local_centres)`. Instantiating this module with either 2D or 3D projective geometric algebra serves planar and spatial rigs with the same equations.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra
from numga.backend.context import Context


# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
# Ideal points represent weightless displacements between points.
Direction = Point.derive.degenerate                 # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Motor = ga.gatype.rotor()                           # Rotor+
Twist = ga.gatype.bivector()                        # Bivector+
Camera = ga.gatype((Point, Point))                  # Point- <- Point-
Quadric = ga.gatype((Plane, Point))                 # Plane+ <- Point-
Information = ga.gatype((Scalar, Twist, Twist))     # Scalar+ <- (Twist+, Twist+)

# Depth-independent uncertainty floor in depth units, added in quadrature with pixel noise that grows with depth.
depth_floor = 0.1
# Fixed-point updates refreshing landmark depths when weighting sight cones into pixel units.
n_reweights = 3


# --- math -----------------------------------------------------------------------------
def sensor_balls(pixels: Point) -> Quadric:
    """Planar sensor quadrics (`[...] Plane+ <- Point-`) measure squared transverse distance on the sensor plane from each pixel in `pixels` (`[...] Point-`)."""
    pixel_planes = mv.x - mv.w * (mv.x & pixels)                         # [...] Plane+
    return pixel_planes * (pixel_planes & Point)                         # [...] Plane+ <- Point-


def shifted_sensor_balls(pixels: Point, principal_point: Point, ball: Quadric) -> Quadric:
    """Translation motors (`[...] Motor+`), formed as the square root of the point ratio from `principal_point` to `pixels`, carry the principal-point quadric `ball` to each pixel (`[...] Plane+ <- Point-`)."""
    translations = (pixels / principal_point).square_root()              # [...] Motor+
    return translations >> ball(translations << Point)                   # [...] Plane+ <- Point-


def triangulation(poses: Motor, cones: Quadric) -> tuple[Point, Quadric]:
    """Carrying local sight `cones` (`[..., ...] Plane+ <- Point-`) into the world by camera `poses` (`[...] Motor+`) and summing across views forms fused splats (`[...] Plane+ <- Point-`) whose centres (`[...] Point-`) reconstruct the landmarks."""
    splats = (poses >> cones(poses << Point)).sum(axis=-1)               # [...] Plane+ <- Point-
    # Each splat's spatial slope vanishes at its centre, and adding the dyad of the plane at infinity makes the centre its pole.
    centres = (splats + mv.w * (mv.w & Point)).solve(mv.w).normalized()  # [...] Point-
    return centres, splats


def alignment_steps(
    poses: Motor, cones: Quadric, pinhole: Point, free_cameras: np.ndarray,
    stretch: float, step_cap: float, landmark_depth: float,
) -> Iterator[tuple[Motor, Quadric, Quadric, Point, Information]]:
    """Yields camera poses, world sight cones, fused splats, triangulated landmarks, and single-camera cost curvature across step-capped alternating Gauss-Newton iterations."""
    while True:
        centres, splats = triangulation(poses, cones)
        world_cones = poses >> cones(poses << Point)                     # [..., ...] Plane+ <- Point-
        # Each landmark in each camera's frame moves by minus its commutator with a camera twist.
        local_centres = poses << centres[:, None]                        # [..., ...] Point-
        motion = -Twist.commutator(local_centres)                        # [..., ...] Point- <- Twist+
        curvature = (cones(motion) & motion).sum(axis=0)                 # [...] Scalar+ <- (Twist+, Twist+)
        slope = (cones(local_centres) & motion).sum(axis=0)              # [...] Scalar+ <- Twist+
        yield poses, world_cones, splats, centres, curvature
        step = curvature.solve(-slope) * free_cameras * stretch          # [...] Twist+
        # Combine pinhole travel with camera rotation weighted at landmark depth so large initial steps stay bounded.
        step_motors = (step / 2).exp()                                   # [...] Motor+
        step_size = ((step_motors >> pinhole) - pinhole).antinorm() + landmark_depth * step.norm()  # [...] Scalar+
        poses = poses * (step * (step_cap / (step_size + step_cap)) / 2).exp()  # [...] Motor+


def pose_uncertainties(
    poses: Motor, curvature: Information, pinhole: Point, transverse_plane: Plane,
    optical_axis: Direction, pixel_noise: float, n_sigmas: float,
) -> tuple[Quadric, Scalar]:
    """Position uncertainty quadrics (`[...] Plane+ <- Point-`) and angular heading spreads (`[...] Scalar+`) at `n_sigmas` standard deviations come from pushing the inverse single-camera `curvature` through pinhole motion and heading rate."""
    information = curvature / pixel_noise**2                             # [...] Scalar+ <- (Twist+, Twist+)
    pinholes = poses >> pinhole                                          # [...] Point-
    pinhole_motion = poses >> Twist.commutator(pinhole)                  # [...] Point- <- Twist+
    heading_rate = transverse_plane & Twist.commutator(optical_axis)     # [] Scalar+ <- Twist+
    # Push the pose covariance forward through pinhole motion and centre the dual quadric on each pinhole.
    dual_ellipsoids = (pinhole_motion(information.solve(Plane & pinhole_motion)) * n_sigmas**2
                       - pinholes * (Plane & pinholes))                  # [...] Point- <- Plane+
    ellipsoids = dual_ellipsoids.inverse()                               # [...] Plane+ <- Point-
    spreads = heading_rate(information.solve(heading_rate)).square_root() * n_sigmas  # [...] Scalar+
    return ellipsoids, spreads


def alternating_alignment(
    poses: Motor, cones: Quadric, n_steps: int, damping: float, free_cameras: np.ndarray,
) -> tuple[Motor, Point, Quadric]:
    """Aligned camera poses (`[...] Motor+`), landmark centres (`[...] Point-`), and fused splats (`[...] Plane+ <- Point-`) result from `n_steps` alternating triangulation and damped Gauss-Newton camera steps.

    Entries of `free_cameras` equal `1` for moving cameras and `0` for anchored cameras that fix the scene gauge.
    """
    for _ in range(n_steps):
        centres, _ = triangulation(poses, cones)
        # Pairing the cone polar of twist-induced landmark motion with the motion gives curvature, and with the landmark gives slope.
        local_centres = poses << centres[:, None]                        # [..., ...] Point-
        motion = -Twist.commutator(local_centres)                        # [..., ...] Point- <- Twist+
        curvature = (cones(motion) & motion).sum(axis=0)                 # [...] Scalar+ <- (Twist+, Twist+)
        slope = (cones(local_centres) & motion).sum(axis=0)              # [...] Scalar+ <- Twist+
        step = curvature.lstsq(-slope, rcond=1e-4) * free_cameras        # [...] Twist+
        poses = poses * (step * (0.5 * damping)).exp()                   # [...] Motor+
    centres, splats = triangulation(poses, cones)
    return poses, centres, splats


def depth_weighted_cones(cameras: Camera, poses: Motor, cones: Quadric) -> Quadric:
    """Sight cones (`[..., ...] Plane+ <- Point-`) rescale into pixel units when divided by the squared landmark depth in each camera plus `depth_floor**2`."""
    weighted_cones = cones                                               # [..., ...] Plane+ <- Point-
    for _ in range(n_reweights):
        centres, _ = triangulation(poses, weighted_cones)
        # Pairing a landmark's projected image with the plane at infinity reads its depth in front of the pinhole.
        depths = mv.w & cameras(poses << centres[:, None])               # [..., ...] Scalar+
        weighted_cones = cones / (depths**2 + depth_floor**2)            # [..., ...] Plane+ <- Point-
    return weighted_cones


def joint_alignment(
    cameras: Camera, poses: Motor, cones: Quadric, n_steps: int, damping: float, free_cameras: np.ndarray,
) -> tuple[Motor, Point, Quadric, Information]:
    """Aligned camera poses, landmark centres, fused splats, and the coupled pose information form (`Scalar+ <- (Twist+[n_cameras], Twist+[n_cameras])`) result from `n_steps` joint Gauss-Newton steps on depth-weighted sight cones.

    Eliminating landmark displacements against their cross-curvature with camera twists folds each shared landmark's compliance into the joint camera information form, coupling every pair of cameras that view the same point while zeroing anchored cameras to fix the gauge.
    """
    n_cameras = len(free_cameras)
    for _ in range(n_steps):
        weighted_cones = depth_weighted_cones(cameras, poses, cones)
        centres, _ = triangulation(poses, weighted_cones)
        # Cameras step by twists and landmarks by weightless directions, leaving homogeneous point scale untouched.
        local_centres = poses << centres[:, None]                        # [..., n_cameras] Point-
        camera_motion = -Twist.commutator(local_centres)                 # [..., n_cameras] Point- <- Twist+
        point_motion = poses << Direction                                # [n_cameras] Direction- <- Direction-
        camera_curvature = (weighted_cones(camera_motion) & camera_motion).sum(axis=0)  # [n_cameras] Scalar+ <- (Twist+, Twist+)
        point_curvature = (weighted_cones(point_motion) & point_motion).sum(axis=1)     # [...] Scalar+ <- (Direction-, Direction-)
        cross_curvature = weighted_cones(camera_motion) & point_motion                  # [..., n_cameras] Scalar+ <- (Twist+, Direction-)

        # Fold each landmark's displacement response back into the camera system as the Schur-complement compliance.
        response = point_curvature[:, None].solve(cross_curvature)       # [..., n_cameras] Direction- <- Twist+
        compliance = cross_curvature[:, :, None](Twist, response[:, None]).sum(axis=0)  # [n_cameras, n_cameras] Scalar+ <- (Twist+, Twist+)
        diagonal_curvature = camera_curvature[:, None] * np.eye(n_cameras)              # [n_cameras, n_cameras] Scalar+ <- (Twist+, Twist+)
        information = ((diagonal_curvature - compliance) * (free_cameras[:, None] * free_cameras)).field(1, 2)  # [] Scalar+ <- (Twist+[n_cameras], Twist+[n_cameras])

        slope = ((weighted_cones(local_centres) & camera_motion).sum(axis=0) * free_cameras).field(1)  # [] Scalar+ <- Twist+[n_cameras]
        step = information.lstsq(-slope, rcond=1e-4).batch()             # [n_cameras] Twist+
        poses = poses * (step * (0.5 * damping)).exp()                   # [n_cameras] Motor+

    centres, splats = triangulation(poses, depth_weighted_cones(cameras, poses, cones))
    return poses, centres, splats, information


def ray_splat_alignment(
    poses: Motor, cones: Quadric, pinhole: Point, pixels: Point, n_steps: int,
    damping: float, free_cameras: np.ndarray,
) -> tuple[Motor, Quadric]:
    """Aligned camera poses (`[...] Motor+`) and fused world splats (`[...] Plane+ <- Point-`) result from `n_steps` Gauss-Newton iterations that minimize each pixel ray's miss against the other cameras' fused splat without instantiating landmarks.

    Minimizing each leave-one-out splat along the sight line from `pinhole` through `pixels` (both in camera coordinates) penalizes misses in proportion to transverse belief sharpness, while the minimizing depth along each ray weights the sight cones into pixel units for the next step.
    """
    # Sight-line geometry and twist derivatives in the camera frame remain fixed across steps.
    headings = pixels - pinhole                                          # [..., ...] Direction-
    rays = pinhole & headings                                            # [..., ...] Line-
    pinhole_motion = Twist.commutator(pinhole)                           # [] Point- <- Twist+
    heading_motion = Twist.commutator(headings)                          # [..., ...] Point- <- Twist+
    ray_motion = Twist.commutator(rays)                                  # [..., ...] Line- <- Twist+
    weighted_cones = cones                                               # [..., ...] Plane+ <- Point-

    for _ in range(n_steps):
        splats = (poses >> weighted_cones(poses << Point)).sum(axis=-1)  # [...] Plane+ <- Point-
        other_splats = (poses << splats[:, None](poses >> Point)) - weighted_cones  # [..., ...] Plane+ <- Point-

        # Meeting the polar planes of the pinhole and heading forms the sight line's polar bivector,
        # whose pairing with the ray over directional stiffness gives the minimum splat cost along the ray.
        polar_pinhole, polar_heading = other_splats(pinhole), other_splats(headings)  # [..., ...] Plane+, Plane+
        polar = polar_pinhole ^ polar_heading                            # [..., ...] Bivector+
        polar_motion = (other_splats(pinhole_motion) ^ polar_heading) + (polar_pinhole ^ other_splats(heading_motion))  # [..., ...] Bivector+ <- Twist+
        stiffness = polar_heading & headings                             # [..., ...] Scalar+

        slope = ((polar & ray_motion) / stiffness).sum(axis=0)           # [...] Scalar+ <- Twist+
        curvature = ((polar_motion & ray_motion) / stiffness).sum(axis=0)  # [...] Scalar+ <- (Twist+, Twist+)
        step = curvature.lstsq(-slope, rcond=1e-4) * free_cameras        # [...] Twist+
        poses = poses * (step * (0.5 * damping)).exp()                   # [...] Motor+

        depths = -(polar_pinhole & headings) / stiffness                 # [..., ...] Scalar+
        weighted_cones = cones / (depths**2 + depth_floor**2)            # [..., ...] Plane+ <- Point-

    return poses, (poses >> weighted_cones(poses << Point)).sum(axis=-1)
