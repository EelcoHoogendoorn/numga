r"""Multi-view reconstruction and camera alignment recover the positions of observed landmarks and the poses of the cameras that see them from image pixels alone. Several cameras view the same landmarks from different places; each image records only the direction toward a landmark, not its depth, and where the sight cones from different cameras cross fixes the landmark while any mismatch between them guides each misplaced camera back into alignment.

A quadric is its polarity map taking a point to its polar plane, with its value at a point given by the regressive product `quadric(point) & point` (the plane comes first because the regressive product of a plane and a point changes sign with the dimension). A pixel's uncertainty on the sensor is a quadric disc, and feeding the camera into that disc and pulling its polar planes back through `camera.adjugate()` lifts it into a sight cone on scene points, `camera.adjugate()(disc(camera))`, which exists even though the camera loses a dimension and moves between frames as `pose >> cone(pose << Point)`. Summing a landmark's cones across the cameras forms its splat, whose polar vanishes at the landmark so `(splats + w * (w & Point)).solve(w)` recovers its centre in closed form, while Gauss-Newton alignment evaluates the cones' curvature and slope directly on the motion of the points under a camera twist. The algebra `ga` is supplied per instance by `examples.instantiate("examples.estimation.multiview.core", PGA2D)` or `PGA3D`, and the same module serves both.

In the matrix notation of projective vision, with a camera matrix $P$ acting on homogeneous points, planes pull back through $P^{\mathsf T}$, a rank-one sensor precision $\mathbf n\, \mathbf n^{\mathsf T}$ lifts to the sight-cone matrix $P^{\mathsf T} \mathbf n\, \mathbf n^{\mathsf T} P$, and the joint camera-and-landmark Gauss-Newton step forms the reduced camera system via the Schur complement $H_{cc} - H_{cp} H_{pp}^{-1} H_{pc}$.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra
from numga.backend.context import Context


# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
# Ideal points: the weightless displacements of points.
Direction = Point.derive.degenerate                 # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Motor = ga.gatype.rotor()                           # Rotor+
Twist = ga.gatype.bivector()                        # Bivector+
Camera = ga.gatype((Point, Point))
Quadric = ga.gatype((Plane, Point))
Information = ga.gatype((Scalar, Twist, Twist))
w = mv.w                                            # [] Plane+
# Uncertainty that does not grow with depth, in units of depth: calibration, the extent of a point and
# whatever else the cones leave out. Pixel noise grows with depth, and the two variances add.
FLOOR = 0.1
# The times the cones are weighted by the depths of the landmarks they place.
N_REWEIGHTS = 3


# --- math -----------------------------------------------------------------------------
def sensor_disc(pixels: Point) -> Quadric:
    """The sensor quadric of each pixel on a planar camera's sensor line: the squared distance from the line through the pixel across the sensor."""
    pixel_planes = mv.x - mv.w * (mv.x & pixels)                         # [...] Plane+
    return pixel_planes * (pixel_planes & Point)                         # [...] Plane+ <- Point-


def sensor_disc_at(pixels: Point, principal_point: Point, disc: Quadric) -> Quadric:
    """A sensor quadric moved from the principal point to each pixel by the translation motor between them, the square root of their ratio."""
    shift = (pixels / principal_point).square_root()                     # [...] Motor+
    return shift >> disc(shift << Point)                                 # [...] Plane+ <- Point-


def triangulate(poses: Motor, cones: Quadric) -> tuple[Point, Quadric]:
    """The landmarks as the centres of their fused splats, found by carrying each landmark's sight cones into the world by the camera poses and summing across the cameras."""
    splats = (poses >> cones(poses << Point)).sum(axis=-1)               # [landmarks] Plane+ <- Point-
    # The splat's polar vanishes at its centre; the dyad on the plane at infinity makes the centre its pole.
    centres = (splats + w * (w & Point)).solve(w).normalized()           # [landmarks] Point-
    return centres, splats


def align(poses: Motor, cones: Quadric, n_steps: int, damping: float, free_cameras: np.ndarray) -> tuple[Motor, Point, Quadric]:
    """The aligned camera poses, landmark centres, and fused splats after alternating closed-form triangulation with damped Gauss-Newton steps on the free cameras.

    The mask `free_cameras` is one for each camera that moves and zero for the anchored cameras that fix the scene frame.
    """
    for _ in range(n_steps):
        centres, _ = triangulate(poses, cones)
        # The landmarks in each camera's frame, and how a small step of the camera moves them: the
        # polar of the motion joined with the motion is the curvature, the landmarks' polar joined with
        # the motion the slope.
        local_centres = poses << centres[:, None]                        # [landmarks, cameras] Point-
        motion = -Twist.commutator(local_centres)                        # [landmarks, cameras] Point- <- Twist+
        curvature = (cones(motion) & motion).sum(axis=0)                 # [cameras] Scalar+ <- (Twist+, Twist+)
        slope = (cones(local_centres) & motion).sum(axis=0)              # [cameras] Scalar+ <- Twist+
        step = curvature.lstsq(-slope, rcond=1e-4) * free_cameras        # [cameras] Twist+
        poses = poses * (step * (0.5 * damping)).exp()                   # [cameras] Motor+
    centres, splats = triangulate(poses, cones)
    return poses, centres, splats


def reweight_cones(cameras: Camera, poses: Motor, cones: Quadric) -> Quadric:
    """The sight cones weighted into pixel units by the squared depth of each triangulated landmark plus the depth-independent variance floor."""
    weighted_cones = cones                                               # [landmarks, cameras] Plane+ <- Point-
    for _ in range(N_REWEIGHTS):
        centres, _ = triangulate(poses, weighted_cones)
        # A landmark's depth in a camera is the weight of its image, the image paired with the plane at
        # infinity.
        depths = w & cameras(poses << centres[:, None])                  # [landmarks, cameras] Scalar+
        weighted_cones = cones / (depths ** 2 + FLOOR ** 2)              # [landmarks, cameras] Plane+ <- Point-
    return weighted_cones


def align_jointly(
    cameras: Camera, poses: Motor, cones: Quadric, n_steps: int, damping: float, free_cameras: np.ndarray,
) -> tuple[Motor, Point, Quadric, Information]:
    """The aligned camera poses, landmark centres, fused splats, and joint pose information form after damped Gauss-Newton steps on the depth-weighted sight cones.

    A step of one camera moves the landmarks it sees, and a displaced landmark changes the cost seen by every
    camera that observes it. Solving the landmarks' curvature against the cross curvature gives each landmark's
    response to a step of each camera; folded into the cameras' own curvature as the Schur complement, it
    couples every pair of cameras through the landmarks they share, and one solve over the field of camera
    steps moves all cameras at once. Cameras held still are zeroed in the form, which fixes where the
    scene lies.
    """
    for _ in range(n_steps):
        weighted_cones = reweight_cones(cameras, poses, cones)
        centres, _ = triangulate(poses, weighted_cones)
        # How a step of each camera moves the landmarks in its frame, and how a displacement of a
        # landmark in the world appears in each camera. Cameras step by twists and landmarks by
        # directions, so no block sees a landmark's homogeneous scale.
        local_centres = poses << centres[:, None]                        # [landmarks, cameras] Point-
        camera_motion = -Twist.commutator(local_centres)                 # [landmarks, cameras] Point- <- Twist+
        point_motion = poses << Direction                                # [cameras] Direction- <- Direction-
        camera_curvature = (weighted_cones(camera_motion) & camera_motion).sum(axis=0)  # [cameras] Scalar+ <- (Twist+, Twist+)
        point_curvature = (weighted_cones(point_motion) & point_motion).sum(axis=1)     # [landmarks] Scalar+ <- (Direction-, Direction-)
        cross_curvature = weighted_cones(camera_motion) & point_motion                  # [landmarks, cameras] Scalar+ <- (Twist+, Direction-)

        # Each landmark's response to a step of each camera that sees it, and through it the coupling of
        # every pair of cameras that share the landmark.
        response = point_curvature[:, None].solve(cross_curvature)       # [landmarks, cameras] Direction- <- Twist+
        compliance = cross_curvature[:, :, None](Twist, response[:, None]).sum(axis=0)   # [cameras, cameras] Scalar+ <- (Twist+, Twist+)
        # Each camera's own curvature on the diagonal, less the compliance, over the free cameras: the
        # information on all poses together, a form over the field of camera steps.
        diagonal_curvature = camera_curvature[:, None] * np.eye(len(free_cameras))       # [cameras, cameras] Scalar+ <- (Twist+, Twist+)
        information = ((diagonal_curvature - compliance) * (free_cameras[:, None] * free_cameras)).field(1, 2)   # [] Scalar+ <- (Twist+[cameras], Twist+[cameras])

        slope = ((weighted_cones(local_centres) & camera_motion).sum(axis=0) * free_cameras).field(1)   # [] Scalar+ <- Twist+[cameras]
        step = information.lstsq(-slope, rcond=1e-4).batch()             # [cameras] Twist+
        poses = poses * (step * (0.5 * damping)).exp()                   # [cameras] Motor+

    centres, splats = triangulate(poses, reweight_cones(cameras, poses, cones))
    return poses, centres, splats, information


def align_rays_to_splats(
    poses: Motor, cones: Quadric, pinhole: Point, pixels: Point, n_steps: int,
    damping: float, free_cameras: np.ndarray,
) -> tuple[Motor, Quadric]:
    """The aligned camera poses and final splats after damped Gauss-Newton steps that bring each pixel's sight line through the other cameras' fused splat without placing landmarks.

    A cone is minus twice the log-likelihood of its pixel as a function of where the landmark lies, flat
    along the sight line, so a splat is minus twice the log of the belief about its landmark: sharp where
    sight lines cross steeply, and elongated where they are nearly parallel. A pixel's cost is the minimum the splat
    of the other cameras takes along its sight line, the belief projected onto the sensor at the pixel: a
    sharp belief penalizes a sight line that misses it, while a broad one barely does. Leaving each pixel's own
    cone out changes neither the cost nor its slope, because that cone vanishes along its own sight line, and
    lets the steps converge as fast as `align_jointly`. The depth at which a splat is minimal along a sight
    line weights the cones into pixel units for the next step. Both `pinhole` and `pixels` are given in the cameras'
    frames.
    """
    # Each pixel's heading and sight line, and how the pinhole, the headings and the sight lines move per
    # unit step of the pose; all fixed in the cameras' frames.
    heading = pixels - pinhole                                           # [landmarks, cameras] Direction-
    ray = pinhole & heading                                              # [landmarks, cameras] Line-
    pinhole_motion = Twist.commutator(pinhole)                           # [] Point- <- Twist+
    heading_motion = Twist.commutator(heading)                           # [landmarks, cameras] Point- <- Twist+
    ray_motion = Twist.commutator(ray)                                   # [landmarks, cameras] Line- <- Twist+
    # The first fusion takes the cones as they are; every later one weights them by the depths read off
    # the beliefs before it.
    weighted_cones = cones                                               # [landmarks, cameras] Plane+ <- Point-

    for _ in range(n_steps):
        # The belief about each landmark, its cones added, and that belief as each camera sees it, less
        # the camera's own cone.
        splats = (poses >> weighted_cones(poses << Point)).sum(axis=-1)  # [landmarks] Plane+ <- Point-
        other_splats = (poses << splats[:, None](poses >> Point)) - weighted_cones  # [landmarks, cameras] Plane+ <- Point-

        # The polar of each sight line, the meet of its points' polar planes, and how it moves; the polar
        # paired with the line over the stiffness along it is the belief's least value on the line.
        polar_pinhole, polar_heading = other_splats(pinhole), other_splats(heading)  # [landmarks, cameras] Plane+, Plane+
        polar = polar_pinhole ^ polar_heading                            # [landmarks, cameras] Line+
        polar_motion = (other_splats(pinhole_motion) ^ polar_heading) + (polar_pinhole ^ other_splats(heading_motion))  # [landmarks, cameras] Line+ <- Twist+
        stiffness = polar_heading & heading                              # [landmarks, cameras] Scalar+

        # The polar joined with the line's motion is the slope, the polar's motion joined with the line's
        # motion the curvature.
        slope = ((polar & ray_motion) / stiffness).sum(axis=0)           # [cameras] Scalar+ <- Twist+
        curvature = ((polar_motion & ray_motion) / stiffness).sum(axis=0)   # [cameras] Scalar+ <- (Twist+, Twist+)
        step = curvature.lstsq(-slope, rcond=1e-4) * free_cameras        # [cameras] Twist+
        poses = poses * (step * (0.5 * damping)).exp()                   # [cameras] Motor+

        # Where along each sight line the belief is least, its depth, puts the next fusion in pixel
        # units; its sign drops out of its square.
        depths = -(polar_pinhole & heading) / stiffness                  # [landmarks, cameras] Scalar+
        weighted_cones = cones / (depths ** 2 + FLOOR ** 2)              # [landmarks, cameras] Plane+ <- Point-

    return poses, (poses >> weighted_cones(poses << Point)).sum(axis=-1)
