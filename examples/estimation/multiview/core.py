r"""Multi-view reconstruction and camera alignment with sight cones: where landmarks are, and where the
cameras that see them stand, from pixels alone.

Each camera is a projective map on points, `Camera: Point <- Point`. A quadric is written as a polarity
map, `Plane <- Point`, a point's polar plane, and its value at a point is the polar joined with the point,
`quadric(p) & p`; the plane comes first, since the join of a plane and a point changes sign with the
dimension.

A pixel's uncertainty is a quadric on the sensor, a disc. Feeding the camera into the disc and carrying
its polar planes back through the camera lifts it into a sight cone on scene points, wider the further
from the camera, `camera.adjugate()(disc(camera))`. The adjugate is the map on planes that incidence
carries over from the camera, `camera.adjugate()(plane) & point == plane & camera(point)`, and it exists
although the camera loses a dimension. A cone moves between frames like any map,
`pose >> cone(pose << Point)`.

A landmark is the centre of its splat, its cones from every camera added: the splat's polar vanishes at
its centre, and a dyad on the plane at infinity makes the centre that plane's pole. The cone is also the
cost, so a camera aligns by Gauss-Newton on the cone's value: the motion of the points under an open step
of the camera, joined with its own polar, is the curvature, and with the points' polar the slope. Three
ways to align are given: alternating the landmarks and the camera steps, `align`; one joint step that lets
the landmarks move with the cameras, `align_jointly`; and aligning each pixel's sight line to the other
cameras' belief about its landmark, with no landmark placed at all, `align_rays_to_splats`.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.estimation.multiview.core", PGA2D)` or PGA3D, and the same module serves
both.
"""

from __future__ import annotations


import numpy as np

from numga import Algebra
from numga.backend.context import Context


# Supplied by examples.instantiate.
ga: Algebra
ctx: Context
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
# Ideal points: the weightless displacements of points.
Direction = Point.derive.degenerate
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
Camera = ga.gatype((Point, Point))
# A quadric as a polarity map, a point to its polar plane, and the curvature of a cost over pose twists.
Quadric = ga.gatype((Plane, Point))                # Plane <- Point
Information = ga.gatype((Scalar, Twist, Twist))    # Scalar <- (Twist, Twist)
w = mv.w
# Uncertainty that does not grow with depth, in units of depth: calibration, the extent of a point and
# whatever else the cones leave out. Pixel noise grows with depth, and the two variances add.
FLOOR = 0.1
# The times the cones are weighted by the depths of the landmarks they place.
N_REWEIGHTS = 3


# --- math -----------------------------------------------------------------------------
def sensor_disc(pixels: Point) -> Quadric:
    """The disc of each pixel on the sensor line of a planar camera: the squared distance from the line
    through the pixel across the sensor."""
    across = mv.x - mv.w * (mv.x & pixels)                               # [...] Plane
    return across * (across & Point)                                     # [...] Plane <- Point


def sensor_disc_at(pixels: Point, principal_point: Point, disc: Quadric) -> Quadric:
    """A disc given at the principal point, moved to each pixel by the translation between them, the
    square root of their ratio."""
    shift = (pixels / principal_point).square_root()                     # [...] Motor
    return shift >> disc(shift << Point)                                 # [...] Plane <- Point


def sight_cones(cameras: Camera, discs: Quadric) -> Quadric:
    """Each pixel's disc carried back through its camera: a sight cone on scene points."""
    return cameras.adjugate()(discs(cameras))                            # [points, cameras] Plane <- Point


def triangulate(poses: Motor, cones: Quadric) -> tuple[Point, Quadric]:
    """The landmarks as the centres of their splats, and the splats: each landmark's cones `[points,
    cameras]`, carried into the world by the poses `[cameras]` and added."""
    splats = (poses >> cones(poses << Point)).sum(axis=-1)               # [points] Plane <- Point
    # The splat's polar vanishes at its centre; the dyad on the plane at infinity makes the centre its pole.
    centres = (splats + w * (w & Point)).solve(w).normalized()           # [points] Point
    return centres, splats


def align(poses: Motor, cones: Quadric, n_steps: int, damping: float,
          moving: np.ndarray) -> tuple[Motor, Point, Quadric]:
    """The poses after alternating the landmarks, placed at their splats' centres, with damped Gauss-Newton
    steps of the moving cameras; and the landmarks and splats there. `moving` is one for each camera that
    moves and zero for those held still, which fix where the scene is."""
    for _ in range(n_steps):
        centres, _ = triangulate(poses, cones)
        # The landmarks in each camera's frame, and how a small step of the camera moves them: the
        # polar of the motion joined with the motion is the curvature, the landmarks' polar joined with
        # the motion the slope.
        seen = poses << centres[:, None]                                 # [points, cameras] Point
        motion = -Twist.commutator(seen)                                 # [points, cameras] Point <- Twist
        curvature = (cones(motion) & motion).sum(axis=0)                 # [cameras] Scalar <- (Twist, Twist)
        slope = (cones(seen) & motion).sum(axis=0)                       # [cameras] Scalar <- Twist
        step = curvature.lstsq(-slope, rcond=1e-4) * moving              # [cameras] Twist
        poses = poses * (step * (0.5 * damping)).exp()                   # [cameras] Motor
    centres, splats = triangulate(poses, cones)
    return poses, centres, splats


def reweight_cones(cameras: Camera, poses: Motor, cones: Quadric) -> Quadric:
    """The cones in pixel units: a cone's value is the squared pixel error times the squared depth, so each
    is divided by the squared depth of the landmark it places, with the floor added. The landmarks depend
    on the weights, so the weighting is repeated."""
    weighted = cones                                                     # [points, cameras] Plane <- Point
    for _ in range(N_REWEIGHTS):
        centres, _ = triangulate(poses, weighted)
        # A landmark's depth in a camera is the weight of its image, the image paired with the plane at
        # infinity.
        depths = w & cameras(poses << centres[:, None])                  # [points, cameras] Scalar
        weighted = cones / (depths ** 2 + FLOOR ** 2)
    return weighted


def align_jointly(cameras: Camera, poses: Motor, cones: Quadric, n_steps: int, damping: float,
                  moving: np.ndarray) -> tuple[Motor, Point, Quadric, Information]:
    """The poses after damped joint Gauss-Newton steps of the moving cameras and the landmarks, with the
    cones in pixel units; the landmarks and splats there, and the information on the poses.

    A step of one camera moves the landmarks it sees, and a moved landmark changes the cost seen by every
    camera that sees it. Solving the landmarks' curvature against the cross curvature gives each landmark's
    response to a step of each camera; folded into the cameras' own curvature, the Schur complement, it
    couples every pair of cameras through the landmarks they share, and one solve over the field of camera
    steps moves all cameras at once. Cameras held still are left out of the form, which fixes where the
    scene is.
    """
    for _ in range(n_steps):
        weighted = reweight_cones(cameras, poses, cones)
        centres, _ = triangulate(poses, weighted)
        # How a step of each camera moves the landmarks in its frame, and how a displacement of a
        # landmark in the world appears in each camera. Cameras step by twists and landmarks by
        # directions, so no block sees a landmark's homogeneous scale.
        seen = poses << centres[:, None]                                 # [points, cameras] Point
        motion = -Twist.commutator(seen)                                 # [points, cameras] Point <- Twist
        moved = poses << Direction                                       # [cameras] Direction <- Direction
        camera_curvature = (weighted(motion) & motion).sum(axis=0)       # [cameras] Scalar <- (Twist, Twist)
        point_curvature = (weighted(moved) & moved).sum(axis=1)          # [points] Scalar <- (Direction, Direction)
        cross_curvature = weighted(motion) & moved                       # [points, cameras] Scalar <- (Twist, Direction)

        # Each landmark's response to a step of each camera that sees it, and through it the coupling of
        # every pair of cameras that share the landmark.
        response = point_curvature[:, None].solve(cross_curvature)      # [points, cameras] Direction <- Twist
        compliance = cross_curvature[:, :, None](Twist, response[:, None]).sum(axis=0)   # [cameras, cameras] Scalar <- (Twist, Twist)
        # Each camera's own curvature on the diagonal, less the compliance, over the moving cameras: the
        # information on all poses together, a form over the field of camera steps.
        own = camera_curvature[:, None] * np.eye(len(moving))           # [cameras, cameras] Scalar <- (Twist, Twist)
        information = ((own - compliance) * (moving[:, None] * moving)).field(1, 2)   # Scalar <- (Twist[cameras], Twist[cameras])

        slope = ((weighted(seen) & motion).sum(axis=0) * moving).field(1)   # Scalar <- Twist[cameras]
        step = information.lstsq(-slope, rcond=1e-4).batch()            # [cameras] Twist
        poses = poses * (step * (0.5 * damping)).exp()                   # [cameras] Motor

    centres, splats = triangulate(poses, reweight_cones(cameras, poses, cones))
    return poses, centres, splats, information


def align_rays_to_splats(poses: Motor, cones: Quadric, pinhole: Point, pixels: Point, n_steps: int,
                         damping: float, moving: np.ndarray) -> tuple[Motor, Quadric]:
    """The poses after damped Gauss-Newton steps that bring each pixel's sight line through the other
    cameras' belief about its landmark, and the splats there; no landmark is placed.

    A cone is minus twice the log-likelihood of its pixel as a function of where the landmark is, flat
    along the sight line, so a splat is minus twice the log of the belief about its landmark: sharp where
    sight lines cross steeply, long where they are nearly parallel. A pixel's cost is the least the splat
    of the other cameras takes along its sight line, the belief projected onto the sensor at the pixel: a
    sharp belief penalizes a sight line that misses it, a blurry one barely does. Leaving each pixel's own
    cone out changes neither the cost nor its slope, since that cone vanishes along its own sight line, and
    lets the steps converge as fast as `align_jointly`'s. The depth at which a splat is least along a sight
    line weights the cones into pixel units for the next step. `pinhole` and `pixels` are in the cameras'
    frames.
    """
    # Each pixel's heading and sight line, and how the pinhole, the headings and the sight lines move per
    # unit step of the pose; all fixed in the cameras' frames.
    heading = pixels - pinhole                                           # [points, cameras] Direction
    ray = pinhole & heading                                              # [points, cameras] Antibivector
    pinhole_motion = Twist.commutator(pinhole)                           # Point <- Twist
    heading_motion = Twist.commutator(heading)                           # [points, cameras] Point <- Twist
    ray_motion = Twist.commutator(ray)                                   # [points, cameras] Antibivector <- Twist
    # The first fusion takes the cones as they are; every later one weights them by the depths read off
    # the beliefs before it.
    weighted = cones                                                     # [points, cameras] Plane <- Point

    for _ in range(n_steps):
        # The belief about each landmark, its cones added, and that belief as each camera sees it, less
        # the camera's own cone.
        splats = (poses >> weighted(poses << Point)).sum(axis=-1)       # [points] Plane <- Point
        others = (poses << splats[:, None](poses >> Point)) - weighted  # [points, cameras] Plane <- Point

        # The polar of each sight line, the meet of its points' polar planes, and how it moves; the polar
        # paired with the line over the stiffness along it is the belief's least value on the line.
        polar_pinhole, polar_heading = others(pinhole), others(heading)
        polar = polar_pinhole ^ polar_heading                            # [points, cameras] Bivector
        polar_motion = (others(pinhole_motion) ^ polar_heading) + (polar_pinhole ^ others(heading_motion))
        stiffness = polar_heading & heading                              # [points, cameras] Scalar

        # The polar joined with the line's motion is the slope, the polar's motion joined with the line's
        # motion the curvature.
        slope = ((polar & ray_motion) / stiffness).sum(axis=0)           # [cameras] Scalar <- Twist
        curvature = ((polar_motion & ray_motion) / stiffness).sum(axis=0)   # [cameras] Scalar <- (Twist, Twist)
        step = curvature.lstsq(-slope, rcond=1e-4) * moving              # [cameras] Twist
        poses = poses * (step * (0.5 * damping)).exp()                   # [cameras] Motor

        # Where along each sight line the belief is least, its depth, puts the next fusion in pixel
        # units; its sign drops out of its square.
        depths = -(polar_pinhole & heading) / stiffness                  # [points, cameras] Scalar
        weighted = cones / (depths ** 2 + FLOOR ** 2)

    return poses, (poses >> weighted(poses << Point)).sum(axis=-1)
