r"""N-camera multi-view reconstruction and bundle adjustment with perspective cone quadrics.

Each camera is a projective map on points, `Camera: Point <- Point`.

A quadric is written as a polarity map, `Plane <- Point`: a point's polar plane. Incidence of
a plane with a point is the plane-first join, `plane & point`, and a quadric's value at a
point is the join of the point's polar with the point, `quadric(p) & p`. That order is a
convention: the regressive product of a plane and a point changes sign with the dimension,
so the plane always comes first.

A pixel measurement on the sensor has a precision disc, a quadric on sensor points. Feeding
the camera into the disc and carrying its polar planes back through the camera lifts it into
a sight cone on scene points, whose cross-section widens with depth:
    cone = camera.adjugate()(disc(camera))              # Plane <- Point
T.adjugate() is the map on planes that incidence carries over from a map on points; it
satisfies T.adjugate()(l) & p == l & T(p) for every plane and point, singular T included.
A cone moves between frames like any map, `pose >> cone(pose << Point)`.

Bundle adjustment works on the cone quadrics directly:
1. Triangulate by summing each point's cones over its cameras. The fused cone's polar of its
   own vertex vanishes; a gauge dyad on the weight makes the vertex the pole of the plane at
   infinity, so `points = (fused + w * (w & Point)).solve(w).normalized()`.
2. Update poses by Gauss-Newton on the cone value. The motion of a local point under a
   pose twist, its polar joined with itself, is the curvature; the point's polar joined with
   the motion is the gradient. The cone is the cost, so no residual metric is chosen.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.estimation.multiview.core", PGA2D)` or PGA3D, and the same
module serves both.
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
ideal = ga.subspace.from_masks(tuple(m for m in Point.output_subspace.masks if m & ga.subspace("w").masks[0]))
# Ideal points: the weightless displacements of points.
Direction = ga.gatype(ideal)
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
euclidean = ga.subspace.from_masks(tuple(m for m in Plane.output_subspace.masks if not m & w.gatype.output_subspace.masks[0]))


def point(coords: np.ndarray) -> Point:
    """Finite points at Euclidean coordinates: the dual of the homogeneous vector."""
    return (mv(euclidean, coords) + w).dual()


def sensor_disk(pixels: Point) -> Quadric:
    """Per-pixel transverse precision dyads on the sensor line of a planar camera.

    The transverse normal line through each pixel, `normal = mv.x - mv.w * (mv.x & pixels)`,
    joined with its own readout is a rank-1 precision disc.
    """
    normal = mv.x - mv.w * (mv.x & pixels)
    return normal * (normal & Point)


def sensor_disk_at(pixels: Point, principal_point: Point, q_sensor: Quadric) -> Quadric:
    """A sensor precision quadric given at the principal point, moved to each pixel.

    The translation from the principal point to a pixel is the square root of their ratio.
    """
    trans = (pixels / principal_point).square_root()
    return trans >> q_sensor(trans << Point)


def make_cones(cameras: Camera, sensor_discs: Quadric) -> Quadric:
    """Pull sensor precision discs back through the camera maps into perspective cones.

    The camera feeds the disc, and its adjugate carries the polar lines back:
    a quadric on scene points whose cross-section widens with depth.
    """
    return cameras.adjugate()(sensor_discs(cameras))     # [n_points, n_cams] Plane <- Point


def triangulate_cones(
    motors: Motor,
    cones: Quadric,
) -> tuple[Point, Quadric]:
    """Triangulate scene points as the vertices of fused perspective cone quadrics.

    The cones are [n_points, n_cams] quadrics in each camera's local frame and the motors
    the [n_cams] camera poses; the result is the [n_points] points and fused quadrics.
    """
    # Move local cones to the world frame and sum them: quadratic constraints add.
    world_cones = motors >> cones(motors << Point)            # [n_points, n_cams] Plane <- Point
    fused = world_cones.sum(axis=-1)                          # [n_points] Plane <- Point

    # The fused cone's polar of its vertex vanishes. Adding the gauge dyad on the weight makes
    # that vertex the pole of the plane at infinity, without altering the spatial gradient:
    points = (fused + w * (w & Point)).solve(w).normalized()  # [n_points] Point
    return points, fused


def bundle_adjust(
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int,
    damping: float,
    free: np.ndarray,
):
    """Jointly optimize camera poses and points purely via perspective cone quadrics.

    Alternates triangulation with damped Gauss-Newton steps on the camera poses. `free` is 1 for
    each camera that moves and 0 for the anchored cameras that fix the gauge.
    """
    motors = initial_motors

    for _ in range(iterations):
        # Triangulate scene points as the fused quadrics' vertices:
        points, _ = triangulate_cones(motors, local_cones)

        # Gauss-Newton on the cone value over each camera's pose twist. A right perturbation of a pose
        # moves its local points by minus the commutator with the twist; the moved point's polar
        # joined with the motion is the curvature, the point's polar joined with the motion the gradient:
        local_points = motors << points[:, None]                  # [n_points, n_cams] Point
        motion = -Twist.commutator(local_points)                  # [n_points, n_cams] Point <- Twist
        curvature = (local_cones(motion) & motion).sum(axis=0)    # [n_cams] Scalar <- (Twist, Twist)
        gradient = (local_cones(local_points) & motion).sum(axis=0)   # [n_cams] Scalar <- Twist

        # Solve the twist steps and hold the anchored cameras to fix gauge freedom:
        step = curvature.lstsq(-gradient, rcond=1e-4) * free    # [n_cams] Twist
        motors = motors * (step * (0.5 * damping)).exp()        # [n_cams] Motor

    points, fused = triangulate_cones(motors, local_cones)
    return motors, points, fused


def reweight_cones(
    cameras: Camera,
    motors: Motor,
    cones: Quadric,
    iterations: int = 3,
) -> Quadric:
    """Reweight perspective cone quadrics into pixel units via Sampson depth scaling.

    Starts from algebraic cone quadrics and iteratively scales each cone by
    `1 / (z ** 2 + FLOOR ** 2)` using point depths: inverse pixel variance where the depth is well
    above the floor, and finite everywhere, the sign of the depth dropping out of its square.
    """
    weighted_cones = cones
    for _ in range(iterations):
        points, _ = triangulate_cones(motors, weighted_cones)
        # A point's depth in a camera is the weight of its projected point, the pairing of the
        # image with the plane at infinity:
        z = w & cameras(motors << points[:, None])                # [n_points, n_cams] Scalar
        weighted_cones = cones / (z ** 2 + FLOOR ** 2)
    return weighted_cones


def bundle_adjust_schur(
    cameras: Camera,
    initial_motors: Motor,
    local_cones: Quadric,
    iterations: int,
    damping: float,
    free: np.ndarray,
):
    """Jointly optimize camera poses with Sampson depth reweighting and a Schur complement.

    The cost is the sum over cameras and points of the cone value `cone(p) & p`, with
    `p = motor << point` the point in its camera's frame. The unknowns are a twist per camera
    and a direction per point. At the triangulated points the cost is stationary over the
    points, so their gradient vanishes and only the camera gradient
    `cones(local_points) & motion` remains. To second order the cost has three curvature
    forms: `h_cam` with both slots twists of one camera, `h_pt` with both slots directions of
    one point, and `h_cross` with a twist slot and a direction slot. The joint Gauss-Newton
    conditions are then, per point,
    `h_pt(direction, Direction) + h_cross(step, Direction) == 0`, and per camera,
    `h_cam(step, Twist) + h_cross(Twist, direction).sum(axis=0) == -gradient`. The point condition
    solves as `response = h_pt.solve(h_cross)`, the map from a camera step to minus the
    point's direction. A point seen by two cameras moves with a step of either, and its move
    changes the cost seen by both: substituting the response into the camera condition folds the
    points out and couples every pair of cameras through the points they share, the compliance
    `h_cross(Twist, response)` of camera i against camera j. The information on all poses
    together is `h_cam` on the diagonal less the compliance summed over the points, a form over
    the field of camera twists, `Scalar <- (Twist[cams], Twist[cams])`, and one solve against the
    gradient gives every camera's step at once. Unlike the alternating solver this accounts for
    the points moving with the cameras. Anchored cameras are left out of both slots of the form
    and of the gradient, and the least-squares step leaves them still, which removes the rig's
    global gauge from the solve.

    Weighting: the value of an algebraic cone at a scene point is the squared pixel distance
    of its image times the square of its depth, because the polar planes were carried back
    through the camera and the image of a point at depth z has weight z. Dividing each cone by
    z squared puts the cost in pixel units, so a far point and a near point count by their
    pixel error alone; the floor added to z squared stands for the uncertainty that does not grow
    with depth. The depth is the weight of the projected point,
    `w & cameras(local_points)`, and since the triangulated points depend on the weighted
    cones the scaling is iterated in `reweight_cones`. A weight is a scalar on a quadric, so
    the weighted solver is the unweighted one with `scaled_cones` in place of `local_cones`:
    the gradient, the three curvature forms, the compliance and the returned information are
    the same expressions, and no weight is carried separately into any of them.

    `free` is 1 for each camera that moves and 0 for the anchored cameras.
    """
    motors = initial_motors

    for _ in range(iterations):
        scaled_cones = reweight_cones(cameras, motors, local_cones)
        points, _ = triangulate_cones(motors, scaled_cones)

        # Curvature blocks over camera twists, over point directions, and across the two. Cameras
        # move by twists and points by directions, ideal points, so neither block ever sees a
        # point's homogeneous scale and no gauge is needed. The motion is where each local point
        # goes per unit right step of its camera's pose; moved carries a world displacement of a
        # point into each camera:
        local_points = motors << points[:, None]                  # [n_points, n_cams] Point
        motion = -Twist.commutator(local_points)                  # [n_points, n_cams] Point <- Twist
        moved = motors << Direction                               # [n_cams] Direction <- Direction
        h_cam = (scaled_cones(motion) & motion).sum(axis=0)       # [n_cams] Scalar <- (Twist, Twist)
        h_pt = (scaled_cones(moved) & moved).sum(axis=1)          # [n_points] Scalar <- (Direction, Direction)
        h_cross = scaled_cones(motion) & moved                    # [n_points, n_cams] Scalar <- (Twist, Direction)

        # Schur complement: fold the points' compliance into the cameras' curvature. Solving the
        # point curvature against the cross term, with the twist slot carried, gives each point's
        # displacement in response to a step of each camera that sees it. That displacement changes
        # the cost seen by every camera that sees the point, so the compliance couples every pair of
        # cameras through the points they share:
        response = h_pt[:, None].solve(h_cross)                   # [n_points, n_cams] Direction <- Twist
        compliance = h_cross[:, :, None](Twist, response[:, None]).sum(axis=0)   # [n_cams, n_cams] Scalar <- (Twist, Twist)
        # Each camera's own curvature on the diagonal, less the compliance, over the free cameras:
        # the information on all poses together, a form over the field of camera twists.
        own = h_cam[:, None] * np.eye(len(free))                 # [n_cams, n_cams] Scalar <- (Twist, Twist)
        information = ((own - compliance) * (free[:, None] * free)).field(1, 2)   # Scalar <- (Twist[cams], Twist[cams])

        # Gauss-Newton step on the reduced curvature, all cameras at once; the anchored cameras, outside
        # the form, are left still, which fixes the gauge freedom:
        gradient = ((scaled_cones(local_points) & motion).sum(axis=0) * free).field(1)   # Scalar <- Twist[cams]
        step = information.lstsq(-gradient, rcond=1e-4).batch()   # [n_cams] Twist
        motors = motors * (step * (0.5 * damping)).exp()          # [n_cams] Motor

    points, fused = triangulate_cones(motors, reweight_cones(cameras, motors, local_cones))
    return motors, points, fused, information



def align_rays_to_splats(
    initial_motors: Motor,
    local_cones: Quadric,
    pinhole: Point,
    pixels: Point,
    iterations: int,
    damping: float,
    free: np.ndarray,
) -> tuple[Motor, Quadric]:
    """Align camera poses so that each pixel's sight line passes through the belief about its point.

    A cone is minus twice the log-likelihood of its pixel, as a function of where the point is:
    flat along the sight line, since moving the point along it does not change what the camera
    sees. Adding the cones of a point over the cameras multiplies their likelihoods, so the fused
    quadric, the splat, is minus twice the log of the belief about the point: a Gaussian wherever
    the sight lines cross at an angle, sharp where they cross steeply, long where they are nearly
    parallel. The cones are weighted into pixel units by the depths of the points their pixels see,
    as in `reweight_cones`, but without extracting a point: the depth is read off the belief itself,
    as where along each sight line its minimum lies, and the cones are weighted by
    `1 / (depth ** 2 + FLOOR ** 2)` for the next fusion.

    A pixel's cost is the splat's minimum along its sight line: minus twice the log-likelihood of
    the most probable point on that line, which is the belief projected onto the sensor and
    evaluated at the pixel. A sharp belief penalizes a sight line that misses it; a blurry one,
    from sight lines that cross at a shallow angle, barely does. No point is ever extracted.

    Similar formulations differ from this one in one step each. Carrying the splat's dual quadric
    through the camera onto the sensor and evaluating the conic there at the pixel gives the same
    cost, at the price of the camera's outermorphism, which the sight line avoids. `bundle_adjust`
    instead places each point at its splat's centre and aligns the cones to those points,
    alternating as here, and `bundle_adjust_schur` lets those points move with the cameras through
    a Schur complement; here no point is shared, and each sight line finds its own best point.
    Each pixel is compared against the splat of the other cameras' cones. Its own cone vanishes
    along its own sight line, so leaving it out changes neither the cost nor its gradient at the
    current poses; its share of the curvature would only hold the step back, and without it the
    steps converge as fast as the Schur complement's. Weighting each
    splat by its value at its centre, the misfit its cones leave between them, is another variant.

    Along the line through the pinhole with heading h, the minimum is the splat on the line over
    the splat on its heading. The splat on the line is the meet of the two points' polar planes,
    paired with the line itself; the splat on the heading, the belief's stiffness along the line,
    is held for each step. The minimum lies at the pinhole's polar paired with the heading, over
    that stiffness, headings away from the pinhole: the depth, in units of the sensor's distance. A camera step moves each sight line as a whole, so sliding along itself,
    which leaves the minimum unchanged, never enters. Alternates fusing the beliefs with damped
    Gauss-Newton steps on the poses. `pinhole` and `pixels` are in the cameras' frames and `free`
    is 1 for each camera that moves and 0 for the anchored cameras that fix the gauge.
    """
    # Each pixel's sight direction and sight line:
    heading = pixels - pinhole                                    # [n_points, n_cams] Direction
    ray = pinhole & heading                                       # [n_points, n_cams] antibivector
    # How the pinhole, the headings and the sight lines move per unit right step of the pose; all
    # fixed in the cameras' frames:
    pinhole_motion = Twist.commutator(pinhole)                    # Point <- Twist
    heading_motion = Twist.commutator(heading)                    # [n_points, n_cams] Point <- Twist
    ray_motion = Twist.commutator(ray)                            # [n_points, n_cams] antibivector <- Twist
    motors = initial_motors
    # The first fusion takes the cones as they are; every later one weights them by the depths read
    # off the beliefs before it:
    weighted = local_cones                                        # [n_points, n_cams] Quadric

    for _ in range(iterations):
        # The belief about each point: its cones summed over the cameras, their likelihoods
        # multiplied. Then each belief as seen from each camera, less that camera's own cone: the
        # belief of the other cameras, which each pixel is compared against:
        splats = (motors >> weighted(motors << Point)).sum(axis=-1)      # [n_points] Plane <- Point
        local = (motors << splats[:, None](motors >> Point)) - weighted   # [n_points, n_cams] Plane <- Point

        # The polar of each sight line, the meet of its points' polar planes, and how it moves; the
        # polar paired with the line over the stiffness is the belief's minimum along the line:
        polar_pinhole, polar_heading = local(pinhole), local(heading)
        polar = polar_pinhole ^ polar_heading                              # [n_points, n_cams]
        polar_motion = (local(pinhole_motion) ^ polar_heading) + (polar_pinhole ^ local(heading_motion))
        stiffness = polar_heading & heading                                # [n_points, n_cams] Scalar

        # How that minimum changes as the camera steps: the polar joined with the line's motion is
        # the gradient, the polar's motion joined with the line's motion the curvature:
        gradient = ((polar & ray_motion) / stiffness).sum(axis=0)          # [n_cams] Scalar <- Twist
        curvature = ((polar_motion & ray_motion) / stiffness).sum(axis=0)  # [n_cams] Scalar <- (Twist, Twist)

        # Solve the twist steps and hold the anchored cameras to fix gauge freedom:
        step = curvature.lstsq(-gradient, rcond=1e-4) * free            # [n_cams] Twist
        motors = motors * (step * (0.5 * damping)).exp()                # [n_cams] Motor

        # Where along each sight line the belief is least, its depth, puts the next fusion in pixel
        # units; its sign drops out of its square:
        depth = -(polar_pinhole & heading) / stiffness                     # [n_points, n_cams] Scalar
        weighted = local_cones / (depth ** 2 + FLOOR ** 2)

    return motors, (motors >> weighted(motors << Point)).sum(axis=-1)
