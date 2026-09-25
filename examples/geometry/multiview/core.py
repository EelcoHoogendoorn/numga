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
    cone = on_planes(camera)(disc(camera))              # Plane <- Point
on_planes(T) is the map on planes induced by a map on points, solved from the incidence
pairing; it satisfies on_planes(T)(l) & p == l & T(p) for every plane and point, singular T
included. A cone moves between frames like any map, `pose >> cone(pose << Point)`.

Bundle adjustment works on the cone quadrics directly:
1. Triangulate by summing each point's cones over its cameras. The fused cone's polar of its
   own vertex vanishes; a gauge dyad on the weight makes the vertex the pole of the plane at
   infinity, so `points = (fused + w * (w & Point)).solve(w).normalized()`.
2. Update poses by Gauss-Newton on the cone value. The motion of a local point under a
   pose twist, its polar joined with itself, is the curvature; the point's polar joined with
   the motion is the gradient. The cone is the cost, so no residual metric is chosen.

The algebra is not fixed here. `ga` is supplied per instance, by
`examples.instantiate("examples.geometry.multiview.core", PGA2D)` or PGA3D, and the same
module serves both.
"""

from __future__ import annotations


import numpy as np

from numga import Algebra, NumpyContext


ga: Algebra                                        # supplied by examples.instantiate
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
ideal = ga.subspace.from_masks(tuple(m for m in Point.output_subspace.masks if m & ga.subspace("w").masks[0]))
Direction = ga.gatype(ideal)                       # ideal points: the weightless displacements of points
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
Camera = ga.gatype((Point, Point))
Quadric = ga.gatype((Plane, Point))                # a quadric as a polarity map: a point's polar plane
Information = ga.gatype((Scalar, Twist, Twist))    # curvature of a cost over pose twists
w = mv.w
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

    The camera feeds the disc, and the induced plane map carries the polar lines back:
    a quadric on scene points whose cross-section widens with depth.
    """
    return on_planes(cameras)(sensor_discs(cameras))     # [n_points, n_cams] Plane <- Point


def on_planes(collineation: Camera):
    """The map on planes induced by a map on points, through incidence.

    Solving the incidence pairing against `Plane & collineation` gives the plane map with
    `on_planes(T)(l) & p == l & T(p)` for every plane l and point p, whether or not T is
    invertible. It carries a quadric's polar planes back through T.
    """
    return (Plane & Point).solve(Plane & collineation)       # Plane <- Plane


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
    1 / z^2 using point depths, converging to true inverse pixel variance units.
    """
    weighted_cones = cones
    for _ in range(iterations):
        points, _ = triangulate_cones(motors, weighted_cones)
        # A point's depth in a camera is the weight of its projected point, the pairing of the
        # image with the plane at infinity; clamped away from the camera plane:
        z = (w & cameras(motors << points[:, None])).abs().clip(0.1, None)   # [n_points, n_cams] Scalar
        weighted_cones = cones / (z ** 2)
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
    `h_pt(direction, .) + h_cross(step, .) == 0`, and per camera,
    `h_cam(step, .) + h_cross(., direction).sum(axis=0) == -gradient`. The point condition
    solves as `response = h_pt.solve(h_cross)`, the map from a camera step to minus the
    point's direction. Substituting it into the camera condition folds the points out:
    `information = h_cam - compliance.sum(axis=0)`, with
    `compliance = cones(motion) & (motors << response)` the cross form evaluated on the
    point's response, and the step solves `information(step, .) == -gradient`. Unlike the
    alternating solver this accounts for the points moving with the cameras, and the reduced
    curvature is the marginal information on each camera's pose. Anchored cameras have their
    steps and information zeroed, which removes the rig's global gauge from the solve.

    Weighting: the value of an algebraic cone at a scene point is the squared pixel distance
    of its image times the square of its depth, because the polar planes were carried back
    through the camera and the image of a point at depth z has weight z. Dividing each cone by
    z squared puts the cost in pixel units, so a far point and a near point count by their
    pixel error alone. The depth is the weight of the projected point,
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
        # displacement in response to a camera step; the moved point's polar joined with that
        # response, pulled into the camera, is the compliance:
        response = h_pt[:, None].solve(h_cross)                   # [n_points, n_cams] Direction <- Twist
        compliance = scaled_cones(motion) & (motors << response)  # [n_points, n_cams] Scalar <- (Twist, Twist)
        information = h_cam - compliance.sum(axis=0)              # [n_cams] Scalar <- (Twist, Twist)

        # Gauss-Newton step on the reduced curvature; hold the anchored cameras to fix gauge freedom:
        gradient = (scaled_cones(local_points) & motion).sum(axis=0)   # [n_cams] Scalar <- Twist
        step = information.lstsq(-gradient, rcond=1e-4) * free    # [n_cams] Twist
        motors = motors * (step * (0.5 * damping)).exp()          # [n_cams] Motor

    points, fused = triangulate_cones(motors, reweight_cones(cameras, motors, local_cones))
    return motors, points, fused, information * free



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
    parallel. This holds in the units the cones carry; `reweight_cones` puts them in pixel units.

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
    Leaving a pixel's own cone out of the splat it is compared against changes neither the cost
    nor its gradient at the current poses, since that cone vanishes along its own sight line; it
    removes only that cone's share of the curvature, which holds the step back. Weighting each
    splat by its value at its centre, the misfit its cones leave between them, is another variant;
    the cost here keeps the units its cones carry.

    Along the line through the pinhole with heading h, the minimum is the splat on the line over
    the splat on its heading. The splat on the line is the meet of the two points' polar planes,
    paired with the line itself; the splat on the heading, the belief's stiffness along the line,
    is held for each step. A camera step moves each sight line as a whole, so sliding along itself,
    which leaves the minimum unchanged, never enters. Alternates fusing the beliefs with damped
    Gauss-Newton steps on the poses. `pinhole` and `pixels` are in the cameras' frames and `free`
    is 1 for each camera that moves and 0 for the anchored cameras that fix the gauge.
    """
    heading = pixels - pinhole                                    # [n_points, n_cams] Direction: each pixel's sight
    ray = pinhole & heading                                       # [n_points, n_cams] sight lines
    # How the pinhole, the headings and the sight lines move per unit right step of the pose; all
    # fixed in the cameras' frames:
    pinhole_motion = Twist.commutator(pinhole)                    # Point <- Twist
    heading_motion = Twist.commutator(heading)                    # [n_points, n_cams] Point <- Twist
    ray_motion = Twist.commutator(ray)                            # [n_points, n_cams] line <- Twist
    motors = initial_motors

    for _ in range(iterations):
        # The belief about each point: its cones summed over the cameras, their likelihoods
        # multiplied. Then each belief as seen from each camera:
        splats = (motors >> local_cones(motors << Point)).sum(axis=-1)    # [n_points] Plane <- Point
        local = motors << splats[:, None](motors >> Point)                # [n_points, n_cams] Plane <- Point

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

    return motors, (motors >> local_cones(motors << Point)).sum(axis=-1)
