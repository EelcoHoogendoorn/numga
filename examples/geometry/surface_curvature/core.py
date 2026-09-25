"""Curvature of quadric surfaces in PGA3D: the second form against the first.

A quadric is a polarity, a map Plane <- Point; its surface is where a point lies on its own polar
plane, p & Q(p) = 0, and that polar plane is the tangent plane. Read on directions, the ideal
points, the same map is the surface's Hessian, and the Euclidean metric is a sum of plane dyads.
The principal curvatures and directions are the eigenpairs of the Hessian restricted to the
tangent plane, against the metric.

The lines of curvature of a central quadric are where it meets its confocal quadrics. The members
of that family through a point are the eigenvalues of one form, the quadric's dual read on
directions minus the point's own dyad, so every pixel of a render finds its place in the net of
curvature lines with one eigensolve.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()                     # finite points, and directions: points with no weight
Plane = ga.gatype.vector()
Direction = ga.gatype(ga.subspace.antivector().degenerate())   # the directions: tangent vectors
Quadric = ga.gatype((Plane, Point))                # a polarity: each point to its polar plane
Form = ga.gatype((Scalar, Direction, Direction))

euclidean = ga.subspace("x y z")
axes = mv(euclidean, np.eye(3))                    # [3] Plane: the coordinate planes
w = mv.w                                           # the plane at infinity

# A direction's dual is the plane through the origin it is normal to; the inner product of those
# planes is the Euclidean metric on directions.
metric = Direction.dual() | Direction.dual()       # [] Scalar <- (Direction, Direction)


# --- plumbing ------------------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Finite points at Euclidean coordinates, shape (..., 3)."""
    return (mv(euclidean, coords) + w).dual()


def direction(coords: np.ndarray) -> Direction:
    """Directions with Euclidean components, shape (..., 3)."""
    return mv(euclidean, coords).dual()


def view_rays(elevation: float, azimuth: float, extent: float, pixels: int) -> tuple[Point, Direction]:
    """An orthographic camera: one ray origin per pixel, far out along the view, and their heading."""
    e, a = np.radians(elevation), np.radians(azimuth)
    back = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(back, right)
    s = np.linspace(-extent, extent, pixels)
    screen = s[None, :, None] * right + s[::-1, None, None] * up + 10 * extent * back
    return point(screen), direction(-back)


def pair(values: Scalar) -> Scalar:
    """Of three eigenvalues on directions, the two that are not the one nearest zero, in order."""
    ordered = values.to_array()
    order = np.argsort(np.abs(ordered), axis=-1)[..., 1:]
    return mv.scalar(np.sort(np.take_along_axis(ordered, order, axis=-1), axis=-1)[..., None])


def nearest_root(a: Scalar, b: Scalar, discriminant: Scalar) -> Scalar:
    """The smaller positive root of a t^2 + 2 b t + c; zero where there is none."""
    root = np.sqrt(np.clip(discriminant.to_array(), 0, None))
    a, b = a.to_array(), b.to_array()
    with np.errstate(divide="ignore", invalid="ignore"):
        roots = np.stack([(-b - root) / a, (-b + root) / a])
    roots = np.where(roots > 0, roots, np.inf).min(axis=0)
    return mv.scalar(np.where(np.isfinite(roots), roots, 0.0)[..., None])


# --- math ----------------------------------------------------------------------------------------
def quadric(weights: np.ndarray) -> Quadric:
    """A central quadric with one weight per axis, as a sum of plane dyads."""
    return (axes * (axes & Point) * weights).sum() - w * (w & Point)


def hit(surface: Quadric, origins: Point, heading: Direction) -> tuple[Point, Scalar]:
    """Where each ray first meets the surface, and the discriminant, negative where it misses.

    The surface's form, bound to a ray in both slots, is a quadratic in the distance along it.
    """
    form = Point & surface(Point)                                   # [] Scalar <- (Point, Point)
    a, b, c = form(heading, heading), form(heading, origins), form(origins, origins)
    discriminant = b * b - a * c
    return (origins + heading * nearest_root(a, b, discriminant)).normalized(), discriminant


def principal(surface: Quadric, points: Point) -> Scalar:
    """The two principal curvatures at points on the surface, in order; convex surfaces curve positively.

    The Hessian form Point & Q(Point) takes the projector onto the tangent plane in both slots: the
    second fundamental form, up to the length of the gradient. Against the metric, the first
    fundamental form, its eigenvalues on directions are the principal curvatures and the normal's,
    zero.
    """
    tangent = surface(points)                                        # [...] Plane
    normal = tangent.dual().cast(Direction)                          # [...] Direction: a plane's dual, less its weight
    project = Direction - normal * ((tangent & Direction) / (tangent & normal))   # [...] onto the tangent plane
    second = -(Point & surface(Point))(project, project) / metric(normal, normal).square_root()
    return pair(second.eigvalsh(metric))                             # [..., principal] Scalar


def confocal(surface: Quadric, points: Point) -> Scalar:
    """The parameters of the confocal quadrics through each point.

    The dual quadric, taken between the planes dual to directions, is the quadric's shape; the
    confocal family shifts it by multiples of the metric. A point lies on a member exactly where the
    shape minus the point's own dyad is singular against the metric, so the members through it are
    that form's eigenvalues: one is the surface itself, and the level sets of the other two are its
    lines of curvature.
    """
    dual_surface = surface.inverse()                                               # [] Point <- Plane
    pole = dual_surface(w)                                                         # [] Point: the pole of the plane at infinity
    centre = pole / (w & pole)                                             # [] Point: at unit weight
    # The plane through the centre normal to each direction:
    through_centre = Direction.dual() - w * (Direction.dual() & centre)    # [] Plane <- Direction
    # Their poles, as directions, and the planes those are normal to: the quadric's shape.
    shape = dual_surface(through_centre).cast(Direction).dual()                    # [] Plane <- Direction
    # Each point's offset from the centre, as the plane it is normal to.
    position = (points - centre).cast(Direction).dual()                    # [...] Plane
    # The shape less the point's own dyad; its eigenvalues are the members through the point.
    through_point = Direction & (shape - position * (position & Direction))   # [...] Form
    return pair(through_point.eigvalsh(metric))                            # [..., members] Scalar
