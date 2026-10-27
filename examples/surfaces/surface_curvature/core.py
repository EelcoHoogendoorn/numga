r"""The principal curvatures and lines of curvature of a quadric surface follow from its polarity and its confocal family.

A quadric is a polarity, a map `Plane <- Point` taking each point to its polar plane, and its surface is where a point lies on its own polar plane, `point & surface(point) == 0`, where that polar plane is the tangent plane. Read on directions, the ideal points, the same bilinear form `Point & surface(Point)` is the surface's Hessian; restricted to the tangent plane on both sides, it is the second fundamental form up to the gradient length, and its eigenvalues against the metric on directions are the principal curvatures. The lines of curvature of a central quadric are where it meets its confocal quadrics; the members of that family through a point are the eigenvalues of one form, the quadric's dual read on directions minus the point's own dyad, placing every pixel in the net of curvature lines with one eigensolve.

In the notation of differential geometry the principal curvatures read as the roots of $\det(\mathrm{II} - \kappa\,\mathrm{I}) = 0$ for the first and second fundamental forms $\mathrm{I}$ and $\mathrm{II}$, and by Dupin's theorem the surfaces of a triply orthogonal confocal family $\sum_i x_i^2 / (a_i^2 - \lambda) = 1$ meet along their lines of curvature.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack, where
from numga.algebras import PGA3D

ga = PGA3D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Finite points carry homogeneous weight, and directions are ideal points of weight zero.
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Direction = Point.derive.degenerate                 # Antivector-
# A polarity: each point to its polar plane.
Quadric = ga.gatype((Plane, Point))

# The three coordinate planes x = 0, y = 0, z = 0, and the plane at infinity.
axes = mv(Plane.derive.nondegenerate, np.eye(3))    # [axes] Plane+
w = mv.w                                            # [] Plane+

# A direction's dual is the plane through the origin it is normal to; the inner product of those
# planes is the Euclidean metric on directions.
metric = Direction.dual() | Direction.dual()        # [] Scalar+ <- (Direction-, Direction-)


# --- math -----------------------------------------------------------------------------
def quadric(weights: np.ndarray) -> Quadric:
    """A central quadric with one weight per coordinate axis, as a sum of plane dyads."""
    return (axes * (axes & Point) * weights).sum() - w * (w & Point)                    # [] Plane+ <- Point-


def hit(surface: Quadric, origins: Point, heading: Direction) -> tuple[Point, Scalar]:
    """Where each ray first meets the surface, and the discriminant, negative where it misses.

    The surface's bilinear form, evaluated on a ray in both inputs, is a quadratic in the distance along it.
    """
    form = Point & surface(Point)                                                       # [] Scalar+ <- (Point-, Point-)
    a, b, c = form(heading, heading), form(heading, origins), form(origins, origins)    # [...] Scalar+, Scalar+, Scalar+
    discriminant = b * b - a * c                                                        # [...] Scalar+
    return (origins + heading * nearest_root(a, b, discriminant)).normalized(), discriminant


def principal(surface: Quadric, points: Point) -> Scalar:
    """The two principal curvatures at points on the surface, in order; convex surfaces curve positively.

    The Hessian form `Point & surface(Point)` takes the projector onto the tangent plane in both inputs: the
    second fundamental form, up to the length of the gradient. Against the metric, the first
    fundamental form, its eigenvalues on directions are the principal curvatures and the normal's,
    zero.
    """
    tangent = surface(points)                                                           # [...] Plane+
    # The normal is the tangent plane's dual less its weight, and the projector removes the normal component.
    normal = tangent.dual().cast(Direction)                                             # [...] Direction-
    projector = Direction - normal * ((tangent & Direction) / (tangent & normal))       # [...] Direction- <- Direction-
    second_form = -(Point & surface(Point))(projector, projector) / metric(normal, normal).square_root()  # [...] Scalar+ <- (Direction-, Direction-)
    return pair(second_form.eigvalsh(metric))                                           # [..., curvatures] Scalar+


def confocal(surface: Quadric, points: Point) -> Scalar:
    """The parameters of the confocal quadrics through each point.

    The dual quadric, taken between the planes dual to directions, is the quadric's shape; the
    confocal family shifts it by multiples of the metric. A point lies on a member exactly where the
    shape minus the point's own dyad is singular against the metric, so the members through it are
    that form's eigenvalues: one is the surface itself, and the level sets of the other two are its
    lines of curvature.
    """
    dual_surface = surface.inverse()                                                    # [] Point- <- Plane+
    # The pole of the plane at infinity is the centre, normalised to unit weight.
    pole = dual_surface(w)                                                              # [] Point-
    centre = pole / (w & pole)                                                          # [] Point-
    # Map each direction to the plane through the centre normal to it, and take the dual of its pole.
    central_planes = Direction.dual() - w * (Direction.dual() & centre)                 # [] Plane+ <- Direction-
    shape = dual_surface(central_planes).dual()                                         # [] Plane+ <- Direction-
    # Each point's offset from the centre, dualised into the plane normal to that displacement.
    offset_planes = (points - centre).dual()                                            # [...] Plane+
    # Subtract the point's own dyad from the shape; the eigenvalues are the confocal members through the point.
    confocal_form = Direction & (shape - offset_planes * (offset_planes & Direction))   # [...] Scalar+ <- (Direction-, Direction-)
    return pair(confocal_form.eigvalsh(metric))                                         # [..., members] Scalar+


# --- plumbing -------------------------------------------------------------------------
def view_rays(elevation: float, azimuth: float, extent: float, n_pixels: int) -> tuple[Point, Direction]:
    """An orthographic camera: one ray origin per pixel, far out along the view, and their heading.

    The camera looks along -x with y to its right and z up, turned up toward z by the elevation and
    then about z by the azimuth.
    """
    turn = (mv.xy * (-np.radians(azimuth) / 2)).exp() * (mv.xz * (-np.radians(elevation) / 2)).exp()  # [] Rotor+
    span = np.linspace(-extent, extent, n_pixels)
    # Place the screen behind the scene along x, across it along y, and up along z.
    screen = np.stack(np.broadcast_arrays(10 * extent, span[None, :], span[::-1, None]), axis=-1)
    return turn >> mv.point(screen), turn >> -mv.x.dual()                               # [pixels, pixels] Point-, [] Direction-


def pair(values: Scalar) -> Scalar:
    """Of three eigenvalues on directions, the two that are not the one nearest zero, in order."""
    zero = values.abs().argmin(axis=-1)
    lower = where(zero == 0, values[..., 1], values[..., 0])                            # [...] Scalar+
    upper = where(zero == 2, values[..., 1], values[..., 2])                            # [...] Scalar+
    return stack([lower, upper], axis=-1)                                               # [..., pair] Scalar+


def nearest_root(a: Scalar, b: Scalar, discriminant: Scalar) -> Scalar:
    """The smaller positive root of `a * t**2 + 2 * b * t + c`, or zero where there is none."""
    zero = discriminant * 0                                                             # [...] Scalar+
    root = where(discriminant > 0, discriminant, zero).square_root()                    # [...] Scalar+
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = (-b - root) / a, (-b + root) / a                                # [...] Scalar+, Scalar+
    positive = where(second > 0, second, zero)                                          # [...] Scalar+
    smaller = where((first > 0) & (~(second > 0) | (first < second)), first, positive)  # [...] Scalar+
    return where(smaller.isfinite(), smaller, zero)                                     # [...] Scalar+
