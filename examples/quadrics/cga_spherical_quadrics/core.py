"""Spherical quadrics in the conformal algebra Cl(3, 1), carried around by a conformal flow.

A circle on S², where the plane `n | x == s` cuts it, is the vector `n + s * w`, and a point of S² is the
dual of its null vector, its unit coordinates on x, y and z plus w: a circle pairs with a point to
`n | x - s`, positive inside the circle, and `mv.w & point == -1`, w squaring to minus one. A quadric Q is a map Plane <- Point,
and a point p is inside where Q(p) & p < 0. Sums of circle dyads give conical donuts, peanuts, twin
islands, lemniscates, crescents, Dupin cyclides, pinched horns, spindles, teardrops, triadic clovers,
hourglasses and parabolic bows. Two off-centre circles c1 and c2 on S² meet in the 2-blade c1 ^ c2, and its
exponential ((c1 ^ c2) * (phase / 2)).exp() is a conformal rotor that moves every quadric along.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+z+w-")
ctx = NumpyContext(ga)
mv = ctx.multivector

# A circle on S², the plane that cuts it out.
Plane = ga.gatype.vector()
# A point of S²: an antivector with intrinsic orientation.
Point = ga.gatype.antivector()
Quadric = ga.gatype((Plane, Point))
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()


def point(xyz: np.ndarray) -> Point:
    """The points of S² at (..., 3) unit coordinates: the duals of their null vectors, x + y + z + w."""
    return ((stack((mv.x, mv.y, mv.z)) * xyz).sum(axis=-1) + mv.w).dual()


def moved(quadric: Quadric, versor: Rotor) -> Quadric:
    """A quadric carried by a versor: pull the point back, push the plane forward."""
    return versor >> quadric(versor << Point)


# --- shapes ---------------------------------------------------------------------------
def make_spherical_donut(r_core: float, r_tube: float) -> Quadric:
    """Spherical donut (torus) extensor on S², a map Plane <- Point: the band between the circles at the
    outer and inner edges of the ring, inside where a point lies inside one and outside the other."""
    outer = mv.z + np.cos(r_core + r_tube) * mv.w
    inner = mv.z + np.cos(r_core - r_tube) * mv.w
    return 0.5 * (outer * (inner & Point) + inner * (outer & Point))


def make_conical_donut(a: float, b: float) -> Quadric:
    """Spherical donut (Limaçon) with central hole meeting in a sharp conical apex on S²."""
    line = mv.w + mv.z + a * mv.x
    return line * (line & Point) - b**2 * (mv.x * (mv.x & Point) + mv.y * (mv.y & Point))


def make_bernoulli_lemniscate(scale_a: float) -> Quadric:
    """True Lemniscate of Bernoulli figure-8 on S²: (w - z)**2 - 2 * scale_a**2 * (x**2 - y**2) < 0, with the
    circle through the pole, `mv.w + mv.z`."""
    pole = mv.w + mv.z
    return pole * (pole & Point) - 2.0 * scale_a**2 * (mv.x * (mv.x & Point) - mv.y * (mv.y & Point))


def make_pinched_horn(r_outer: float) -> Quadric:
    """Pinched horn cyclide on S²: the donut whose inner radius shrinks to a single cusp at the pole."""
    return make_spherical_donut(r_outer / 2.0, r_outer / 2.0)


def make_eccentric_cyclide(r_core: float, r_tube: float, boost_beta: float) -> Quadric:
    """Eccentric Dupin cyclide on S² with unequal tube width, by a dilation."""
    return moved(make_spherical_donut(r_core, r_tube), (mv.xw * (boost_beta / 2.0)).exp())


def make_spherical_cassini(alpha1: float, alpha2: float, c_threshold: float) -> Quadric:
    """Spherical Cassini oval quadric on S²: a point p is inside where
    (p1 & p) * (p2 & p) < c_threshold * (mv.w & p)**2, with p1 and p2 the planes tangent to S²
    at the two foci.

    Yields twin islands (c_threshold small), figure-8 lemniscates (c_threshold near pinch),
    pinched-waist peanuts (c_threshold larger), or asymmetric teardrops (alpha1 != alpha2).
    """
    # The plane z turned toward x by the first angle, and away from it by the second.
    p1 = mv.w + ((mv.zx * (-alpha1 / 2)).exp() >> mv.z)
    p2 = mv.w + ((mv.zx * (alpha2 / 2)).exp() >> mv.z)
    return 0.5 * (p1 * (p2 & Point) + p2 * (p1 & Point)) - c_threshold * mv.w * (mv.w & Point)


def make_spherical_crescent(r_outer: float, r_inner: float, offset: float) -> Quadric:
    """Spherical crescent moon (sickle) bounded by two eccentric circles on S²."""
    c_outer = mv.z + np.cos(r_outer) * mv.w
    c_inner = ((mv.zx * (-offset / 2)).exp() >> mv.z) + np.cos(r_inner) * mv.w
    return 0.5 * (c_outer * (c_inner & Point) + c_inner * (c_outer & Point))


def make_spherical_spindle(weight_z: float, weight_xy: float, bias: float) -> Quadric:
    """Spherical spindle with two opposite conical poles on S²."""
    return (
        weight_z * mv.z * (mv.z & Point)
        - weight_xy * mv.x * (mv.x & Point)
        - weight_xy * mv.y * (mv.y & Point)
        - bias * mv.w * (mv.w & Point)
    )


def make_spherical_clover(tilt_angle: float, radius: float, bias: float) -> Quadric:
    """Triadic 3-lobed rounded deltoid (cloverleaf) on S²: three circle dyads, 120° apart."""
    angles = np.radians([0.0, 120.0, 240.0])
    # The plane z tilted toward x, then turned about z to each of the three angles.
    tilted = (mv.xy * (-angles / 2)).exp() * (mv.zx * (-tilt_angle / 2)).exp() >> mv.z
    circles = tilted + np.cos(radius) * mv.w
    return (circles * (circles & Point)).sum(axis=0) - bias * mv.w * (mv.w & Point)


def make_spherical_hourglass(theta: float, c_waist: float) -> Quadric:
    """True vertical hourglass on S²: two symmetric bulbs connected by a narrow waist."""
    # The plane z turned toward y and away from it by the same angle.
    p1 = mv.w + ((mv.zy * (-theta / 2)).exp() >> mv.z)
    p2 = mv.w + ((mv.zy * (theta / 2)).exp() >> mv.z)
    return 0.5 * (p1 * (p2 & Point) + p2 * (p1 & Point)) - c_waist * mv.w * (mv.w & Point)


def make_spherical_parabola(weight_y: float, linear_x: float, bias: float) -> Quadric:
    """Parabolic bow curve on S²."""
    return (
        weight_y * mv.y * (mv.y & Point)
        + 0.5 * linear_x * (mv.x * (mv.w & Point) + mv.w * (mv.x & Point))
        - bias * mv.w * (mv.w & Point)
    )


# --- math -----------------------------------------------------------------------------
def make_circle_intersection_vortex(
    center1: Plane, radius1: float, center2: Plane, radius2: float,
) -> Bivector:
    """Intersection 2-blade of two off-center circles on S²: each circle is the vector of its
    centre with weight the cosine of its radius."""
    c1 = center1 + mv.w * np.cos(radius1)
    c2 = center2 + mv.w * np.cos(radius2)
    return (c1 ^ c2).normalized()


def flow(quadrics: Quadric, generator: Bivector, phases: np.ndarray) -> Quadric:
    """The quadrics carried by (generator * (phase / 2)).exp(), batched as (phases, quadrics)."""
    rotors = (generator * (phases / 2.0)).exp()
    return moved(quadrics, rotors[:, None])
