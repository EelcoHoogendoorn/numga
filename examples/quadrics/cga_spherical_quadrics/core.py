"""Spherical quadrics in the conformal algebra Cl(3, 1), carried around by a conformal flow.

A circle on S², where the plane `n | x == s` cuts it, is the vector `n + s * mv.w`. A point of S² is a null
antivector, the sphere's centre `mv.w.dual()` plus its unit direction on `mv.x.dual()`, `mv.y.dual()` and
`mv.z.dual()`: a circle pairs with a point to `n | x - s`, positive inside the circle, and `mv.w & point == -1`,
w squaring to minus one. A quadric is a map from points to planes, and a point lies inside where
`quadric(point) & point < 0`. Sums of circle dyads give conical donuts, peanuts, twin islands, lemniscates,
crescents, Dupin cyclides, pinched horns, spindles, teardrops, triadic clovers, hourglasses and parabolic bows.
Two off-centre circles on S² meet in the 2-blade `first ^ second`, and its exponential is a conformal rotor that
moves every quadric along.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+z+w-")
ctx = NumpyContext(ga)
mv = ctx.multivector

# A circle on S², the plane that cuts it out.
Plane = ga.gatype.vector()                                                     # Vector+
# A point of S²: an antivector with intrinsic orientation.
Point = ga.gatype.antivector()                                                 # Antivector-
Quadric = ga.gatype((Plane, Point))
Bivector = ga.gatype.bivector()                                                # Bivector+


# --- plumbing -------------------------------------------------------------------------
def point(xyz: np.ndarray) -> Point:
    """The points of S² at `[..., 3]` unit coordinates: the sphere's centre plus their unit directions."""
    return mv.w.dual() + (stack((mv.x.dual(), mv.y.dual(), mv.z.dual())) * xyz).sum(axis=-1)


# --- shapes ---------------------------------------------------------------------------
def make_spherical_donut(r_core: float, r_tube: float) -> Quadric:
    """A spherical donut is the band between the circles at the outer and inner edges of a ring on S², inside where
    a point lies inside one circle and outside the other."""
    outer = mv.z + np.cos(r_core + r_tube) * mv.w                              # [] Plane+
    inner = mv.z + np.cos(r_core - r_tube) * mv.w                              # [] Plane+
    return 0.5 * (outer * (inner & Point) + inner * (outer & Point))


def make_conical_donut(tilt: float, width: float) -> Quadric:
    """A conical donut, a limaçon on S², is a ring whose central hole closes in a sharp conical apex at the pole."""
    # A circle through the pole, tilted toward x.
    through_pole = mv.w + mv.z + tilt * mv.x                                   # [] Plane+
    return through_pole * (through_pole & Point) - width**2 * (mv.x * (mv.x & Point) + mv.y * (mv.y & Point))


def make_bernoulli_lemniscate(spread: float) -> Quadric:
    """The lemniscate of Bernoulli on S² is a figure eight: the dyad of the point circle at the pole less a weighted
    difference of the dyads of the planes x and y."""
    pole = mv.w + mv.z                                                         # [] Plane+
    return pole * (pole & Point) - 2.0 * spread**2 * (mv.x * (mv.x & Point) - mv.y * (mv.y & Point))


def make_pinched_horn(r_outer: float) -> Quadric:
    """A pinched horn cyclide on S² is the donut whose inner radius shrinks to a single cusp at the pole."""
    return make_spherical_donut(r_outer / 2.0, r_outer / 2.0)


def make_eccentric_cyclide(r_core: float, r_tube: float, boost: float) -> Quadric:
    """An eccentric Dupin cyclide on S² is a donut dilated toward x, its tube wider on one side than the other."""
    dilation = (mv.xw * (boost / 2.0)).exp()                                   # [] Rotor+
    # A versor carries a map as it carries anything: the point pulled back, the plane pushed forward.
    return dilation >> make_spherical_donut(r_core, r_tube)(dilation << Point)


def make_spherical_cassini(first_angle: float, second_angle: float, threshold: float) -> Quadric:
    """A Cassini oval on S² holds the points whose pairings with the planes tangent at its two foci multiply to less
    than the threshold, in units of the point's weight squared.

    A small threshold gives twin islands, one near the pinch a figure eight, a larger one a peanut with a pinched
    waist, and unequal angles an asymmetric teardrop.
    """
    # The plane z turned toward x by the first angle, and away from it by the second; each, with w, is the plane
    # tangent to S² at a focus.
    first_tangent = mv.w + ((mv.zx * (-first_angle / 2)).exp() >> mv.z)        # [] Plane+
    second_tangent = mv.w + ((mv.zx * (second_angle / 2)).exp() >> mv.z)       # [] Plane+
    return (0.5 * (first_tangent * (second_tangent & Point) + second_tangent * (first_tangent & Point))
            - threshold * mv.w * (mv.w & Point))


def make_spherical_crescent(r_outer: float, r_inner: float, offset: float) -> Quadric:
    """A crescent on S² is the sickle between two eccentric circles, inside the outer and outside the inner."""
    outer = mv.z + np.cos(r_outer) * mv.w                                      # [] Plane+
    inner = ((mv.zx * (-offset / 2)).exp() >> mv.z) + np.cos(r_inner) * mv.w   # [] Plane+
    return 0.5 * (outer * (inner & Point) + inner * (outer & Point))


def make_spherical_spindle(weight_z: float, weight_xy: float, bias: float) -> Quadric:
    """A spindle on S² is a region with two opposite conical poles."""
    return (
        weight_z * mv.z * (mv.z & Point)
        - weight_xy * mv.x * (mv.x & Point)
        - weight_xy * mv.y * (mv.y & Point)
        - bias * mv.w * (mv.w & Point)
    )


def make_spherical_clover(tilt_angle: float, radius: float, bias: float) -> Quadric:
    """A clover on S² is a rounded deltoid with three lobes: the sum of three circle dyads a third of a turn apart."""
    angles = np.radians([0.0, 120.0, 240.0])
    # The plane z tilted toward x, then turned in the plane xy to each of the three angles.
    turns = (mv.xy * (-angles / 2)).exp() * (mv.zx * (-tilt_angle / 2)).exp()  # [lobes] Rotor+
    tilted = turns >> mv.z                                                     # [lobes] Plane+
    circles = tilted + np.cos(radius) * mv.w                                   # [lobes] Plane+
    return (circles * (circles & Point)).sum(axis=0) - bias * mv.w * (mv.w & Point)


def make_spherical_hourglass(tilt: float, waist: float) -> Quadric:
    """An hourglass on S² is a Cassini oval of two foci tilted equally either way along y: two symmetric bulbs
    joined by a narrow waist."""
    # The plane z turned toward y and away from it by the same angle; each, with w, is the plane tangent to S² at a
    # focus.
    first_tangent = mv.w + ((mv.zy * (-tilt / 2)).exp() >> mv.z)               # [] Plane+
    second_tangent = mv.w + ((mv.zy * (tilt / 2)).exp() >> mv.z)               # [] Plane+
    return (0.5 * (first_tangent * (second_tangent & Point) + second_tangent * (first_tangent & Point))
            - waist * mv.w * (mv.w & Point))


def make_spherical_parabola(weight_y: float, linear_x: float, bias: float) -> Quadric:
    """A parabolic bow on S² holds the points whose pairing with the plane y, squared, stays below a linear function
    of their pairing with the plane x."""
    return (
        weight_y * mv.y * (mv.y & Point)
        + 0.5 * linear_x * (mv.x * (mv.w & Point) + mv.w * (mv.x & Point))
        - bias * mv.w * (mv.w & Point)
    )


# --- math -----------------------------------------------------------------------------
def flow(quadrics: Quadric, generator: Bivector, phases: np.ndarray) -> Quadric:
    """The quadrics carried by `(generator * (phase / 2)).exp()`, batched as `[phases, quadrics]`."""
    rotors = (generator * (phases / 2.0)).exp()[:, None]                       # [phases, 1] Rotor+
    return rotors >> quadrics(rotors << Point)
