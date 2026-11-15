"""Spherical quadrics in the conformal algebra of the sphere are polarity maps carried around by a conformal flow.

A circle on the unit sphere, cut out by a plane `n | x == s`, is modelled as the vector `n + s * w`. A point of the
sphere is modelled as a null antivector equal to the sphere's centre `w.dual()` plus a unit spatial direction, so a
circle pairs with a point to `n | x - s`, positive inside the circle, while `w & point == -1` because `w` squares to
minus one. A quadric is a symmetric polarity map `Plane <- Point` built as a sum of circle dyads, and a point lies
inside the region where `quadric(point) & point < 0`. Sums of circle dyads produce rings, conical donuts, Cassini
ovals, Bernoulli lemniscates, crescents, eccentric Dupin cyclides, pinched horns, spindles, clovers, hourglasses, and
parabolic bows. Two intersecting circles meet in the 2-blade `first ^ second`, whose exponential is a conformal rotor
that carries every quadric as `rotor >> quadric(rotor << Point)`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("x+y+z+w-")
ctx = NumpyContext(ga)
mv = ctx.multivector

# A circle on the unit sphere is the plane that cuts it out.
Plane = ga.gatype.vector()                                                     # Vector+
# A point of the unit sphere is a null antivector with intrinsic orientation.
Point = ga.gatype.antivector()                                                 # Antivector-
Quadric = ga.gatype((Plane, Point))                                            # Plane+ <- Point-
Bivector = ga.gatype.bivector()                                                # Bivector+


# --- math -----------------------------------------------------------------------------
def spherical_donut(r_core: float, r_tube: float) -> Quadric:
    """Two concentric circles at angular radii `r_core + r_tube` and `r_core - r_tube` around the north pole form a
    ring quadric (`[] Plane+ <- Point-`) whose quadratic form is negative inside the band between them."""
    outer = mv.z + np.cos(r_core + r_tube) * mv.w                              # [] Plane+
    inner = mv.z + np.cos(r_core - r_tube) * mv.w                              # [] Plane+
    return 0.5 * (outer * (inner & Point) + inner * (outer & Point))


def conical_donut(tilt: float, width: float) -> Quadric:
    """A tilted circle through the north pole minus weighted coordinate-plane dyads defines a limaçon ring quadric
    (`[] Plane+ <- Point-`) whose central hole closes in a conical apex at the pole."""
    # A circle through the north pole, tilted toward x.
    through_pole = mv.w + mv.z + tilt * mv.x                                   # [] Plane+
    return through_pole * (through_pole & Point) - width**2 * (mv.x * (mv.x & Point) + mv.y * (mv.y & Point))


def bernoulli_lemniscate(spread: float) -> Quadric:
    """The point circle at the north pole minus a hyperbolic pair of coordinate-plane dyads defines a figure-eight
    lemniscate quadric (`[] Plane+ <- Point-`) of angular spread `spread`."""
    pole = mv.w + mv.z                                                         # [] Plane+
    return pole * (pole & Point) - 2 * spread**2 * (mv.x * (mv.x & Point) - mv.y * (mv.y & Point))


def pinched_horn(r_outer: float) -> Quadric:
    """A spherical donut whose inner circle shrinks to the north pole forms a pinched horn cyclide quadric
    (`[] Plane+ <- Point-`) with outer angular diameter `r_outer`."""
    return spherical_donut(r_outer / 2, r_outer / 2)


def eccentric_cyclide(r_core: float, r_tube: float, boost: float) -> Quadric:
    """A conformal dilation along `x` carries a concentric spherical donut into an eccentric Dupin cyclide quadric
    (`[] Plane+ <- Point-`) whose tube widens toward `+x`."""
    dilation = (mv.xw * (boost / 2)).exp()                                     # [] Rotor+
    # A versor carries a polarity map by pulling the input point back and pushing the output plane forward.
    return dilation >> spherical_donut(r_core, r_tube)(dilation << Point)


def spherical_cassini(first_angle: float, second_angle: float, threshold: float) -> Quadric:
    """Two tangent circles at focal angles `first_angle` and `-second_angle` along `x` define a spherical Cassini oval
    quadric (`[] Plane+ <- Point-`) at level `threshold`.

    Increasing `threshold` transitions the region from twin islands through a figure-eight pinch to a single peanut,
    while unequal focal angles produce an asymmetric teardrop.
    """
    # Turning z toward +x and -x gives the two focal directions; adding w forms the tangent circles at the two foci.
    first_turn = (mv.zx * (-first_angle / 2)).exp()                            # [] Rotor+
    second_turn = (mv.zx * (second_angle / 2)).exp()                           # [] Rotor+
    first_tangent = mv.w + (first_turn >> mv.z)                                # [] Plane+
    second_tangent = mv.w + (second_turn >> mv.z)                              # [] Plane+
    return (0.5 * (first_tangent * (second_tangent & Point) + second_tangent * (first_tangent & Point))
            - threshold * mv.w * (mv.w & Point))


def spherical_crescent(r_outer: float, r_inner: float, offset: float) -> Quadric:
    """An outer circle of radius `r_outer` and an eccentric inner circle shifted by `offset` form a crescent quadric
    (`[] Plane+ <- Point-`) negative inside the outer circle and outside the inner one."""
    outer = mv.z + np.cos(r_outer) * mv.w                                      # [] Plane+
    shift = (mv.zx * (-offset / 2)).exp()                                      # [] Rotor+
    inner = (shift >> mv.z) + np.cos(r_inner) * mv.w                           # [] Plane+
    return 0.5 * (outer * (inner & Point) + inner * (outer & Point))


def spherical_spindle(weight_z: float, weight_xy: float, bias: float) -> Quadric:
    """Opposing `z` and equatorial `x`, `y` plane dyads shifted by `bias` define a spindle quadric
    (`[] Plane+ <- Point-`) with two opposite conical poles."""
    return (
        weight_z * mv.z * (mv.z & Point)
        - weight_xy * mv.x * (mv.x & Point)
        - weight_xy * mv.y * (mv.y & Point)
        - bias * mv.w * (mv.w & Point)
    )


def spherical_clover(tilt_angle: float, radius: float, bias: float) -> Quadric:
    """Three circle dyads spaced a third of a turn apart at polar tilt `tilt_angle` sum to a three-lobed clover quadric
    (`[] Plane+ <- Point-`)."""
    angles = np.radians([0, 120, 240])
    # Tilting z toward x and turning around xy places the three lobe centres symmetrically around the pole.
    turns = (mv.xy * (-angles / 2)).exp() * (mv.zx * (-tilt_angle / 2)).exp()  # [3] Rotor+
    tilted = turns >> mv.z                                                     # [3] Plane+
    circles = tilted + np.cos(radius) * mv.w                                   # [3] Plane+
    return (circles * (circles & Point)).sum(axis=0) - bias * mv.w * (mv.w & Point)


def spherical_hourglass(tilt: float, waist: float) -> Quadric:
    """Two tangent circles tilted by `tilt` and `-tilt` along `y` define a symmetric hourglass Cassini quadric
    (`[] Plane+ <- Point-`) with waist parameter `waist`."""
    # Turning z either way in the yz plane places the two foci symmetrically above and below the pole.
    first_turn = (mv.zy * (-tilt / 2)).exp()                                   # [] Rotor+
    second_turn = (mv.zy * (tilt / 2)).exp()                                   # [] Rotor+
    first_tangent = mv.w + (first_turn >> mv.z)                                # [] Plane+
    second_tangent = mv.w + (second_turn >> mv.z)                              # [] Plane+
    return (0.5 * (first_tangent * (second_tangent & Point) + second_tangent * (first_tangent & Point))
            - waist * mv.w * (mv.w & Point))


def spherical_parabola(weight_y: float, linear_x: float, bias: float) -> Quadric:
    """A squared `y`-plane pairing balanced against a linear `x`-plane pairing defines a parabolic bow quadric
    (`[] Plane+ <- Point-`) on the sphere."""
    return (
        weight_y * mv.y * (mv.y & Point)
        + 0.5 * linear_x * (mv.x * (mv.w & Point) + mv.w * (mv.x & Point))
        - bias * mv.w * (mv.w & Point)
    )


def conformal_flow(quadrics: Quadric, generator: Bivector, n_frames: int) -> Iterator[Quadric]:
    """Yields the `[n_quadrics]` quadrics (`[n_quadrics] Plane+ <- Point-`) across `n_frames` uniformly spaced phases
    of one full conformal turn generated by `generator`."""
    for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        rotor = (generator * (phase / 2)).exp()                                # [] Rotor+
        yield rotor >> quadrics(rotor << Point)


# --- plumbing -------------------------------------------------------------------------
def sphere_points(xyz: np.ndarray) -> Point:
    """Unit Cartesian coordinates of shape `[..., 3]` lift to null conformal points (`[...] Point-`) on the unit sphere
    by adding the sphere's centre to their spatial directions."""
    return mv.w.dual() + (stack((mv.x.dual(), mv.y.dual(), mv.z.dual())) * xyz).sum(axis=-1)
