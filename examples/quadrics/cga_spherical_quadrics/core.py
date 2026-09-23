"""Spherical quadrics in the conformal algebra Cl(3, 1), carried around by a conformal flow.

A point of S² is a null vector (x, y, z, 1). A quadric is a map Q : Vector -> Plane, and the
point is inside where Q(p) & p < 0. Sums of plane dyads give conical donuts, peanuts, twin
islands, lemniscates, crescents, Dupin cyclides, pinched horns, spindles, teardrops, triadic
clovers, hourglasses and parabolic bows. Two off-centre circles on S² meet in a 2-blade, and
its exponential exp(θ (c1 ^ c2) / 2) is a conformal rotor that moves every quadric along.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+w-")
ctx = NumpyContext(ga)
mv = ctx.multivector

Vector = ga.gatype.vector()
Plane = ga.gatype.antivector()
Quadric = ga.gatype((Plane, Vector))
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()

# Dual basis planes via pseudoscalar I = xyzw
I = mv.xyzw
px = (mv.x * I).cast(Plane.output_subspace)
py = (mv.y * I).cast(Plane.output_subspace)
pz = (mv.z * I).cast(Plane.output_subspace)
pw = (-mv.w * I).cast(Plane.output_subspace)
p_wz = pw - pz


def point(xyz: np.ndarray) -> Vector:
    """Null vectors (x, y, z, 1) of the points of S² at (..., 3) unit coordinates."""
    return mv.x * xyz[..., 0] + mv.y * xyz[..., 1] + mv.z * xyz[..., 2] + mv.w


def moved(Q: Quadric, versor: Rotor) -> Quadric:
    """A quadric carried by a versor: pull the point back, push the plane forward."""
    return versor >> Q(versor << Vector)


# --- shapes ---------------------------------------------------------------------------
def make_spherical_donut(r_core: float, r_tube: float) -> Quadric:
    """Spherical donut (torus) extensor Q : Vector -> Plane on S²."""
    c_out = np.cos(r_core + r_tube)
    c_in = np.cos(r_core - r_tube)
    z0 = (c_out + c_in) / 2.0
    dz = (c_in - c_out) / 2.0
    return (
        pz * (pz & Vector)
        - z0 * (pz * (pw & Vector) + pw * (pz & Vector))
        + (z0**2 - dz**2) * pw * (pw & Vector)
    )


def make_conical_donut(a: float, b: float) -> Quadric:
    """Spherical donut (Limaçon) with central hole meeting in a razor conical apex on S²."""
    line = pw - pz - a * px
    return line * (line & Vector) - b**2 * (px * (px & Vector) + py * (py & Vector))


def make_bernoulli_lemniscate(scale_a: float) -> Quadric:
    """True Lemniscate of Bernoulli figure-8 on S²: (w - z)² - 2 a² (x² - y²) < 0."""
    return p_wz * (p_wz & Vector) - 2.0 * scale_a**2 * (px * (px & Vector) - py * (py & Vector))


def make_pinched_horn(r_outer: float) -> Quadric:
    """Pinched horn cyclide on S²: the donut whose inner radius shrinks to a single cusp at the pole."""
    return make_spherical_donut(r_outer / 2.0, r_outer / 2.0)


def make_eccentric_cyclide(r_core: float, r_tube: float, boost_beta: float) -> Quadric:
    """Eccentric Dupin cyclide on S² with unequal tube width via Lorentz boost."""
    return moved(make_spherical_donut(r_core, r_tube), (mv.xw * (boost_beta / 2.0)).exp())


def make_spherical_cassini(alpha1: float, alpha2: float, c_threshold: float) -> Quadric:
    """Spherical Cassini oval quadric (1 - n1.x)(1 - n2.x) < C * w² on S².

    Yields twin islands (C small), figure-8 lemniscates (C near pinch),
    pinched-waist peanuts (C larger), or asymmetric teardrops (alpha1 != alpha2).
    """
    p1 = pw - (np.sin(alpha1) * px + np.cos(alpha1) * pz)
    p2 = pw - (-np.sin(alpha2) * px + np.cos(alpha2) * pz)
    return 0.5 * (p1 * (p2 & Vector) + p2 * (p1 & Vector)) - c_threshold * pw * (pw & Vector)


def make_spherical_crescent(r_outer: float, r_inner: float, offset: float) -> Quadric:
    """Spherical crescent moon (sickle) bounded by two eccentric circles on S²."""
    c_outer = pz - np.cos(r_outer) * pw
    c_inner = (np.sin(offset) * px + np.cos(offset) * pz) - np.cos(r_inner) * pw
    return 0.5 * (c_outer * (c_inner & Vector) + c_inner * (c_outer & Vector))


def make_spherical_spindle(weight_z: float, weight_xy: float, bias: float) -> Quadric:
    """Spherical spindle with two opposite conical poles on S²."""
    return (
        weight_z * pz * (pz & Vector)
        - weight_xy * px * (px & Vector)
        - weight_xy * py * (py & Vector)
        - bias * pw * (pw & Vector)
    )


def make_spherical_clover(tilt_angle: float, radius: float, bias: float) -> Quadric:
    """Triadic 3-lobed rounded deltoid (cloverleaf) on S²: three circle dyads, 120° apart."""
    angles = np.radians([0.0, 120.0, 240.0])
    circles = (px * (np.sin(tilt_angle) * np.cos(angles)) + py * (np.sin(tilt_angle) * np.sin(angles))
               + np.cos(tilt_angle) * pz - np.cos(radius) * pw)
    return (circles * (circles & Vector)).sum(axis=0) - bias * pw * (pw & Vector)


def make_spherical_hourglass(theta: float, c_waist: float) -> Quadric:
    """True vertical hourglass on S²: two symmetric bulbs connected by a narrow waist."""
    p1 = pw - (np.sin(theta) * py + np.cos(theta) * pz)
    p2 = pw - (-np.sin(theta) * py + np.cos(theta) * pz)
    return 0.5 * (p1 * (p2 & Vector) + p2 * (p1 & Vector)) - c_waist * pw * (pw & Vector)


def make_spherical_parabola(weight_y: float, linear_x: float, bias: float) -> Quadric:
    """Parabolic bow curve on S²."""
    return (
        weight_y * py * (py & Vector)
        - 0.5 * linear_x * (px * (pw & Vector) + pw * (px & Vector))
        - bias * pw * (pw & Vector)
    )


# --- math -----------------------------------------------------------------------------
def make_circle_intersection_vortex(
    center1: np.ndarray, radius1: float, center2: np.ndarray, radius2: float,
) -> Bivector:
    """Intersection 2-blade of two off-center circles on S²: each circle is the vector of its
    centre with weight cos(radius)."""
    c1 = mv.x * center1[0] + mv.y * center1[1] + mv.z * center1[2] + mv.w * np.cos(radius1)
    c2 = mv.x * center2[0] + mv.y * center2[1] + mv.z * center2[2] + mv.w * np.cos(radius2)
    return (c1 ^ c2).normalized()


def flow(quadrics: Quadric, generator: Bivector, phases: np.ndarray) -> Quadric:
    """The quadrics carried by exp(phase · generator / 2), batched as (phases, quadrics)."""
    rotors = (generator * (phases / 2.0)).exp()
    return moved(quadrics, rotors[:, None])
