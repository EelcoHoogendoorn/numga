"""Ray tracing Dupin cyclides on the 3-sphere, in the conformal model of S³. Core mathematics.

A point of S³ is a unit vector p of R⁴, carried as the null vector `p + mv.e` in the algebra
`x+y+z+w+e-`; spheres are vectors, and points are their duals. The eye sits at w and looks along
-x. Turning it towards a direction d moves it along a great circle,
`origin + ray_linear(d) * np.sin(t) + ray_quadratic(d, d) * (1 - np.cos(t))`; in the half angle
`u = np.tan(t / 2)` that circle is a parabola whose bend is the antipode of the eye. Substituting
the parabola into a quadric's form gives a polynomial in u of degree four, solved per pixel for
the nearest hit along the whole circle.

The surfaces start as a spherical cylinder, a tube around a great circle, and a spherical
cone. A conformal dilation bends the cylinder's core into a small circle: a torus when the dilation
keeps the spin about the core, a Dupin cyclide when it does not. A dilation of the cone pulls its
two vertices together into a spindle cyclide.

This module never imports a plotting library.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext

ga = Algebra("x+y+z+w+e-")
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
# Spheres and great spheres are vectors, points the duals of null vectors, and directions at the eye
# the vectors in x, y and z. Motors rotate R⁴ and dilate conformally; a quadric maps a point to its
# polar sphere.
Sphere = ga.gatype.vector()
Point = ga.gatype.antivector()
Direction = ga.gatype.from_blades("x y z")
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Sphere, Point))          # Sphere <- Point
PointMap = ga.gatype((Point, Point))

# The eye, at w on S³, and the point opposite it.
origin = (mv.w + mv.e).dual()                 # [] Point
antipode = (-mv.w + mv.e).dual()              # [] Point

# A flat chart about the eye, the stereographic projection from its antipode: the eye's null vector is the chart's
# origin and the antipode's is its point at infinity. Constructions of the flat conformal model, written with these
# two, draw the same surfaces on S³.
chart_origin = (mv.w + mv.e) * 0.5
chart_infinity = mv.e - mv.w

# The eye turned by t towards a unit d: origin + ray_linear(d) * np.sin(t) + ray_quadratic(d, d) * (1 - np.cos(t)).
metric = Direction | Direction
ray_rotation = 0.5 * (Direction ^ mv.w)
ray_linear = ray_rotation.commutator(origin) * 2
ray_quadratic = ray_rotation.commutator(ray_linear) * 2
# Times 1 + u ** 2, with u = np.tan(t / 2): origin + ray_linear(d) * 2 * u + ray_bend(d, d) * u ** 2, the bend the
# antipode.
ray_bend = origin * metric + ray_quadratic * 2


# --- plumbing -------------------------------------------------------------------------
def pixel_grid(shape: tuple[int, int], fov: float) -> np.ndarray:
    """(n_pixels, 3) x, y, z coordinates of pinhole pixel directions looking along -x, row-major."""
    height, width = shape
    u = (2 * (np.arange(width) + 0.5) / width - 1)[None, :]
    v = (1 - 2 * (np.arange(height) + 0.5) / height)[:, None] * height / width
    x, y, z = np.broadcast_arrays(-1.0, u * np.tan(fov / 2), v * np.tan(fov / 2))
    return np.stack([x, y, z], axis=-1).reshape(-1, 3)


def sensor(shape: tuple[int, int], fov: float) -> Direction:
    """Unit pixel directions at the eye, row-major."""
    return mv(Direction, pixel_grid(shape, fov)).normalized()


def nearest_angle(*coefficients: Scalar) -> np.ndarray:
    """Smallest positive angle of a great-circle hit, or inf on a miss.

    Arguments are ascending coefficients of a polynomial in `u = np.tan(t / 2)`. The reversed
    polynomial is solved, for `r = 1 / u`; each real root maps to `t = 2 * np.arctan2(1, r)`
    between 0 and `2 * np.pi`, so hits past the antipode (`r < 0`) and at it (`r == 0`) are
    ordered along the whole circle.
    """
    c = np.stack(np.broadcast_arrays(*(c.to_array() for c in coefficients)), axis=-1)
    degree = c.shape[-1] - 1
    companion = np.zeros(c.shape[:-1] + (degree, degree))
    companion[..., 1:, :-1] = np.eye(degree - 1)
    companion[..., :, -1] = -c[..., degree:0:-1] / c[..., :1]
    roots = np.linalg.eigvals(companion)
    angles = 2 * np.arctan2(1.0, roots.real)
    # Near-double roots come back with round-off imaginary parts; they are grazing hits.
    real = np.abs(roots.imag) <= 1e-6 * (1 + np.abs(roots.real))
    return np.where(real, angles, np.inf).min(axis=-1)


# --- shapes ---------------------------------------------------------------------------
def cylinder(tube: np.ndarray) -> Quadric:
    """All points at angle `tube` from the great circle in the xy plane: `z ** 2 + w ** 2 == np.sin(tube) ** 2`."""
    return mv.z * (mv.z & Point) + mv.w * (mv.w & Point) - np.sin(tube) ** 2 * mv.e * (mv.e & Point)


def cone(opening: float) -> Quadric:
    """The cone with vertex z, its axis towards x and half-opening `opening`; its geodesics from z
    meet again at -z, its second vertex."""
    return mv.x * (mv.x & Point) + np.cos(opening) ** 2 * (mv.z * (mv.z & Point) - mv.e * (mv.e & Point))


def dilation(aim: Sphere, strength: np.ndarray) -> Motor:
    """The conformal map of S³ that pushes points towards `aim`, a point as a unit vector, keeping it
    and its antipode fixed. Seen from `aim` it is a uniform scaling."""
    return ((aim ^ mv.e) * (strength / 2)).exp()


# --- math -----------------------------------------------------------------------------
def trace(surfaces: Quadric, pixels: Direction) -> tuple[Scalar, np.ndarray]:
    """Headlight facing, and the angle t of the nearest hit, per surface along the leading axis and per unit pixel
    direction; t is inf on a miss.

    Facing is the cosine between the ray and the surface normal at the nearest hit.
    """
    # Zero on the surface:
    form = Point & surfaces                                                 # [surfaces] Scalar <- (Point, Point)

    # form(X, X) along X = origin + ray_linear(d) * 2 * u + ray_bend(d, d) * u ** 2, one form per power of u:
    constant = form(origin, origin)
    linear = 4 * form(origin, ray_linear)
    quadratic = 4 * form(ray_linear, ray_linear) + 2 * form(origin, ray_bend)
    cubic = 4 * form(ray_linear, ray_bend)
    quartic = form(ray_bend, ray_bend)
    angle = nearest_angle(constant[:, None], linear[:, None](pixels), quadratic[:, None](pixels, pixels),
                          cubic[:, None](pixels, pixels, pixels), quartic[:, None](pixels, pixels, pixels, pixels))
    t = np.where(np.isfinite(angle), angle, 0.0)
    hit = origin + ray_linear(pixels) * np.sin(t) + ray_quadratic(pixels, pixels) * (1 - np.cos(t))
    velocity = ray_linear(pixels) * np.cos(t) + ray_quadratic(pixels, pixels) * np.sin(t)

    # At the hit the polar sphere is the tangent sphere; its length is that of the surface gradient.
    polar = surfaces[:, None](hit)
    return -(polar & velocity) / (polar | polar).abs().square_root(), angle
