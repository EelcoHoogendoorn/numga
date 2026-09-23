"""Ray tracing Dupin cyclides on the 3-sphere, in the conformal model of S³. Core mathematics.

A point of S³ is a unit vector p of R⁴, carried as the null vector p + e in R_{4,1};
spheres are vectors, and points are their duals. The eye sits at w and looks along -x.
Turning it towards a direction d moves it along a great circle, origin + sin(t) linear(d) +
(1 - cos(t)) quadratic(d, d); in the half angle u = tan(t / 2) that circle is a parabola
whose bend is the antipode of the eye. Substituting the parabola into a quadric's form gives
a polynomial in u of degree four, solved per pixel for the nearest hit along the whole circle.

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
Sphere = ga.gatype.vector()                   # spheres, and great spheres
Point = ga.gatype.antivector()                # points: duals of null vectors
Direction = ga.gatype.from_blades("x y z")    # directions at the eye
Motor = ga.gatype.rotor()                     # rotations of R⁴ and conformal dilations
Quadric = ga.gatype((Sphere, Point))          # polar sphere <- point
PointMap = ga.gatype((Point, Point))

origin = (mv.w + mv.e).dual()                 # the eye, at w on S³
antipode = (-mv.w + mv.e).dual()              # the point opposite the eye

# The eye turned by t towards a unit d: origin + sin(t) ray_linear(d) + (1 - cos(t)) ray_quadratic(d, d).
metric = Direction | Direction
ray_rotation = 0.5 * (Direction ^ mv.w)
ray_linear = ray_rotation.commutator(origin) * 2
ray_quadratic = ray_rotation.commutator(ray_linear) * 2
# Times 1 + u², with u = tan(t / 2): origin + 2 u ray_linear(d) + u² ray_bend(d, d), the bend the antipode.
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

    Arguments are ascending coefficients of a polynomial in u = tan(t / 2). The reversed
    polynomial is solved, for r = 1 / u; each real root maps to t = 2 * arctan2(1, r) in
    (0, 2 pi), so hits past the antipode (r < 0) and at it (r = 0) are ordered along the
    whole circle.
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
    """All points at angle `tube` from the great circle in the xy plane: z² + w² = sin²(tube)."""
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
    """Headlight facing and hit mask per surface along the leading axis and per unit pixel direction.

    Facing is the cosine between the ray and the surface normal at the nearest hit.
    """
    camera_map = mv.rotor() >> Point                                        # the eye stays at w
    form = camera_map & surfaces(camera_map)                                # zero on the surface

    # form(X, X) along X = origin + 2 u ray_linear(d) + u² ray_bend(d, d), one form per power of u:
    constant = form(origin, origin)
    linear = 4 * form(origin, ray_linear)
    quadratic = 4 * form(ray_linear, ray_linear) + 2 * form(origin, ray_bend)
    cubic = 4 * form(ray_linear, ray_bend)
    quartic = form(ray_bend, ray_bend)
    angle = nearest_angle(constant[:, None], linear[:, None](pixels), quadratic[:, None](pixels, pixels),
                          cubic[:, None](pixels, pixels, pixels), quartic[:, None](pixels, pixels, pixels, pixels))
    visible = np.isfinite(angle)
    t = np.where(visible, angle, 0.0)
    hit = origin + ray_linear(pixels) * np.sin(t) + ray_quadratic(pixels, pixels) * (1 - np.cos(t))
    velocity = ray_linear(pixels) * np.cos(t) + ray_quadratic(pixels, pixels) * np.sin(t)

    # At the hit the polar sphere is the tangent sphere; its length is that of the surface gradient.
    polar = surfaces[:, None](hit)
    return -(polar & velocity) / (polar | polar).abs().square_root(), visible
