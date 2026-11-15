"""Ray tracing Dupin cyclides on the 3-sphere in conformal geometric algebra.

A point of the 3-sphere is carried as a null antivector adding its unit four-space direction to one
unit of the ambient centre, while spheres are vectors and quadrics are polarity maps from points to
their polar spheres. Turning the eye at `w` toward a tangent direction traces a great circle that
becomes a parabola in the half-angle tangent `u` whose quadratic bend is the antipode of the eye.
Substituting that parabola into a quadric's bilinear form yields a quartic in `u` whose five
coefficient maps are assembled once before binding pixel directions.

Conformal dilations bend a spherical cylinder around a great circle into symmetric tori or lopsided
Dupin cyclides, and pull the two vertices of a spherical cone together into spindle cyclides.

This module never imports a plotting library.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, concatenate
from numga.algebras import CGA3D
from examples.quadrics.cyclides.numerical import nearest_hit_angles, pixel_offsets

ga = CGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()                                                    # Scalar+
# Spheres and great spheres are vectors, and points of S³ are null antivectors. Tangent directions at
# the eye lie in the duals of x, y, and z, doubling as displacements in the flat chart about the eye.
Sphere = ga.gatype.vector()                                                    # Vector+
Point = ga.gatype.antivector()                                                 # Antivector-
Upright = Sphere.derive.within(Algebra("x+y+z+"))                              # Vector+
Direction = Upright.derive.dual                                                # Antivector-
Motor = ga.gatype.rotor()                                                      # Rotor+
Bivector = ga.gatype.bivector()                                                # Bivector+
Quadric = ga.gatype((Sphere, Point))                                           # Sphere+ <- Point-

# The eye sits at w on S³ and its antipode sits opposite at -w.
origin = mv.w.dual() + mv.e.dual()                                             # [] Point-
antipode = mv.e.dual() - mv.w.dual()                                           # [] Point-

# Point spheres at the eye and its antipode serve as origin and infinity of the stereographic flat chart.
chart_origin = (mv.w + mv.e) * 0.5                                             # [] Sphere+
chart_infinity = mv.e - mv.w                                                   # [] Sphere+

# Joining w with a tangent direction d spans the R⁴ rotation plane whose dual generates the great-circle ray.
ray_rotation = 0.5 * (mv.w.dual() & Direction).dual()                          # [] Bivector+ <- Direction-
ray_linear = ray_rotation.commutator(origin) * 2                               # [] Point- <- Direction-
ray_quadratic = ray_rotation.commutator(ray_linear) * 2                        # [] Point- <- (Direction-, Direction-)
# Clearing 1 + u**2 in u = tan(t / 2) turns the great circle into a parabola whose u**2 bend is the antipode.
ray_bend = ray_quadratic * 2 + origin * Direction.dot(Direction)               # [] Point- <- (Direction-, Direction-)


# --- math -----------------------------------------------------------------------------
def cylinder_quadrics(tubes: np.ndarray) -> Quadric:
    """Points at angular radius `tubes` from the `xy` great circle satisfy a spherical-cylinder
    polarity map (`[...] Sphere+ <- Point-`)."""
    return mv.z * (mv.z & Point) + mv.w * (mv.w & Point) - np.sin(tubes) ** 2 * mv.e * (mv.e & Point)


def cone_quadric(opening: float) -> Quadric:
    """Geodesics leaving `z` and `-z` at half-opening `opening` around the `x` axis form a
    two-vertex spherical cone (`[] Sphere+ <- Point-`)."""
    return mv.x * (mv.x & Point) + np.cos(opening) ** 2 * (mv.z * (mv.z & Point) - mv.e * (mv.e & Point))


def placement_motors(toward: Point, ahead: np.ndarray, tilt: np.ndarray) -> Motor:
    """Motors (`[...] Motor+`) tilt about the `yw` plane and carry `toward` onto the point `ahead`
    radians along `-x` in front of the eye."""
    ahead_turn = (mv.wx * (ahead / 2)).exp()                                   # [...] Motor+
    target = ahead_turn >> mv.w.dual()                                         # [...] Point-
    onto_target = (1 + target * toward.inverse()).normalized()                 # [...] Motor+
    tilting = (mv.yw * (tilt / 2)).exp()                                       # [...] Motor+
    return onto_target * tilting


def coordinate_rotation(angles: list[float]) -> Motor:
    """Six coordinate-plane angles in `xy`, `xz`, `xw`, `yz`, `yw`, and `zw` compose into a
    four-dimensional rotation (`[] Motor+`)."""
    rotor = mv.rotor()                                                         # [] Motor+
    for plane, angle in zip((mv.xy, mv.xz, mv.xw, mv.yz, mv.yw, mv.zw), angles):
        step = (plane * angle).exp()                                           # [] Motor+
        rotor = rotor * step                                                   # [] Motor+
    return rotor


def camera_view_motor(centre: Point, angles: list[float], ahead: float, yaw: float, pitch: float) -> Motor:
    """Rotating by `angles`, carrying the rotated `centre` to `ahead` radians in front of the eye,
    and yawing and pitching at `w` gives a view motor (`[] Motor+`)."""
    body_turn = coordinate_rotation(angles)                                    # [] Motor+
    ahead_turn = (mv.wx * (ahead / 2)).exp()                                   # [] Motor+
    target = ahead_turn >> mv.w.dual()                                         # [] Point-
    rotated_centre = body_turn >> centre                                       # [] Point-
    carry = (1 + target * rotated_centre.inverse()).normalized()               # [] Motor+
    yawing = (mv.xy * (yaw / 2)).exp()                                         # [] Motor+
    pitching = (mv.xz * (pitch / 2)).exp()                                     # [] Motor+
    return yawing * pitching * carry * body_turn


def flat_camera_motor(position: Direction, target: Direction) -> Motor:
    """Inverting the pose of a pinhole camera at `position` aimed at `target` brings the flat
    chart's camera onto the eye at `w` looking along `-x` (`[] Motor+`)."""
    sight = target - position                                                  # [] Direction-
    forward = sight / sight.antinorm()                                         # [] Direction-
    upright = mv.z.dual() - (mv.z.dual() | forward) * forward.inverse()        # [] Direction-
    up = upright / upright.antinorm()                                          # [] Direction-
    aim = (1 + forward * (-mv.x.dual()).inverse()).normalized()                # [] Motor+
    aimed_up = aim >> mv.z.dual()                                              # [] Direction-
    roll = (1 + up * aimed_up.inverse()).normalized()                          # [] Motor+
    translation = ((antipode & position).dual() * (-1 / 2)).exp()              # [] Motor+
    return (translation * roll * aim).inverse()


def surface_trace(surfaces: Quadric, pixels: Direction) -> tuple[Scalar, Scalar]:
    """Tracing `surfaces` along unit `pixels` gives the headlight cosine and nearest hit angle
    (`[...] Scalar+`, `[...] Scalar+`), with `inf` on a miss."""
    form = Point & surfaces                                                    # [...] Scalar+ <- (Point-, Point-)

    # Evaluating form(X, X) along the parabolic ray collects five direction maps, one per power of u = tan(t / 2).
    constant = form(origin, origin)                                            # [...] Scalar+
    linear = 4 * form(origin, ray_linear)                                      # [...] Scalar+ <- Direction-
    quadratic = 4 * form(ray_linear, ray_linear) + 2 * form(origin, ray_bend)  # [...] Scalar+ <- (Direction-, Direction-)
    cubic = 4 * form(ray_linear, ray_bend)                                     # [...] Scalar+ <- (Direction-, Direction-, Direction-)
    quartic = form(ray_bend, ray_bend)                                         # [...] Scalar+ <- (Direction-, Direction-, Direction-, Direction-)

    angles = nearest_hit_angles([
        constant[:, None],
        linear[:, None](pixels),
        quadratic[:, None](pixels, pixels),
        cubic[:, None](pixels, pixels, pixels),
        quartic[:, None](pixels, pixels, pixels, pixels),
    ])                                                                         # [...] Scalar+
    clamped = angles.clip(0.0, 2 * np.pi)                                      # [...] Scalar+
    hits = (origin + ray_linear(pixels) * clamped.sin()
            + ray_quadratic(pixels, pixels) * (1 - clamped.cos()))             # [...] Point-
    velocities = (ray_linear(pixels) * clamped.cos()
                  + ray_quadratic(pixels, pixels) * clamped.sin())             # [...] Point-

    # The polar sphere at a surface point is tangent there, and its reverse norm scales with the surface gradient.
    polars = surfaces[:, None](hits)                                           # [...] Sphere+
    facings = -(polars & velocities) / (polars | polars).abs().square_root()   # [...] Scalar+
    return facings, angles


def vortex_states(torus_surface: Quadric, circle: Bivector, pixels: Direction,
                  n_frames: int) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields `(facings, angles)` (`[1, ...] Scalar+`) as `torus_surface` flows once around unit
    bivector `circle` over `n_frames` frames."""
    phases = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    flows = (circle * (phases / 2)).exp()                                      # [n_frames] Motor+
    for surface in flows >> torus_surface(flows << Point):
        yield surface_trace(surface.reshape(1), pixels)


def linked_vortex_states(cyclide_surface: Quadric, ring_surface: Quadric, circle: Bivector,
                         camera: Motor, pixels: Direction, n_frames: int) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields `(facings, angles)` (`[2, ...] Scalar+`) as `cyclide_surface` flows once around unit
    bivector `circle` linked with static `ring_surface` over `n_frames` frames."""
    placed_ring = (camera >> ring_surface(camera << Point)).reshape(1)         # [1] Sphere+ <- Point-
    ring_facing, ring_angle = surface_trace(placed_ring, pixels)               # [1, ...] Scalar+, [1, ...] Scalar+
    phases = np.linspace(0.0, 2 * np.pi, n_frames, endpoint=False)
    flows = (circle * (phases / 2)).exp()                                      # [n_frames] Motor+
    for motion in camera * flows:
        placed_cyclide = (motion >> cyclide_surface(motion << Point)).reshape(1)  # [1] Sphere+ <- Point-
        facing, angle = surface_trace(placed_cyclide, pixels)                  # [1, ...] Scalar+, [1, ...] Scalar+
        yield concatenate([facing, ring_facing]).abs(), concatenate([angle, ring_angle])


# --- plumbing -------------------------------------------------------------------------
def sensor_directions(image_shape: tuple[int, int], fov: float) -> Direction:
    """Unit pixel directions (`[...] Direction-`) at the eye look along `-x` with `z` up in
    row-major order."""
    right, up = pixel_offsets(image_shape, fov)
    rays = -mv.x.dual() + mv.y.dual() * right + mv.z.dual() * up               # [...] Direction-
    return rays / rays.antinorm()
