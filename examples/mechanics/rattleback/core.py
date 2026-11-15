"""Rolling dynamics and quadric inertia of a rattleback on a horizontal table in 3D projective geometric algebra.

Points and directions are intrinsic antivectors, while planes are extrinsic vectors. The ellipsoidal hull's support
map sends the table normal in the body frame to the displacement from the hull centre to the lowest point,
`-shape(normal) / height`, with `height = (normal & shape(normal)).square_root()`. Translating a body spin from the
centre of mass at the origin to that contact point builds `rolling = contact_shift >> Spin`, the map from spins to
rigid rates that hold the contact instantaneously at rest. Pulling forques back through `rolling.adjugate()` removes
the unknown contact reaction because forces through a resting contact do no work on rolling rates, while
`rolling.adjugate()(inertia(rolling))` forms the reduced rolling inertia on spins. Differentiating the moving contact
point across the hull supplies the rate correction `rolling_change` alongside the Euler commutator in the acceleration
solve.
"""

from __future__ import annotations

from collections.abc import Iterator

from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga

ctx = NumpyContext(ga)
mv = ctx.multivector

Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
Direction = Point.derive.degenerate
Spin = ga.gatype.bivector().derive.nondegenerate
Rate = ga.gatype.bivector()
Motor = ga.gatype.rotor()
Forque = ga.gatype.antibivector()
Shape = ga.gatype((Direction, Plane))
Rolling = ga.gatype((Rate, Spin))
Inertia = ga.gatype((Forque, Rate))
Quadric = ga.gatype((Plane, Point))

infinity = mv.w                                                       # [] Plane+
origin = infinity.dual()                                              # [] Point-
up = mv.z.dual()                                                      # [] Direction-
mirrors = stack((mv.x, mv.y, mv.z))                                   # [3] Plane+


# --- math

def half_space(plane: Plane) -> Quadric:
    """Symmetrizing `plane` with `infinity` forms a quadric negative on `plane`'s inner side (`[...] Plane+ <- Point-`)."""
    return (plane * (infinity & Point) + infinity * (plane & Point)) / 2


def ball(centre: Point, radius: float) -> Quadric:
    """Coordinate planes offset through unit-weight `centre` points define solid sphere quadrics (`[...] Plane+ <- Point-`) of `radius`."""
    sphere_planes = mirrors - infinity * (mirrors & centre[..., None])  # [..., 3] Plane+
    return (sphere_planes * (sphere_planes & Point)).sum(axis=-1) / radius**2 - infinity * (infinity & Point)


def mass_distribution(wood: Quadric, balls: Quadric, grid: Point, wood_density: float,
                      steel_density: float, mass: float) -> tuple[Point, Point, Inertia]:
    """Interior grid points (`[...] Point-`) weighted by wood and steel densities determine the centre of mass (`[] Point-`) and body inertia (`[] Forque- <- Rate+`)."""
    wood_mask = ((wood[:, None](grid) & grid) <= 0).all(axis=0)
    steel_mask = ((balls[:, None](grid) & grid) <= 0).any(axis=0)
    densities = wood_density * wood_mask + steel_density * steel_mask
    body_mask = densities > 0
    samples = grid[body_mask]                                         # [...] Point-
    sample_masses = mass * densities[body_mask] / densities[body_mask].sum()
    mass_centre = (samples * (sample_masses / mass)).sum(axis=0)      # [] Point-
    # Joining each mass-weighted point with its velocity under Rate sums the translational and rotational inertia.
    body_points = origin + (samples - mass_centre)                    # [...] Point-
    inertia = (body_points * sample_masses).regressive(body_points.commutator(Rate)).sum(axis=0)  # [] Forque- <- Rate+
    return samples, mass_centre, inertia


def contact_geometry(motor: Motor, shape: Shape, centre: Direction,
                     up: Direction) -> tuple[Plane, Direction, Scalar, Rolling]:
    """The body-frame table normal determines the ellipsoid support offset, hull centre height, and rolling map (`[...] Rate+ <- Spin+`)."""
    normal = (motor << up).dual_inverse()                             # [...] Plane+
    height = (normal & shape(normal)).square_root()                   # [...] Scalar+
    support = -shape(normal) / height                                 # [...] Direction-

    # Translating a spin from the centre of mass to the contact point turns the body instantaneously about the contact.
    contact_shift = ((centre + support) * origin.inverse() / 2).exp()  # [...] Motor+
    rolling = contact_shift >> Spin                                   # [...] Rate+ <- Spin+
    return normal, support, height, rolling


def dynamics(motor: Motor, spin: Spin, shape: Shape, centre: Direction, inertia: Inertia,
             mass: float, gravity: Direction, up: Direction) -> tuple[Rate, Spin]:
    """Pulling the force and Euler balance back across the rolling map gives the body rate (`[...] Rate+`) and angular acceleration (`[...] Spin+`)."""
    normal, support, height, rolling = contact_geometry(motor, shape, centre, up)
    rate = rolling(spin)                                              # [...] Rate+

    # Differentiate the support point as the table normal rotates in the body while the hull centre stays fixed.
    normal_rate = spin.commutator(normal)                             # [...] Plane+
    height_rate = (normal & shape(normal_rate)) / height              # [...] Scalar+
    contact_rate = -shape(normal_rate) / height - support * height_rate / height  # [...] Direction-
    rolling_change = (contact_rate * origin.inverse()).commutator(spin)  # [...] Rate+
    force = (origin & (motor << gravity)) * mass                      # [...] Forque-

    # Pulling forques back across the rolling map annihilates the workless reaction at the resting contact.
    pullback = rolling.adjugate()                                     # [...] Forque- <- Forque-
    effective_inertia = pullback(inertia(rolling))                    # [...] Forque- <- Spin+
    acceleration = effective_inertia.solve(pullback(
        force - inertia(rate).commutator(rate) - inertia(rolling_change)
    ))                                                                # [...] Spin+
    return rate, acceleration


def energy(motor: Motor, spin: Spin, shape: Shape, centre: Direction, inertia: Inertia,
           mass: float, gravity: Direction, up: Direction) -> Scalar:
    """Kinetic and gravitational potential contributions sum to the total mechanical energy (`[...] Scalar+`) of the rolling body."""
    _, _, _, rolling = contact_geometry(motor, shape, centre, up)
    rate = rolling(spin)                                              # [...] Rate+
    potential = -mass * (gravity.dual_inverse() & (motor >> origin))  # [...] Scalar+
    return (rate & inertia(rate)) / 2 + potential


# --- plumbing

def rkmk_step(motor: Motor, spin: Spin, dt: float, shape: Shape, centre: Direction,
              inertia: Inertia, mass: float, gravity: Direction, up: Direction) -> tuple[Motor, Spin]:
    """A fourth-order Runge-Kutta-Munthe-Kaas step advances `motor` and body-frame `spin` over `dt`."""
    stage_increment = (spin * 0).retyped(Rate)                        # [...] Rate+
    stage_spin = spin                                                 # [...] Spin+
    total_increment = stage_increment                                 # [...] Rate+
    total_spin_change = spin * 0                                      # [...] Spin+
    for step_fraction, weight in ((0.5, 1 / 6), (0.5, 1 / 3), (1.0, 1 / 3), (0.0, 1 / 6)):
        rate, acceleration = dynamics(motor * stage_increment.exp(), stage_spin,
                                      shape, centre, inertia, mass, gravity, up)
        adjoint = stage_increment.commutator(Rate) * 2                # [...] Rate+ <- Rate+
        dexp_inverse = Rate + adjoint / 2 + adjoint(adjoint) / 12     # [...] Rate+ <- Rate+
        increment_rate = dexp_inverse(-rate / 2)                      # [...] Rate+
        total_increment = total_increment + increment_rate * (dt * weight)  # [...] Rate+
        total_spin_change = total_spin_change + acceleration * (dt * weight)  # [...] Spin+
        stage_increment = increment_rate * (dt * step_fraction)       # [...] Rate+
        stage_spin = spin + acceleration * (dt * step_fraction)       # [...] Spin+
    return motor * total_increment.exp(), spin + total_spin_change


def trajectory(motor: Motor, spin: Spin, dt: float, shape: Shape, centre: Direction,
               inertia: Inertia, mass: float, gravity: Direction,
               up: Direction) -> Iterator[tuple[Motor, Spin]]:
    """Yields rigid placements (`[...] Motor+`) and body-frame spins (`[...] Spin+`) at steps of `dt`."""
    while True:
        yield motor, spin
        motor, spin = rkmk_step(motor, spin, dt, shape, centre, inertia, mass, gravity, up)
