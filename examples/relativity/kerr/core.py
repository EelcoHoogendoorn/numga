"""Light rays in the Kerr geometry, driven by geometric derivatives of their Hamiltonian.

Events and the directions between them are antivectors, and a ray's momentum is a covector, a vector
that pairs with a direction to a number, `momentum & direction`. The geometry is gauge theory gravity's:
a flat background, and a position gauge field that carries the directions of a local frame, measured flat,
to the directions of the Kerr geometry, `gauge(geometry)`: the identity plus the profile times the null
dyad. The dyad squares to zero, so its inverse is the identity less the dyad, and no metric is written
down: an observer's interval is the flat square of its direction carried back into the local frame, and a
ray's Hamiltonian is half the flat square of its momentum raised and carried into it. Differentiating
with respect to momentum gives the tangent, a direction, and differentiating with respect to position
gives minus the momentum rate, a covector. The rotation gauge field, `rotation(geometry, direction)`,
is the plane the local frame turns in per step along a direction. The scalar profile's gradient is a
covector, paired with a small step to give the profile's change, and the null direction's derivative is a
map from a step to the null direction's change along it; together they give the Hamiltonian's gradient,
minus the force on the ray, without component equations. Camera rays are followed until each is captured
or escapes; the disk takes a share of each ray it crosses, by its optical depth along the ray. What each
ray met is decided here, and the renderer only colours the pixels.

Ingoing Kerr–Schild coordinates cover the future horizon. Units set G and c to one; the spatial
spin bivector has magnitude J/M. The positive radial branch is used, away from the singular disk.

Reference: Pelle et al., Skylight (2022), equations 24–26,
https://academic.oup.com/mnras/article/515/1/1316/6631564
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import STA as ga
from numga.extensor import Extensor

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
Spin = Bivector.derive.within("x y z")
Rotor = ga.gatype.rotor()
# Directions on the camera's screen, along its own x and y: the antivectors that hold both t and z.
Screen = Antivector.derive.containing("t z")
ScreenFrame = ga.gatype((Antivector, Screen))                # Antivector <- Screen
Coherency = ga.gatype((Screen, Screen))                      # Screen <- Screen
DirectionGradient = ga.gatype((Antivector, Antivector))      # Antivector <- Antivector
# The position gauge field: from a direction of the local frame to one of the Kerr geometry.
GaugeField = ga.gatype((Antivector, Antivector))             # Antivector <- Antivector
# One unit along the lab's time, the direction across the lab's slice of space `mv.t`, and the part of a
# direction within that slice.
time = -mv.t.dual()                                          # [] Antivector
space = Antivector - time * (mv.t & Antivector)              # Antivector <- Antivector


# --- math -----------------------------------------------------------------------------
def field(position: Antivector, spin: Spin, mass: float) -> Field:
    """The Kerr profile, null direction and their derivatives at a batch of events."""
    spatial = space(position)                                                               # [...] Antivector
    # The volume the position spans with the spin plane: the spin times the position's height off its
    # plane, held by its complement along time. Wedged with the spin it is that height along the spin's
    # axis, times the spin squared; the spin enters twice, so its sense of turning drops out.
    volume = spatial.anticommutator(spin)                                                   # [...] Vector
    height = volume ^ spin                                                                  # [...] Antivector
    spin_squared = -spin.squared()                                                          # [...] Scalar
    offset = spatial.squared() - spin_squared                                               # [...] Scalar
    discriminant = (offset.squared() - 4 * volume.squared()).square_root()                  # [...] Scalar
    radius_squared = (offset + discriminant) / 2                                            # [...] Scalar
    radius = radius_squared.square_root()                                                   # [...] Scalar
    # Paired with a small step, how far the step changes the oblate radius.
    radius_gradient = (radius_squared * spatial - height).dual() / (radius * discriminant)  # [...] Vector
    # The hole turns as `(spin * (-angle / 2)).exp()` does, and drags the frame with it: the outgoing
    # direction twists against the turning, by the position's commutator with the spin.
    direction = (radius * spatial - spatial.commutator(spin) - height / radius) / (radius_squared + spin_squared)  # [...] Antivector
    lever = spatial - 2 * radius * direction + height / radius_squared                      # [...] Antivector
    direction_gradient = (
        lever * (radius_gradient & Antivector)
        + radius * space - space.commutator(spin)
        - (space.anticommutator(spin) ^ spin) / radius
    ) / (radius_squared + spin_squared)                                                     # [...] Antivector <- Antivector
    denominator = radius_squared.squared() - volume.squared()                               # [...] Scalar
    profile = mass * radius**3 / denominator                                                # [...] Scalar
    profile_gradient = profile * (3 * radius_gradient / radius
        - (4 * radius**3 * radius_gradient - 2 * height.dual()) / denominator)              # [...] Vector
    # The curl of the null covector, `-(Vector ^ null_gradient(Antivector).dual()).trace(1, 2)`, piece by
    # piece: the rank-one piece gives its two covectors wedged, the twist by the spin twice the spin, the
    # rest none.
    null_curl = ((radius_gradient ^ lever.dual()) - 2 * spin) / (radius_squared + spin_squared)   # [...] Bivector
    return Field(radius, profile, time - direction, profile_gradient, -direction_gradient, null_curl)  # [...] Field


def dyad(geometry: Field) -> GaugeField:
    """The null dyad, weighted by the profile: the null direction, read by its null plane."""
    return geometry.profile * geometry.null * (geometry.null_plane & Antivector)            # [...] Antivector <- Antivector


def gauge(geometry: Field) -> GaugeField:
    """The position gauge field: the identity plus the weighted null dyad, from a direction of the local
    frame to one of the Kerr geometry."""
    return Antivector + dyad(geometry)                                                      # [...] Antivector <- Antivector


def gauge_inverse(geometry: Field) -> GaugeField:
    """The gauge field's inverse: the dyad squares to zero, so the identity less the dyad."""
    return Antivector - dyad(geometry)                                                      # [...] Antivector <- Antivector


def local(geometry: Field, momentum: Vector) -> Antivector:
    """A covector momentum's direction in the local frame: raised flat, `-momentum.dual()`, and carried
    through the gauge field. Half its flat square is the ray's Hamiltonian."""
    return gauge(geometry)(-momentum.dual())                                                # [...] Antivector


def proper_squared(geometry: Field, direction: Antivector) -> Scalar:
    """The squared proper length of a timelike direction: minus the flat square of the direction carried
    back into the local frame."""
    return -gauge_inverse(geometry)(direction).squared()                                    # [...] Scalar


def overlap_gradient(geometry: Field, momentum: Vector) -> Vector:
    """The gradient of the momentum's pairing with the null direction, `momentum & null`: the momentum
    pulled back through the null direction's derivative."""
    return geometry.null_gradient.adjugate()(momentum)                                      # [...] Vector


def motion(geometry: Field, momentum: Vector) -> tuple[Antivector, Vector]:
    """The two Hamiltonian derivatives at a field: the ray tangent and minus its position gradient."""
    overlap = momentum & geometry.null                                                      # [...] Scalar
    # Half the flat square of the momentum's local direction is the Hamiltonian; its momentum derivative
    # is that direction carried through the gauge field once more.
    tangent = gauge(geometry)(local(geometry, momentum))                                    # [...] Antivector
    # The gradient of profile * overlap squared, by the product rule.
    gradient = (geometry.profile_gradient * overlap.squared()
                + 2 * geometry.profile * overlap * overlap_gradient(geometry, momentum))      # [...] Vector
    return tangent, -gradient                                                               # [...] Antivector, [...] Vector


def rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """The two Hamiltonian derivatives: ray tangent and minus its position gradient."""
    return motion(field(position, spin, mass), momentum)                                    # [...] Antivector, [...] Vector


def camera_rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Past-directed Hamiltonian motion per unit of decreasing coordinate time."""
    tangent, momentum_rate = rates(position, momentum, spin, mass)                          # [...] Antivector, [...] Vector
    clock = -(mv.t & tangent)                                                               # [...] Scalar
    return tangent / clock, momentum_rate / clock                                           # [...] Antivector, [...] Vector


def rotation(geometry: Field, direction: Antivector) -> Bivector:
    """The rotation gauge field: the plane the local frame turns or boosts in, per unit step along a
    direction."""
    # The null plane's reading of the direction.
    null = geometry.null_plane                                                              # [...] Vector
    along = null & direction                                                                # [...] Scalar
    # The gradient of profile * along.
    turning = geometry.profile_gradient * along + geometry.profile * overlap_gradient(geometry, -direction.dual())  # [...] Vector
    return -(turning ^ null) - geometry.profile * geometry.null_curl * along                # [...] Bivector


def polarized_rates(position: Antivector, momentum: Vector, screen: ScreenFrame,
                    spin: Spin, mass: float) -> tuple[Antivector, Vector, ScreenFrame]:
    """Hamiltonian ray motion and parallel transport of its camera screen, per unit of decreasing
    coordinate time."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    tangent, momentum_rate = motion(geometry, momentum)                                     # [...] Antivector, [...] Vector
    screen_rate = -rotation(geometry, local(geometry, momentum)).commutator(screen)         # [...] Antivector <- Screen
    clock = -(mv.t & tangent)                                                               # [...] Scalar
    return tangent / clock, momentum_rate / clock, screen_rate / clock                      # [...] Antivector, [...] Vector, [...] Antivector <- Screen


def disk_coherency(position: Antivector, momentum: Vector, screen: ScreenFrame,
                  normal: Vector, spin: Spin, mass: float, polarization_degree: float) -> Coherency:
    """Unit-intensity disk emission resolved along the parallel-transported camera axes; `normal` is the
    disk's plane, the covector that measures height above it."""
    heading = local(field(position, spin, mass), momentum)                                  # [...] Antivector
    direction = space(heading) / (mv.t & heading)                                           # [...] Antivector
    # The magnetic direction circles the normal, square to the plane through the normal and the
    # emitter. Emission is polarized perpendicular to it across the ray: in that plane, square to the ray.
    meridian = space(position).dual() ^ normal                                              # [...] Bivector
    polarization_axis = meridian.commutator(direction) / meridian.norm()                    # [...] Antivector
    polarization = screen.adjoint()(polarization_axis)                                      # [...] Screen
    # Polarization vanishes smoothly when looking along the magnetic direction.
    polarized_fraction = polarization_degree * polarization.squared()                       # [...] Scalar
    return ((1 - polarized_fraction) / 2 * Screen
            + polarization_degree * polarization * (polarization | Screen))                 # [...] Screen <- Screen


def transmitted(coherency: Coherency, analyzers: Screen) -> Scalar:
    """The power each unit analyzer passes: its quadratic readout of the coherency."""
    return analyzers | coherency(analyzers)                                                 # [...] Scalar


def frequency_ratio(position: Antivector, momentum: Vector, camera: Antivector, orbit: Bivector,
                    spin: Spin, mass: float) -> Scalar:
    """The frequency a camera at rest at `camera` sees, over the one emitted by gas on a circular
    orbit in the unit plane `orbit`, for rays whose momenta are given at the gas."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector
    at_rest = time / proper_squared(field(camera, spin, mass), time).square_root()         # [] Antivector
    # The momentum paired with time is conserved along the ray, so the camera's frequency reads here.
    return (momentum & at_rest) / (momentum & gas)                                          # [...] Scalar


def orbiting_gas(position: Antivector, orbit: Bivector, spin: Spin, mass: float) -> Antivector:
    """The unit velocity of gas on a circular equatorial orbit in the unit plane `orbit`, at each event."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    angular_speed = orbital_speed(geometry.radius, orbit, spin, mass)                       # [...] Scalar
    # A point turned by `(orbit * (-angle / 2)).exp()` moves by its commutator with the orbit's plane.
    flow = time + angular_speed * space(position).commutator(orbit)                         # [...] Antivector
    return flow / proper_squared(geometry, flow).square_root()                              # [...] Antivector


def orbital_speed(radius: Scalar, orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """Kepler's angular speed, per coordinate time, of a circular equatorial orbit in the unit plane
    `orbit` at each oblate radius, the spin counted along the orbit."""
    alignment = -spin.scalar_product(orbit)                                                 # [] Scalar
    return np.sqrt(mass) / (radius * radius.square_root() + alignment * np.sqrt(mass))      # [...] Scalar


def carried(position: Antivector, radius: Scalar, orbit: Bivector, spin: Spin, mass: float,
            elapsed: np.ndarray) -> Antivector:
    """The gas the camera sees at each disk event once a coordinate time `elapsed` has passed,
    followed back along its orbit to time zero, `[elapsed..., events]`. Each event's own time counts
    too, earlier for light that travelled further, and inner rings lap outer ones."""
    emitted = elapsed[..., None] + (mv.t & position)                                        # [elapsed..., events] Scalar
    angle = orbital_speed(radius, orbit, spin, mass) * emitted                              # [elapsed..., events] Scalar
    return (orbit * (-angle / 2)).exp() >> position                                         # [elapsed..., events] Antivector


def innermost_stable_orbit(orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """The radius of the innermost stable circular equatorial orbit in the unit plane `orbit`, turning
    with the hole, the spin counted along the orbit: 6M without spin, M at the extreme (Bardeen, Press
    and Teukolsky, 1972)."""
    ratio = -spin.scalar_product(orbit) / mass                                              # [] Scalar
    first = 1 + cube_root(1 - ratio**2) * (cube_root(1 + ratio) + cube_root(1 - ratio))     # [] Scalar
    second = (3 * ratio**2 + first**2).square_root()                                        # [] Scalar
    return mass * (3 + second - ((3 - first) * (3 + first + 2 * second)).square_root())    # [] Scalar


def disk_temperature(radius: Scalar, orbit: Bivector, spin: Spin, mass: float, scale: float) -> Scalar:
    """A relativistic thin disk's temperature at each oblate radius, from the flux of Page and Thorne
    (1974), for gas orbiting in the unit plane `orbit`, the spin counted along the orbit: zero at the
    innermost stable orbit, falling as the radius to the power -3/4 far out. `scale` is
    (accretion rate c^6 / (σ G^2 M^2))^(1/4), K."""
    ratio = -spin.scalar_product(orbit) / mass                                              # [] Scalar
    root_radius = (radius / mass).square_root()                                             # [...] Scalar
    edge = (innermost_stable_orbit(orbit, spin, mass) / mass).square_root()                 # [] Scalar
    # The roots of the cubic `root_radius**3 - 3 * root_radius + 2 * ratio`, and the weight of each in
    # the torque; the product of a root's differences from the other two is the cubic's derivative there.
    roots = 2 * ((ratio.arccos() + np.array([-np.pi, np.pi, 3 * np.pi])) / 3).cos()       # [3] Scalar
    weights = (roots - ratio) ** 2 / (roots * (roots ** 2 - 1))                             # [3] Scalar
    # The torque the disk carries, zero at the innermost stable orbit; round-off kept non-negative.
    torque = (root_radius - edge - 1.5 * ratio * (root_radius / edge).log()
              - (weights * ((root_radius[..., None] - roots) / (edge - roots)).log()).sum(axis=-1)).clip(0, np.inf)  # [...] Scalar
    flux = 3 / (8 * np.pi) * torque / (root_radius ** 4 * (root_radius ** 3 - 3 * root_radius + 2 * ratio))  # [...] Scalar
    return scale * flux.square_root().square_root()                                         # [...] Scalar


def disk_depth(radius: Scalar, inner_radius: Scalar, outer_radius: float, depth: float) -> Scalar:
    """The disk's optical depth straight through its thickness at each oblate radius: `depth` at the
    inner edge, where the gas is densest, thinning outwards to none at the outer edge."""
    fraction = (radius - inner_radius) / (outer_radius - inner_radius)                      # [...] Scalar
    return depth * (1 - fraction) * (1 - fraction)                                          # [...] Scalar


def disk_opacity(position: Antivector, momentum: Vector, depth: Scalar, normal: Vector,
                 orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """The share of a ray's light the disk absorbs where the ray crosses it. Its optical depth along
    the ray is the depth through its thickness over the cosine of the ray's slant as the orbiting gas
    sees it: the ray's energy in the gas's frame over its momentum along the normal. Gas moving away
    from the light's source sees the ray more obliquely, and absorbs more."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector
    slant = ((momentum & gas) / (momentum & normal.dual())).abs()                           # [...] Scalar
    return 1 - (-depth * slant).exp()                                                       # [...] Scalar


def crossing_fraction(start: Antivector, end: Antivector, normal: Vector) -> Scalar:
    """How far along a ray segment it crosses the plane `normal`, the covector of height above it."""
    return -(normal & start) / (normal & (end - start))                                     # [...] Scalar


def disk_crossing(start: tuple[Extensor, ...], end: tuple[Extensor, ...], normal: Vector,
                  spin: Spin, mass: float) -> tuple[tuple[Extensor, ...], Scalar]:
    """A ray's state, its event first, where its segment from `start` to `end` crosses the central disk
    plane `normal`, and its oblate radius there."""
    fraction = crossing_fraction(start[0], end[0], normal)                                  # [...] Scalar
    state = tuple(value + fraction * (stop - value) for value, stop in zip(start, end))     # [...] state
    return state, field(state[0], spin, mass).radius                                        # [...] state, [...] Scalar


def resolutions(steps: Generator[Step, np.ndarray, None], pixel_count: int, normal: Vector,
                inner_radius: Scalar, outer_radius: float, opacity: Callable[[Antivector, Vector, Scalar], Scalar],
                escape_radius: float, capture_radius: float, spin: Spin, mass: float) -> Iterator[Image]:
    """The camera rays resolved at each step. Each crossing of the disk, between its radii, gives its
    state there and its weight: the disk's opacity, for the crossing's event, momentum and oblate
    radius, times the share of the ray not yet absorbed. Rays
    escaping give their directions, weighted by what is left of them. A ray ends when absorbed,
    escaped or captured; the rays still unresolved are sent back to the integrator."""
    pixels = np.arange(pixel_count)
    remaining = mv.scalar(np.ones((pixel_count, 1)))                                        # [rays] Scalar
    # Sending nothing starts the trace; every later send names the rays still traced.
    keep = None
    while pixels.size:
        step = steps.send(keep)
        crossing = step.crossing(normal)
        at_plane, radii = disk_crossing(tuple(value[crossing] for value in step.start),
                                        tuple(value[crossing] for value in step.end), normal, spin, mass)  # [crossings] state, [crossings] Scalar
        on_disk = np.flatnonzero((radii >= inner_radius) & (radii <= outer_radius))
        crossing, at_plane, radii = crossing[on_disk], tuple(value[on_disk] for value in at_plane), radii[on_disk]
        weights = remaining[crossing] * opacity(at_plane[0], at_plane[1], radii)            # [crossings] Scalar
        remaining = replaced(remaining, crossing, remaining[crossing] - weights)            # [rays] Scalar
        escaped, keep = step.outcomes(remaining, escape_radius, capture_radius)
        # An escaped ray's direction is its tangent, its momentum carried through the gauge field.
        directions, _ = rates(step.end[0][escaped], step.end[1][escaped], spin, mass)       # [escaped] Antivector
        yield Image(pixels[crossing], at_plane, radii, weights, pixels[escaped], directions, remaining[escaped])
        pixels, remaining = pixels[keep], remaining[keep]


def resolve(steps: Generator[Step, np.ndarray, None], pixel_count: int, normal: Vector,
            inner_radius: Scalar, outer_radius: float, opacity: Callable[[Antivector, Vector, Scalar], Scalar],
            escape_radius: float, capture_radius: float, spin: Spin, mass: float) -> Image:
    """What every camera ray met, gathered over all steps."""
    parts = tuple(resolutions(steps, pixel_count, normal, inner_radius, outer_radius, opacity,
                              escape_radius, capture_radius, spin, mass))
    return Image(
        np.concatenate([part.disk_pixels for part in parts]),
        tuple(concatenate(list(values), axis=0) for values in zip(*(part.disk for part in parts))),
        concatenate([part.disk_radii for part in parts], axis=0),
        concatenate([part.disk_weights for part in parts], axis=0),
        np.concatenate([part.sky_pixels for part in parts]),
        concatenate([part.sky_directions for part in parts], axis=0),
        concatenate([part.sky_weights for part in parts], axis=0),
    )


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Field:
    radius: Scalar
    profile: Scalar
    null: Antivector
    profile_gradient: Vector
    null_gradient: DirectionGradient
    null_curl: Bivector

    @property
    def null_plane(self) -> Vector:
        """The null direction's plane, the light cone's tangent plane along it, which contains the
        direction: minus its dual."""
        return -self.null.dual()                                                            # [...] Vector


@dataclass(frozen=True)
class Step:
    """One integration step of a batch of rays: their state at its start and at its end, and their
    oblate radius at the end."""
    start: tuple[Extensor, ...]
    end: tuple[Extensor, ...]
    radius: Scalar

    def crossing(self, normal: Vector) -> np.ndarray:
        """The rays whose segment crosses the plane `normal`."""
        return np.flatnonzero((normal & self.start[0]) * (normal & self.end[0]) <= 0)

    def outcomes(self, remaining: Scalar, escape_radius: float, capture_radius: float) -> tuple[np.ndarray, np.ndarray]:
        """The rays escaping at the end of the step with light left, and the rays still traced: those
        with light left, neither escaped nor captured."""
        lit = remaining > 0
        escaped = np.flatnonzero(lit & (self.radius > escape_radius))
        traced = np.flatnonzero(lit & (self.radius <= escape_radius) & (self.radius >= capture_radius))
        return escaped, traced


@dataclass(frozen=True)
class Image:
    """What camera rays met: their crossings of the disk, by pixel, with the ray's state there (event,
    momentum and, when transported, screen), the oblate radius and the share of the pixel's light the
    crossing gives; and the pixels whose rays escaped, with their directions and the share left for
    the sky. A pixel seen through the disk's clear edge appears more than once."""
    disk_pixels: np.ndarray
    disk: tuple[Extensor, ...]
    disk_radii: Scalar
    disk_weights: Scalar
    sky_pixels: np.ndarray
    sky_directions: Antivector
    sky_weights: Scalar


def replaced(values: Extensor, indices: np.ndarray, entries: Extensor) -> Extensor:
    """`values` along their first axis with those at `indices` replaced by `entries`, built anew."""
    rest = np.setdiff1d(np.arange(values.shape[0]), indices)
    order = np.argsort(np.concatenate([rest, indices]))
    return concatenate([values[rest], entries], axis=0)[order]


def cube_root(value: Scalar) -> Scalar:
    """The real cube root of each non-negative scalar."""
    return (value.log() / 3).exp()


def runge_kutta(rates: Callable[..., tuple[Extensor, ...]], state: tuple[Extensor, ...],
                step: Scalar) -> tuple[Extensor, ...]:
    """One fourth-order step of a state, a tuple of extensors, along its rates."""
    first = rates(*state)
    second = rates(*(value + rate * step / 2 for value, rate in zip(state, first)))
    third = rates(*(value + rate * step / 2 for value, rate in zip(state, second)))
    fourth = rates(*(value + rate * step for value, rate in zip(state, third)))
    return tuple(value + (a + 2 * b + 2 * c + d) * step / 6
                 for value, a, b, c, d in zip(state, first, second, third, fourth))


def initial_momentum(position: Antivector, direction: Antivector, spin: Spin, mass: float) -> Vector:
    """A future null covector with the given unit spatial direction in the Kerr–Schild frame."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    local_null = time + direction                                                           # [...] Antivector
    # Carried back out of the local frame and lowered flat.
    momentum = -gauge_inverse(geometry)(local_null).dual()                                  # [...] Vector
    return momentum / (momentum & time)                                                     # [...] Vector


def trace(position: Antivector, momentum: Vector, spin: Spin, mass: float,
          step_size: float, steps: int, stop_fraction: float,
          dynamics: Callable[[Antivector, Vector, Spin, float], tuple[Antivector, Vector]]) -> Iterator[tuple[Antivector, Vector]]:
    """Batched fourth-order steps; captured rays slow towards a surface inside the horizon.

    The affine step is reduced only inside the horizon. This changes sampling, not the ray path,
    and keeps every stage away from the singularity. Returned events include the starting state.
    """
    horizon = mass + (mass**2 + spin.squared()).square_root()
    state = (position, momentum)
    yield state
    for _ in range(steps):
        radius = field(state[0], spin, mass).radius
        clock = ((radius / horizon - stop_fraction) / (1 - stop_fraction)).clip(0, 1)
        state = runge_kutta(lambda *values: dynamics(*values, spin, mass), state, step_size * clock)
        yield state


def camera_rays(eye: Antivector, orientation: Rotor, pixel_width: int, pixel_height: int,
                half_view: float, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Pixel-centre rays looking along local -z, with right +y and up -x, flattened over pixels."""
    horizontal = ((np.arange(pixel_width) + 0.5) / pixel_width * 2 - 1) * half_view * pixel_width / pixel_height
    vertical = (1 - (np.arange(pixel_height) + 0.5) / pixel_height * 2) * half_view
    x, y, z = mv.x.dual(), mv.y.dual(), mv.z.dual()
    direction = orientation >> (-z + y * horizontal[None, :] - x * vertical[:, None])
    direction = direction / direction.squared().square_root()
    momentum = -initial_momentum(eye, -direction, spin, mass)
    momentum = momentum.reshape(-1)
    return eye.broadcast_to(momentum.shape), momentum


def camera_screen(position: Antivector, momentum: Vector, orientation: Rotor,
                  spin: Spin, mass: float) -> ScreenFrame:
    """Orthonormal transverse camera axes, with horizontal x and vertical y screen inputs."""
    heading = local(field(position, spin, mass), momentum)                                  # [...] Antivector
    direction = space(heading) / -(mv.t & heading)                                          # [...] Antivector
    # The camera's own right and up, local +y and -x, each less its parts along the directions before
    # it; every direction here is a unit of space, squaring to one.
    right = orientation >> mv.y.dual()
    right = right - direction * (direction | right)
    right = right / right.squared().square_root()
    up = orientation >> -mv.x.dual()
    up = up - direction * (direction | up) - right * (right | up)
    up = up / up.squared().square_root()
    # A screen direction is read by pairing with the camera's slices x and y.
    return right * (mv.x & Screen) + up * (mv.y & Screen)                                   # [...] Antivector <- Screen


def camera_trace(state: tuple[Extensor, ...], spin: Spin, mass: float, step_fraction: float, steps: int,
                 dynamics: Callable[..., tuple[Extensor, ...]]) -> Generator[Step, np.ndarray, None]:
    """Fourth-order steps of camera rays, each a fraction of the oblate radius; the consumer sends
    back the indices of the rays still traced. The state is an event and a momentum, and the rays'
    screen when `dynamics` transports one."""
    radius = field(state[0], spin, mass).radius
    for _ in range(steps):
        end = runge_kutta(lambda *values: dynamics(*values, spin, mass), state, step_fraction * radius)
        end_radius = field(end[0], spin, mass).radius
        keep = yield Step(state, end, end_radius)
        state, radius = tuple(value[keep] for value in end), end_radius[keep]
