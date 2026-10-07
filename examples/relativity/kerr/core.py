"""Light rays in the Kerr geometry, driven by geometric derivatives of their Hamiltonian.

The inverse metric is an identity plus a null dyad. Its quadratic form gives the ray Hamiltonian;
differentiating with respect to momentum gives the tangent, and differentiating with respect to
position gives minus the momentum rate. Momentum is a covector represented by its Minkowski dual.
The scalar profile's gradient is a vector, whose inner product with a step is the profile's change,
and the null direction's derivative is a map with the step left open; together they give the
Hamiltonian's gradient, minus the force on the ray, without component equations.
Camera rays are followed until each is captured, escapes, or crosses the disk; where each ended is
decided here, and the renderer only colours the pixels.

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
Spin = ga.gatype.from_blades("xy xz yz")
Rotor = ga.gatype.rotor()
Screen = ga.gatype.from_blades("x y")
ScreenFrame = ga.gatype((Vector, Screen))
Coherency = ga.gatype((Screen, Screen))
VectorGradient = ga.gatype((Vector, Vector))
spatial_identity = mv.t | (mv.t ^ Vector)


# --- math -----------------------------------------------------------------------------
def field(position: Vector, spin: Spin, mass: float) -> Field:
    """The Kerr profile, null direction and their derivatives at a batch of events."""
    spatial = mv.t | (mv.t ^ position)                                                      # [...] Vector
    normal = spin * mv.xyz                                                                  # [...] Vector
    projection = normal | spatial                                                           # [...] Scalar
    spin_squared = -spin.squared()                                                          # [...] Scalar
    offset = -spatial.squared() - spin_squared                                              # [...] Scalar
    discriminant = (offset.squared() + 4 * projection.squared()).square_root()              # [...] Scalar
    radius_squared = (offset + discriminant) / 2                                            # [...] Scalar
    radius = radius_squared.square_root()                                                   # [...] Scalar
    # Its inner product with a small step is how far the step changes the oblate radius.
    radius_gradient = -(radius_squared * spatial - normal * projection) / (radius * discriminant)  # [...] Vector
    direction = (radius * spatial + (spatial | spin) - normal * projection / radius) / (radius_squared + spin_squared)  # [...] Vector
    lever = spatial - 2 * radius * direction + normal * projection / radius_squared          # [...] Vector
    direction_gradient = (
        lever * (radius_gradient | Vector)
        + radius * spatial_identity + (spatial_identity | spin)
        - normal * (normal | Vector) / radius
    ) / (radius_squared + spin_squared)                                                     # [...] Vector <- Vector
    denominator = radius_squared.squared() + projection.squared()                           # [...] Scalar
    profile = mass * radius**3 / denominator                                                # [...] Scalar
    profile_gradient = profile * (3 * radius_gradient / radius
        - (4 * radius**3 * radius_gradient + 2 * projection * normal) / denominator)       # [...] Vector
    # The gradient's curl, `(Vector ^ null_gradient(Vector)).contract(1, 2)`, piece by piece: the
    # rank-one piece gives its two vectors wedged, the turn by the spin twice the spin, the rest none.
    null_curl = -((radius_gradient ^ lever) + 2 * spin) / (radius_squared + spin_squared)    # [...] Bivector
    return Field(radius, profile, mv.t - direction, profile_gradient, -direction_gradient, null_curl)  # [...] Field


def metric(geometry: Field) -> VectorGradient:
    """The identity minus the null dyad: a vector to its covector."""
    return Vector - 2 * geometry.profile * geometry.null * (geometry.null | Vector)         # [...] Vector <- Vector


def inverse_metric(geometry: Field) -> VectorGradient:
    """The identity plus the null dyad: covector momentum to the ray's tangent."""
    return Vector + 2 * geometry.profile * geometry.null * (geometry.null | Vector)         # [...] Vector <- Vector


def frame(geometry: Field) -> VectorGradient:
    """The identity plus half the dyad: covector momentum to its direction in the local frame."""
    return Vector + geometry.profile * geometry.null * (geometry.null | Vector)             # [...] Vector <- Vector


def overlap_gradient(geometry: Field, vector: Vector) -> Vector:
    """The gradient of the null direction's overlap with a fixed vector, `null | vector`: the vector
    pulled back through the null direction's derivative, whose changes are spatial."""
    return geometry.null_gradient.adjoint()(mv.t | (mv.t ^ vector))                         # [...] Vector


def motion(geometry: Field, momentum: Vector) -> tuple[Vector, Vector]:
    """The two Hamiltonian derivatives at a field: the ray tangent and minus its position gradient."""
    overlap = geometry.null | momentum                                                      # [...] Scalar
    # Half the momentum paired with its image under the inverse metric, the identity plus the null
    # dyad, is the Hamiltonian; its momentum derivative is that image.
    tangent = momentum + 2 * geometry.profile * geometry.null * overlap                     # [...] Vector
    # The gradient of profile * overlap squared, by the product rule.
    gradient = (geometry.profile_gradient * overlap.squared()
                + 2 * geometry.profile * overlap * overlap_gradient(geometry, momentum))      # [...] Vector
    return tangent, -gradient                                                               # [...] Vector, [...] Vector


def rates(position: Vector, momentum: Vector, spin: Spin, mass: float) -> tuple[Vector, Vector]:
    """The two Hamiltonian derivatives: ray tangent and minus its position gradient."""
    return motion(field(position, spin, mass), momentum)                                    # [...] Vector, [...] Vector


def camera_rates(position: Vector, momentum: Vector, spin: Spin, mass: float) -> tuple[Vector, Vector]:
    """Past-directed Hamiltonian motion per unit of decreasing coordinate time."""
    tangent, momentum_rate = rates(position, momentum, spin, mass)                          # [...] Vector, [...] Vector
    clock = -(mv.t | tangent)                                                               # [...] Scalar
    return tangent / clock, momentum_rate / clock                                           # [...] Vector, [...] Vector


def connection(geometry: Field, direction: Vector) -> Bivector:
    """The plane the local Kerr–Schild frame turns or boosts in, per unit step along a direction."""
    along = geometry.null | direction                                                       # [...] Scalar
    # The gradient of profile * along.
    turning = geometry.profile_gradient * along + geometry.profile * overlap_gradient(geometry, direction)  # [...] Vector
    return (turning ^ geometry.null) + geometry.profile * geometry.null_curl * along        # [...] Bivector


def polarized_rates(position: Vector, momentum: Vector, screen: ScreenFrame,
                    spin: Spin, mass: float) -> tuple[Vector, Vector, ScreenFrame]:
    """Hamiltonian ray motion and parallel transport of its camera screen, per unit of decreasing
    coordinate time."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    tangent, momentum_rate = motion(geometry, momentum)                                     # [...] Vector, [...] Vector
    # The frame, the identity plus half the dyad, takes covector momentum to its local null direction.
    local_momentum = momentum + geometry.profile * geometry.null * (geometry.null | momentum)  # [...] Vector
    screen_rate = -connection(geometry, local_momentum).commutator(screen)                  # [...] Vector <- Screen
    clock = -(mv.t | tangent)                                                               # [...] Scalar
    return tangent / clock, momentum_rate / clock, screen_rate / clock                      # [...] Vector, [...] Vector, [...] Vector <- Screen


def disk_coherency(position: Vector, momentum: Vector, screen: ScreenFrame,
                  normal: Vector, spin: Spin, mass: float, polarization_degree: float) -> Coherency:
    """Unit-intensity disk emission resolved along the parallel-transported camera axes."""
    local_momentum = frame(field(position, spin, mass))(momentum)                            # [...] Vector
    direction = (mv.t | (mv.t ^ local_momentum)) / (mv.t | local_momentum)                  # [...] Vector
    magnetic = position | (normal * mv.xyz)                                                 # [...] Vector
    magnetic = magnetic / (-magnetic.squared()).square_root()                               # [...] Vector
    # Emission is polarized perpendicular to the magnetic direction projected across the ray.
    electric = (direction ^ magnetic) * mv.xyz                                              # [...] Vector
    polarization = screen.adjoint()(electric)                                               # [...] Screen
    # Polarization vanishes smoothly when looking along the magnetic direction.
    polarized_fraction = -polarization_degree * polarization.squared()                      # [...] Scalar
    return ((1 - polarized_fraction) / 2 * Screen
            - polarization_degree * polarization * (polarization | Screen))                 # [...] Screen <- Screen


def transmitted(coherency: Coherency, analyzers: Screen) -> Scalar:
    """The power each unit analyzer passes: its quadratic readout of the coherency."""
    return -(analyzers | coherency(analyzers))                                              # [...] Scalar


def frequency_ratio(position: Vector, momentum: Vector, camera: Vector, orbit: Bivector,
                    spin: Spin, mass: float) -> Scalar:
    """The frequency a camera at rest at `camera` sees, over the one emitted by gas on a circular
    orbit in the unit plane `orbit`, for rays whose momenta are given at the gas."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    # Kepler's angular speed of a circular equatorial orbit, the spin counted along the orbit.
    alignment = -spin.scalar_product(orbit)                                                 # [] Scalar
    angular_speed = np.sqrt(mass) / (geometry.radius * geometry.radius.square_root() + alignment * np.sqrt(mass))  # [...] Scalar
    flow = mv.t + angular_speed * orbit.commutator(mv.t | (mv.t ^ position))                # [...] Vector
    gas = flow / (flow | metric(geometry)(flow)).square_root()                               # [...] Vector
    at_rest = mv.t / (mv.t | metric(field(camera, spin, mass))(mv.t)).square_root()          # [] Vector
    # The momentum paired with time is conserved along the ray, so the camera's frequency reads here.
    return (momentum | at_rest) / (momentum | gas)                                          # [...] Scalar


def disk_temperature(radius: Scalar, inner_radius: float, hottest: float) -> Scalar:
    """A thin disk's temperature at each oblate radius: zero at its inner edge, `hottest` at 49/36 of
    that radius, and falling as the radius to the power -3/4 beyond."""
    ratio = inner_radius / radius                                                           # [...] Scalar
    root = ratio.square_root()                                                              # [...] Scalar
    # ratio^(3/4) (1 - ratio^(1/2))^(1/4), divided by its value at the hottest radius.
    profile = (ratio * root).square_root() * (1 - root).square_root().square_root()         # [...] Scalar
    return profile * (hottest / ((36 / 49) ** 0.75 * (1 / 7) ** 0.25))                      # [...] Scalar


def crossing_fraction(start: Vector, end: Vector, normal: Vector) -> Scalar:
    """How far along a ray segment it crosses the plane normal to `normal`."""
    return -(start | normal) / ((end - start) | normal)                                     # [...] Scalar


def disk_crossing(start: Vector, end: Vector, normal: Vector,
                  spin: Spin, mass: float) -> tuple[Vector, Scalar]:
    """The crossing of a ray segment and the central disk plane, and its oblate radius."""
    point = start + crossing_fraction(start, end, normal) * (end - start)                   # [...] Vector
    return point, field(point, spin, mass).radius                                           # [...] Vector, [...] Scalar


def resolutions(steps: Generator[Step, np.ndarray, None], pixel_count: int, normal: Vector,
                inner_radius: float, outer_radius: float, escape_radius: float,
                capture_radius: float, spin: Spin, mass: float) -> Iterator[Image]:
    """The camera rays resolved at each step: those crossing the disk between its radii, with their
    state at the crossing, and those escaping, with their directions. Captured rays end unseen. The
    rays still unresolved are sent back to the integrator."""
    pixels = np.arange(pixel_count)
    try:
        step = next(steps)
        while pixels.size:
            position, end = step.start[0], step.end[0]
            crossing = np.flatnonzero((normal | position) * (normal | end) <= 0)
            fraction = crossing_fraction(position[crossing], end[crossing], normal)         # [crossings] Scalar
            at_plane = tuple(start[crossing] + fraction * (stop[crossing] - start[crossing])
                             for start, stop in zip(step.start, step.end))
            radii = field(at_plane[0], spin, mass).radius                                   # [crossings] Scalar
            on_disk = (radii >= inner_radius) & (radii <= outer_radius)
            resolved = np.zeros(pixels.size, dtype=bool)
            resolved[crossing[on_disk]] = True
            escaped = np.flatnonzero((step.radius > escape_radius) & ~resolved)
            # An escaped ray's direction is its tangent, the inverse metric of its momentum.
            directions = inverse_metric(field(end[escaped], spin, mass))(step.end[1][escaped])  # [escaped] Vector
            yield Image(pixels[crossing[on_disk]], tuple(value[on_disk] for value in at_plane),
                        radii[on_disk], pixels[escaped], directions)
            resolved[escaped] = True
            resolved |= step.radius < capture_radius
            keep = np.flatnonzero(~resolved)
            pixels = pixels[keep]
            if pixels.size:
                step = steps.send(keep)
    except StopIteration as error:
        raise RuntimeError(f"camera integration left {pixels.size} rays unresolved") from error


def resolve(steps: Generator[Step, np.ndarray, None], pixel_count: int, normal: Vector,
            inner_radius: float, outer_radius: float, escape_radius: float,
            capture_radius: float, spin: Spin, mass: float) -> Image:
    """Where every camera ray ended, gathered over all steps."""
    parts = tuple(resolutions(steps, pixel_count, normal, inner_radius, outer_radius,
                              escape_radius, capture_radius, spin, mass))
    return Image(
        np.concatenate([part.disk_pixels for part in parts]),
        tuple(concatenate(list(values), axis=0) for values in zip(*(part.disk for part in parts))),
        concatenate([part.disk_radii for part in parts], axis=0),
        np.concatenate([part.sky_pixels for part in parts]),
        concatenate([part.sky_directions for part in parts], axis=0),
    )


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Field:
    radius: Scalar
    profile: Scalar
    null: Vector
    profile_gradient: Vector
    null_gradient: VectorGradient
    null_curl: Bivector


@dataclass(frozen=True)
class Step:
    """One integration step of a batch of rays: their state at its start and at its end, and their
    oblate radius at the end."""
    start: tuple[Extensor, ...]
    end: tuple[Extensor, ...]
    radius: Scalar


@dataclass(frozen=True)
class Image:
    """Where camera rays ended: the pixels whose rays crossed the disk, with their state there (event,
    momentum and, when transported, screen) and oblate radius; and the pixels whose rays escaped,
    with their directions."""
    disk_pixels: np.ndarray
    disk: tuple[Extensor, ...]
    disk_radii: Scalar
    sky_pixels: np.ndarray
    sky_directions: Vector


def runge_kutta(rates: Callable[..., tuple[Extensor, ...]], state: tuple[Extensor, ...],
                step: Scalar) -> tuple[Extensor, ...]:
    """One fourth-order step of a state, a tuple of extensors, along its rates."""
    first = rates(*state)
    second = rates(*(value + rate * step / 2 for value, rate in zip(state, first)))
    third = rates(*(value + rate * step / 2 for value, rate in zip(state, second)))
    fourth = rates(*(value + rate * step for value, rate in zip(state, third)))
    return tuple(value + (a + 2 * b + 2 * c + d) * step / 6
                 for value, a, b, c, d in zip(state, first, second, third, fourth))


def initial_momentum(position: Vector, direction: Vector, spin: Spin, mass: float) -> Vector:
    """A future null covector with the given unit spatial direction in the Kerr–Schild frame."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    local_null = mv.t + direction
    # The null dyad squares to zero, so the inverse frame just changes its sign.
    momentum = local_null - geometry.profile * geometry.null * (geometry.null | local_null)
    return momentum / (momentum | mv.t)


def trace(position: Vector, momentum: Vector, spin: Spin, mass: float,
          step_size: float, steps: int, stop_fraction: float,
          dynamics: Callable[[Vector, Vector, Spin, float], tuple[Vector, Vector]]) -> Iterator[tuple[Vector, Vector]]:
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


def camera_rays(eye: Vector, orientation: Rotor, pixel_width: int, pixel_height: int,
                half_view: float, spin: Spin, mass: float) -> tuple[Vector, Vector]:
    """Pixel-centre rays looking along local -z, with right +y and up -x, flattened over pixels."""
    horizontal = ((np.arange(pixel_width) + 0.5) / pixel_width * 2 - 1) * half_view * pixel_width / pixel_height
    vertical = (1 - (np.arange(pixel_height) + 0.5) / pixel_height * 2) * half_view
    direction = orientation >> (-mv.z + mv.y * horizontal[None, :] - mv.x * vertical[:, None])
    direction = direction / (-direction.squared()).square_root()
    momentum = -initial_momentum(eye, -direction, spin, mass)
    momentum = momentum.reshape(-1)
    return eye.broadcast_to(momentum.shape), momentum


def camera_screen(position: Vector, momentum: Vector, orientation: Rotor,
                  spin: Spin, mass: float) -> ScreenFrame:
    """Orthonormal transverse camera axes, with horizontal x and vertical y screen inputs."""
    local_momentum = frame(field(position, spin, mass))(momentum)                            # [...] Vector
    direction = (mv.t | (mv.t ^ local_momentum)) / -(mv.t | local_momentum)
    right = orientation >> mv.y
    right = right + direction * (direction | right)
    right = right / (-right.squared()).square_root()
    up = (direction ^ right) * mv.xyz
    return -right * (mv.x | Screen) - up * (mv.y | Screen)


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
