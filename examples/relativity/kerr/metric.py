"""Kerr light rays and polarization from the metric and its Levi-Civita connection.

The metric maps coordinate directions to covectors. Its inverse gives the ray Hamiltonian and
its momentum derivative, the tangent. The metric's directional derivatives give the Christoffel
connection, which carries coordinate screen directions along the ray. Momenta and gradients
read directions through the regressive pairing; observers are normalized with the physical metric.
An orthonormal source frame specifies camera angles and the prescribed disk polarization only.

Ingoing Cartesian Kerr–Schild coordinates, G = c = 1, on the positive radial branch away from the
singular disk. The disk model takes subextremal spin, |a|/M < 1, the gas orbiting with the hole for positive spin and against it for negative.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

from numga import NumpyContext, concatenate, where
from numga.algebras import STA as ga
from numga.extensor import Extensor

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
# The hole's spin per unit mass: intrinsic, the patch of the plane it turns in, a plane through time.
Spin = Bivector.derive.containing("t").derive.flipped_orientation
Rotor = ga.gatype.rotor()
# Directions on the camera's screen, along its own x and y: the antivectors that hold both t and z.
Screen = Antivector.derive.containing("t z")
# The camera's slices x and y, which read screen directions: vectors, extrinsic.
ScreenSlice = Vector.derive.within("x y")
ScreenFrame = ga.gatype((Antivector, Screen))                # Antivector- <- Screen
# The light's polarization covariance, from the slices that read the screen to screen directions.
Coherency = ga.gatype((Screen, ScreenSlice))                 # Screen <- ScreenSlice
DirectionGradient = ga.gatype((Antivector, Antivector))      # Antivector- <- Antivector-
# The metric and inverse metric connect complementary types.
Metric = ga.gatype((Vector, Antivector))                    # Vector+ <- Antivector-
InverseMetric = ga.gatype((Antivector, Vector))             # Antivector- <- Vector+
# One unit along the lab's time, the direction across the lab's slice of space `mv.t`, and the part of a
# direction within that slice.
time = -mv.t.dual()                                          # [] Antivector-
space = Antivector - time * (mv.t & Antivector)              # Antivector- <- Antivector-
# The camera's own axes in its rest frame, before it is turned: it looks along -z, with its right +y and its up -x.
camera_forward = -mv.z.dual()                                # [] Antivector-
camera_right = mv.y.dual()                                   # [] Antivector-
camera_up = -mv.x.dual()                                     # [] Antivector-
# The crossings of the disk a camera ray keeps: its direct, secondary and tertiary images, and one more.
CROSSINGS = 4


# --- math -----------------------------------------------------------------------------
def field(position: Antivector, spin: Spin, mass: float) -> Field:
    """The Kerr profile, null direction and their derivatives at a batch of events."""
    spatial = space(position)                                                               # [...] Antivector-
    # The plane the hole turns in, whose dual is its spin.
    turning = spin.dual_inverse()                                                           # [...] Bivector+
    # The volume the position spans with the turning plane: the plane times the position's height off it,
    # held by its complement along time. Wedged with the plane it is that height along the spin, times the
    # spin squared; the plane enters twice, so its sense of turning drops out.
    volume = spatial.anticommutator(turning)                                                # [...] Vector+
    height = volume ^ turning                                                               # [...] Antivector-
    spin_squared = spin.squared()                                                           # [...] Scalar+
    offset = spatial.squared() - spin_squared                                               # [...] Scalar+
    discriminant = (offset.squared() - 4 * volume.squared()).square_root()                  # [...] Scalar+
    radius_squared = (offset + discriminant) / 2                                            # [...] Scalar+
    radius = radius_squared.square_root()                                                   # [...] Scalar+
    # Paired with a small step, how far the step changes the oblate radius.
    radius_gradient = (radius_squared * spatial - height).dual() / (radius * discriminant)  # [...] Vector+
    # The hole turns as `(turning * (-angle / 2)).exp()` does, and drags the frame with it: the outgoing
    # direction twists against the turning, by the position's commutator with the plane.
    direction = (radius * spatial - spatial.commutator(turning) - height / radius) / (radius_squared + spin_squared)  # [...] Antivector-
    lever = spatial - 2 * radius * direction + height / radius_squared                      # [...] Antivector-
    direction_gradient = (
        lever * (radius_gradient & Antivector)
        + radius * space - space.commutator(turning)
        - (space.anticommutator(turning) ^ turning) / radius
    ) / (radius_squared + spin_squared)                                                     # [...] Antivector- <- Antivector-
    denominator = radius_squared.squared() - volume.squared()                               # [...] Scalar+
    profile = mass * radius**3 / denominator                                                # [...] Scalar+
    profile_gradient = profile * (3 * radius_gradient / radius
        - (4 * radius**3 * radius_gradient - 2 * height.dual()) / denominator)              # [...] Vector+
    return Field(radius, profile, time - direction, profile_gradient, -direction_gradient)


def metric(geometry: Field) -> Metric:
    """The physical metric: a direction's covector, with positive timelike square."""
    plane = geometry.null_plane
    return -Antivector.dual() - 2 * geometry.profile * plane * (plane & Antivector)


def inverse_metric(geometry: Field) -> InverseMetric:
    """The inverse metric, taking a covector momentum to its coordinate tangent."""
    null = geometry.null
    return -Vector.dual() + 2 * geometry.profile * null * (Vector & null)


def source_frame(geometry: Field) -> DirectionGradient:
    """An orthonormal frame specifying the camera and prescribed emission, in coordinates.

    This choice fixes the same local source directions as the Kerr–Schild frame. Gravitational
    evolution uses the metric and Christoffel connection; this frame only defines local observations.
    """
    return Antivector + geometry.profile * geometry.null * (geometry.null_plane & Antivector)


def source_components(geometry: Field) -> DirectionGradient:
    """Coordinate directions resolved in the specified orthonormal source frame."""
    return Antivector - geometry.profile * geometry.null * (geometry.null_plane & Antivector)


def proper_squared(geometry: Field, direction: Antivector) -> Scalar:
    """The metric's reading of a direction on itself, positive for a timelike direction."""
    return metric(geometry)(direction) & direction


def hamiltonian(geometry: Field, momentum: Vector) -> Scalar:
    """Half the momentum's reading of its image under the inverse metric."""
    return (momentum & inverse_metric(geometry)(momentum)) / 2


def overlap_gradient(geometry: Field, momentum: Vector) -> Vector:
    """The gradient of the momentum's pairing with the null direction, `momentum & null`: the momentum
    pulled back through the null direction's derivative."""
    return geometry.null_gradient.adjugate()(momentum)                                      # [...] Vector+


def motion(geometry: Field, momentum: Vector) -> tuple[Antivector, Vector]:
    """The two Hamiltonian derivatives at a field: the ray tangent and minus its position gradient."""
    overlap = momentum & geometry.null                                                      # [...] Scalar+
    # Differentiating the quadratic Hamiltonian with respect to momentum applies the inverse metric.
    tangent = inverse_metric(geometry)(momentum)                                    # [...] Antivector-
    # The gradient of profile * overlap squared, by the product rule.
    gradient = (geometry.profile_gradient * overlap.squared()
                + 2 * geometry.profile * overlap * overlap_gradient(geometry, momentum))      # [...] Vector+
    return tangent, -gradient                                                               # [...] Antivector-, [...] Vector+


def rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """The two Hamiltonian derivatives: ray tangent and minus its position gradient."""
    return motion(field(position, spin, mass), momentum)                                    # [...] Antivector-, [...] Vector+


def camera_rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Past-directed Hamiltonian motion per unit of decreasing coordinate time."""
    tangent, momentum_rate = rates(position, momentum, spin, mass)                          # [...] Antivector-, [...] Vector+
    clock = -(mv.t & tangent)                                                               # [...] Scalar+
    return tangent / clock, momentum_rate / clock                                           # [...] Antivector-, [...] Vector+


def metric_change(geometry: Field, along: Antivector, direction: Antivector) -> Vector:
    """The metric's directional derivative along a coordinate step, applied to a fixed direction."""
    plane = geometry.null_plane
    plane_change = -geometry.null_gradient(along).dual()
    return -2 * (
        (geometry.profile_gradient & along) * plane * (plane & direction)
        + geometry.profile * (plane_change * (plane & direction)
                              + plane * (plane_change & direction)))


def christoffel(geometry: Field, along: Antivector, direction: Antivector) -> Antivector:
    """The torsion-free metric connection in the fixed Cartesian coordinate basis."""
    plane = geometry.null_plane
    along_reading, direction_reading = plane & along, plane & direction
    along_gradient = geometry.null_gradient.adjugate()(-along.dual())
    direction_gradient = geometry.null_gradient.adjugate()(-direction.dual())
    # Differentiate the scalar metric reading with both coordinate directions held fixed.
    reading_gradient = -2 * (
        geometry.profile_gradient * along_reading * direction_reading
        + geometry.profile * (along_gradient * direction_reading
                              + along_reading * direction_gradient))
    # Two directional derivatives and the scalar reading's gradient give the lowered connection;
    # the inverse metric returns its direction.
    return inverse_metric(geometry)(
        metric_change(geometry, along, direction)
        + metric_change(geometry, direction, along) - reading_gradient) / 2


def polarized_rates(position: Antivector, momentum: Vector, screen: ScreenFrame,
                    spin: Spin, mass: float) -> tuple[Antivector, Vector, ScreenFrame]:
    """Hamiltonian motion and coordinate-screen transport, per unit of decreasing coordinate time."""
    geometry = field(position, spin, mass)
    tangent, momentum_rate = motion(geometry, momentum)
    screen_rate = -christoffel(geometry, tangent, screen)
    clock = -(mv.t & tangent)
    return tangent / clock, momentum_rate / clock, screen_rate / clock


def disk_coherency(position: Antivector, momentum: Vector, screen: ScreenFrame,
                  normal: Vector, spin: Spin, mass: float, polarization_degree: float) -> Coherency:
    """Unit-intensity disk emission resolved along the parallel-transported camera axes; `normal` is the
    disk's plane, the covector that measures height above it."""
    geometry = field(position, spin, mass)
    # The prescribed magnetic pattern is specified in this orthonormal source frame.
    heading = source_components(geometry)(inverse_metric(geometry)(momentum))
    screen = source_components(geometry)(screen)                                  # [...] Antivector-
    direction = space(heading) / (mv.t & heading)                                           # [...] Antivector-
    # The magnetic direction circles the normal, square to the plane through the normal and the
    # emitter. Emission is polarized perpendicular to it across the ray: in that plane, square to the ray.
    meridian = space(position).dual() ^ normal                                              # [...] Bivector+
    polarization_axis = meridian.commutator(direction) / meridian.norm()                    # [...] Antivector-
    polarization = screen.adjoint()(polarization_axis)                                      # [...] Screen
    # Polarization vanishes smoothly when looking along the magnetic direction.
    polarized_fraction = polarization_degree * polarization.squared()                       # [...] Scalar+
    # Unpolarized light is round, where the metric enters; the polarized part weights the polarization by
    # each slice's reading of it.
    return ((1 - polarized_fraction) / 2 * ScreenSlice.dual()
            + polarization_degree * polarization * (ScreenSlice & polarization))            # [...] Screen <- ScreenSlice


def transmitted(coherency: Coherency, analyzers: ScreenSlice) -> Scalar:
    """The power each unit analyzer passes, an analyzer being the slice that reads the screen across its
    axis's normal: its reading of the coherency's image of itself."""
    return analyzers & coherency(analyzers)                                                 # [...] Scalar+


def frequency_ratio(position: Antivector, momentum: Vector, camera: Antivector, orbit: Bivector,
                    spin: Spin, mass: float) -> Scalar:
    """The frequency a camera at rest at `camera` sees, over the one emitted by gas on a circular
    orbit in the unit plane `orbit`, for rays whose momenta are given at the gas."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector-
    at_rest = time / proper_squared(field(camera, spin, mass), time).square_root()         # [] Antivector-
    # The momentum paired with time is conserved along the ray, so the camera's frequency reads here.
    return (momentum & at_rest) / (momentum & gas)                                          # [...] Scalar+


def orbiting_gas(position: Antivector, orbit: Bivector, spin: Spin, mass: float) -> Antivector:
    """The unit velocity of gas on a circular equatorial orbit in the unit plane `orbit`, at each event."""
    geometry = field(position, spin, mass)                                                  # [...] Field
    angular_speed = orbital_speed(geometry.radius, orbit, spin, mass)                       # [...] Scalar+
    # A point turned by `(orbit * (-angle / 2)).exp()` moves by its commutator with the orbit's plane.
    flow = time + angular_speed * space(position).commutator(orbit)                         # [...] Antivector-
    return flow / proper_squared(geometry, flow).square_root()                              # [...] Antivector-


def orbital_speed(radius: Scalar, orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """Kepler's angular speed, per coordinate time, of a circular equatorial orbit in the unit plane
    `orbit` at each oblate radius, the spin counted along the orbit."""
    alignment = orbit & spin                                                                # [] Scalar+
    return np.sqrt(mass) / (radius * radius.square_root() + alignment * np.sqrt(mass))      # [...] Scalar+


def carried(position: Antivector, radius: Scalar, orbit: Bivector, spin: Spin, mass: float,
            distance: float, elapsed: np.ndarray) -> Antivector:
    """The gas the camera sees at each disk event once a coordinate time `elapsed` has passed,
    followed back along its orbit to time zero, `[elapsed..., events]`. Each event's own time counts
    too, earlier for light that travelled further, and inner rings lap outer ones; the clock starts when
    light leaves the hole's centre for the camera at its `distance`, so that moving the camera does not
    turn the disk."""
    emitted = elapsed[..., None] + (mv.t & position) + distance                             # [elapsed..., events] Scalar+
    angle = orbital_speed(radius, orbit, spin, mass) * emitted                              # [elapsed..., events] Scalar+
    return (orbit * (angle / 2)).exp() >> position                                          # [elapsed..., events] Antivector-


def innermost_stable_orbit(orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """The radius of the innermost stable circular equatorial orbit in the unit plane `orbit`, the spin
    counted along the orbit, negative for gas orbiting against the hole: 6M without spin, and at the
    extreme M with the hole and 9M against it (Bardeen, Press and Teukolsky, 1972)."""
    ratio = (orbit & spin) / mass                                                           # [] Scalar+
    first = 1 + cube_root(1 - ratio**2) * (cube_root(1 + ratio) + cube_root(1 - ratio))     # [] Scalar+
    second = (3 * ratio**2 + first**2).square_root()                                        # [] Scalar+
    # Gas orbiting with the hole reaches inward of the radius without spin, gas orbiting against it outward.
    reach = ((3 - first) * (3 + first + 2 * second)).square_root()                      # [] Scalar+
    return mass * (3 + second - where(ratio < 0, -reach, reach))                         # [] Scalar+


def disk_temperature(radius: Scalar, orbit: Bivector, spin: Spin, mass: float, scale: float) -> Scalar:
    """A relativistic thin disk's temperature at each oblate radius, from the flux of Page and Thorne
    (1974), for gas orbiting in the unit plane `orbit`, the spin counted along the orbit: zero at the
    innermost stable orbit, falling as the radius to the power -3/4 far out. `scale` is
    (accretion rate c^6 / (σ G^2 M^2))^(1/4), K."""
    ratio = (orbit & spin) / mass                                                           # [] Scalar+
    root_radius = (radius / mass).square_root()                                             # [...] Scalar+
    edge = (innermost_stable_orbit(orbit, spin, mass) / mass).square_root()                 # [] Scalar+
    # The roots of the cubic `root_radius**3 - 3 * root_radius + 2 * ratio`, and the weight of each in
    # the torque; the product of a root's differences from the other two is the cubic's derivative there.
    roots = 2 * ((ratio.arccos() + np.array([-np.pi, np.pi, 3 * np.pi])) / 3).cos()       # [3] Scalar+
    weights = (roots - ratio) ** 2 / (roots * (roots ** 2 - 1))                             # [3] Scalar+
    # The torque the disk carries, zero at the innermost stable orbit; round-off kept non-negative.
    torque = (root_radius - edge - 1.5 * ratio * (root_radius / edge).log()
              - (weights * ((root_radius[..., None] - roots) / (edge - roots)).log()).sum(axis=-1)).clip(0, np.inf)  # [...] Scalar+
    flux = 3 / (8 * np.pi) * torque / (root_radius ** 4 * (root_radius ** 3 - 3 * root_radius + 2 * ratio))  # [...] Scalar+
    return scale * flux.square_root().square_root()                                         # [...] Scalar+


def disk_depth(radius: Scalar, inner_radius: Scalar, outer_radius: float, depth: float) -> Scalar:
    """The disk's optical depth straight through its thickness at each oblate radius: `depth` at the
    inner edge, where the gas is densest, thinning outwards to none at the outer edge."""
    fraction = (radius - inner_radius) / (outer_radius - inner_radius)                      # [...] Scalar+
    return depth * (1 - fraction) * (1 - fraction)                                          # [...] Scalar+


def disk_opacity(position: Antivector, momentum: Vector, depth: Scalar, normal: Vector,
                 orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """The share of a ray's light the disk absorbs where the ray crosses it. Its optical depth along
    the ray is the depth through its thickness over the cosine of the ray's slant as the orbiting gas
    sees it: the ray's energy in the gas's frame over its momentum along the normal. Gas moving away
    from the light's source sees the ray more obliquely, and absorbs more."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector-
    slant = ((momentum & gas) / (momentum & normal.dual())).abs()                           # [...] Scalar+
    return 1 - (-depth * slant).exp()                                                       # [...] Scalar+


def crossing_fraction(start: Antivector, end: Antivector, normal: Vector) -> Scalar:
    """How far along a ray segment it crosses the plane `normal`, the covector of height above it."""
    return -(normal & start) / (normal & (end - start))                                     # [...] Scalar+


def disk_crossing(start: tuple[Extensor, ...], end: tuple[Extensor, ...], normal: Vector,
                  spin: Spin, mass: float) -> tuple[tuple[Extensor, ...], Scalar]:
    """A ray's state, its event first, where its segment from `start` to `end` crosses the central disk
    plane `normal`, and its oblate radius there."""
    fraction = crossing_fraction(start[0], end[0], normal)                                  # [...] Scalar+
    state = tuple(value + fraction * (stop - value) for value, stop in zip(start, end))     # [...] state
    return state, field(state[0], spin, mass).radius                                        # [...] state, [...] Scalar


def resolve(state: tuple[Extensor, ...], spin: Spin, mass: float, step_fraction: float, n_steps: int,
            dynamics: Callable[..., tuple[Extensor, ...]], normal: Vector, inner_radius: Scalar, outer_radius: float,
            opacity: Callable[[Antivector, Vector, Scalar], Scalar], escape_radius: float,
            capture_radius: float) -> tuple[Image, tuple[Extensor, ...]]:
    """The camera rays traced and resolved together, by fourth-order steps each a fraction of the oblate radius, in
    one compiled loop. Every ray keeps its place and stops where it ends: absorbed, escaped or captured. Each crossing
    of the disk between its radii takes the disk's opacity there of the light the ray has left, kept as the next of its
    `CROSSINGS` crossings; a ray escaping leaves its direction, its tangent given by the inverse
    metric of its momentum, and what light is left for the sky. The state is an event and a momentum, and the rays' screen when
    `dynamics` transports one; the image comes with the state each ray stopped in."""
    import jax
    from numga.backend.jax import JaxContext

    # Sparse execution spells each product as its nonzero terms, which XLA fuses across the wide batch of rays.
    context = JaxContext(ga, execution="sparse")
    n_rays = state[0].shape[0]
    # The loop carries the event in the full antivector layout, whatever layout the rays start in.
    event, *rest = state
    traced = tuple(value.materialize(context) for value in (event.retyped(Antivector), *rest))
    remaining = mv.scalar(np.ones((n_rays, 1))).materialize(context)                       # [rays] Scalar+
    # Each ray's crossings, empty while their weight is zero, and its escape, empty likewise.
    crossed = tuple(value[:, None].broadcast_to((n_rays, CROSSINGS)) for value in traced)    # [rays, crossings] state
    crossed_radii, crossed_weights = remaining[:, None] * np.zeros(CROSSINGS), remaining[:, None] * np.zeros(CROSSINGS)
    filled = np.zeros(n_rays, dtype=int)                                                   # [rays]
    sky, sky_weights = traced[0], remaining * 0                                            # [rays] Antivector-, Scalar+

    def going(state: tuple[Extensor, ...], remaining: Scalar) -> np.ndarray:
        """The rays still traced: with light left, neither escaped nor captured."""
        radius = field(state[0], spin, mass).radius                                        # [rays] Scalar+
        return (remaining > 0) & (radius <= escape_radius) & (radius >= capture_radius)    # [rays]

    def step(carry):
        index, state, remaining, crossed, crossed_radii, crossed_weights, filled, sky, sky_weights = carry
        traced = going(state, remaining)                                                   # [rays]
        # A ray that has ended takes no step.
        size = where(traced, step_fraction * field(state[0], spin, mass).radius, remaining * 0)   # [rays] Scalar+
        end = runge_kutta(lambda *values: dynamics(*values, spin, mass), state, size)
        # A crossing of the disk between its radii, where the segment changes side of its plane.
        at_plane, radii = disk_crossing(state, end, normal, spin, mass)                    # [rays] state, Scalar
        crossing = traced & ((normal & state[0]) * (normal & end[0]) <= 0) & (radii >= inner_radius) & (radii <= outer_radius)
        weights = where(crossing, remaining * opacity(at_plane[0], at_plane[1], radii), remaining * 0)
        remaining = remaining - weights
        # The crossing is stored as its ray's next one.
        into = crossing[:, None] & (np.arange(CROSSINGS) == filled[:, None])               # [rays, crossings]
        crossed = tuple(where(into, value[:, None], crossed_value) for value, crossed_value in zip(at_plane, crossed))
        crossed_radii, crossed_weights = where(into, radii[:, None], crossed_radii), where(into, weights[:, None], crossed_weights)
        filled = filled + crossing
        # A ray leaving the escape radius with light left gives the sky its direction and that light.
        escaping = traced & (remaining > 0) & (field(end[0], spin, mass).radius > escape_radius)
        tangent, _ = rates(end[0], end[1], spin, mass)
        sky, sky_weights = where(escaping, tangent, sky), where(escaping, remaining, sky_weights)
        return index + 1, end, remaining, crossed, crossed_radii, crossed_weights, filled, sky, sky_weights

    def unfinished(carry) -> bool:
        """Steps are left, and some ray is still traced."""
        index, state, remaining = carry[:3]
        return (index < n_steps) & going(state, remaining).any()

    carry = (0, traced, remaining, crossed, crossed_radii, crossed_weights, filled, sky, sky_weights)
    _, *carry = jax.jit(lambda carry: jax.lax.while_loop(unfinished, step, carry))(carry)
    traced, _, crossed, crossed_radii, crossed_weights, _, sky, sky_weights = jax.tree_util.tree_map(
        lambda value: value.materialize(mv.context) if isinstance(value, Extensor) else np.asarray(value), carry,
        is_leaf=lambda value: isinstance(value, Extensor))
    rays, crossing_idx = np.nonzero(crossed_weights > 0)
    escaped = np.flatnonzero(sky_weights > 0)
    image = Image(rays, tuple(value[rays, crossing_idx] for value in crossed), crossed_radii[rays, crossing_idx], crossed_weights[rays, crossing_idx],
                  escaped, sky[escaped], sky_weights[escaped])
    return image, traced


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Field:
    radius: Scalar
    profile: Scalar
    null: Antivector
    profile_gradient: Vector
    null_gradient: DirectionGradient

    @property
    def null_plane(self) -> Vector:
        """The null direction's plane, the light cone's tangent plane along it, which contains the
        direction: minus its dual."""
        return -self.null.dual()                                                            # [...] Vector+


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
    local_null = time + direction                                                           # [...] Antivector-
    # A coordinate null direction is lowered with the physical metric.
    momentum = metric(geometry)(source_frame(geometry)(local_null))                                  # [...] Vector+
    return momentum / (momentum & time)                                                     # [...] Vector+


def trace(position: Antivector, momentum: Vector, spin: Spin, mass: float,
          step_size: float, n_steps: int, stop_fraction: float,
          dynamics: Callable[[Antivector, Vector, Spin, float], tuple[Antivector, Vector]]) -> tuple[Antivector, Vector]:
    """Batched fourth-order steps in one compiled loop; captured rays slow towards a surface inside the horizon.
    The events and momenta along the paths, `[steps + 1, ...]`, include the starting state.

    The affine step is reduced only inside the horizon. This changes sampling, not the ray path,
    and keeps every stage away from the singularity.
    """
    import jax
    from numga.backend.jax import JaxContext

    horizon = mass + (mass**2 - spin.squared()).square_root()

    def step(state: tuple[Antivector, Vector], _) -> tuple[tuple[Antivector, Vector], tuple[Antivector, Vector]]:
        radius = field(state[0], spin, mass).radius
        clock = ((radius / horizon - stop_fraction) / (1 - stop_fraction)).clip(0, 1)
        state = runge_kutta(lambda *values: dynamics(*values, spin, mass), state, step_size * clock)
        return state, state

    # Sparse execution spells each product as its nonzero terms, which XLA fuses across the wide batch of rays.
    context = JaxContext(ga, execution="sparse")
    start = (position.retyped(Antivector).materialize(context), momentum.retyped(Vector).materialize(context))
    _, later = jax.jit(lambda start: jax.lax.scan(step, start, length=n_steps))(start)    # [steps, ...] state
    return tuple(concatenate([first[None], after]).materialize(mv.context) for first, after in zip(start, later))


def camera_boost(geometry: Field) -> Rotor:
    """A coordinate-static camera's rest frame in the Kerr–Schild frame, outside the static limit."""
    observer = source_components(geometry)(time) / proper_squared(geometry, time).square_root()  # [...] Antivector-
    return (1 - observer * time).normalized()                                               # [...] Rotor+


def camera_rays(eye: Antivector, orientation: Rotor, pixel_width: int, pixel_height: int,
                half_view: float, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Pixel-centre rays of a coordinate-static camera, looking along its -z, with right +y and up -x."""
    horizontal = ((np.arange(pixel_width) + 0.5) / pixel_width * 2 - 1) * half_view * pixel_width / pixel_height
    vertical = (1 - (np.arange(pixel_height) + 0.5) / pixel_height * 2) * half_view
    direction = orientation >> (camera_forward + camera_right * horizontal[None, :] + camera_up * vertical[:, None])
    direction = direction / direction.squared().square_root()
    geometry = field(eye, spin, mass)
    # The future photon arrives opposite the direction the camera looks; its rest frame is boosted
    # into the Kerr–Schild frame before its heading is lowered into a coordinate momentum.
    heading = camera_boost(geometry) >> (time - direction)
    momentum = metric(geometry)(source_frame(geometry)(heading))
    momentum = -momentum / (momentum & time)
    momentum = momentum.reshape(-1)
    # Every ray starts at the eye, as an event of the whole of spacetime, the type its steps carry.
    return eye.broadcast_to(momentum.shape), momentum


def camera_screen(position: Antivector, momentum: Vector, orientation: Rotor,
                  spin: Spin, mass: float) -> ScreenFrame:
    """Orthonormal transverse camera axes, with horizontal x and vertical y screen inputs."""
    geometry = field(position, spin, mass)
    boost = camera_boost(geometry)
    heading = boost.reverse() >> source_components(geometry)(inverse_metric(geometry)(momentum))                                 # [...] Antivector-
    direction = space(heading) / -(mv.t & heading)                                          # [...] Antivector-
    # The turn carrying the camera's forward onto the direction it looks to see the ray: the reflection in
    # the forward, then in the direction halfway between the two.
    forward = orientation >> camera_forward                                                 # [] Antivector-
    halfway = direction + forward                                                           # [...] Antivector-
    turn = halfway * forward / halfway.squared().square_root()                              # [...] Rotor+
    # The camera's own right and up, turned with it; a screen direction is read by pairing with the
    # camera's slices x and y.
    return source_frame(geometry)((boost * turn * orientation) >> (camera_right * (mv.x & Screen) + camera_up * (mv.y & Screen)))  # [...] Antivector- <- Screen


