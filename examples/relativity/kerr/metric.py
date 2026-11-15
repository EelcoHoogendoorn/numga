"""Light rays and polarization in the Kerr geometry are traced from the metric and its Levi-Civita connection.

Events and directions are intrinsic antivectors, while momenta and spatial gradients are extrinsic covectors that pair with directions via `momentum & direction`. The physical metric maps coordinate directions to covectors, and its inverse maps covector momenta to coordinate ray tangents, giving the ray Hamiltonian as half the momentum's pairing with its inverse-metric tangent. Directional derivatives of the metric define the torsion-free Christoffel connection, which transports coordinate screen directions along each ray, while an orthonormal Kerr–Schild source frame specifies the static camera's angles and the disk's prescribed emission pattern.

Ingoing Cartesian Kerr–Schild coordinates cover the future horizon on the positive radial branch away from the singular ring, in units with gravitational constant and speed of light set to one, for subextremal black-hole spins.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, where
from numga.algebras import STA as ga
from numga.extensor import Extensor
from examples.relativity.kerr.plumbing import Image, crossing_fraction, resolve, runge_kutta, trace

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
# The hole's spin per unit mass: intrinsic, the patch of the plane it turns in, a plane through time.
Spin = Bivector.derive.within(Algebra("x+y+z+")).derive.dual
Rotor = ga.gatype.rotor()
# The camera's slices x and y, which read screen directions: vectors, extrinsic.
ScreenSlice = Vector.derive.within(Algebra("x+y+"))
# Directions on the camera's screen, along its own x and y: the antivectors that hold both t and z.
Screen = ScreenSlice.derive.dual
ScreenFrame = ga.gatype((Antivector, Screen))                # Antivector- <- Screen-
# The light's polarization covariance, from the slices that read the screen to screen directions.
Coherency = ga.gatype((Screen, ScreenSlice))                 # Screen- <- ScreenSlice+
DirectionGradient = ga.gatype((Antivector, Antivector))      # Antivector- <- Antivector-
# The metric and inverse metric connect complementary types.
Metric = ga.gatype((Vector, Antivector))                     # Vector+ <- Antivector-
InverseMetric = ga.gatype((Antivector, Vector))              # Antivector- <- Vector+
# One unit along the lab's time, the direction across the lab's slice of space, and the part of a
# direction within that slice.
time = -mv.t.dual()                                          # [] Antivector-
space = Antivector - time * (mv.t & Antivector)              # [] Antivector- <- Antivector-
# The camera's own axes in its rest frame, before it is turned: it looks along -z, with its right +y and its up -x.
camera_forward = -mv.z.dual()                                # [] Antivector-
camera_right = mv.y.dual()                                   # [] Antivector-
camera_up = -mv.x.dual()                                     # [] Antivector-


# --- math -----------------------------------------------------------------------------
def field_components(position: Antivector, spin: Spin, mass: float) -> tuple[Scalar, Scalar, Antivector, Vector, DirectionGradient]:
    """Kerr–Schild geometry at `position` comprises the oblate radius, scalar profile, ingoing null direction, and their gradients (`[...]`)."""
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
    return radius, profile, time - direction, profile_gradient, -direction_gradient


def metric(profile: Scalar, null: Antivector) -> Metric:
    """Forms the physical Kerr–Schild metric (`[...] Vector+ <- Antivector-`) with positive timelike square."""
    plane = -null.dual()                                                                    # [...] Vector+
    return -Antivector.dual() - 2 * profile * plane * (plane & Antivector)                  # [...] Vector+ <- Antivector-


def inverse_metric(profile: Scalar, null: Antivector) -> InverseMetric:
    """Forms the inverse Kerr–Schild metric (`[...] Antivector- <- Vector+`) mapping a covector momentum to its tangent."""
    return -Vector.dual() + 2 * profile * null * (Vector & null)                            # [...] Antivector- <- Vector+


def source_frame(profile: Scalar, null: Antivector) -> DirectionGradient:
    """Maps orthonormal Kerr–Schild source-frame directions to coordinate directions (`[...] Antivector- <- Antivector-`)."""
    plane = -null.dual()                                                                    # [...] Vector+
    return Antivector + profile * null * (plane & Antivector)                               # [...] Antivector- <- Antivector-


def source_components(profile: Scalar, null: Antivector) -> DirectionGradient:
    """Resolves coordinate directions in the orthonormal Kerr–Schild source frame (`[...] Antivector- <- Antivector-`)."""
    plane = -null.dual()                                                                    # [...] Vector+
    return Antivector - profile * null * (plane & Antivector)                               # [...] Antivector- <- Antivector-


def proper_squared(profile: Scalar, null: Antivector, direction: Antivector) -> Scalar:
    """Pairs `metric(profile, null)(direction)` with `direction` to read squared proper length (`[...] Scalar+`)."""
    return metric(profile, null)(direction) & direction                                     # [...] Scalar+


def hamiltonian(profile: Scalar, null: Antivector, momentum: Vector) -> Scalar:
    """Gives the ray Hamiltonian (`[...] Scalar+`) as half the momentum's pairing with its inverse-metric tangent."""
    return (momentum & inverse_metric(profile, null)(momentum)) / 2                         # [...] Scalar+


def overlap_gradient(null_gradient: DirectionGradient, momentum: Vector) -> Vector:
    """Pulls `momentum` back through `null_gradient.adjugate()` to form the gradient (`[...] Vector+`) of `momentum & null`."""
    return null_gradient.adjugate()(momentum)                                               # [...] Vector+


def motion(profile: Scalar, null: Antivector, profile_gradient: Vector,
           null_gradient: DirectionGradient, momentum: Vector) -> tuple[Antivector, Vector]:
    """Differentiates the Hamiltonian into the ray tangent (`[...] Antivector-`) and minus its position gradient (`[...] Vector+`)."""
    overlap = momentum & null                                                               # [...] Scalar+
    # Differentiating the quadratic Hamiltonian with respect to momentum applies the inverse metric.
    tangent = inverse_metric(profile, null)(momentum)                                       # [...] Antivector-
    # The gradient of profile * overlap squared, by the product rule.
    gradient = (profile_gradient * overlap.squared()
                + 2 * profile * overlap * overlap_gradient(null_gradient, momentum))        # [...] Vector+
    return tangent, -gradient                                                               # [...] Antivector-, [...] Vector+


def rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Hamiltonian ray dynamics at `position` pairs the ray tangent (`[...] Antivector-`) with the momentum rate (`[...] Vector+`)."""
    radius, profile, null, profile_gradient, null_gradient = field_components(position, spin, mass)
    del radius
    return motion(profile, null, profile_gradient, null_gradient, momentum)                 # [...] Antivector-, [...] Vector+


def camera_rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Past-directed Hamiltonian motion (`[...] Antivector-`, `[...] Vector+`) scales by decreasing coordinate time."""
    tangent, momentum_rate = rates(position, momentum, spin, mass)                          # [...] Antivector-, [...] Vector+
    clock = -(mv.t & tangent)                                                               # [...] Scalar+
    return tangent / clock, momentum_rate / clock                                           # [...] Antivector-, [...] Vector+


def metric_change(profile: Scalar, null: Antivector, profile_gradient: Vector,
                  null_gradient: DirectionGradient, step_direction: Antivector, direction: Antivector) -> Vector:
    """Directional differentiation along `step_direction` gives the metric's rate of change (`[...] Vector+`) on a fixed coordinate `direction`."""
    plane = -null.dual()                                                                    # [...] Vector+
    plane_change = -null_gradient(step_direction).dual()                                    # [...] Vector+
    return -2 * (
        (profile_gradient & step_direction) * plane * (plane & direction)
        + profile * (plane_change * (plane & direction)
                     + plane * (plane_change & direction)))                                 # [...] Vector+


def christoffel(profile: Scalar, null: Antivector, profile_gradient: Vector,
                null_gradient: DirectionGradient, step_direction: Antivector, direction: Antivector) -> Antivector:
    """Metric derivatives in the Cartesian coordinate basis assemble the torsion-free Levi-Civita connection (`[...] Antivector-`)."""
    plane = -null.dual()                                                                    # [...] Vector+
    step_reading, direction_reading = plane & step_direction, plane & direction             # [...] Scalar+, [...] Scalar+
    step_gradient = null_gradient.adjugate()(-step_direction.dual())                        # [...] Vector+
    direction_gradient = null_gradient.adjugate()(-direction.dual())                        # [...] Vector+
    # Differentiate the scalar metric reading with both coordinate directions held fixed.
    reading_gradient = -2 * (
        profile_gradient * step_reading * direction_reading
        + profile * (step_gradient * direction_reading
                     + step_reading * direction_gradient))                                  # [...] Vector+
    # Two directional derivatives and the scalar reading's gradient give the lowered connection;
    # the inverse metric returns its direction.
    return inverse_metric(profile, null)(
        metric_change(profile, null, profile_gradient, null_gradient, step_direction, direction)
        + metric_change(profile, null, profile_gradient, null_gradient, direction, step_direction) - reading_gradient) / 2  # [...] Antivector-


def polarized_rates(position: Antivector, momentum: Vector, screen: ScreenFrame,
                    spin: Spin, mass: float) -> tuple[Antivector, Vector, ScreenFrame]:
    """Past-directed ray motion carries the coordinate screen (`[...] Antivector- <- Screen-`) by Levi-Civita transport per unit of coordinate time."""
    radius, profile, null, profile_gradient, null_gradient = field_components(position, spin, mass)
    del radius
    tangent, momentum_rate = motion(profile, null, profile_gradient, null_gradient, momentum)
    screen_rate = -christoffel(profile, null, profile_gradient, null_gradient, tangent, screen)  # [...] Antivector- <- Screen-
    clock = -(mv.t & tangent)                                                               # [...] Scalar+
    return tangent / clock, momentum_rate / clock, screen_rate / clock                      # [...] Antivector-, [...] Vector+, [...] Antivector- <- Screen-


def disk_coherency(position: Antivector, momentum: Vector, screen: ScreenFrame,
                   normal: Vector, spin: Spin, mass: float, polarization_degree: float) -> Coherency:
    """Forms the unit-intensity polarization covariance (`[...] Screen- <- ScreenSlice+`) on the transported coordinate screen."""
    radius, profile, null, _, _ = field_components(position, spin, mass)
    del radius
    # The prescribed magnetic pattern is specified in this orthonormal source frame.
    heading = source_components(profile, null)(inverse_metric(profile, null)(momentum))     # [...] Antivector-
    local_screen = source_components(profile, null)(screen)                                 # [...] Antivector- <- Screen-
    direction = space(heading) / (mv.t & heading)                                           # [...] Antivector-
    # The magnetic direction circles the normal, square to the plane through the normal and the
    # emitter. Emission is polarized perpendicular to it across the ray: in that plane, square to the ray.
    meridian = space(position).dual() ^ normal                                              # [...] Bivector+
    polarization_axis = meridian.commutator(direction) / meridian.norm()                    # [...] Antivector-
    polarization = local_screen.adjoint()(polarization_axis)                                # [...] Screen-
    # Polarization vanishes smoothly when looking along the magnetic direction.
    polarized_fraction = polarization_degree * polarization.squared()                       # [...] Scalar+
    # Unpolarized light is round, where the metric enters; the polarized part weights the polarization by
    # each slice's reading of it.
    return ((1 - polarized_fraction) / 2 * ScreenSlice.dual()
            + polarization_degree * polarization * (ScreenSlice & polarization))            # [...] Screen- <- ScreenSlice+


def transmitted_power(coherency: Coherency, analyzers: ScreenSlice) -> Scalar:
    """Reads the power (`[...] Scalar+`) transmitted by each unit analyzer slice across `coherency`."""
    return analyzers & coherency(analyzers)                                                 # [...] Scalar+


def orbital_speed(radius: Scalar, orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """Gives the Keplerian coordinate angular speed (`[...] Scalar+`) of equatorial gas orbiting in `orbit`."""
    alignment = orbit & spin                                                                # [] Scalar+
    return np.sqrt(mass) / (radius * radius.square_root() + alignment * np.sqrt(mass))      # [...] Scalar+


def orbiting_gas(position: Antivector, orbit: Bivector, spin: Spin, mass: float) -> Antivector:
    """Gives the unit four-velocity (`[...] Antivector-`) of equatorial gas on a circular orbit in `orbit`."""
    radius, profile, null, _, _ = field_components(position, spin, mass)
    angular_speed = orbital_speed(radius, orbit, spin, mass)                                # [...] Scalar+
    # A point turned by `(orbit * (-angle / 2)).exp()` moves by its commutator with the orbit's plane.
    flow = time + angular_speed * space(position).commutator(orbit)                         # [...] Antivector-
    return flow / proper_squared(profile, null, flow).square_root()                         # [...] Antivector-


def frequency_ratio(position: Antivector, momentum: Vector, camera: Antivector, orbit: Bivector,
                    spin: Spin, mass: float) -> Scalar:
    """Gives the observed-to-emitted frequency ratio (`[...] Scalar+`) between a static camera at `camera` and orbiting gas."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector-
    camera_radius, camera_profile, camera_null, _, _ = field_components(camera, spin, mass)
    del camera_radius
    at_rest = time / proper_squared(camera_profile, camera_null, time).square_root()        # [] Antivector-
    # The momentum paired with time is conserved along the ray, so the camera's frequency reads here.
    return (momentum & at_rest) / (momentum & gas)                                          # [...] Scalar+


def orbital_origins(position: Antivector, radius: Scalar, orbit: Bivector, spin: Spin, mass: float,
                    distance: float, elapsed: np.ndarray) -> Antivector:
    """Turns disk events (`[...] Antivector-`) back along their orbits through `elapsed` plus light-travel delay."""
    emitted = elapsed[..., None] + (mv.t & position) + distance                             # [...] Scalar+
    angle = orbital_speed(radius, orbit, spin, mass) * emitted                              # [...] Scalar+
    rotors = (orbit * (angle / 2)).exp()                                                    # [...] Rotor+
    return rotors >> position                                                               # [...] Antivector-


def cube_root(value: Scalar) -> Scalar:
    """Gives the real cube root (`[...] Scalar+`) of each non-negative scalar."""
    return (value.log() / 3).exp()                                                          # [...] Scalar+


def innermost_stable_orbit(orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """Gives the innermost stable circular equatorial orbit radius (`[] Scalar+`) of Bardeen, Press and Teukolsky (1972)."""
    ratio = (orbit & spin) / mass                                                           # [] Scalar+
    first = 1 + cube_root(1 - ratio**2) * (cube_root(1 + ratio) + cube_root(1 - ratio))     # [] Scalar+
    second = (3 * ratio**2 + first**2).square_root()                                        # [] Scalar+
    # Gas orbiting with the hole reaches inward of the radius without spin, gas orbiting against it outward.
    reach = ((3 - first) * (3 + first + 2 * second)).square_root()                          # [] Scalar+
    return mass * (3 + second - where(ratio < 0, -reach, reach))                            # [] Scalar+


def disk_temperature(radius: Scalar, orbit: Bivector, spin: Spin, mass: float, scale: float) -> Scalar:
    """Gives the Page and Thorne (1974) thin-disk temperature (`[...] Scalar+`) at each oblate `radius`."""
    ratio = (orbit & spin) / mass                                                           # [] Scalar+
    root_radius = (radius / mass).square_root()                                             # [...] Scalar+
    edge = (innermost_stable_orbit(orbit, spin, mass) / mass).square_root()                 # [] Scalar+
    # The roots of the cubic `root_radius**3 - 3 * root_radius + 2 * ratio`, and the weight of each in
    # the torque; the product of a root's differences from the other two is the cubic's derivative there.
    roots = 2 * ((ratio.arccos() + np.array([-np.pi, np.pi, 3 * np.pi])) / 3).cos()         # [3] Scalar+
    weights = (roots - ratio) ** 2 / (roots * (roots ** 2 - 1))                             # [3] Scalar+
    # The torque the disk carries, zero at the innermost stable orbit; round-off kept non-negative.
    torque = (root_radius - edge - 1.5 * ratio * (root_radius / edge).log()
              - (weights * ((root_radius[..., None] - roots) / (edge - roots)).log()).sum(axis=-1)).clip(0, np.inf)  # [...] Scalar+
    flux = 3 / (8 * np.pi) * torque / (root_radius ** 4 * (root_radius ** 3 - 3 * root_radius + 2 * ratio))  # [...] Scalar+
    return scale * flux.square_root().square_root()                                         # [...] Scalar+


def disk_depth(radius: Scalar, inner_radius: Scalar, outer_radius: float, depth: float) -> Scalar:
    """Gives the perpendicular optical depth (`[...] Scalar+`) tapering quadratically from `depth` at `inner_radius` to zero at `outer_radius`."""
    fraction = (radius - inner_radius) / (outer_radius - inner_radius)                      # [...] Scalar+
    return depth * (1 - fraction) * (1 - fraction)                                          # [...] Scalar+


def disk_opacity(position: Antivector, momentum: Vector, depth: Scalar, normal: Vector,
                 orbit: Bivector, spin: Spin, mass: float) -> Scalar:
    """Gives the absorbed fraction (`[...] Scalar+`) of a ray crossing the disk at slant optical depth in the gas rest frame."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector-
    slant = ((momentum & gas) / (momentum & normal.dual())).abs()                           # [...] Scalar+
    return 1 - (-depth * slant).exp()                                                       # [...] Scalar+


def disk_crossing(start: tuple[Extensor, ...], end: tuple[Extensor, ...], normal: Vector,
                  spin: Spin, mass: float) -> tuple[tuple[Extensor, ...], Scalar]:
    """Interpolates a ray's state tuple across `normal` and evaluates its oblate radius (`[...] Scalar+`) at the crossing."""
    fraction = crossing_fraction(start[0], end[0], normal)                                  # [...] Scalar+
    state = tuple(value + fraction * (stop - value) for value, stop in zip(start, end))     # [...] state
    radius = field_components(state[0], spin, mass)[0]                                      # [...] Scalar+
    return state, radius                                                                    # [...] state, [...] Scalar+


def camera_boost(profile: Scalar, null: Antivector) -> Rotor:
    """Forms the boost rotor (`[...] Rotor+`) from a coordinate-static camera's rest frame into the local Kerr–Schild frame."""
    observer = source_components(profile, null)(time) / proper_squared(profile, null, time).square_root()  # [...] Antivector-
    return (1 - observer * time).normalized()                                               # [...] Rotor+


def initial_momentum(position: Antivector, direction: Antivector, spin: Spin, mass: float) -> Vector:
    """Forms a future-directed null covector (`[...] Vector+`) with unit spatial `direction` in the local Kerr–Schild frame."""
    radius, profile, null, _, _ = field_components(position, spin, mass)
    del radius
    local_null = time + direction                                                           # [...] Antivector-
    # A coordinate null direction is lowered with the physical metric.
    momentum = metric(profile, null)(source_frame(profile, null)(local_null))               # [...] Vector+
    return momentum / (momentum & time)                                                     # [...] Vector+


def camera_rays(eye: Antivector, orientation: Rotor, n_width: int, n_height: int,
                half_view: float, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Forms pixel-centre ray origins (`[n_height * n_width] Antivector-`) and past-directed momenta (`[n_height * n_width] Vector+`)."""
    horizontal = ((np.arange(n_width) + 0.5) / n_width * 2 - 1) * half_view * n_width / n_height
    vertical = (1 - (np.arange(n_height) + 0.5) / n_height * 2) * half_view
    direction = orientation >> (camera_forward + camera_right * horizontal[None, :] + camera_up * vertical[:, None])  # [n_height, n_width] Antivector-
    direction = direction / direction.antinorm()                                            # [n_height, n_width] Antivector-
    radius, profile, null, _, _ = field_components(eye, spin, mass)
    del radius
    boost = camera_boost(profile, null)                                                     # [] Rotor+
    # The future photon arrives opposite the direction the camera looks; its rest frame is boosted
    # into the Kerr–Schild frame before its heading is lowered into a coordinate momentum.
    heading = boost >> (time - direction)                                                   # [n_height, n_width] Antivector-
    momentum = metric(profile, null)(source_frame(profile, null)(heading))                  # [n_height, n_width] Vector+
    momentum = -momentum / (momentum & time)                                                # [n_height, n_width] Vector+
    momentum = momentum.reshape(-1)                                                         # [n_height * n_width] Vector+
    # Every ray starts at the eye, as an event of the whole of spacetime, the type its steps carry.
    return eye.broadcast_to(momentum.shape), momentum


def camera_screen(position: Antivector, momentum: Vector, orientation: Rotor,
                  spin: Spin, mass: float) -> ScreenFrame:
    """Forms orthonormal transverse camera screen frames (`[...] Antivector- <- Screen-`) in coordinate directions across each ray."""
    radius, profile, null, _, _ = field_components(position, spin, mass)
    del radius
    boost = camera_boost(profile, null)                                                     # [...] Rotor+
    unboost = boost.reverse()                                                               # [...] Rotor+
    heading = unboost >> source_components(profile, null)(inverse_metric(profile, null)(momentum))  # [...] Antivector-
    direction = space(heading) / -(mv.t & heading)                                          # [...] Antivector-
    # The turn carrying the camera's forward onto the direction it looks to see the ray: the reflection in
    # the forward, then in the direction halfway between the two.
    forward = orientation >> camera_forward                                                 # [] Antivector-
    halfway = direction + forward                                                           # [...] Antivector-
    turn = halfway * forward / halfway.antinorm()                                           # [...] Rotor+
    frame_rotor = boost * turn * orientation                                                # [...] Rotor+
    # The camera's own right and up, turned with it; a screen direction is read by pairing with the
    # camera's slices x and y.
    return source_frame(profile, null)(frame_rotor >> (camera_right * (mv.x & Screen) + camera_up * (mv.y & Screen)))  # [...] Antivector- <- Screen-
