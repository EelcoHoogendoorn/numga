"""Light rays in the Kerr geometry are traced by geometric derivatives of their Hamiltonian and parallel transport of the camera screen.

Events and directions are intrinsic antivectors, while momenta and spatial gradients are extrinsic vectors that pair with directions via `momentum & direction`. Gauge theory gravity writes the curved geometry on a flat background through a position gauge field `gauge` that maps local-frame directions to Kerr directions as the identity plus a scalar profile times a null dyad. Because the null dyad squares to zero, `gauge_inverse` is the identity minus that dyad. Carrying a direction back into the local frame gives its squared proper length as minus the flat square, and raising a momentum flat before carrying it into the local frame gives the ray Hamiltonian as minus half its flat square. Differentiating with respect to momentum gives the ray tangent, while pulling the momentum back through the null direction's derivative by its adjugate combines with the profile gradient to give minus the momentum rate without component equations. Along each ray, the rotation gauge field gives the plane in which the local frame turns or boosts per step, transporting the camera screen by a bivector commutator while the thin accretion disk absorbs and emits polarized blackbody light at each equatorial crossing.

Ingoing Cartesian Kerr–Schild coordinates cover the future horizon on the positive radial branch away from the singular ring, in units with gravitational constant and speed of light set to one. The hole's spin per unit mass is an intrinsic spatial patch whose dual inverse is the turning plane, following Pelle et al., Skylight (2022), equations 24–26, https://arxiv.org/abs/2206.06429.
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
# The position gauge field: from a direction of the local frame to one of the Kerr geometry.
GaugeField = ga.gatype((Antivector, Antivector))             # Antivector- <- Antivector-
# One unit along the lab's time, the direction across the lab's slice of space, and the part of a
# direction within that slice.
time = -mv.t.dual()                                          # [] Antivector-
space = Antivector - time * (mv.t & Antivector)              # [] Antivector- <- Antivector-
# The camera's own axes in its rest frame, before it is turned: it looks along -z, with its right +y and its up -x.
camera_forward = -mv.z.dual()                                # [] Antivector-
camera_right = mv.y.dual()                                   # [] Antivector-
camera_up = -mv.x.dual()                                     # [] Antivector-


# --- math -----------------------------------------------------------------------------
def field_components(position: Antivector, spin: Spin, mass: float) -> tuple[Scalar, Scalar, Antivector, Vector, DirectionGradient, Bivector]:
    """Kerr–Schild geometry at `position` comprises the oblate radius, scalar profile, ingoing null direction, their gradients, and the null curl (`[...]`)."""
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
    # The curl of the null vector, `-(Vector ^ null_gradient(Antivector).dual()).trace(1, 2)`, piece by
    # piece: the rank-one piece gives its two vectors wedged, the twist by the plane twice the plane, the
    # rest none.
    null_curl = ((radius_gradient ^ lever.dual()) - 2 * turning) / (radius_squared + spin_squared)  # [...] Bivector+
    return radius, profile, time - direction, profile_gradient, -direction_gradient, null_curl


def dyad(profile: Scalar, null: Antivector) -> GaugeField:
    """Forms the profile-weighted null dyad (`[...] Antivector- <- Antivector-`) from the null direction and its tangent plane `-null.dual()`."""
    null_plane = -null.dual()                                                               # [...] Vector+
    return profile * null * (null_plane & Antivector)                                       # [...] Antivector- <- Antivector-


def gauge(profile: Scalar, null: Antivector) -> GaugeField:
    """Maps local-frame directions to Kerr–Schild coordinate directions (`[...] Antivector- <- Antivector-`) by adding the weighted null dyad."""
    return Antivector + dyad(profile, null)                                                 # [...] Antivector- <- Antivector-


def gauge_inverse(profile: Scalar, null: Antivector) -> GaugeField:
    """Maps Kerr–Schild coordinate directions back to the local frame (`[...] Antivector- <- Antivector-`) by subtracting the square-zero null dyad."""
    return Antivector - dyad(profile, null)                                                 # [...] Antivector- <- Antivector-


def local_direction(profile: Scalar, null: Antivector, momentum: Vector) -> Antivector:
    """Raises `momentum` with the flat metric and carries it through the gauge field into the local frame (`[...] Antivector-`)."""
    return gauge(profile, null)(-momentum.dual())                                           # [...] Antivector-


def proper_squared(profile: Scalar, null: Antivector, direction: Antivector) -> Scalar:
    """Gives the squared proper length (`[...] Scalar+`) of a timelike direction as minus its flat square in the local frame."""
    return -gauge_inverse(profile, null)(direction).squared()                               # [...] Scalar+


def hamiltonian(profile: Scalar, null: Antivector, momentum: Vector) -> Scalar:
    """Gives the ray Hamiltonian (`[...] Scalar+`) as minus half the flat square of the momentum's local direction."""
    return -local_direction(profile, null, momentum).squared() / 2                          # [...] Scalar+


def overlap_gradient(null_gradient: DirectionGradient, momentum: Vector) -> Vector:
    """Pulls `momentum` back through `null_gradient.adjugate()` to form the gradient (`[...] Vector+`) of `momentum & null`."""
    return null_gradient.adjugate()(momentum)                                               # [...] Vector+


def motion(profile: Scalar, null: Antivector, profile_gradient: Vector,
           null_gradient: DirectionGradient, momentum: Vector) -> tuple[Antivector, Vector]:
    """Differentiates the Hamiltonian into the ray tangent (`[...] Antivector-`) and minus its position gradient (`[...] Vector+`)."""
    overlap = momentum & null                                                               # [...] Scalar+
    # Minus half the flat square of the momentum's local direction is the Hamiltonian; its momentum derivative
    # is that direction carried through the gauge field once more.
    tangent = gauge(profile, null)(local_direction(profile, null, momentum))                # [...] Antivector-
    # The gradient of profile * overlap squared, by the product rule.
    gradient = (profile_gradient * overlap.squared()
                + 2 * profile * overlap * overlap_gradient(null_gradient, momentum))        # [...] Vector+
    return tangent, -gradient                                                               # [...] Antivector-, [...] Vector+


def rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Hamiltonian ray dynamics at `position` pairs the ray tangent (`[...] Antivector-`) with the momentum rate (`[...] Vector+`)."""
    radius, profile, null, profile_gradient, null_gradient, null_curl = field_components(position, spin, mass)
    del radius, null_curl
    return motion(profile, null, profile_gradient, null_gradient, momentum)                 # [...] Antivector-, [...] Vector+


def camera_rates(position: Antivector, momentum: Vector, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Past-directed Hamiltonian motion (`[...] Antivector-`, `[...] Vector+`) scales by decreasing coordinate time."""
    tangent, momentum_rate = rates(position, momentum, spin, mass)                          # [...] Antivector-, [...] Vector+
    clock = -(mv.t & tangent)                                                               # [...] Scalar+
    return tangent / clock, momentum_rate / clock                                           # [...] Antivector-, [...] Vector+


def rotation(profile: Scalar, null: Antivector, profile_gradient: Vector,
             null_gradient: DirectionGradient, null_curl: Bivector, direction: Antivector) -> Bivector:
    """The rotation gauge bivector (`[...] Bivector+`) gives the plane in which the local frame turns or boosts along `direction`."""
    null_plane = -null.dual()                                                               # [...] Vector+
    projection = null_plane & direction                                                     # [...] Scalar+
    slope = profile_gradient * projection + profile * overlap_gradient(null_gradient, -direction.dual())  # [...] Vector+
    return -(slope ^ null_plane) - profile * null_curl * projection                         # [...] Bivector+


def polarized_rates(position: Antivector, momentum: Vector, screen: ScreenFrame,
                    spin: Spin, mass: float) -> tuple[Antivector, Vector, ScreenFrame]:
    """Past-directed ray motion carries the camera screen (`[...] Antivector- <- Screen-`) by parallel transport per unit of coordinate time."""
    radius, profile, null, profile_gradient, null_gradient, null_curl = field_components(position, spin, mass)
    del radius
    tangent, momentum_rate = motion(profile, null, profile_gradient, null_gradient, momentum)
    heading = local_direction(profile, null, momentum)                                      # [...] Antivector-
    connection = rotation(profile, null, profile_gradient, null_gradient, null_curl, heading)  # [...] Bivector+
    screen_rate = -connection.commutator(screen)                                            # [...] Antivector- <- Screen-
    clock = -(mv.t & tangent)                                                               # [...] Scalar+
    return tangent / clock, momentum_rate / clock, screen_rate / clock                      # [...] Antivector-, [...] Vector+, [...] Antivector- <- Screen-


def disk_coherency(position: Antivector, momentum: Vector, screen: ScreenFrame,
                   normal: Vector, spin: Spin, mass: float, polarization_degree: float) -> Coherency:
    """Forms the unit-intensity polarization covariance (`[...] Screen- <- ScreenSlice+`) on the transported camera screen."""
    radius, profile, null, profile_gradient, null_gradient, null_curl = field_components(position, spin, mass)
    del radius, profile_gradient, null_gradient, null_curl
    heading = local_direction(profile, null, momentum)                                      # [...] Antivector-
    direction = space(heading) / (mv.t & heading)                                           # [...] Antivector-
    # The magnetic direction circles the normal, square to the plane through the normal and the
    # emitter. Emission is polarized perpendicular to it across the ray: in that plane, square to the ray.
    meridian = space(position).dual() ^ normal                                              # [...] Bivector+
    polarization_axis = meridian.commutator(direction) / meridian.norm()                    # [...] Antivector-
    polarization = screen.adjoint()(polarization_axis)                                      # [...] Screen-
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
    radius, profile, null, profile_gradient, null_gradient, null_curl = field_components(position, spin, mass)
    del profile_gradient, null_gradient, null_curl
    angular_speed = orbital_speed(radius, orbit, spin, mass)                                # [...] Scalar+
    # A point turned by `(orbit * (-angle / 2)).exp()` moves by its commutator with the orbit's plane.
    flow = time + angular_speed * space(position).commutator(orbit)                         # [...] Antivector-
    return flow / proper_squared(profile, null, flow).square_root()                         # [...] Antivector-


def frequency_ratio(position: Antivector, momentum: Vector, camera: Antivector, orbit: Bivector,
                    spin: Spin, mass: float) -> Scalar:
    """Gives the observed-to-emitted frequency ratio (`[...] Scalar+`) between a static camera at `camera` and orbiting gas."""
    gas = orbiting_gas(position, orbit, spin, mass)                                         # [...] Antivector-
    camera_radius, camera_profile, camera_null, _, _, _ = field_components(camera, spin, mass)
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
    observer = gauge_inverse(profile, null)(time) / proper_squared(profile, null, time).square_root()  # [...] Antivector-
    return (1 - observer * time).normalized()                                               # [...] Rotor+


def initial_momentum(position: Antivector, direction: Antivector, spin: Spin, mass: float) -> Vector:
    """Forms a future-directed null momentum (`[...] Vector+`) with unit spatial `direction` in the local Kerr–Schild frame."""
    radius, profile, null, _, _, _ = field_components(position, spin, mass)
    del radius
    local_null = time + direction                                                           # [...] Antivector-
    # Carried back out of the local frame and lowered flat.
    momentum = -gauge_inverse(profile, null)(local_null).dual()                             # [...] Vector+
    return momentum / (momentum & time)                                                     # [...] Vector+


def camera_rays(eye: Antivector, orientation: Rotor, n_width: int, n_height: int,
                half_view: float, spin: Spin, mass: float) -> tuple[Antivector, Vector]:
    """Forms pixel-centre ray origins (`[n_height * n_width] Antivector-`) and past-directed momenta (`[n_height * n_width] Vector+`)."""
    horizontal = ((np.arange(n_width) + 0.5) / n_width * 2 - 1) * half_view * n_width / n_height
    vertical = (1 - (np.arange(n_height) + 0.5) / n_height * 2) * half_view
    direction = orientation >> (camera_forward + camera_right * horizontal[None, :] + camera_up * vertical[:, None])  # [n_height, n_width] Antivector-
    direction = direction / direction.antinorm()                                            # [n_height, n_width] Antivector-
    radius, profile, null, _, _, _ = field_components(eye, spin, mass)
    del radius
    boost = camera_boost(profile, null)                                                     # [] Rotor+
    # The future photon arrives opposite the direction the camera looks; its rest frame is boosted
    # into the Kerr–Schild frame before its heading is lowered into a coordinate momentum.
    heading = boost >> (time - direction)                                                   # [n_height, n_width] Antivector-
    momentum = -gauge_inverse(profile, null)(heading).dual()                                # [n_height, n_width] Vector+
    momentum = -momentum / (momentum & time)                                                # [n_height, n_width] Vector+
    momentum = momentum.reshape(-1)                                                         # [n_height * n_width] Vector+
    # Every ray starts at the eye, as an event of the whole of spacetime, the type its steps carry.
    return eye.broadcast_to(momentum.shape), momentum


def camera_screen(position: Antivector, momentum: Vector, orientation: Rotor,
                  spin: Spin, mass: float) -> ScreenFrame:
    """Forms orthonormal transverse camera screen frames (`[...] Antivector- <- Screen-`) across each ray."""
    radius, profile, null, _, _, _ = field_components(position, spin, mass)
    del radius
    boost = camera_boost(profile, null)                                                     # [...] Rotor+
    unboost = boost.reverse()                                                               # [...] Rotor+
    heading = unboost >> local_direction(profile, null, momentum)                           # [...] Antivector-
    direction = space(heading) / -(mv.t & heading)                                          # [...] Antivector-
    # The turn carrying the camera's forward onto the direction it looks to see the ray: the reflection in
    # the forward, then in the direction halfway between the two.
    forward = orientation >> camera_forward                                                 # [] Antivector-
    halfway = direction + forward                                                           # [...] Antivector-
    turn = halfway * forward / halfway.antinorm()                                           # [...] Rotor+
    frame_rotor = boost * turn * orientation                                                # [...] Rotor+
    # The camera's own right and up, turned with it; a screen direction is read by pairing with the
    # camera's slices x and y.
    return frame_rotor >> (camera_right * (mv.x & Screen) + camera_up * (mv.y & Screen))    # [...] Antivector- <- Screen-
