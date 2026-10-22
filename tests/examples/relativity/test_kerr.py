"""Kerr geometry, its geometric derivatives, and null-ray evolution."""

import numpy as np

from numga import stack
from examples.relativity.kerr import core


ROUND_OFF = 1e-11
mv = core.mv
# Unit directions along time and the three axes, and the slices that read their coordinates.
T, X, Y, Z = core.time, mv.x.dual(), mv.y.dual(), mv.z.dual()
DIRECTIONS = stack([T, X, Y, Z])                                     # [4] Antivector
SLICES = stack([mv.t, mv.x, mv.y, mv.z])                             # [4] Vector


def events(coordinates) -> core.Antivector:
    """Events, or directions, at the given (t, x, y, z) on the last axis."""
    return (DIRECTIONS * np.asarray(coordinates, dtype=float)).sum(axis=-1)


def matrix(direction_map: core.DirectionGradient) -> np.ndarray:
    """Maps on directions in coordinates, `[..., out, in]`: each image of an axis read by each slice."""
    return (SLICES[:, None] & direction_map[..., None, None](DIRECTIONS[None, :])).to_array()


def test_null_congruence_and_exact_inverse_gauge():
    positions = events([[0, 4, 2, 1], [3, -3, 1, 2], [-1, 2, -4, -1]])
    spin = 0.7 * mv.xy + 0.2 * mv.yz
    mass = 1.0
    geometry = core.field(positions, spin, mass)
    momentum = core.initial_momentum(positions, X, spin, mass)

    # The null direction follows straight lines in the background geometry;
    # its dyad squares to zero, so the inverse changes only the sign.
    np.testing.assert_allclose(geometry.null.squared().kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(geometry.null_gradient(geometry.null).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    # The closed-form curl is the derivative of the null covector: a wedge and a trace.
    curl = -(core.Vector ^ geometry.null_gradient(core.Antivector).dual()).trace(1, 2)
    np.testing.assert_allclose((geometry.null_curl - curl).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((core.gauge(geometry)(core.gauge_inverse(geometry)) - core.Antivector).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    # A light ray's momentum, carried into the local frame, is flat null.
    np.testing.assert_allclose(core.local(geometry, momentum).squared().kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((momentum & T).kernel, 1, atol=ROUND_OFF, rtol=0)

    # With zero spin the profile is the mass divided by ordinary spatial distance.
    schwarzschild = core.field(positions, 0 * spin, mass)
    radius = core.space(positions).squared().square_root()
    np.testing.assert_allclose((schwarzschild.radius - radius).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((schwarzschild.profile - mass / radius).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_directional_derivatives_drive_hamiltonian_rates():
    position = events([1, 4, 2, 1])
    momentum = mv.vector([1, 0.8, -0.3, 0.2])
    spin = 0.7 * mv.xy + 0.2 * mv.yz
    mass = 1.0
    increment = 1e-4
    geometry = core.field(position, spin, mass)
    ahead = core.field(position + increment * DIRECTIONS, spin, mass)
    behind = core.field(position - increment * DIRECTIONS, spin, mass)
    tangent, force = core.rates(position, momentum, spin, mass)

    def hamiltonian(geometry, momentum):
        return -(momentum & momentum.dual()) / 2 + geometry.profile * (momentum & geometry.null).squared()

    profile_change = (ahead.profile - behind.profile) / (2 * increment)
    null_change = (ahead.null - behind.null) / (2 * increment)
    hamiltonian_change = (hamiltonian(ahead, momentum) - hamiltonian(behind, momentum)) / (2 * increment)
    velocity_change = (hamiltonian(geometry, momentum + increment * SLICES)
                       - hamiltonian(geometry, momentum - increment * SLICES)) / (2 * increment)

    # These tolerances measure centered-difference accuracy, not round-off alone.
    np.testing.assert_allclose((profile_change - (geometry.profile_gradient & DIRECTIONS)).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((null_change - geometry.null_gradient(DIRECTIONS)).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((hamiltonian_change + (force & DIRECTIONS)).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((velocity_change - (SLICES & tangent)).kernel,
                               0, atol=2e-9, rtol=0)


def test_rays_conserve_their_invariants_and_spin_reversal_mirrors_the_paths():
    mass = 1.0
    spin = mv.xy * np.array([0.7, -0.7])[:, None]
    offsets = np.array([-7.0, 2.0, 7.0])
    initial = -8 * X + Y * offsets
    # The mirror across the plane y = 0, on events and on covectors; it reverses the spin.
    mirror = core.Antivector - 2 * Y * (mv.y & core.Antivector)
    covector_mirror = core.Vector - 2 * mv.y * (core.Vector & Y)
    position = stack([initial, mirror(initial)])
    momentum = core.initial_momentum(position, X, spin, mass)
    step_size = 0.04
    steps = 250
    stop_fraction = 0.8
    positions, momenta = core.trace(position, momentum, spin, mass, step_size, steps, stop_fraction, core.rates)
    geometry = core.field(positions, spin, mass)
    hamiltonian = -(momenta & momenta.dual()) / 2 + geometry.profile * (momenta & geometry.null).squared()
    energy = momenta & T
    # The momentum along the direction the spin turns each event in.
    angular_momentum = momenta & spin.commutator(positions)
    horizon = mass + (mass**2 + spin.squared()).square_root()

    # Null propagation and axial angular momentum constrain the integrated dynamics.
    # Their tolerances measure the accuracy of the finite RK4 steps.
    np.testing.assert_allclose(hamiltonian.kernel, 0, atol=2e-6, rtol=0)
    np.testing.assert_allclose((angular_momentum - angular_momentum[0]).kernel,
                               0, atol=1e-7, rtol=0)
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror(positions[:, 0]) - positions[:, 1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((covector_mirror(momenta[:, 0]) - momenta[:, 1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    # The central ray crosses the horizon and stops safely inside it; the outer rays escape.
    final_radius = geometry.radius[-1]
    assert np.all((final_radius[:, 1] - horizon[:, 0]).kernel < 0)
    assert np.all((final_radius[:, 1] - stop_fraction * horizon[:, 0]).kernel > 0)
    assert np.all(final_radius[:, [0, 2]].kernel > 5)


def test_camera_and_disk_geometry_rotate_with_the_black_hole():
    mass = 1.0
    spin = 0.7 * mv.xy
    orientation = (0.31 * mv.xz).exp()
    eye = (orientation >> Z) * 25
    rotation = (0.23 * mv.yz).exp()
    pixel_width, pixel_height = 5, 3
    half_view = 0.4
    position, momentum = core.camera_rays(
        eye, orientation, pixel_width, pixel_height, half_view, spin, mass)
    rotated_position, rotated_momentum = core.camera_rays(
        rotation >> eye, rotation * orientation, pixel_width, pixel_height,
        half_view, rotation >> spin, mass)
    tangent, _ = core.rates(position, momentum, spin, mass)

    # Camera rays point into the past and remain null in the curved metric.
    np.testing.assert_allclose((momentum & tangent).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((rotated_position - (rotation >> position)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((rotated_momentum - (rotation >> momentum)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    assert np.all((mv.t & tangent).kernel < 0)
    center = (pixel_width * pixel_height) // 2
    heading = core.space(core.local(core.field(position[center], spin, mass), momentum[center]))
    center_direction = heading / heading.squared().square_root()
    toward_hole = -eye / eye.squared().square_root()
    np.testing.assert_allclose((center_direction - toward_hole).kernel,
                               0, atol=ROUND_OFF, rtol=0)

    # Intersect a tilted disk between events whose heights have opposite signs.
    start = rotation >> events([1, 3, 4, 2])
    end = rotation >> events([6, 8, -1, -3])
    (crossing,), radius = core.disk_crossing((start,), (end,), rotation >> mv.z,
                                             rotation >> spin, mass)
    expected = rotation >> events([3, 5, 2, 0])
    np.testing.assert_allclose((crossing - expected).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(radius.kernel, np.sqrt(29 - 0.7**2),
                               atol=ROUND_OFF, rtol=0)


def test_backward_camera_rays_resolve_the_schwarzschild_capture_boundary():
    mass = 1.0
    spin = 0 * mv.xy
    eye = 25 * X
    slopes = np.array([0.14, 0.18, 0.20, 0.28])
    direction = -X + Y * slopes
    direction = direction / direction.squared().square_root()
    momentum = -core.initial_momentum(eye, -direction, spin, mass)
    position = eye.broadcast_to(momentum.shape)

    def angular(position, momentum):
        """The momentum along the direction a turn in the xy plane moves each event."""
        return momentum & mv.xy.commutator(position)

    energy = momentum & T
    angular_momentum = angular(position, momentum)
    impact = (angular_momentum / energy).kernel[:, 0]
    expected_capture = np.abs(impact) < 3 * np.sqrt(3) * mass
    # Inside the photon sphere, where no ray turns back out.
    capture_radius = 2.1 * mass
    escape_radius = 40 * mass
    step_fraction = 0.1
    steps = 300

    def clear(event: core.Antivector, momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        """No disk: every ray keeps its light until it escapes or is captured."""
        return radius * 0
    image, (final_position, final_momentum) = core.resolve(
        (position, momentum), spin, mass, step_fraction, steps, core.camera_rates, mv.z, 0.0, escape_radius, clear,
        escape_radius, capture_radius)
    final_radius = core.field(final_position, spin, mass).radius.kernel[:, 0]
    escape = final_radius > escape_radius
    tangent, _ = core.rates(final_position, final_momentum, spin, mass)

    # Every ray ends, escaped with all its light or captured.
    assert np.all(escape | (final_radius < capture_radius))
    np.testing.assert_array_equal(image.sky_pixels, np.flatnonzero(escape))
    np.testing.assert_array_equal(~escape, expected_capture)
    # The finite-step check constrains the escaped rays after their full encounter.
    np.testing.assert_allclose((final_momentum & tangent)[escape].kernel, 0, atol=2e-5, rtol=0)
    np.testing.assert_allclose(angular(final_position, final_momentum)[escape].kernel,
                               angular_momentum[escape].kernel, atol=2e-5, rtol=0)
    np.testing.assert_allclose((final_momentum & T)[escape].kernel, energy[escape].kernel, atol=ROUND_OFF, rtol=0)


def test_connection_matches_directional_metric_variation():
    position = events([1, 4, 2, 1])
    spin = 0.7 * mv.xy + 0.2 * mv.yz
    mass = 1.0
    velocity = events([1, 0.4, -0.2, 0.3])
    increment = 1e-4
    geometry = core.field(position, spin, mass)
    ahead = core.field(position + increment * DIRECTIONS, spin, mass)
    behind = core.field(position - increment * DIRECTIONS, spin, mass)

    def dyad(geometry):
        return core.dyad(geometry)

    def metric(geometry):
        # The coordinate metric: minus the flat products of the axes carried back into the local frame.
        carried = core.gauge_inverse(geometry)[..., None](DIRECTIONS)
        return -(carried[..., :, None] | carried[..., None, :]).to_array()

    # Independently recover transport from finite changes of the metric and frame, in coordinates:
    # the frame takes local directions to coordinate ones, and the Christoffel symbols follow from
    # the metric's changes along each axis, `metric_gradient[axis, a, b]`.
    frame = matrix(core.Antivector + dyad(geometry))
    frame_gradient = (matrix(dyad(ahead)) - matrix(dyad(behind))) / (2 * increment)
    metric_gradient = (metric(ahead) - metric(behind)) / (2 * increment)
    terms = metric_gradient.transpose(0, 2, 1) + metric_gradient.transpose(2, 0, 1) - metric_gradient.transpose(1, 2, 0)
    christoffel = np.einsum("ad,bcd->abc", np.linalg.inv(metric(geometry)), terms) / 2
    components = (SLICES & velocity).to_array()
    expected = -np.linalg.solve(frame,
        np.einsum("b,bij->ij", components, frame_gradient)
        + np.einsum("abc,b,cd->ad", christoffel, components, frame))
    local_velocity = (core.Antivector - dyad(geometry))(velocity)
    transported = -core.rotation(geometry, local_velocity).commutator(core.Antivector)
    flat_connection = core.rotation(core.field(position, spin, 0.0), local_velocity)

    # The finite-difference tolerance checks the connection, including its transport sign.
    np.testing.assert_allclose(matrix(transported), expected, atol=2e-8, rtol=0)
    np.testing.assert_allclose(flat_connection.kernel, 0, atol=ROUND_OFF, rtol=0)


def test_parallel_transport_preserves_the_transverse_screen_along_kerr_rays():
    mass = 1.0
    spin = 0.7 * mv.xy + 0.2 * mv.yz
    orientation = (0.31 * mv.xz).exp()
    eye = (orientation >> Z) * 18
    pixel_width, pixel_height = 5, 3
    half_view = 0.4
    inner_radius, outer_radius = 2.02, 40.0
    step_fraction = 0.1
    steps = 250
    position, momentum = core.camera_rays(
        eye, orientation, pixel_width, pixel_height, half_view, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)

    def clear(event: core.Antivector, momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        """No disk: every ray is traced until it escapes or is captured."""
        return radius * 0
    _, (end, momentum, screen) = core.resolve(
        (position, momentum, screen), spin, mass, step_fraction, steps, core.polarized_rates, mv.z, 0.0, outer_radius,
        clear, outer_radius, inner_radius)
    heading = core.local(core.field(end, spin, mass), momentum)
    heading = heading / -(mv.t & heading)

    # Finite-step transport keeps both screen axes unit, perpendicular, and transverse, to the fourth-order
    # error of the steps, about 2e-5 here, at the end of every ray.
    np.testing.assert_allclose((screen | heading).kernel, 0, atol=5e-5, rtol=0)
    np.testing.assert_allclose(((screen | screen) - (core.Screen | core.Screen)).kernel, 0, atol=5e-5, rtol=0)


def test_disk_coherency_through_a_rotating_polarizer():
    position = 6 * X
    spin = 0.7 * mv.xy
    mass = 0.0
    orientation = mv.rotor()
    polarization_degrees = np.array([0.0, 0.6, 1.0])
    angles = np.linspace(0, np.pi, 17)
    momentum = -core.initial_momentum(position, Z, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)
    coherency = core.disk_coherency(
        position, momentum, screen, mv.z, spin, mass, polarization_degrees[:, None])
    analyzer = (mv.xy * (angles / 2)).exp() >> X
    opposite = (mv.xy * ((angles + np.pi) / 2)).exp() >> X
    transmitted = core.transmitted(coherency, analyzer)
    repeated = core.transmitted(coherency, opposite)
    expected = ((1 - polarization_degrees[:, None]) / 2
                + polarization_degrees[:, None] * np.sin(angles) ** 2)

    # Azimuthal magnetic field and a normal ray give maximal linear polarization.
    np.testing.assert_allclose(coherency.trace().kernel, 1, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(transmitted.kernel[..., 0], expected, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((repeated - transmitted).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_every_camera_ray_shares_its_light_once_between_disk_sky_and_hole():
    mass = 1.0
    spin = 0.7 * mv.xy
    orientation = (0.6 * mv.zx).exp()
    eye = (orientation >> Z) * 25
    pixel_width, pixel_height = 9, 7
    pixel_count = pixel_width * pixel_height
    inner_radius, outer_radius = 3.4, 12.0
    depth = 40.0
    capture_radius = 1.05 * (mass + np.sqrt(mass**2 - 0.7**2))

    def opacity(event: core.Antivector, momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        layer = core.disk_depth(radius, inner_radius, outer_radius, depth)
        return core.disk_opacity(event, momentum, layer, mv.z, mv.xy, spin, mass)
    position, momentum = core.camera_rays(eye, orientation, pixel_width, pixel_height, 0.35, spin, mass)
    image, _ = core.resolve((position, momentum), spin, mass, 0.2, 300, core.camera_rates, mv.z, inner_radius,
                            outer_radius, opacity, 100.0, capture_radius)
    disk_share = np.bincount(image.disk_pixels, image.disk_weights.kernel[..., 0], pixel_count)
    sky_share = np.bincount(image.sky_pixels, image.sky_weights.kernel[..., 0], pixel_count)

    # checks: crossings lie in the disk plane between its radii, each giving part of its pixel's light.
    np.testing.assert_allclose((mv.z & image.disk[0]).kernel, 0, atol=ROUND_OFF, rtol=0)
    assert np.all((image.disk_radii > inner_radius) & (image.disk_radii < outer_radius))
    assert np.all(image.disk_weights > 0)
    # A ray escapes at most once; escaped rays share all their light, captured ones lose the rest.
    assert len(np.unique(image.sky_pixels)) == len(image.sky_pixels)
    np.testing.assert_allclose((disk_share + sky_share)[image.sky_pixels], 1, atol=ROUND_OFF, rtol=0)
    assert np.all(disk_share + sky_share <= 1 + ROUND_OFF)
    # Some rays meet the disk where it clears, and its body absorbs others all but whole.
    assert np.any((disk_share > 0) & (disk_share < 0.5))
    assert np.any(disk_share > 0.999)
    # The central ray, aimed at the hole, passes inside the disk's inner edge and is captured.
    assert disk_share[pixel_count // 2] == 0 and sky_share[pixel_count // 2] == 0


def test_disk_opacity_face_on_and_where_its_gas_recedes():
    mass, depth = 1.0, 0.7
    spin = 0.8 * mv.xy
    far, near = 1e6, 6.0
    # Far out the gas barely moves and the field is flat: a ray along the normal meets the bare depth.
    face_on = core.initial_momentum(X * far, -Z, spin, mass)
    # Two rays at one slant through opposite sides of the orbit: the gas at +x moves along +y, towards
    # where the light goes, and the gas at -x moves away from it.
    sides = stack([X * near, -X * near])                                                   # [2] Antivector
    slanted = core.initial_momentum(sides, (Y - Z) / np.sqrt(2), spin, mass)

    straight = core.disk_opacity(X * far, face_on, mv.scalar([depth]), mv.z, mv.xy, spin, mass)
    approaching, receding = core.disk_opacity(sides, slanted, mv.scalar([depth]), mv.z, mv.xy, spin, mass)

    # checks: this tolerance tests the far field's residual orbital speed and curvature, about 1e-6.
    np.testing.assert_allclose(straight.kernel[..., 0], 1 - np.exp(-depth), atol=1e-4, rtol=0)
    assert receding.kernel[0] > approaching.kernel[0]


def test_frequency_ratio_of_orbiting_gas_seen_by_a_camera_at_rest():
    mass, radius, distance = 1.0, 8.0, 1e6
    position = radius * X
    no_spin = 0 * mv.xy
    radial = core.initial_momentum(position, X, no_spin, mass)
    schwarzschild = core.frequency_ratio(position, radial, distance * X, mv.xy, no_spin, mass)
    flat = core.frequency_ratio(position, radial, distance * X, mv.xy, no_spin, 0.0)
    spin = 0.8 * mv.xy
    toward = core.initial_momentum(position, Y, spin, mass)
    away = core.initial_momentum(position, -Y, spin, mass)
    approaching = core.frequency_ratio(position, toward, distance * Y, mv.xy, spin, mass)
    receding = core.frequency_ratio(position, away, -distance * Y, mv.xy, spin, mass)

    # A radial photon from a circular Schwarzschild orbit: time dilation of the orbit and the well.
    expected = np.sqrt(1 - 3 * mass / radius) / np.sqrt(1 - 2 * mass / distance)
    np.testing.assert_allclose(schwarzschild.kernel, expected, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(flat.kernel, 1, atol=ROUND_OFF, rtol=0)
    # Gas coming towards the camera is blueshifted, gas going away redshifted.
    assert approaching.kernel[0] > 1 > receding.kernel[0]


def test_relativistic_thin_disk_temperature():
    mass, scale = 1.0, 1.0
    orbit = mv.xy
    radii = np.array([6.5, 8.0, 12.0, 40.0])
    nearly_still = core.disk_temperature(mv.scalar(radii[:, None]), orbit, orbit * 1e-9, mass, scale).kernel[:, 0]
    root3, root6 = np.sqrt(3), np.sqrt(6)
    logarithm = np.log((np.sqrt(radii) + root3) * (root6 - root3) / ((np.sqrt(radii) - root3) * (root6 + root3)))
    schwarzschild = (3 / (8 * np.pi * radii**3) / (1 - 3 / radii)
                     * (1 - np.sqrt(6 / radii) + np.sqrt(3 / (4 * radii)) * logarithm)) ** 0.25

    # Without spin, the Kerr flux is Page and Thorne's Schwarzschild one.
    np.testing.assert_allclose(nearly_still, schwarzschild, rtol=1e-6, atol=0)
    # Zero at the innermost stable orbit; a faster spin reaches deeper and runs hotter.
    edge = core.innermost_stable_orbit(orbit, orbit * 0.8, mass)
    assert core.disk_temperature(edge, orbit, orbit * 0.8, mass, scale).kernel[0] < 1e-3
    peaks = [core.disk_temperature(mv.scalar(np.geomspace(core.innermost_stable_orbit(orbit, orbit * spin, mass).kernel[0], 30, 2000)[:, None]),
                                   orbit, orbit * spin, mass, scale).kernel.max() for spin in (0.5, 0.8, 0.95)]
    assert peaks[0] < peaks[1] < peaks[2]


def test_innermost_stable_orbit():
    orbit = mv.xy
    # At the extreme the cube root of zero is taken through its logarithm, minus infinity.
    with np.errstate(divide="ignore"):
        radii = core.innermost_stable_orbit(orbit, orbit * np.array([0.0, 0.8, 1.0]), 1.0)
    # 6M without spin, about 2.91M at spin 0.8M, and the hole's own mass at the extreme.
    np.testing.assert_allclose(radii.kernel[..., 0], [6.0, 2.9066, 1.0], atol=1e-4, rtol=0)
    np.testing.assert_allclose(radii.kernel[[0, 2], 0], [6.0, 1.0], atol=ROUND_OFF, rtol=0)
