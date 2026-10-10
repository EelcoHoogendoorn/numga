"""Kerr geometry, its geometric derivatives, and null-ray evolution."""

import numpy as np

from numga import stack
from examples.relativity.kerr import core


ROUND_OFF = 1e-11


def test_null_congruence_and_exact_inverse_metric():
    positions = core.mv.vector([[0, 4, 2, 1], [3, -3, 1, 2], [-1, 2, -4, -1]])
    spin = 0.7 * core.mv.xy + 0.2 * core.mv.yz
    mass = 1.0
    geometry = core.field(positions, spin, mass)
    dyad = geometry.profile * geometry.null * (geometry.null | core.Vector)
    metric = core.Vector - 2 * dyad
    inverse_metric = core.Vector + 2 * dyad
    momentum = core.initial_momentum(positions, core.mv.x, spin, mass)

    # The null direction follows straight lines in the background geometry;
    # its dyad squares to zero, so the inverse changes only the sign.
    np.testing.assert_allclose(geometry.null.squared().kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(geometry.null_gradient(geometry.null).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    # The closed-form curl is the null direction's gradient contracted against an open vector.
    np.testing.assert_allclose(
        (geometry.null_curl - (core.Vector ^ geometry.null_gradient(core.Vector)).contract(1, 2)).kernel,
        0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((metric(inverse_metric) - core.Vector).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((momentum | inverse_metric(momentum)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((momentum | core.mv.t).kernel, 1, atol=ROUND_OFF, rtol=0)

    # With zero spin the profile is the mass divided by ordinary spatial distance.
    schwarzschild = core.field(positions, 0 * spin, mass)
    spatial = positions - core.mv.t * (core.mv.t | positions)
    radius = (-spatial.squared()).square_root()
    np.testing.assert_allclose((schwarzschild.radius - radius).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((schwarzschild.profile - mass / radius).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_directional_derivatives_drive_hamiltonian_rates():
    position = core.mv.t + 4 * core.mv.x + 2 * core.mv.y + core.mv.z
    momentum = core.mv.t + 0.8 * core.mv.x - 0.3 * core.mv.y + 0.2 * core.mv.z
    directions = stack([core.mv.t, core.mv.x, core.mv.y, core.mv.z])
    spin = 0.7 * core.mv.xy + 0.2 * core.mv.yz
    mass = 1.0
    increment = 1e-4
    geometry = core.field(position, spin, mass)
    ahead = core.field(position + increment * directions, spin, mass)
    behind = core.field(position - increment * directions, spin, mass)
    tangent, force = core.rates(position, momentum, spin, mass)

    profile_change = (ahead.profile - behind.profile) / (2 * increment)
    null_change = (ahead.null - behind.null) / (2 * increment)
    hamiltonian_change = (ahead.profile * (ahead.null | momentum).squared()
                          - behind.profile * (behind.null | momentum).squared()) / (2 * increment)
    momentum_ahead = momentum + increment * directions
    momentum_behind = momentum - increment * directions
    velocity_change = ((momentum_ahead.squared() - momentum_behind.squared()) / 2
                       + geometry.profile * ((geometry.null | momentum_ahead).squared()
                                             - (geometry.null | momentum_behind).squared())) / (2 * increment)

    # These tolerances measure centered-difference accuracy, not round-off alone.
    np.testing.assert_allclose((profile_change - (geometry.profile_gradient | directions)).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((null_change - geometry.null_gradient(directions)).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((hamiltonian_change + (force | directions)).kernel,
                               0, atol=2e-9, rtol=0)
    np.testing.assert_allclose((velocity_change - (tangent | directions)).kernel,
                               0, atol=2e-9, rtol=0)


def test_rays_conserve_their_invariants_and_spin_reversal_mirrors_the_paths():
    mass = 1.0
    spin = core.mv.xy * np.array([0.7, -0.7])[:, None]
    offsets = np.array([-7.0, 2.0, 7.0])
    initial = -8 * core.mv.x + core.mv.y * offsets
    mirror = core.Vector + 2 * core.mv.y * (core.mv.y | core.Vector)
    position = stack([initial, mirror(initial)])
    momentum = core.initial_momentum(position, core.mv.x, spin, mass)
    step_size = 0.04
    steps = 250
    stop_fraction = 0.8
    states = tuple(core.trace(position, momentum, spin, mass, step_size, steps, stop_fraction, core.rates))
    positions = stack([state[0] for state in states])
    momenta = stack([state[1] for state in states])
    geometry = core.field(positions, spin, mass)
    hamiltonian = momenta.squared() / 2 + geometry.profile * (geometry.null | momenta).squared()
    energy = core.mv.t | momenta
    angular_momentum = (positions ^ momenta) | spin
    horizon = mass + (mass**2 + spin.squared()).square_root()

    # Null propagation and axial angular momentum constrain the integrated dynamics.
    # Their tolerances measure the accuracy of the finite RK4 steps.
    np.testing.assert_allclose(hamiltonian.kernel, 0, atol=2e-6, rtol=0)
    np.testing.assert_allclose((angular_momentum - angular_momentum[0]).kernel,
                               0, atol=1e-7, rtol=0)
    np.testing.assert_allclose((energy - energy[0]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror(positions[:, 0]) - positions[:, 1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((mirror(momenta[:, 0]) - momenta[:, 1]).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    # The central ray crosses the horizon and stops safely inside it; the outer rays escape.
    final_radius = geometry.radius[-1]
    assert np.all((final_radius[:, 1] - horizon[:, 0]).kernel < 0)
    assert np.all((final_radius[:, 1] - stop_fraction * horizon[:, 0]).kernel > 0)
    assert np.all(final_radius[:, [0, 2]].kernel > 5)


def test_camera_and_disk_geometry_rotate_with_the_black_hole():
    mass = 1.0
    spin = 0.7 * core.mv.xy
    orientation = (0.31 * core.mv.xz).exp()
    eye = (orientation >> core.mv.z) * 25
    rotation = (0.23 * core.mv.yz).exp()
    pixel_width, pixel_height = 5, 3
    half_view = 0.4
    position, momentum = core.camera_rays(
        eye, orientation, pixel_width, pixel_height, half_view, spin, mass)
    rotated_position, rotated_momentum = core.camera_rays(
        rotation >> eye, rotation * orientation, pixel_width, pixel_height,
        half_view, rotation >> spin, mass)
    tangent, _ = core.rates(position, momentum, spin, mass)

    # Camera rays point into the past and remain null in the curved metric.
    np.testing.assert_allclose((momentum | tangent).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((rotated_position - (rotation >> position)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((rotated_momentum - (rotation >> momentum)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    assert np.all((core.mv.t | tangent).kernel < 0)
    center = (pixel_width * pixel_height) // 2
    geometry = core.field(position[center], spin, mass)
    local_null = momentum[center] + geometry.profile * geometry.null * (geometry.null | momentum[center])
    center_direction = core.mv.t | (core.mv.t ^ local_null)
    center_direction = center_direction / (-center_direction.squared()).square_root()
    toward_hole = -eye / (-eye.squared()).square_root()
    np.testing.assert_allclose((center_direction - toward_hole).kernel,
                               0, atol=ROUND_OFF, rtol=0)

    # Intersect a tilted disk between events whose heights have opposite signs.
    start = rotation >> (core.mv.t + 3 * core.mv.x + 4 * core.mv.y + 2 * core.mv.z)
    end = rotation >> (6 * core.mv.t + 8 * core.mv.x - core.mv.y - 3 * core.mv.z)
    (crossing,), radius = core.disk_crossing((start,), (end,), rotation >> core.mv.z,
                                             rotation >> spin, mass)
    expected = rotation >> (3 * core.mv.t + 5 * core.mv.x + 2 * core.mv.y)
    np.testing.assert_allclose((crossing - expected).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(radius.kernel, np.sqrt(29 - 0.7**2),
                               atol=ROUND_OFF, rtol=0)


def test_backward_camera_rays_resolve_the_schwarzschild_capture_boundary():
    mass = 1.0
    spin = 0 * core.mv.xy
    eye = 25 * core.mv.x
    slopes = np.array([0.14, 0.18, 0.20, 0.28])
    direction = -core.mv.x + core.mv.y * slopes
    direction = direction / (-direction.squared()).square_root()
    momentum = -core.initial_momentum(eye, -direction, spin, mass)
    position = eye.broadcast_to(momentum.shape)
    energy = core.mv.t | momentum
    angular_momentum = (position ^ momentum) | core.mv.xy
    impact = (angular_momentum / energy).kernel[:, 0]
    expected_capture = np.abs(impact) < 3 * np.sqrt(3) * mass
    # Inside the photon sphere, where no ray turns back out.
    capture_radius = 2.1 * mass
    escape_radius = 40 * mass
    step_fraction = 0.1
    steps = 300
    active = np.arange(len(slopes))
    captured = np.zeros(len(slopes), dtype=bool)
    finished = np.zeros(len(slopes), dtype=bool)
    rays = core.camera_trace((position, momentum), spin, mass, step_fraction, steps, core.camera_rates)
    segment = next(rays)

    while len(active):
        radius = segment.radius.kernel[:, 0]
        capture = radius < capture_radius
        escape = radius > escape_radius
        done = capture | escape
        captured[active[capture]] = True
        finished[active[done]] = True

        # The finite-step check constrains the escaped rays after their full encounter.
        if np.any(escape):
            final_position = segment.end[0][escape]
            final_momentum = segment.end[1][escape]
            tangent, _ = core.rates(final_position, final_momentum, spin, mass)
            np.testing.assert_allclose((final_momentum | tangent).kernel,
                                       0, atol=2e-5, rtol=0)
            np.testing.assert_allclose(((final_position ^ final_momentum) | core.mv.xy).kernel,
                                       angular_momentum[active[escape]].kernel, atol=2e-5, rtol=0)
            np.testing.assert_allclose((core.mv.t | final_momentum).kernel,
                                       energy[active[escape]].kernel, atol=ROUND_OFF, rtol=0)
        keep = np.flatnonzero(~done)
        active = active[keep]
        if len(active):
            segment = rays.send(keep)

    assert np.all(finished)
    np.testing.assert_array_equal(captured, expected_capture)


def test_connection_matches_directional_metric_variation():
    position = core.mv.t + 4 * core.mv.x + 2 * core.mv.y + core.mv.z
    spin = 0.7 * core.mv.xy + 0.2 * core.mv.yz
    mass = 1.0
    velocity = core.mv.t + 0.4 * core.mv.x - 0.2 * core.mv.y + 0.3 * core.mv.z
    basis = stack([core.mv.t, core.mv.x, core.mv.y, core.mv.z])
    increment = 1e-4
    metric_signs = np.diag([1.0, -1.0, -1.0, -1.0])
    geometry = core.field(position, spin, mass)
    ahead = core.field(position + increment * basis, spin, mass)
    behind = core.field(position - increment * basis, spin, mass)
    dyad = geometry.profile * geometry.null * (geometry.null | core.Vector)
    ahead_dyad = ahead.profile * ahead.null * (ahead.null | core.Vector)
    behind_dyad = behind.profile * behind.null * (behind.null | core.Vector)

    # Independently recover transport from finite changes of the metric and frame.
    frame = (core.Vector + dyad).kernel
    metric = metric_signs @ (core.Vector - 2 * dyad).kernel
    frame_gradient = (ahead_dyad - behind_dyad).kernel / (2 * increment)
    metric_gradient = -2 * metric_signs @ frame_gradient
    christoffel = np.einsum(
        "ad,bcd->abc", np.linalg.inv(metric),
        metric_gradient + metric_gradient.swapaxes(0, 1) - metric_gradient.transpose(1, 2, 0),
    ) / 2
    expected = -np.linalg.solve(frame,
        np.einsum("b,bij->ij", velocity.kernel, frame_gradient)
        + np.einsum("abc,b,cd->ad", christoffel, velocity.kernel, frame))
    local_velocity = (core.Vector - dyad)(velocity)
    transported = -core.connection(geometry, local_velocity).commutator(core.Vector)
    flat_connection = core.connection(core.field(position, spin, 0.0), local_velocity)

    # The finite-difference tolerance checks the connection, including its transport sign.
    np.testing.assert_allclose(transported.kernel, expected, atol=2e-8, rtol=0)
    np.testing.assert_allclose(flat_connection.kernel, 0, atol=ROUND_OFF, rtol=0)


def test_parallel_transport_preserves_the_transverse_screen_along_kerr_rays():
    mass = 1.0
    spin = 0.7 * core.mv.xy + 0.2 * core.mv.yz
    orientation = (0.31 * core.mv.xz).exp()
    eye = (orientation >> core.mv.z) * 18
    pixel_width, pixel_height = 5, 3
    half_view = 0.4
    inner_radius, outer_radius = 2.02, 40.0
    step_fraction = 0.1
    steps = 250
    position, momentum = core.camera_rays(
        eye, orientation, pixel_width, pixel_height, half_view, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)
    rays = core.camera_trace((position, momentum, screen), spin, mass,
                             step_fraction, steps, core.polarized_rates)
    segment = next(rays)
    transverse_error = 0.0
    gram_error = 0.0

    while True:
        end, momentum, screen = segment.end
        local_momentum = core.frame(core.field(end, spin, mass))(momentum)
        local_momentum = local_momentum / -(core.mv.t | local_momentum)
        transverse_error = max(transverse_error,
            np.max(np.abs((screen | local_momentum).kernel)))
        gram_error = max(gram_error,
            np.max(np.abs(((screen | screen) - (core.Screen | core.Screen)).kernel)))
        radius = segment.radius.kernel[..., 0]
        keep = np.flatnonzero((radius > inner_radius) & (radius < outer_radius))
        if not len(keep):
            break
        segment = rays.send(keep)

    # Finite-step transport keeps both screen axes unit, perpendicular, and transverse.
    np.testing.assert_allclose([transverse_error, gram_error], 0, atol=1e-5, rtol=0)


def test_disk_coherency_through_a_rotating_polarizer():
    position = 6 * core.mv.x
    spin = 0.7 * core.mv.xy
    mass = 0.0
    orientation = core.mv.rotor()
    polarization_degrees = np.array([0.0, 0.6, 1.0])
    angles = np.linspace(0, np.pi, 17)
    momentum = -core.initial_momentum(position, core.mv.z, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)
    coherency = core.disk_coherency(
        position, momentum, screen, core.mv.z, spin, mass, polarization_degrees[:, None])
    analyzer = (core.mv.xy * (angles / 2)).exp() >> core.mv.x
    opposite = (core.mv.xy * ((angles + np.pi) / 2)).exp() >> core.mv.x
    transmitted = -(analyzer | coherency(analyzer))
    repeated = -(opposite | coherency(opposite))
    expected = ((1 - polarization_degrees[:, None]) / 2
                + polarization_degrees[:, None] * np.sin(angles) ** 2)

    # Azimuthal magnetic field and a normal ray give maximal linear polarization.
    np.testing.assert_allclose(coherency.trace().kernel, 1, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(transmitted.kernel[..., 0], expected, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((repeated - transmitted).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_every_camera_ray_shares_its_light_once_between_disk_sky_and_hole():
    mass = 1.0
    spin = 0.7 * core.mv.xy
    orientation = (0.6 * core.mv.zx).exp()
    eye = (orientation >> core.mv.z) * 25
    pixel_width, pixel_height = 9, 7
    pixel_count = pixel_width * pixel_height
    inner_radius, outer_radius = 3.4, 12.0
    depth = 40.0
    capture_radius = 1.05 * (mass + np.sqrt(mass**2 - 0.7**2))

    def opacity(event: core.Vector, momentum: core.Vector, radius: core.Scalar) -> core.Scalar:
        layer = core.disk_depth(radius, inner_radius, outer_radius, depth)
        return core.disk_opacity(event, momentum, layer, core.mv.z, core.mv.xy, spin, mass)
    position, momentum = core.camera_rays(eye, orientation, pixel_width, pixel_height, 0.35, spin, mass)
    steps = core.camera_trace((position, momentum), spin, mass, 0.2, 300, core.camera_rates)
    image = core.resolve(steps, pixel_count, core.mv.z, inner_radius, outer_radius, opacity, 100.0, capture_radius, spin, mass)
    disk_share = np.bincount(image.disk_pixels, image.disk_weights.kernel[..., 0], pixel_count)
    sky_share = np.bincount(image.sky_pixels, image.sky_weights.kernel[..., 0], pixel_count)

    # checks: crossings lie in the disk plane between its radii, each giving part of its pixel's light.
    np.testing.assert_allclose((core.mv.z | image.disk[0]).kernel, 0, atol=ROUND_OFF, rtol=0)
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
    spin = 0.8 * core.mv.xy
    far, near = 1e6, 6.0
    # Far out the gas barely moves and the field is flat: a ray along the normal meets the bare depth.
    face_on = core.initial_momentum(core.mv.x * far, -core.mv.z, spin, mass)
    # Two rays at one slant through opposite sides of the orbit: the gas at +x moves along +y, towards
    # where the light goes, and the gas at -x moves away from it.
    sides = stack([core.mv.x * near, -core.mv.x * near])                                   # [2] Vector
    slanted = core.initial_momentum(sides, (core.mv.y - core.mv.z) / np.sqrt(2), spin, mass)

    straight = core.disk_opacity(core.mv.x * far, face_on, core.mv.scalar([depth]), core.mv.z, core.mv.xy, spin, mass)
    approaching, receding = core.disk_opacity(sides, slanted, core.mv.scalar([depth]), core.mv.z, core.mv.xy, spin, mass)

    # checks: this tolerance tests the far field's residual orbital speed and curvature, about 1e-6.
    np.testing.assert_allclose(straight.kernel[..., 0], 1 - np.exp(-depth), atol=1e-4, rtol=0)
    assert receding.kernel[0] > approaching.kernel[0]


def test_frequency_ratio_of_orbiting_gas_seen_by_a_camera_at_rest():
    mass, radius, distance = 1.0, 8.0, 1e6
    position = radius * core.mv.x
    no_spin = 0 * core.mv.xy
    radial = core.initial_momentum(position, core.mv.x, no_spin, mass)
    schwarzschild = core.frequency_ratio(position, radial, distance * core.mv.x, core.mv.xy, no_spin, mass)
    flat = core.frequency_ratio(position, radial, distance * core.mv.x, core.mv.xy, no_spin, 0.0)
    spin = 0.8 * core.mv.xy
    toward = core.initial_momentum(position, core.mv.y, spin, mass)
    away = core.initial_momentum(position, -core.mv.y, spin, mass)
    approaching = core.frequency_ratio(position, toward, distance * core.mv.y, core.mv.xy, spin, mass)
    receding = core.frequency_ratio(position, away, -distance * core.mv.y, core.mv.xy, spin, mass)

    # A radial photon from a circular Schwarzschild orbit: time dilation of the orbit and the well.
    expected = np.sqrt(1 - 3 * mass / radius) / np.sqrt(1 - 2 * mass / distance)
    np.testing.assert_allclose(schwarzschild.kernel, expected, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(flat.kernel, 1, atol=ROUND_OFF, rtol=0)
    # Gas coming towards the camera is blueshifted, gas going away redshifted.
    assert approaching.kernel[0] > 1 > receding.kernel[0]


def test_relativistic_thin_disk_temperature():
    mass, scale = 1.0, 1.0
    orbit = core.mv.xy
    radii = np.array([6.5, 8.0, 12.0, 40.0])
    nearly_still = core.disk_temperature(core.mv.scalar(radii[:, None]), orbit, orbit * 1e-9, mass, scale).kernel[:, 0]
    root3, root6 = np.sqrt(3), np.sqrt(6)
    logarithm = np.log((np.sqrt(radii) + root3) * (root6 - root3) / ((np.sqrt(radii) - root3) * (root6 + root3)))
    schwarzschild = (3 / (8 * np.pi * radii**3) / (1 - 3 / radii)
                     * (1 - np.sqrt(6 / radii) + np.sqrt(3 / (4 * radii)) * logarithm)) ** 0.25

    # Without spin, the Kerr flux is Page and Thorne's Schwarzschild one.
    np.testing.assert_allclose(nearly_still, schwarzschild, rtol=1e-6, atol=0)
    # Zero at the innermost stable orbit; a faster spin reaches deeper and runs hotter.
    edge = core.innermost_stable_orbit(orbit, orbit * 0.8, mass)
    assert core.disk_temperature(edge, orbit, orbit * 0.8, mass, scale).kernel[0] < 1e-3
    peaks = [core.disk_temperature(core.mv.scalar(np.geomspace(core.innermost_stable_orbit(orbit, orbit * spin, mass).kernel[0], 30, 2000)[:, None]),
                                   orbit, orbit * spin, mass, scale).kernel.max() for spin in (0.5, 0.8, 0.95)]
    assert peaks[0] < peaks[1] < peaks[2]


def test_innermost_stable_orbit():
    orbit = core.mv.xy
    # At the extreme the cube root of zero is taken through its logarithm, minus infinity.
    with np.errstate(divide="ignore"):
        radii = core.innermost_stable_orbit(orbit, orbit * np.array([0.0, 0.8, 1.0]), 1.0)
    # 6M without spin, about 2.91M at spin 0.8M, and the hole's own mass at the extreme.
    np.testing.assert_allclose(radii.kernel[..., 0], [6.0, 2.9066, 1.0], atol=1e-4, rtol=0)
    np.testing.assert_allclose(radii.kernel[[0, 2], 0], [6.0, 1.0], atol=ROUND_OFF, rtol=0)
