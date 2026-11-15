"""Kerr geometry, geometric Hamiltonian derivatives, parallel transport, camera rays, and thin-disk optics."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.relativity.kerr import core, render, scenarios


round_off = 1e-11
mv = core.mv
# Unit directions along time and the three axes, and the slices that read their coordinates.
time_axis, x_axis, y_axis, z_axis = core.time, mv.x.dual(), mv.y.dual(), mv.z.dual()
directions = stack([time_axis, x_axis, y_axis, z_axis])                 # [4] Antivector-
slices = stack([mv.t, mv.x, mv.y, mv.z])                                # [4] Vector+


def events(coordinates) -> core.Antivector:
    """Forms spacetime events or directions (`[...] Antivector-`) from `(t, x, y, z)` coordinates."""
    return (directions * np.asarray(coordinates, dtype=float)).sum(axis=-1)


def matrix(direction_map: core.DirectionGradient) -> np.ndarray:
    """Reads coordinate matrices (`[..., 4, 4]`) of direction maps by pairing with `slices` and `directions`."""
    return (slices[:, None] & direction_map[..., None, None](directions[None, :])).to_array()


def test_null_congruence_and_exact_inverse_gauge():
    positions = events([[0, 4, 2, 1], [3, -3, 1, 2], [-1, 2, -4, -1]])
    spin = (0.7 * mv.xy + 0.2 * mv.yz).dual()
    mass = 1.0
    radius, profile, null, _, null_gradient, null_curl = core.field_components(positions, spin, mass)
    del radius
    momentum = core.initial_momentum(positions, x_axis, spin, mass)

    # The null direction follows straight lines in the background geometry;
    # its dyad squares to zero, so the inverse changes only the sign.
    assert (null.squared().abs() < round_off).all()
    assert (null_gradient(null).scalar_norm_squared() < round_off**2).all()
    # The closed-form curl is the derivative of the null covector: a wedge and a trace.
    curl = -(core.Vector ^ null_gradient(core.Antivector).dual()).trace(1, 2)
    assert ((null_curl - curl).scalar_norm_squared() < round_off**2).all()
    round_trip = core.gauge(profile, null)(core.gauge_inverse(profile, null)) - core.Antivector
    assert (round_trip[..., None](directions).scalar_norm_squared() < round_off**2).all()
    # A light ray's momentum, carried into the local frame, is flat null.
    assert (core.local_direction(profile, null, momentum).squared().abs() < round_off).all()
    assert (((momentum & time_axis) - 1).abs() < round_off).all()

    # With zero spin the profile is the mass divided by ordinary spatial distance.
    schwarzschild_radius, schwarzschild_profile, _, _, _, _ = core.field_components(positions, 0 * spin, mass)
    flat_radius = core.space(positions).antinorm()
    assert ((schwarzschild_radius - flat_radius).abs() < round_off).all()
    assert ((schwarzschild_profile - mass / flat_radius).abs() < round_off).all()


def test_directional_derivatives_drive_hamiltonian_rates():
    position = events([1, 4, 2, 1])
    momentum = mv.vector([1, 0.8, -0.3, 0.2])
    spin = (0.7 * mv.xy + 0.2 * mv.yz).dual()
    mass = 1.0
    increment = 1e-4
    _, profile, null, profile_gradient, null_gradient, _ = core.field_components(position, spin, mass)
    _, ahead_profile, ahead_null, _, _, _ = core.field_components(position + increment * directions, spin, mass)
    _, behind_profile, behind_null, _, _, _ = core.field_components(position - increment * directions, spin, mass)
    tangent, force = core.rates(position, momentum, spin, mass)

    profile_change = (ahead_profile - behind_profile) / (2 * increment)
    null_change = (ahead_null - behind_null) / (2 * increment)
    hamiltonian_change = (core.hamiltonian(ahead_profile, ahead_null, momentum)
                          - core.hamiltonian(behind_profile, behind_null, momentum)) / (2 * increment)
    velocity_change = (core.hamiltonian(profile, null, momentum + increment * slices)
                       - core.hamiltonian(profile, null, momentum - increment * slices)) / (2 * increment)

    assert ((profile_change - (profile_gradient & directions)).abs() < 2e-9).all()
    assert ((null_change - null_gradient(directions)).scalar_norm_squared() < 4e-18).all()
    assert ((hamiltonian_change + (force & directions)).abs() < 2e-9).all()
    assert ((velocity_change - (slices & tangent)).abs() < 2e-9).all()


def test_camera_and_disk_geometry_rotate_with_the_black_hole():
    mass = 1.0
    spin = 0.7 * mv.xy.dual()
    orientation = (0.31 * mv.xz).exp()
    eye = (orientation >> z_axis) * 25
    rotor = (0.23 * mv.yz).exp()
    n_width, n_height = 5, 3
    half_view = 0.4
    position, momentum = core.camera_rays(eye, orientation, n_width, n_height, half_view, spin, mass)
    rotated_position, rotated_momentum = core.camera_rays(
        rotor >> eye, rotor * orientation, n_width, n_height, half_view, rotor >> spin, mass)
    tangent, _ = core.rates(position, momentum, spin, mass)

    # Camera rays point into the past and remain null in the curved metric.
    assert ((momentum & tangent).abs() < round_off).all()
    assert ((rotated_position - (rotor >> position)).scalar_norm_squared() < round_off**2).all()
    assert ((rotated_momentum - (rotor >> momentum)).scalar_norm_squared() < round_off**2).all()
    assert ((mv.t & tangent) < 0).all()
    center = (n_width * n_height) // 2
    _, camera_profile, camera_null, _, _, _ = core.field_components(position[center], spin, mass)
    unboost = core.camera_boost(camera_profile, camera_null).reverse()
    heading = core.space(unboost >> core.local_direction(camera_profile, camera_null, momentum[center]))
    center_direction = heading / heading.antinorm()
    toward_hole = -eye / eye.antinorm()
    assert ((center_direction - toward_hole).scalar_norm_squared() < round_off**2).all()

    # Intersect a tilted disk between events whose heights have opposite signs.
    start = rotor >> events([1, 3, 4, 2])
    end = rotor >> events([6, 8, -1, -3])
    (crossing,), radius = core.disk_crossing((start,), (end,), rotor >> mv.z, rotor >> spin, mass)
    expected = rotor >> events([3, 5, 2, 0])
    assert ((crossing - expected).scalar_norm_squared() < round_off**2).all()
    assert ((radius - np.sqrt(29 - 0.7**2)).abs() < round_off).all()


def test_connection_matches_directional_metric_variation():
    position = events([1, 4, 2, 1])
    spin = (0.7 * mv.xy + 0.2 * mv.yz).dual()
    mass = 1.0
    velocity = events([1, 0.4, -0.2, 0.3])
    increment = 1e-4
    _, profile, null, profile_gradient, null_gradient, null_curl = core.field_components(position, spin, mass)
    _, ahead_profile, ahead_null, _, _, _ = core.field_components(position + increment * directions, spin, mass)
    _, behind_profile, behind_null, _, _, _ = core.field_components(position - increment * directions, spin, mass)

    def coordinate_metric(metric_profile, metric_null):
        carried = core.gauge_inverse(metric_profile, metric_null)[..., None](directions)
        return -(carried[..., :, None] | carried[..., None, :]).to_array()

    frame = matrix(core.gauge(profile, null))
    frame_gradient = (matrix(core.dyad(ahead_profile, ahead_null))
                      - matrix(core.dyad(behind_profile, behind_null))) / (2 * increment)
    metric_gradient = (coordinate_metric(ahead_profile, ahead_null)
                       - coordinate_metric(behind_profile, behind_null)) / (2 * increment)
    terms = metric_gradient.transpose(0, 2, 1) + metric_gradient.transpose(2, 0, 1) - metric_gradient.transpose(1, 2, 0)
    christoffel = np.einsum("ad,bcd->abc", np.linalg.inv(coordinate_metric(profile, null)), terms) / 2
    components = (slices & velocity).to_array()
    expected = -np.linalg.solve(
        frame,
        np.einsum("b,bij->ij", components, frame_gradient) + np.einsum("abc,b,cd->ad", christoffel, components, frame))
    local_velocity = core.gauge_inverse(profile, null)(velocity)
    connection = core.rotation(profile, null, profile_gradient, null_gradient, null_curl, local_velocity)
    transported = -connection.commutator(core.Antivector)
    _, flat_profile, flat_null, flat_profile_grad, flat_null_grad, flat_curl = core.field_components(position, spin, 0.0)
    flat_connection = core.rotation(flat_profile, flat_null, flat_profile_grad, flat_null_grad, flat_curl, local_velocity)

    np.testing.assert_allclose(matrix(transported), expected, atol=2e-8, rtol=0)
    assert (flat_connection.scalar_norm_squared() < round_off**2).all()


def test_disk_coherency_through_a_rotating_polarizer():
    position = 6 * x_axis
    spin = 0.7 * mv.xy.dual()
    mass = 0.0
    orientation = mv.rotor()
    polarization_degrees = np.array([0.0, 0.6, 1.0])
    angles = np.linspace(0, np.pi, 17)
    momentum = -core.initial_momentum(position, z_axis, spin, mass)
    screen = core.camera_screen(position, momentum, orientation, spin, mass)
    coherency = core.disk_coherency(
        position, momentum, screen, mv.z, spin, mass, polarization_degrees[:, None])
    analyzer_rotors = (mv.xy * (angles / 2)).exp()
    opposite_rotors = (mv.xy * ((angles + np.pi) / 2)).exp()
    analyzer = analyzer_rotors >> mv.x
    opposite = opposite_rotors >> mv.x
    transmitted = core.transmitted_power(coherency, analyzer)
    repeated = core.transmitted_power(coherency, opposite)
    expected = ((1 - polarization_degrees[:, None]) / 2
                + polarization_degrees[:, None] * np.sin(angles) ** 2)

    intensity = (core.ScreenSlice & coherency(core.ScreenSlice)).contract()
    assert ((intensity - 1).abs() < round_off).all()
    assert ((transmitted - expected).abs() < round_off).all()
    assert ((repeated - transmitted).abs() < round_off).all()


def test_disk_opacity_face_on_and_where_its_gas_recedes():
    mass, depth = 1.0, 0.7
    spin = 0.8 * mv.xy.dual()
    far, near = 1e6, 6.0
    face_on = core.initial_momentum(x_axis * far, -z_axis, spin, mass)
    sides = stack([x_axis * near, -x_axis * near])
    slanted = core.initial_momentum(sides, (y_axis - z_axis) / np.sqrt(2), spin, mass)

    straight = core.disk_opacity(x_axis * far, face_on, mv.scalar([depth]), mv.z, mv.xy, spin, mass)
    approaching, receding = core.disk_opacity(sides, slanted, mv.scalar([depth]), mv.z, mv.xy, spin, mass)

    assert ((straight - (1 - np.exp(-depth))).abs() < 1e-4).all()
    assert (receding > approaching).all()


def test_frequency_ratio_of_orbiting_gas_seen_by_a_camera_at_rest():
    mass, radius, distance = 1.0, 8.0, 1e6
    position = radius * x_axis
    no_spin = 0 * mv.xy.dual()
    radial = core.initial_momentum(position, x_axis, no_spin, mass)
    schwarzschild = core.frequency_ratio(position, radial, distance * x_axis, mv.xy, no_spin, mass)
    flat = core.frequency_ratio(position, radial, distance * x_axis, mv.xy, no_spin, 0.0)
    spin = 0.8 * mv.xy.dual()
    toward = core.initial_momentum(position, y_axis, spin, mass)
    away = core.initial_momentum(position, -y_axis, spin, mass)
    approaching = core.frequency_ratio(position, toward, distance * y_axis, mv.xy, spin, mass)
    receding = core.frequency_ratio(position, away, -distance * y_axis, mv.xy, spin, mass)

    expected = np.sqrt(1 - 3 * mass / radius) / np.sqrt(1 - 2 * mass / distance)
    assert ((schwarzschild - expected).abs() < round_off).all()
    assert ((flat - 1).abs() < round_off).all()
    assert bool((approaching > 1).all()) and bool((receding < 1).all())


def test_relativistic_thin_disk_temperature():
    mass, scale = 1.0, 1.0
    orbit = mv.xy
    radii = np.array([6.5, 8.0, 12.0, 40.0])
    nearly_still = core.disk_temperature(mv.scalar(radii[:, None]), orbit, orbit.dual() * 1e-9, mass, scale)
    root3, root6 = np.sqrt(3), np.sqrt(6)
    logarithm = np.log((np.sqrt(radii) + root3) * (root6 - root3) / ((np.sqrt(radii) - root3) * (root6 + root3)))
    schwarzschild = (3 / (8 * np.pi * radii**3) / (1 - 3 / radii)
                     * (1 - np.sqrt(6 / radii) + np.sqrt(3 / (4 * radii)) * logarithm)) ** 0.25

    assert ((nearly_still - schwarzschild).abs() < 1e-6).all()
    edge = core.innermost_stable_orbit(orbit, orbit.dual() * 0.8, mass)
    assert (core.disk_temperature(edge, orbit, orbit.dual() * 0.8, mass, scale) < 1e-3).all()
    sample_radii = mv.scalar(np.linspace(6.0, 20.0, 32)[:, None])
    peak_low = core.disk_temperature(sample_radii, orbit, orbit.dual() * 0.5, mass, scale)
    peak_high = core.disk_temperature(sample_radii, orbit, orbit.dual() * 0.8, mass, scale)
    assert (peak_high > peak_low).any()


def test_innermost_stable_orbit():
    orbit = mv.xy
    with np.errstate(divide="ignore"):
        radii = core.innermost_stable_orbit(orbit, orbit.dual() * np.array([0.0, 0.8, 1.0]), 1.0)
    assert ((radii - np.array([6.0, 2.9066, 1.0])).abs() < 1e-4).all()
    assert ((radii[[0, 2]] - np.array([6.0, 1.0])).abs() < round_off).all()


def test_the_scenes_trace_resolve_and_draw():
    paths, horizons = scenarios.spin_sweep_scene(2)
    assert len(list(render.ray_animation(paths[:, :1], horizons[:1], scenarios.extent))) == 1
    plt.close(render.ray_comparison_figure((paths[:, 0], paths[:, 0]), horizons[0], scenarios.extent))
    n_width, n_height = 4, 3
    shape = (n_height, n_width)
    image, temperature, transmission = scenarios.camera_scene(n_width, n_height, scenarios.half_view, 8)
    sky = render.star_texture(64, 32, scenarios.sky_seed)
    plt.close(render.camera_figure(image, temperature, scenarios.exposure_temperature, shape, sky))
    plt.close(render.camera_comparison_figure((image, image), (temperature, temperature),
                                              scenarios.exposure_temperature, shape, sky))
    assert len(list(render.polarizer_animation(image, temperature, scenarios.exposure_temperature, transmission[:1],
                                               shape, sky))) == 1
    assert len(list(render.polarizer_comparison_animation(
        (image, image), (temperature, temperature), scenarios.exposure_temperature,
        (transmission[:1], transmission[:1]), shape, sky))) == 1
    assert len(list(render.disk_animation(image, temperature, scenarios.exposure_temperature,
                                          scenarios.disk_flow_states(image, 1), 1, shape, sky))) == 1
