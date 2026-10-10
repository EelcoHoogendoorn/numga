"""Constitutive optical layers satisfy Maxwell, interface scattering, and energy conservation."""

import numpy as np
import pytest

from numga import stack
from examples.optics.coating import core


ROUND_OFF = 1e-10


def test_bare_interface_matches_both_fresnel_polarizations_and_brewster_angle():
    incident_index = 1.0
    substrate_index = 1.5
    angles = np.array([0.0, 0.4, 0.9, np.arctan(substrate_index / incident_index)])
    tangent = incident_index * np.sin(angles)
    parallel = core.mv.t + core.mv.x * tangent
    polarizations = stack([core.mv.y, core.mv.x])[:, None]
    incident_normal = incident_index * np.cos(angles)
    substrate_normal = np.sqrt(substrate_index**2 - tangent**2)
    incident_admittance = np.stack([incident_normal, incident_index**2 / incident_normal])
    substrate_admittance = np.stack([substrate_normal, substrate_index**2 / substrate_normal])

    incident = core.Ports.isotropic(incident_index, parallel)
    substrate = core.Ports.isotropic(substrate_index, parallel)
    result = core.scatter(substrate.outgoing, incident)
    reflected = (incident_admittance - substrate_admittance) / (incident_admittance + substrate_admittance)
    transmitted = 2 * incident_admittance / (incident_admittance + substrate_admittance)
    reflectance, transmittance = result.powers(polarizations, incident, substrate)

    np.testing.assert_allclose((result.reflection(polarizations) - polarizations * reflected).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((result.transmission(polarizations) - polarizations * transmitted).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(reflectance.kernel[..., 0], reflected**2, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(transmittance.kernel[..., 0],
                               substrate_admittance / incident_admittance * transmitted**2,
                               atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(result.reflection(core.mv.x)[-1].kernel, 0, atol=ROUND_OFF, rtol=0)


def test_quarter_wave_matching_layer_cancels_reflection_with_the_correct_phase():
    incident_index = 1.0
    substrate_index = 1.52
    coating_index = np.sqrt(incident_index * substrate_index)
    optical_distance = np.array(np.pi / (2 * coating_index))
    parallel = core.mv.t
    polarization = (core.mv.x + 1j * core.mv.y) / np.sqrt(2)
    chi = core.dielectric(np.full(3, coating_index**2), np.array(1.0))

    medium = core.Medium.from_chi(chi, parallel)
    incident = core.Ports.isotropic(incident_index, parallel)
    substrate = core.Ports.isotropic(substrate_index, parallel)
    result = core.scatter(core.propagate(medium.generator, optical_distance)(substrate.outgoing), incident)
    transmitted = polarization * (-1j * np.sqrt(incident_index / substrate_index))
    reflectance, transmittance = result.powers(polarization, incident, substrate)

    np.testing.assert_allclose(result.reflection.kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((result.transmission(polarization) - transmitted).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(reflectance.kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(transmittance.kernel, 1, atol=ROUND_OFF, rtol=0)


@pytest.mark.parametrize("pairs", [1, 3, 6])
def test_quarter_wave_mirror_respects_layer_order_and_impedance_inversion(pairs):
    incident_index = 1.0
    substrate_index = 1.52
    high_index = 2.3
    low_index = 1.46
    # Two stacks: high-index first and low-index first.
    indices = np.tile([[high_index, low_index], [low_index, high_index]], (pairs, 1))
    optical_distance = np.pi / (2 * indices)
    ratio = np.array([high_index / low_index, low_index / high_index])
    effective = substrate_index * ratio ** (2 * pairs)
    parallel = core.mv.t
    chi = core.dielectric(indices[..., None] ** 2 * np.ones(3), np.array(1.0))

    medium = core.Medium.from_chi(chi, parallel)
    propagation = core.compose(core.propagate(medium.generator, optical_distance))
    incident = core.Ports.isotropic(incident_index, parallel)
    substrate = core.Ports.isotropic(substrate_index, parallel)
    result = core.scatter(propagation(substrate.outgoing), incident)
    reflected = (incident_index - effective) / (incident_index + effective)
    transmitted = (-1) ** pairs * 2 * incident_index / (incident_index / ratio**pairs + substrate_index * ratio**pairs)
    reflectance, transmittance = result.powers(core.mv.x, incident, substrate)

    np.testing.assert_allclose((result.reflection(core.mv.x) - core.mv.x * reflected).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((result.transmission(core.mv.x) - core.mv.x * transmitted).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((reflectance + transmittance).kernel, 1, atol=ROUND_OFF, rtol=0)


def test_oblique_multilayer_matches_recursive_multiple_reflections():
    incident_index = 1.0
    substrate_index = 1.52
    indices = np.array([1.38, 2.2, 1.65])
    thicknesses = np.array([137.0, 82.0, 110.0])
    wavelengths = np.array([450.0, 550.0, 710.0])
    angles = np.array([0.0, 0.4, 0.85])
    tangent = incident_index * np.sin(angles)
    parallel = core.mv.t + core.mv.x * tangent
    polarizations = stack([core.mv.y, core.mv.x])[:, None, None]
    normal = np.sqrt(indices[:, None] ** 2 - tangent**2)
    incident_normal = incident_index * np.cos(angles)
    substrate_normal = np.sqrt(substrate_index**2 - tangent**2)
    incident_admittance = np.stack([incident_normal, incident_index**2 / incident_normal], axis=-1)
    substrate_admittance = np.stack([substrate_normal, substrate_index**2 / substrate_normal], axis=-1)
    admittances = np.stack([normal, indices[:, None] ** 2 / normal], axis=-1)
    optical_distance = 2 * np.pi * thicknesses[:, None, None] / wavelengths[None, :, None]
    phases = normal[:, None, :, None] * optical_distance[..., None]
    chi = core.dielectric(indices[:, None, None] ** 2 * np.ones(3), np.array(1.0))

    medium = core.Medium.from_chi(chi, parallel)
    layers = core.propagate(medium.generator[:, None], optical_distance)  # [layers, wavelengths, angles]
    incident = core.Ports.isotropic(incident_index, parallel)
    substrate = core.Ports.isotropic(substrate_index, parallel)
    result = core.scatter(core.compose(layers)(substrate.outgoing), incident)

    # Independently sum repeated reflections within each film, starting at the substrate.
    reflected = (admittances[-1] - substrate_admittance) / (admittances[-1] + substrate_admittance)
    transmitted = 2 * admittances[-1] / (admittances[-1] + substrate_admittance)
    left_admittances = np.concatenate([incident_admittance[None], admittances[:-1]])
    for index in range(len(indices) - 1, -1, -1):
        left = left_admittances[index]
        right = admittances[index]
        interface_reflection = (left - right) / (left + right)
        interface_transmission = 2 * left / (left + right)
        travel = np.exp(-1j * phases[index])
        denominator = 1 + interface_reflection * reflected * travel**2
        transmitted = interface_transmission * transmitted * travel / denominator
        reflected = (interface_reflection + reflected * travel**2) / denominator
    reflected = np.moveaxis(reflected, -1, 0)
    transmitted = np.moveaxis(transmitted, -1, 0)
    reflectance, transmittance = result.powers(polarizations, incident, substrate)

    np.testing.assert_allclose((result.reflection(polarizations) - polarizations * reflected).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((result.transmission(polarizations) - polarizations * transmitted).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(reflectance.kernel[..., 0], abs(reflected)**2, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((reflectance + transmittance).kernel, 1, atol=ROUND_OFF, rtol=0)


def test_anisotropic_chi_reconstruction_satisfies_full_maxwell_and_rotates_covariantly():
    permittivity = np.array([2.2, 3.1, 4.3])
    reluctivity = np.array(0.8)
    axion = 0.37
    material_turn = (0.23 * core.mv.xy + 0.17 * core.mv.yz).exp()
    frame_turn = (0.31 * core.mv.xy).exp()
    parallel = core.mv.t + 0.4 * core.mv.x + 0.1 * core.mv.y
    base = core.dielectric(permittivity, reluctivity)
    chi = (material_turn >> base(material_turn << core.Bivector)) + axion * core.Bivector

    medium = core.Medium.from_chi(chi, parallel)
    indices, modes = medium.generator.eig()
    fields = medium.reconstruct(modes)
    wavevectors = parallel + core.mv.z * indices
    moved_chi = frame_turn >> chi(frame_turn << core.Bivector)
    moved = core.Medium.from_chi(moved_chi, frame_turn >> parallel)

    np.testing.assert_allclose((medium.boundary(medium.reconstruct) - core.Boundary).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((wavevectors ^ fields).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((wavevectors ^ chi(fields)).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((moved.generator - (frame_turn >> medium.generator(frame_turn << core.Boundary))).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((moved.reconstruct(frame_turn >> core.Boundary) - (frame_turn >> medium.reconstruct)).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_axion_jump_changes_interfaces_without_changing_bulk_wave_indices():
    index = 1.5
    axion_jump = np.array([0.0, 0.2, 0.7, 2.0])
    common_axion = 0.43
    parallel = core.mv.t
    base = core.dielectric(np.full(3, index**2), np.array(1.0))
    axion_map = core.mv.scalar(axion_jump[:, None]) * core.Bivector
    incident_medium = core.Medium.from_chi(base, parallel)
    substrate_medium = core.Medium.from_chi(base + axion_map, parallel)
    incident = core.Ports.from_medium(incident_medium, index, parallel)
    substrate = core.Ports.from_medium(substrate_medium, index, parallel)

    result = core.scatter(substrate.outgoing, incident)
    denominator = 4 * index**2 + axion_jump**2
    reflected = core.mv.x * (-axion_jump**2 / denominator) + core.mv.y * (2 * index * axion_jump / denominator)
    reflectance, transmittance = result.powers(core.mv.x, incident, substrate)
    shifted_base = base + common_axion * core.Bivector
    shifted_incident = core.Ports.from_medium(core.Medium.from_chi(shifted_base, parallel), index, parallel)
    shifted_substrate = core.Ports.from_medium(core.Medium.from_chi(shifted_base + axion_map, parallel), index, parallel)
    shifted = core.scatter(shifted_substrate.outgoing, shifted_incident)

    np.testing.assert_allclose((result.reflection(core.mv.x) - reflected).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((result.transmission(core.mv.x) - core.mv.x - reflected).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose(reflectance.kernel[..., 0], axion_jump**2 / denominator, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((reflectance + transmittance).kernel, 1, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((substrate_medium.generator(substrate_medium.generator) - index**2 * core.Boundary).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((shifted.reflection - result.reflection).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((shifted.transmission - result.transmission).kernel, 0, atol=ROUND_OFF, rtol=0)


def test_twisted_chi_selects_circular_polarization_and_remains_reciprocal():
    ordinary_index = 1.5
    extraordinary_index = 1.8
    ambient_index = (ordinary_index + extraordinary_index) / 2
    pitch = 400.0
    turns = 6
    slices_per_turn = 16
    thickness = pitch / slices_per_turn
    windings = np.array([-1.0, 1.0])
    angles = (np.arange(turns * slices_per_turn) + 0.5) * (2 * np.pi / slices_per_turn)
    wavelengths = np.array([500.0, 660.0, 850.0])
    parallel = core.mv.t
    circular = stack([(core.mv.x + 1j * core.mv.y) / np.sqrt(2),
                      (core.mv.x - 1j * core.mv.y) / np.sqrt(2)])[:, None, None]
    base = core.dielectric(np.array([extraordinary_index**2, ordinary_index**2, ordinary_index**2]), np.array(1.0))
    rotations = (core.mv.xy * (angles[:, None] * windings / 2)).exp()
    chi = rotations >> base(rotations << core.Bivector)

    medium = core.Medium.from_chi(chi, parallel)
    layers = core.propagate(medium.generator[..., None], 2 * np.pi * thickness / wavelengths)
    ports = core.Ports.isotropic(ambient_index, parallel)
    result = core.scatter(core.compose(layers)(ports.outgoing), ports)
    reverse = core.scatter(core.compose(layers[::-1])(ports.outgoing), ports)
    reflectance, transmittance = result.powers(circular, ports, ports)

    # Reversing the twist exchanges the selected circular polarization.
    np.testing.assert_allclose((reflectance[0, 0] - reflectance[1, 1]).kernel, 0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((reflectance[1, 0] - reflectance[0, 1]).kernel, 0, atol=ROUND_OFF, rtol=0)
    assert reflectance.kernel[1, 0, 1, 0] > 0.98
    assert reflectance.kernel[0, 0, 1, 0] < 0.01
    assert np.max(reflectance.kernel[..., [0, 2], 0]) < 0.2
    np.testing.assert_allclose((reflectance + transmittance).kernel, 1, atol=ROUND_OFF, rtol=0)
    # Reciprocity transposes the amplitude map; it does not conjugate the optical phase.
    np.testing.assert_allclose((reverse.transmission - result.transmission.adjoint()).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_interior_fields_match_every_interface_and_conserve_normal_power():
    incident_index = 1.2
    substrate_index = 1.6
    permittivities = np.array([[2.2, 3.1, 4.3], [3.3, 2.5, 3.8], [2.8, 3.7, 2.4]])
    reluctivities = np.array([0.8, 1.1, 0.9])
    axions = np.array([-0.2, 0.3, 0.1])
    rotations = (core.mv.xy * np.array([0.2, -0.3, 0.1])
                 + core.mv.yz * np.array([0.15, 0.1, -0.2])).exp()
    thicknesses = np.array([80.0, 140.0, 65.0])
    wavelength = 630.0
    samples = 9
    fractions = np.linspace(0, 1, samples)
    parallel = core.mv.t + 0.3 * core.mv.x
    polarizations = stack([(core.mv.x + 1j * core.mv.y) / np.sqrt(2),
                           (core.mv.x - 1j * core.mv.y) / np.sqrt(2)])
    normal = core.mv.z.real()
    spatial_volume = core.mv.xyz.real()
    base = core.dielectric(permittivities, reluctivities)
    chi = (rotations >> base(rotations << core.Bivector)) + core.mv.scalar(axions[:, None]) * core.Bivector
    incident_chi = core.dielectric(np.full(3, incident_index**2), np.array(1.0))
    substrate_chi = core.dielectric(np.full(3, substrate_index**2), np.array(1.0))

    medium = core.Medium.from_chi(chi, parallel)
    incident_medium = core.Medium.from_chi(incident_chi, parallel)
    substrate_medium = core.Medium.from_chi(substrate_chi, parallel)
    incident = core.Ports.from_medium(incident_medium, incident_index, parallel)
    substrate = core.Ports.from_medium(substrate_medium, substrate_index, parallel)
    propagation = core.compose(core.propagate(medium.generator, 2 * np.pi * thicknesses / wavelength))
    result = core.scatter(propagation(substrate.outgoing), incident)
    transmitted = result.transmission(polarizations)
    exit_state = substrate.outgoing(transmitted)
    right_field = substrate_medium.reconstruct(exit_state)
    right_excitation = substrate_chi(right_field)
    transmitted_power = substrate.power(transmitted)

    fields = medium.interior(thicknesses, wavelength, exit_state, fractions)
    for index in range(len(thicknesses) - 1, -1, -1):
        field = fields[:, index]
        excitation = chi[index](field)
        # Maxwell's jump conditions compare physical fields across different materials.
        np.testing.assert_allclose((core.mv.z ^ (field[:, -1] - right_field)).kernel,
                                   0, atol=ROUND_OFF, rtol=0)
        np.testing.assert_allclose((core.mv.z ^ (excitation[:, -1] - right_excitation)).kernel,
                                   0, atol=ROUND_OFF, rtol=0)
        electric = core.mv.t | field
        magnetic = core.mv.t | excitation
        electric_real, electric_imaginary = electric.real(), (-1j * electric).real()
        magnetic_real, magnetic_imaginary = magnetic.real(), (-1j * magnetic).real()
        flux = ((electric_real ^ magnetic_real ^ normal).scalar_product(spatial_volume)
                + (electric_imaginary ^ magnetic_imaginary ^ normal).scalar_product(spatial_volume))
        np.testing.assert_allclose((flux - transmitted_power[:, None]).kernel,
                                   0, atol=ROUND_OFF, rtol=0)
        right_field, right_excitation = field[:, 0], excitation[:, 0]

    # At the entrance the sampled field meets the incident and reflected exterior waves.
    forward_wavevector = parallel + core.mv.z * incident.normal_index
    backward_wavevector = parallel - core.mv.z * incident.normal_index
    backward_electric = core.Polarization - core.mv.z * (parallel | core.Polarization) / incident.normal_index
    exterior_field = ((forward_wavevector ^ incident.electric_field(polarizations))
                      + (backward_wavevector ^ backward_electric(result.reflection(polarizations))))
    np.testing.assert_allclose((core.mv.z ^ (right_field - exterior_field)).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((core.mv.z ^ (right_excitation - incident_chi(exterior_field))).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_selected_circular_wave_decays_through_the_twisted_reflector():
    ordinary_index = 1.5
    extraordinary_index = 1.8
    ambient_index = (ordinary_index + extraordinary_index) / 2
    pitch = 400.0
    turns = 6
    slices_per_turn = 16
    layers = turns * slices_per_turn
    thicknesses = np.full(layers, pitch / slices_per_turn)
    angles = (np.arange(layers) + 0.5) * (2 * np.pi / slices_per_turn)
    wavelength = 660.0
    samples = 9
    fractions = np.linspace(0, 1, samples)
    parallel = core.mv.t
    circular = stack([(core.mv.x + 1j * core.mv.y) / np.sqrt(2),
                      (core.mv.x - 1j * core.mv.y) / np.sqrt(2)])
    base = core.dielectric(np.array([extraordinary_index**2, ordinary_index**2, ordinary_index**2]), np.array(1.0))
    rotations = (core.mv.xy * (angles / 2)).exp()
    chi = rotations >> base(rotations << core.Bivector)

    medium = core.Medium.from_chi(chi, parallel)
    ports = core.Ports.isotropic(ambient_index, parallel)
    propagation = core.compose(core.propagate(medium.generator, 2 * np.pi * thicknesses / wavelength))
    result = core.scatter(propagation(ports.outgoing), ports)
    exit_state = ports.outgoing(result.transmission(circular))
    electric = core.mv.t | medium.interior(thicknesses, wavelength, exit_state, fractions)
    intensity = -(electric.real().scalar_norm_squared()
                  + (-1j * electric).real().scalar_norm_squared())
    first_pitch = intensity[:, :slices_per_turn].mean(axis=1).mean(axis=1)
    last_pitch = intensity[:, -slices_per_turn:].mean(axis=1).mean(axis=1)
    penetration = last_pitch / first_pitch

    assert penetration.kernel[0, 0] < 0.05
    assert 0.9 < penetration.kernel[1, 0] < 1.1


def test_stepped_pulse_keeps_its_energy_and_returns_the_hand_turning_with_the_helix():
    from examples.optics.coating import scenarios
    _, _, _, _, before, reflected, transmitted = scenarios.time_domain()
    # The leapfrog's energy, measured once the pulses have left the helix: a method-accuracy check.
    np.testing.assert_allclose(((reflected + transmitted) / before).kernel, 1, atol=1e-3)
    # Turning with the helix comes back, turning against it passes, as the reflectance band says.
    fractions = (reflected / before).kernel[:, 0]
    assert fractions[0] > 0.98 and fractions[1] < 0.01
