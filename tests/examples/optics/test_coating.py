"""Constitutive optical layers satisfy Maxwell's equations, Fresnel and multilayer scattering, reciprocity, and energy conservation."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numga import stack
from examples.optics.coating import core, render, scenarios

round_off = 1e-10


def test_bare_interface_matches_both_fresnel_polarizations_and_brewster_angle() -> None:
    """Scattering across a bare dielectric interface matches s- and p-polarization Fresnel amplitudes and vanishes at Brewster's angle."""
    incident_index = 1.0
    substrate_index = 1.5
    angles = np.array([0.0, 0.4, 0.9, np.arctan(substrate_index / incident_index)])
    tangent = incident_index * np.sin(angles)
    parallel = core.unit_frequency + core.mv.x * tangent
    polarizations = stack([core.mv.yt, core.mv.xt])[:, None]
    incident_normal = incident_index * np.cos(angles)
    substrate_normal = np.sqrt(substrate_index ** 2 - tangent ** 2)
    incident_admittance = np.stack([incident_normal, incident_index ** 2 / incident_normal])
    substrate_admittance = np.stack([substrate_normal, substrate_index ** 2 / substrate_normal])

    incident_response = core.dielectric_response(np.full(3, incident_index ** 2), 1.0)
    substrate_response = core.dielectric_response(np.full(3, substrate_index ** 2), 1.0)
    incident_wave = core.exterior_wave(incident_index, parallel)
    substrate_wave = core.exterior_wave(substrate_index, parallel)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(substrate_wave)
    reflection, transmission = core.interface_scattering(substrate_outgoing, incident_response, incident_index, parallel)
    reflected = (incident_admittance - substrate_admittance) / (incident_admittance + substrate_admittance)
    transmitted = 2 * incident_admittance / (incident_admittance + substrate_admittance)
    reflectance, transmittance = core.scattering_powers(
        reflection, transmission, polarizations, incident_wave, incident_response, substrate_wave, substrate_response
    )

    np.testing.assert_allclose((reflection(polarizations) - polarizations * reflected).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((transmission(polarizations) - polarizations * transmitted).kernel, 0, atol=round_off, rtol=0)
    assert ((reflectance - reflected ** 2).abs() < round_off).all()
    assert ((transmittance - substrate_admittance / incident_admittance * transmitted ** 2).abs() < round_off).all()
    np.testing.assert_allclose(reflection(core.mv.xt)[-1].kernel, 0, atol=round_off, rtol=0)


def test_quarter_wave_matching_layer_cancels_reflection_with_the_correct_phase() -> None:
    """An ideal quarter-wave matching film cancels normal-incidence reflection and transmits unit power with a quarter-cycle phase shift."""
    incident_index = 1.0
    substrate_index = 1.52
    coating_index = np.sqrt(incident_index * substrate_index)
    optical_distance = np.array(np.pi / (2 * coating_index))
    parallel = core.unit_frequency
    polarization = (core.mv.xt + 1j * core.mv.yt) / np.sqrt(2)
    film_response = core.dielectric_response(np.full(3, coating_index ** 2), np.array(1.0))

    _, generator = core.layer_operators(film_response, parallel)
    incident_response = core.dielectric_response(np.full(3, incident_index ** 2), 1.0)
    substrate_response = core.dielectric_response(np.full(3, substrate_index ** 2), 1.0)
    incident_wave = core.exterior_wave(incident_index, parallel)
    substrate_wave = core.exterior_wave(substrate_index, parallel)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(substrate_wave)
    reflection, transmission = core.interface_scattering(
        core.layer_propagation(generator, optical_distance)(substrate_outgoing),
        incident_response,
        incident_index,
        parallel,
    )
    transmitted = polarization * (-1j * np.sqrt(incident_index / substrate_index))
    reflectance, transmittance = core.scattering_powers(
        reflection, transmission, polarization, incident_wave, incident_response, substrate_wave, substrate_response
    )

    np.testing.assert_allclose(reflection.kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((transmission(polarization) - transmitted).kernel, 0, atol=round_off, rtol=0)
    assert (reflectance.abs() < round_off).all()
    assert ((transmittance - 1).abs() < round_off).all()


@pytest.mark.parametrize("n_pairs", [1, 3])
def test_quarter_wave_mirror_respects_layer_order_and_impedance_inversion(n_pairs: int) -> None:
    """Alternating quarter-wave dielectric stacks invert the effective substrate impedance according to layer order."""
    incident_index = 1.0
    substrate_index = 1.52
    high_index = 2.3
    low_index = 1.46
    indices = np.tile([[high_index, low_index], [low_index, high_index]], (n_pairs, 1))
    optical_distance = np.pi / (2 * indices)
    ratio = np.array([high_index / low_index, low_index / high_index])
    effective = substrate_index * ratio ** (2 * n_pairs)
    parallel = core.unit_frequency
    response = core.dielectric_response(indices[..., None] ** 2 * np.ones(3), np.array(1.0))

    _, generator = core.layer_operators(response, parallel)
    propagation = core.stack_propagation(core.layer_propagation(generator, optical_distance))
    incident_response = core.dielectric_response(np.full(3, incident_index ** 2), 1.0)
    substrate_response = core.dielectric_response(np.full(3, substrate_index ** 2), 1.0)
    incident_wave = core.exterior_wave(incident_index, parallel)
    substrate_wave = core.exterior_wave(substrate_index, parallel)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(substrate_wave)
    reflection, transmission = core.interface_scattering(
        propagation(substrate_outgoing), incident_response, incident_index, parallel
    )
    reflected = (incident_index - effective) / (incident_index + effective)
    transmitted = (-1) ** n_pairs * 2 * incident_index / (incident_index / ratio ** n_pairs + substrate_index * ratio ** n_pairs)
    reflectance, transmittance = core.scattering_powers(
        reflection, transmission, core.mv.xt, incident_wave, incident_response, substrate_wave, substrate_response
    )

    np.testing.assert_allclose((reflection(core.mv.xt) - core.mv.xt * reflected).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((transmission(core.mv.xt) - core.mv.xt * transmitted).kernel, 0, atol=round_off, rtol=0)
    assert ((reflectance + transmittance - 1).abs() < round_off).all()


def test_oblique_multilayer_matches_recursive_multiple_reflections() -> None:
    """Oblique propagation through a three-layer isotropic stack matches recursive Airy multiple-reflection amplitudes."""
    incident_index = 1.0
    substrate_index = 1.52
    indices = np.array([1.38, 2.2, 1.65])
    thicknesses = np.array([137.0, 82.0, 110.0])
    wavelengths = np.array([450.0, 550.0, 710.0])
    angles = np.array([0.0, 0.4, 0.85])
    tangent = incident_index * np.sin(angles)
    parallel = core.unit_frequency + core.mv.x * tangent
    polarizations = stack([core.mv.yt, core.mv.xt])[:, None, None]
    normal = np.sqrt(indices[:, None] ** 2 - tangent ** 2)
    incident_normal = incident_index * np.cos(angles)
    substrate_normal = np.sqrt(substrate_index ** 2 - tangent ** 2)
    incident_admittance = np.stack([incident_normal, incident_index ** 2 / incident_normal], axis=-1)
    substrate_admittance = np.stack([substrate_normal, substrate_index ** 2 / substrate_normal], axis=-1)
    admittances = np.stack([normal, indices[:, None] ** 2 / normal], axis=-1)
    optical_distance = 2 * np.pi * thicknesses[:, None, None] / wavelengths[None, :, None]
    phases = normal[:, None, :, None] * optical_distance[..., None]
    response = core.dielectric_response(indices[:, None, None] ** 2 * np.ones(3), np.array(1.0))

    _, generator = core.layer_operators(response, parallel)
    layers = core.layer_propagation(generator[:, None], optical_distance)
    incident_response = core.dielectric_response(np.full(3, incident_index ** 2), 1.0)
    substrate_response = core.dielectric_response(np.full(3, substrate_index ** 2), 1.0)
    incident_wave = core.exterior_wave(incident_index, parallel)
    substrate_wave = core.exterior_wave(substrate_index, parallel)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(substrate_wave)
    reflection, transmission = core.interface_scattering(
        core.stack_propagation(layers)(substrate_outgoing), incident_response, incident_index, parallel
    )

    reflected = (admittances[-1] - substrate_admittance) / (admittances[-1] + substrate_admittance)
    transmitted = 2 * admittances[-1] / (admittances[-1] + substrate_admittance)
    left_admittances = np.concatenate([incident_admittance[None], admittances[:-1]])
    for index in range(len(indices) - 1, -1, -1):
        left = left_admittances[index]
        right = admittances[index]
        interface_reflection = (left - right) / (left + right)
        interface_transmission = 2 * left / (left + right)
        travel = np.exp(-1j * phases[index])
        denominator = 1 + interface_reflection * reflected * travel ** 2
        transmitted = interface_transmission * transmitted * travel / denominator
        reflected = (interface_reflection + reflected * travel ** 2) / denominator
    reflected = np.moveaxis(reflected, -1, 0)
    transmitted = np.moveaxis(transmitted, -1, 0)
    reflectance, transmittance = core.scattering_powers(
        reflection, transmission, polarizations, incident_wave, incident_response, substrate_wave, substrate_response
    )

    np.testing.assert_allclose((reflection(polarizations) - polarizations * reflected).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((transmission(polarizations) - polarizations * transmitted).kernel, 0, atol=round_off, rtol=0)
    assert ((reflectance - np.abs(reflected) ** 2).abs() < round_off).all()
    assert ((reflectance + transmittance - 1).abs() < round_off).all()


def test_anisotropic_chi_reconstruction_satisfies_full_maxwell_and_rotates_covariantly() -> None:
    """Reconstructed eigenmodes satisfy both exterior Maxwell equations and transform covariantly under spatial rotations."""
    permittivity = np.array([2.2, 3.1, 4.3])
    reluctivity = np.array(0.8)
    axion = 0.37
    material_turn = (0.23 * core.mv.xy + 0.17 * core.mv.yz).exp()
    frame_turn = (0.31 * core.mv.xy).exp()
    parallel = core.unit_frequency + 0.4 * core.mv.x + 0.1 * core.mv.y
    base = core.dielectric_response(permittivity, reluctivity)
    response = (material_turn >> base(material_turn << core.Field)) + core.mv.antipseudoscalar([axion]) * core.Field

    boundary = core.field_boundary + core.excitation_boundary(response)
    reconstruction, generator = core.layer_operators(response, parallel)
    indices, modes = generator.eig()
    fields = reconstruction(modes)
    wavevectors = parallel + core.mv.z * indices
    moved_response = frame_turn >> response(frame_turn << core.Field)
    moved_reconstruction, moved_generator = core.layer_operators(moved_response, frame_turn >> parallel)

    np.testing.assert_allclose((boundary(reconstruction) - core.Boundary).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((wavevectors ^ fields).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((wavevectors ^ response(fields)).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((moved_generator - (frame_turn >> generator(frame_turn << core.Boundary))).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((moved_reconstruction(frame_turn >> core.Boundary) - (frame_turn >> reconstruction)).kernel, 0, atol=round_off, rtol=0)


def test_axion_jump_changes_interfaces_without_changing_bulk_wave_indices() -> None:
    """An axion coupling jump rotates interface polarization while leaving bulk propagation speeds and common axion shifts invariant."""
    index = 1.5
    axion_jump = np.array([0.0, 0.2, 0.7, 2.0])
    common_axion = 0.43
    parallel = core.unit_frequency
    base = core.dielectric_response(np.full(3, index ** 2), np.array(1.0))
    axion_map = core.mv.antipseudoscalar(axion_jump[:, None]) * core.Field
    substrate_response = base + axion_map
    wave = core.exterior_wave(index, parallel)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(wave)
    _, substrate_generator = core.layer_operators(substrate_response, parallel)

    reflection, transmission = core.interface_scattering(substrate_outgoing, base, index, parallel)
    denominator = 4 * index ** 2 + axion_jump ** 2
    reflected = core.mv.xt * (-axion_jump ** 2 / denominator) + core.mv.yt * (2 * index * axion_jump / denominator)
    reflectance, transmittance = core.scattering_powers(
        reflection, transmission, core.mv.xt, wave, base, wave, substrate_response
    )

    shifted_base = base + core.mv.antipseudoscalar([common_axion]) * core.Field
    shifted_substrate = shifted_base + axion_map
    shifted_outgoing = (core.field_boundary + core.excitation_boundary(shifted_substrate))(wave)
    shifted_reflection, shifted_transmission = core.interface_scattering(shifted_outgoing, shifted_base, index, parallel)

    np.testing.assert_allclose((reflection(core.mv.xt) - reflected).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((transmission(core.mv.xt) - core.mv.xt - reflected).kernel, 0, atol=round_off, rtol=0)
    assert ((reflectance - axion_jump ** 2 / denominator).abs() < round_off).all()
    assert ((reflectance + transmittance - 1).abs() < round_off).all()
    np.testing.assert_allclose((substrate_generator(substrate_generator) - index ** 2 * core.Boundary).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((shifted_reflection - reflection).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((shifted_transmission - transmission).kernel, 0, atol=round_off, rtol=0)


def test_twisted_chi_selects_circular_polarization_and_remains_reciprocal() -> None:
    """Opposite cholesteric twists exchange circular Bragg stopbands while preserving Lorentz reciprocity."""
    ordinary_index = 1.5
    extraordinary_index = 1.8
    ambient_index = (ordinary_index + extraordinary_index) / 2
    pitch = 400.0
    n_turns = 6
    n_slices_per_turn = 16
    thickness = pitch / n_slices_per_turn
    windings = np.array([-1.0, 1.0])
    angles = (np.arange(n_turns * n_slices_per_turn) + 0.5) * (2 * np.pi / n_slices_per_turn)
    wavelengths = np.array([500.0, 660.0, 850.0])
    parallel = core.unit_frequency
    circular = stack([(core.mv.xt + 1j * core.mv.yt) / np.sqrt(2),
                      (core.mv.xt - 1j * core.mv.yt) / np.sqrt(2)])[:, None, None]
    base = core.dielectric_response(np.array([extraordinary_index ** 2, ordinary_index ** 2, ordinary_index ** 2]), np.array(1.0))
    rotations = (core.mv.xy * (-angles[:, None] * windings / 2)).exp()
    response = rotations >> base(rotations << core.Field)

    _, generator = core.layer_operators(response, parallel)
    layers = core.layer_propagation(generator[..., None], 2 * np.pi * thickness / wavelengths)
    ambient_response = core.dielectric_response(np.full(3, ambient_index ** 2), 1.0)
    ambient_wave = core.exterior_wave(ambient_index, parallel)
    ambient_outgoing = (core.field_boundary + core.excitation_boundary(ambient_response))(ambient_wave)
    reflection, transmission = core.interface_scattering(
        core.stack_propagation(layers)(ambient_outgoing), ambient_response, ambient_index, parallel
    )
    _, reverse_transmission = core.interface_scattering(
        core.stack_propagation(layers[::-1])(ambient_outgoing), ambient_response, ambient_index, parallel
    )
    reflectance, transmittance = core.scattering_powers(
        reflection, transmission, circular, ambient_wave, ambient_response, ambient_wave, ambient_response
    )

    assert ((reflectance[0, 0] - reflectance[1, 1]).abs() < round_off).all()
    assert ((reflectance[1, 0] - reflectance[0, 1]).abs() < round_off).all()
    assert (reflectance[1, 0, 1].real() > 0.98).all()
    assert (reflectance[0, 0, 1].real() < 0.01).all()
    assert (reflectance[:, :, [0, 2]].real() < 0.2).all()
    assert ((reflectance + transmittance - 1).abs() < round_off).all()
    np.testing.assert_allclose((reverse_transmission - transmission.adjoint()).kernel, 0, atol=round_off, rtol=0)


def test_interior_fields_match_every_interface_and_conserve_normal_power() -> None:
    """Reconstructed interior fields satisfy tangential Maxwell continuity across every face and conserve normal energy flux."""
    incident_index = 1.2
    substrate_index = 1.6
    permittivities = np.array([[2.2, 3.1, 4.3], [3.3, 2.5, 3.8], [2.8, 3.7, 2.4]])
    reluctivities = np.array([0.8, 1.1, 0.9])
    axions = np.array([-0.2, 0.3, 0.1])
    rotations = (core.mv.xy * np.array([0.2, -0.3, 0.1])
                 + core.mv.yz * np.array([0.15, 0.1, -0.2])).exp()
    thicknesses = np.array([80.0, 140.0, 65.0])
    wavelength = 630.0
    n_samples = 5
    fractions = np.linspace(0, 1, n_samples)
    parallel = core.unit_frequency + 0.3 * core.mv.x
    polarizations = stack([(core.mv.xt + 1j * core.mv.yt) / np.sqrt(2),
                           (core.mv.xt - 1j * core.mv.yt) / np.sqrt(2)])
    base = core.dielectric_response(permittivities, reluctivities)
    response = (rotations >> base(rotations << core.Field)) + core.mv.antipseudoscalar(axions[:, None]) * core.Field
    incident_response = core.dielectric_response(np.full(3, incident_index ** 2), np.array(1.0))
    substrate_response = core.dielectric_response(np.full(3, substrate_index ** 2), np.array(1.0))

    reconstruction, generator = core.layer_operators(response, parallel)
    substrate_reconstruction, _ = core.layer_operators(substrate_response, parallel)
    incident_wave = core.exterior_wave(incident_index, parallel)
    substrate_wave = core.exterior_wave(substrate_index, parallel)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(substrate_wave)
    propagation = core.stack_propagation(core.layer_propagation(generator, 2 * np.pi * thicknesses / wavelength))
    reflection, transmission = core.interface_scattering(propagation(substrate_outgoing), incident_response, incident_index, parallel)
    transmitted = transmission(polarizations)
    exit_state = substrate_outgoing(transmitted)
    right_field = substrate_reconstruction(exit_state)
    right_excitation = substrate_response(right_field)
    transmitted_power = core.wave_power(substrate_wave(transmitted), substrate_response)

    fields = core.interior_fields(reconstruction, generator, thicknesses, wavelength, exit_state, fractions)
    for index in range(len(thicknesses) - 1, -1, -1):
        field = fields[:, index]
        excitation = response[index](field)
        np.testing.assert_allclose((core.mv.z ^ (field[:, -1] - right_field)).kernel, 0, atol=round_off, rtol=0)
        np.testing.assert_allclose((core.mv.z ^ (excitation[:, -1] - right_excitation)).kernel, 0, atol=round_off, rtol=0)
        flux = core.wave_power(field, response[index])
        assert ((flux - transmitted_power[:, None]).abs() < round_off).all()
        right_field, right_excitation = field[:, 0], excitation[:, 0]

    normal_index = (incident_index ** 2 - (parallel - core.unit_frequency).squared()).square_root()
    backward_wavevector = parallel - core.mv.z * normal_index
    tangential = core.unit_frequency.inverse() | core.Polarization
    backward_potential = tangential + core.mv.z * (parallel | tangential) / normal_index
    exterior_field = incident_wave(polarizations) + (backward_wavevector ^ backward_potential(reflection(polarizations)))
    np.testing.assert_allclose((core.mv.z ^ (right_field - exterior_field)).kernel, 0, atol=round_off, rtol=0)
    np.testing.assert_allclose((core.mv.z ^ (right_excitation - incident_response(exterior_field))).kernel, 0, atol=round_off, rtol=0)


def test_render_figures_and_one_pulse_frame() -> None:
    """Chart renderers return clean figures and the ray-traced pulse renderer streams a single frame."""
    wavelengths = np.array([500.0, 600.0])
    angles = np.array([0.0, 0.3])
    jumps = np.array([-1.0, 1.0])
    depths = np.array([[0.0, 10.0], [10.0, 20.0]])
    spectra = core.mv.scalar(np.zeros((3, 2, 1)))
    handed = core.mv.scalar(np.zeros((2, 2, 2, 1)))
    angular = core.mv.scalar(np.zeros((2, 2, 2, 1)))
    fields = core.mv.xt * np.ones((2, 2, 2))
    for figure in (
        render.spectra_figure(wavelengths, spectra),
        render.handedness_figure(wavelengths, handed),
        render.axion_figure(jumps, spectra),
        render.angular_figure(wavelengths, angles, angular),
        render.internal_field_figure(depths, fields),
    ):
        plt.close(figure)

    _, _, nodes, slab_nodes, optic_axes, _, _, _, state_stream = scenarios.pulse_setup(1, 1)
    frame = next(render.pulse_animation(nodes, slab_nodes, optic_axes, 2, state_stream))
    assert frame.ndim == 3 and frame.shape[2] == 3
