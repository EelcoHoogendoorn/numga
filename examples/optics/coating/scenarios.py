"""Scenario parameters and verification runs compute quarter-wave antireflection spectra, cholesteric Bragg reflection, internal wave patterns, time-domain pulse splitting, and axion-jump cross-polarization."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import stack
from numga.topology import Topology
from examples.optics.coating import core

# Wavelengths and thicknesses in nanometres.
design_wavelength = 550.0
n_wavelengths = 401
wavelengths = np.linspace(450, 850, n_wavelengths)
incident_index = 1.0
substrate_index = 1.5
coating_indices = np.array([substrate_index, np.sqrt(incident_index * substrate_index), 1.38])
optical_thicknesses = np.array([0.0, design_wavelength / 4, design_wavelength / 4])
coating_thicknesses = optical_thicknesses / coating_indices
ordinary_index = 1.5
extraordinary_index = 1.8
surrounding_index = (ordinary_index + extraordinary_index) / 2
# Pitch of one full turn of the crystal optic axis.
pitch = 400.0
n_turns = 6
n_slices_per_turn = 16
n_layers = n_turns * n_slices_per_turn
slice_thickness = pitch / n_slices_per_turn
slice_depths = (np.arange(n_layers) + 0.5) * slice_thickness
crystal_permittivity = np.array([extraordinary_index ** 2, ordinary_index ** 2, ordinary_index ** 2])
twist_signs = np.array([1.0, -1.0])
polarization_signs = np.array([1.0, -1.0])
band_wavelength = pitch * surrounding_index

# Time-domain pulse parameters on a 1D periodic loop of nodes.
node_spacing = 40.0
n_nodes = 1000
courant = 1.0
slab_start = 18000.0
pulse_start = 10000.0
pulse_width = 1000.0
light_travel = 27000.0
n_steps = round(light_travel / (courant * node_spacing))
n_frames = 168
hand_signs = np.array([1.0, -1.0])
n_samples_per_layer = 5
frame_duration_ms = 40
n_axion_jumps = 241
axion_jumps = np.linspace(-3, 3, n_axion_jumps)
n_angles = 41
incidence_angles = np.linspace(0, np.deg2rad(70), n_angles)
n_map_wavelengths = 121
map_wavelengths = np.linspace(450, 850, n_map_wavelengths)

# Circular inputs turning from xt toward yt along +z and in the opposite sense.
circular_polarizations = (core.mv.xt + core.mv.yt * (1j * polarization_signs)) / np.sqrt(2)  # [2] Polarization+


# --- math -----------------------------------------------------------------------------
def helix_media(depths: np.ndarray) -> tuple[core.Direction, core.Constitutive]:
    """Transporting a uniaxial dielectric across `depths` (`[n_depths]`) by opposite spatial rotors yields the crystal optic axes (`[n_depths, 2] Direction-`) and constitutive maps (`[n_depths, 2] Excitation- <- Field+`)."""
    angles = 2 * np.pi * depths[:, None] / pitch * twist_signs
    rotors = (core.mv.xy * (-angles / 2)).exp()                                   # [..., 2] Rotor+
    crystal = core.dielectric_response(crystal_permittivity, 1.0)                 # [] Excitation- <- Field+
    return rotors >> core.mv.x.dual(), rotors >> crystal(rotors << core.Field)


def antireflection_spectra() -> tuple[core.Scalar, core.Scalar]:
    """Normal-incidence scattering across bare glass, an ideal quarter-wave matching film, and magnesium fluoride returns reflectance and transmittance (`[3, n_wavelengths] Scalar+`)."""
    incident_response = core.dielectric_response(np.full(3, incident_index ** 2), 1.0)
    substrate_response = core.dielectric_response(np.full(3, substrate_index ** 2), 1.0)
    incident_wave = core.exterior_wave(incident_index, core.unit_frequency)
    substrate_wave = core.exterior_wave(substrate_index, core.unit_frequency)
    substrate_outgoing = (core.field_boundary + core.excitation_boundary(substrate_response))(substrate_wave)
    film_response = core.dielectric_response(coating_indices[:, None] ** 2 * np.ones(3), 1.0)
    _, generator = core.layer_operators(film_response, core.unit_frequency)
    distance = 2 * np.pi * coating_thicknesses[:, None] / wavelengths
    coating = core.layer_propagation(generator[:, None], distance)                # [3, n_wavelengths] Boundary+ <- Boundary+
    reflection, transmission = core.interface_scattering(
        coating(substrate_outgoing), incident_response, incident_index, core.unit_frequency
    )
    return core.scattering_powers(
        reflection, transmission, core.mv.xt, incident_wave, incident_response, substrate_wave, substrate_response
    )


def twisted_spectra() -> tuple[core.Scalar, core.Scalar]:
    """Normal-incidence Bragg reflection across opposite cholesteric helices returns circular reflectance and transmittance (`[2, 2, n_wavelengths] Scalar+`)."""
    _, response = helix_media(slice_depths)
    _, generator = core.layer_operators(response, core.unit_frequency)
    distance = 2 * np.pi * slice_thickness / wavelengths
    layers = core.layer_propagation(generator[..., None], distance)               # [n_layers, 2, n_wavelengths] Boundary+ <- Boundary+
    propagation = core.stack_propagation(layers)                                  # [2, n_wavelengths] Boundary+ <- Boundary+
    surrounding_response = core.dielectric_response(np.full(3, surrounding_index ** 2), 1.0)
    surrounding_wave = core.exterior_wave(surrounding_index, core.unit_frequency)
    surrounding_outgoing = (core.field_boundary + core.excitation_boundary(surrounding_response))(surrounding_wave)
    reflection, transmission = core.interface_scattering(
        propagation(surrounding_outgoing), surrounding_response, surrounding_index, core.unit_frequency
    )
    return core.scattering_powers(
        reflection, transmission, circular_polarizations[:, None, None],
        surrounding_wave, surrounding_response, surrounding_wave, surrounding_response,
    )


def axion_reflectance() -> core.Scalar:
    """Scattering x-polarized light off an axion-coupling jump between matched dielectrics returns co-polarized, cross-polarized, and total reflectance (`[3, n_axion_jumps] Scalar+`)."""
    surrounding_response = core.dielectric_response(np.full(3, surrounding_index ** 2), 1.0)
    surrounding_wave = core.exterior_wave(surrounding_index, core.unit_frequency)
    axion_response = surrounding_response + core.mv.antipseudoscalar(axion_jumps[:, None]) * core.Field
    axion_outgoing = (core.field_boundary + core.excitation_boundary(axion_response))(surrounding_wave)
    reflection, _ = core.interface_scattering(axion_outgoing, surrounding_response, surrounding_index, core.unit_frequency)
    reflected_field = reflection(core.mv.xt)                                      # [n_axion_jumps] Polarization+
    along_incident = core.mv.x.dual() & core.time                                 # Excitation-
    across_incident = core.mv.y.dual() & core.time                                # Excitation-
    parallel_part = core.mv.xt * (reflected_field & along_incident)               # [n_axion_jumps] Polarization+
    perpendicular_part = core.mv.yt * (reflected_field & across_incident)         # [n_axion_jumps] Polarization+
    incident_power = core.wave_power(surrounding_wave(core.mv.xt), surrounding_response)
    return stack([
        core.wave_power(surrounding_wave(parallel_part), surrounding_response),
        core.wave_power(surrounding_wave(perpendicular_part), surrounding_response),
        core.wave_power(surrounding_wave(reflected_field), surrounding_response),
    ]) / incident_power


def angular_reflectance() -> core.Scalar:
    """Oblique illumination of the positive cholesteric helix across `incidence_angles` and `map_wavelengths` returns circular reflectance (`[2, n_angles, n_map_wavelengths] Scalar+`)."""
    _, response = helix_media(slice_depths)
    parallel = core.unit_frequency + core.mv.x * (surrounding_index * np.sin(incidence_angles))
    _, generator = core.layer_operators(response[:, 0, None], parallel)
    distance = 2 * np.pi * slice_thickness / map_wavelengths
    layers = (core.layer_propagation(layer_generator[:, None], distance) for layer_generator in generator)
    propagation = core.stack_propagation(layers)                                  # [n_angles, n_map_wavelengths] Boundary+ <- Boundary+
    surrounding_response = core.dielectric_response(np.full(3, surrounding_index ** 2), 1.0)
    surrounding_wave = core.exterior_wave(surrounding_index, parallel[:, None])
    surrounding_outgoing = (core.field_boundary + core.excitation_boundary(surrounding_response))(surrounding_wave)
    reflection, transmission = core.interface_scattering(
        propagation(surrounding_outgoing), surrounding_response, surrounding_index, parallel[:, None]
    )
    tilt = (core.mv.zx * (incidence_angles / 2)).exp()                            # [n_angles] Rotor+
    beam_fields = tilt[None, :] >> circular_polarizations[:, None]                # [2, n_angles] Field+
    tangential = core.mv.z.inverse() | (core.mv.z ^ beam_fields)                  # [2, n_angles] Polarization+
    return core.scattering_powers(
        reflection, transmission, tangential[..., None],
        surrounding_wave, surrounding_response, surrounding_wave, surrounding_response,
    )[0]


def internal_fields() -> tuple[np.ndarray, core.Field]:
    """Reconstructing the wave inside the positive helix at `band_wavelength` returns sample depths (`[n_layers, n_samples_per_layer]`) and field bivectors (`[2, n_layers, n_samples_per_layer] Field+`)."""
    fractions = np.linspace(0, 1, n_samples_per_layer)
    depths = (np.arange(n_layers)[:, None] + fractions) * slice_thickness
    _, response = helix_media(slice_depths)
    reconstruction, generator = core.layer_operators(response[:, 0], core.unit_frequency)
    surrounding_response = core.dielectric_response(np.full(3, surrounding_index ** 2), 1.0)
    surrounding_wave = core.exterior_wave(surrounding_index, core.unit_frequency)
    surrounding_outgoing = (core.field_boundary + core.excitation_boundary(surrounding_response))(surrounding_wave)
    layers = core.layer_propagation(generator, np.full(n_layers, 2 * np.pi * slice_thickness / band_wavelength))
    _, transmission = core.interface_scattering(
        core.stack_propagation(layers)(surrounding_outgoing), surrounding_response, surrounding_index, core.unit_frequency
    )
    exit_state = surrounding_outgoing(transmission(circular_polarizations))       # [2] Boundary+
    fields = core.interior_fields(
        reconstruction, generator, np.full(n_layers, slice_thickness), band_wavelength, exit_state, fractions
    )
    return depths, fields


def initial_pulse(positions: np.ndarray) -> core.Field:
    """Gaussian wavepackets at `positions` (`[V]`) build co-rotating, counter-rotating, and linearly polarized electric fields (`[3, V] Field+`)."""
    envelope = np.exp(-((positions - pulse_start) / pulse_width) ** 2 / 2)
    phases = 2 * np.pi * surrounding_index * positions / band_wavelength
    rotors = (core.real.xy * (-hand_signs[:, None] * phases / 2)).exp()           # [2, ...] Rotor+
    carriers = rotors >> core.real.xt                                             # [2, ...] Field+
    return envelope * stack([carriers[0], carriers[1], carriers.sum(axis=0) / 2])


def pulse_setup(n_run_steps: int, n_run_frames: int) -> tuple[
    Topology,
    np.ndarray,
    core.Direction,
    core.Direction,
    core.Direction,
    core.Constitutive,
    core.Constitutive,
    core.Scalar,
    Iterator[tuple[core.Field, core.Field]],
]:
    """Preparing the 1D periodic loop and initial forward wavepacket returns the loop topology, node positions, display geometry, medium fields, initial total energy, and strided leapfrog state stream."""
    topology = Topology.loops([n_nodes])
    positions = np.arange(n_nodes) * node_spacing
    inside = ((positions >= slab_start) & (positions < slab_start + n_turns * pitch)).astype(float)
    directors, response = helix_media(positions - slab_start)
    surrounding = core.dielectric_response(np.full(3, surrounding_index ** 2), 1.0)
    media = (inside * response[:, 0] + (1 - inside) * surrounding).real()         # [n_nodes] Excitation- <- Field+
    node_response = media.field()                                                 # (Excitation- <- Field+)[V]
    node_inverse = media.inverse().field()                                        # (Field+ <- Excitation-)[V]
    edge_response = surrounding.real()                                            # [] Excitation- <- Field+

    interval = courant * node_spacing
    electric = initial_pulse(positions).field()                                   # [3] Field+[V]
    shifted = initial_pulse(positions + node_spacing / 2 + interval / (2 * surrounding_index))
    magnetic = (surrounding_index * (core.real.t.inverse() | (core.real.z ^ shifted))).field()  # [3] Field+[E]
    initial_energy = core.electromagnetic_energy(
        electric, magnetic, node_response, edge_response, node_spacing
    ).sites.sum()                                                                 # [3] Scalar+
    stride = n_run_steps // n_run_frames
    states = islice(
        core.leapfrog_states(topology.d[0], node_inverse, edge_response, electric, magnetic, node_spacing, interval, n_run_steps),
        stride - 1,
        None,
        stride,
    )
    slab_indices = np.flatnonzero(inside)
    nodes = positions * core.real.z.dual()                                        # [n_nodes] Direction-
    return topology, positions, nodes, nodes[slab_indices], directors[slab_indices, 0].real(), node_response, edge_response, initial_energy, states


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Saves the five optical-coating figures and the 3D pulse-splitting animation, then checks energy conservation and handedness selectivity."""
    from examples.util.animation import save_animation, save_figure
    from examples.optics.coating import render

    reflection, transmission = antireflection_spectra()
    twisted_reflection, twisted_transmission = twisted_spectra()
    axion_reflection = axion_reflectance()
    angle_reflection = angular_reflectance()
    depths, fields = internal_fields()
    save_figure(render.spectra_figure(wavelengths, reflection), "coating_chi_matching")
    save_figure(render.handedness_figure(wavelengths, twisted_reflection), "coating_handedness")
    save_figure(render.axion_figure(axion_jumps, axion_reflection), "coating_axion")
    save_figure(render.angular_figure(map_wavelengths, incidence_angles, angle_reflection), "coating_polarization_map")
    save_figure(render.internal_field_figure(depths, fields), "coating_internal_field")

    topology, positions, nodes, slab_nodes, optic_axes, node_response, edge_response, initial_energy, state_stream = pulse_setup(n_steps, n_frames)
    kept_states = []

    def recorded_states() -> Iterator[tuple[core.Field, core.Field]]:
        """Yields each state from `state_stream` while recording it for post-run energy checks."""
        for state in state_stream:
            kept_states.append(state)
            yield state

    save_animation(render.pulse_animation(nodes, slab_nodes, optic_axes, 2, recorded_states()), "coating_pulse", frame_duration_ms)
    final_energy = core.electromagnetic_energy(*kept_states[-1], node_response, edge_response, node_spacing)  # [3] Scalar+[V]
    front = topology.reading(np.flatnonzero(positions < slab_start), 0)
    back = topology.reading(np.flatnonzero(positions >= slab_start + n_turns * pitch), 0)
    pulse_reflectance = (front * final_energy).sites.sum() / initial_energy       # [3] Scalar+
    pulse_transmittance = (back * final_energy).sites.sum() / initial_energy      # [3] Scalar+

    # checks
    assert ((reflection + transmission - 1).abs() < 1e-10).all()
    assert ((twisted_reflection + twisted_transmission - 1).abs() < 1e-10).all()
    assert ((pulse_reflectance + pulse_transmittance - 1).abs() < 5e-3).all()
    assert (pulse_reflectance[0] > 0.95).all() and (pulse_reflectance[1] < 0.01).all()


if __name__ == "__main__":
    main()
