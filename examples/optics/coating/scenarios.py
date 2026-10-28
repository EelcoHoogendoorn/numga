"""The scenes are a matching film, a twisted birefringent reflector with a pulse stepped through it, and an
axion interface."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.optics.coating import core

# Lengths are in nm throughout.
design_wavelength = 550.0
n_spectrum = 401
spectrum = np.linspace(450, 850, n_spectrum)
incident_index = 1.0
substrate_index = 1.5
coating_names = ("Bare glass", "Ideal matching layer", "MgF₂ layer")
coating_indices = np.array([substrate_index, np.sqrt(incident_index * substrate_index), 1.38])
optical_thicknesses = np.array([0.0, design_wavelength / 4, design_wavelength / 4])
coating_thicknesses = optical_thicknesses / coating_indices
ordinary_index = 1.5
extraordinary_index = 1.8
surrounding_index = (ordinary_index + extraordinary_index) / 2
# The length of one full turn of the optic axis.
pitch = 400.0
n_turns = 6
n_slices_per_turn = 16
n_layers = n_turns * n_slices_per_turn
slice_thickness = pitch / n_slices_per_turn
# The depths of the slices' centres.
slice_depths = (np.arange(n_layers) + 0.5) * slice_thickness
crystal_permittivity = np.array([extraordinary_index ** 2, ordinary_index ** 2, ordinary_index ** 2])
twist_signs = np.array([1.0, -1.0])
polarization_signs = np.array([1.0, -1.0])
# The circular inputs by the sense their field turns in along the beam, from `xt` towards `yt` or back.
polarization_names = ("x to y", "y to x")
# The reflection band's centre.
band_wavelength = pitch * surrounding_index
# The time domain: the band's pulse sent along a line of nodes through the same helix, light speed one. The
# spacing between nodes, their count, the time step per node spacing, where the slab starts, the pulse's
# centre and its envelope's standard deviation, and the distance light travels in the run.
node_spacing = 40.0
n_nodes = 1000
courant = 1.0
slab_start = 18000.0
pulse_start = 10000.0
pulse_width = 1000.0
light_travel = 27000.0
n_frames = 120
# The pulse's carriers turning with and against the helix.
hand_signs = np.array([1.0, -1.0])
# Field samples across each slice, and an animation frame's duration.
n_samples_per_layer = 5
frame_duration_ms = 50
n_axion_jumps = 241
axion_jumps = np.linspace(-3, 3, n_axion_jumps)
axion_names = ("Parallel to incident", "Perpendicular to incident", "Total reflection")
n_angles = 41
incidence_angles = np.linspace(0, np.deg2rad(70), n_angles)
n_map_wavelengths = 121
map_wavelengths = np.linspace(450, 850, n_map_wavelengths)


# --- math -----------------------------------------------------------------------------
def helix(depths: np.ndarray) -> tuple[core.Direction, core.Constitutive]:
    """The optic axis and the crystal's map at the given depths, for both senses of twist: [depths, helices]."""
    # Both helices use the same crystal; only the rotor transporting chi changes.
    angles = 2 * np.pi * depths[:, None] / pitch * twist_signs
    rotors = (core.mv.xy * (-angles / 2)).exp()                 # [layers, helices] Rotor+
    crystal = core.dielectric(crystal_permittivity, 1.0)
    return rotors >> core.mv.x.dual(), rotors >> crystal(rotors << core.Bivector)


# The two circular inputs, their fields turning from `xt` towards `yt` along the beam and back.
circular_polarizations = (core.mv.xt + core.mv.yt * (1j * polarization_signs)) / np.sqrt(2)   # [polarizations] Polarization+


def antireflection() -> tuple[core.Scalar, core.Scalar]:
    """The reflected and transmitted power of x-polarized light at normal incidence on bare glass and on
    glass under each quarter-wave coating, over the wavelengths: `[coatings, wavelengths]`."""
    incident = core.Ports.isotropic(incident_index, core.unit_frequency)
    substrate = core.Ports.isotropic(substrate_index, core.unit_frequency)
    chi = core.dielectric(coating_indices[:, None] ** 2 * np.ones(3), 1.0)
    medium = core.Medium.from_chi(chi, core.unit_frequency)
    distance = 2 * np.pi * coating_thicknesses[:, None] / spectrum
    coating = core.propagate(medium.generator[:, None], distance)
    result = core.scatter(coating(substrate.outgoing), incident)
    return result.powers(core.mv.xt, incident, substrate)


def twisted() -> tuple[core.Scalar, core.Scalar]:
    """The twisted reflector at normal incidence: the reflected and transmitted power of both circular polarizations
    through both helices, over the wavelengths."""
    _, chi = helix(slice_depths)
    medium = core.Medium.from_chi(chi, core.unit_frequency)
    distance = 2 * np.pi * slice_thickness / spectrum
    layers = core.propagate(medium.generator[..., None], distance)
    propagation = core.compose(layers)                        # [helices, wavelengths] Boundary+ <- Boundary+
    ports = core.Ports.isotropic(surrounding_index, core.unit_frequency)
    result = core.scatter(propagation(ports.outgoing), ports)
    reflected, transmitted = result.powers(circular_polarizations[:, None, None], ports, ports)   # [polarizations, helices, wavelengths] Scalar+, Scalar+
    return reflected, transmitted


def axion() -> core.Scalar:
    """The power reflected off a jump in the axion coupling alone, for x-polarized light at normal
    incidence: along x, across it, and in all, for each jump."""
    # Identical ordinary dielectric responses: all reflection comes from the axion jump.
    chi = core.dielectric(np.full(3, surrounding_index ** 2), 1.0)
    incident = core.Ports.isotropic(surrounding_index, core.unit_frequency)
    axion_chi = chi + core.mv(core.AxionCoupling, axion_jumps[:, None]) * core.Bivector
    substrate_medium = core.Medium.from_chi(axion_chi, core.unit_frequency)
    substrate = core.Ports.from_medium(substrate_medium, surrounding_index, core.unit_frequency)
    result = core.scatter(substrate.outgoing, incident)
    reflected = result.reflection(core.mv.xt)                  # [jumps] Polarization+
    # The planes x and y sweep through time read the reflected field along and across the incident one.
    along_incident = core.mv.x.dual() & core.time              # [] Antibivector-
    across_incident = core.mv.y.dual() & core.time             # [] Antibivector-
    parallel = core.mv.xt * (reflected & along_incident)      # [jumps] Polarization+
    perpendicular = core.mv.yt * (reflected & across_incident)   # [jumps] Polarization+
    return stack([incident.power(parallel), incident.power(perpendicular), incident.power(reflected)]) / incident.power(core.mv.xt)


def angular() -> core.Scalar:
    """The positive helix's reflection of both circular polarizations over the angles of incidence and
    the wavelengths."""
    _, chi = helix(slice_depths)
    parallel = core.unit_frequency + core.mv.x * (surrounding_index * np.sin(incidence_angles))
    medium = core.Medium.from_chi(chi[:, 0, None], parallel)
    distance = 2 * np.pi * slice_thickness / map_wavelengths
    # One layer at a time, each batched over angles and wavelengths.
    layers = (core.propagate(generator[:, None], distance) for generator in medium.generator)
    propagation = core.compose(layers)                        # [angles, wavelengths] Boundary+ <- Boundary+
    ports = core.Ports.isotropic(surrounding_index, parallel[:, None])
    scattering = core.scatter(propagation(ports.outgoing), ports)
    tilt = (core.mv.zx * (incidence_angles / 2)).exp()        # [angles] Rotor+
    beam_fields = tilt[None, :] >> circular_polarizations[:, None]   # [polarizations, angles] Bivector+
    # The field's planes across the normal: its rejection by z.
    tangential = core.mv.z.inverse() | (core.mv.z ^ beam_fields)
    return scattering.powers(tangential[..., None], ports, ports)[0]


def penetration() -> tuple[np.ndarray, core.Bivector]:
    """The field at the band's centre through the positive helix, `[cases, layers, samples]`, at the
    depths `[layers, samples]`."""
    wavelength = band_wavelength
    fractions = np.linspace(0, 1, n_samples_per_layer)
    depths = (np.arange(n_layers)[:, None] + fractions) * slice_thickness
    _, chi = helix(slice_depths)
    medium = core.Medium.from_chi(chi[:, 0], core.unit_frequency)
    ports = core.Ports.isotropic(surrounding_index, core.unit_frequency)
    layers = core.propagate(medium.generator, np.full(n_layers, 2 * np.pi * slice_thickness / wavelength))
    scattering = core.scatter(core.compose(layers)(ports.outgoing), ports)
    exit_state = ports.outgoing(scattering.transmission(circular_polarizations))
    fields = medium.interior(np.full(n_layers, slice_thickness), wavelength, exit_state, fractions)   # [cases, layers, samples] Bivector+
    return depths, fields


def pulse(at: np.ndarray) -> core.Bivector:
    """The incoming pulse's electric planes at the given positions, [cases, positions]: the band's
    carrier turning with the helix, against it, and their average, which does not turn."""
    envelope = np.exp(-((at - pulse_start) / pulse_width) ** 2 / 2)
    phases = 2 * np.pi * surrounding_index * at / band_wavelength
    carriers = (core.real.xy * (-hand_signs[:, None] * phases / 2)).exp() >> core.real.xt   # [hands, positions] Bivector+
    cases = stack([carriers[0], carriers[1], carriers.sum(axis=0) / 2])              # [cases, positions] Bivector+
    return envelope * cases


def time_domain() -> tuple[core.Direction, core.Direction, core.Direction, core.Bivector, core.Scalar, core.Scalar,
                           core.Scalar]:
    """The pulse through the positively twisted helix, stepped in time.

    The result holds the nodes, as displacements along the line from its start, the slab's nodes and optic
    axes, the electric planes at each frame [frames, cases, nodes], and per case the energy before, reflected
    out of the front and transmitted out of the back.
    """
    positions = np.arange(n_nodes) * node_spacing
    inside = ((positions >= slab_start) & (positions < slab_start + n_turns * pitch)).astype(float)
    directors, chi = helix(positions - slab_start)
    surrounding_chi = core.dielectric(np.full(3, surrounding_index ** 2), 1.0)
    media = (inside * chi[:, 0] + (1 - inside) * surrounding_chi).real()            # [nodes] Antibivector- <- Bivector+
    line = core.Line.from_chi(media, surrounding_chi.real(), node_spacing)

    # A pulse already travelling towards +z: its magnetic planes are its electric planes contracted
    # through z and t, times the index, half a node ahead and half a step earlier.
    interval = courant * node_spacing
    electric = pulse(positions)
    magnetic = surrounding_index * (core.real.t.inverse() | (core.real.z ^ pulse(positions + node_spacing / 2 + interval / (2 * surrounding_index))))
    before = line.energy(electric, magnetic).sum(axis=-1)
    steps = round(light_travel / interval)
    stride = steps // n_frames
    states = list(islice(line.evolve(electric, magnetic, interval, steps), stride - 1, None, stride))
    electric, magnetic = stack([e for e, _ in states]), stack([m for _, m in states])  # [frames, cases, nodes] Bivector+, Bivector+

    energy = line.energy(electric[-1], magnetic[-1])                                  # [cases, nodes] Scalar+
    reflected = (energy * (positions < slab_start)).sum(axis=-1)
    transmitted = (energy * (positions >= slab_start + n_turns * pitch)).sum(axis=-1)
    slab = np.flatnonzero(inside)
    nodes = positions * core.real.z.dual()                                           # [nodes] Direction-
    return nodes, nodes[slab], directors[slab, 0].real(), electric, before, reflected, transmitted


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.coating import render

    reflection, transmission = antireflection()
    twisted_reflection, twisted_transmission = twisted()
    axion_reflection = axion()
    angle_reflection = angular()
    depths, fields = penetration()
    save_figure(render.spectra(spectrum, reflection, coating_names), "coating_chi_matching")
    save_figure(render.handedness(spectrum, twisted_reflection, polarization_names), "coating_handedness")
    save_figure(render.axion(axion_jumps, axion_reflection, axion_names), "coating_axion")
    save_figure(render.angular(map_wavelengths, incidence_angles, angle_reflection), "coating_polarization_map")
    save_figure(render.field(depths, fields), "coating_internal_field")
    nodes, slab, directors, pulses, before, pulse_reflected, pulse_transmitted = time_domain()
    # The pulse that does not turn: one handedness comes back, the other passes.
    save_animation(render.pulse(nodes, slab, directors, pulses[:, 2]), "coating_pulse", frame_duration_ms)

    # checks
    # Lossless layers conserve power: what is not reflected is transmitted.
    np.testing.assert_allclose((reflection + transmission).kernel, 1, atol=1e-10)
    np.testing.assert_allclose((twisted_reflection + twisted_transmission).kernel, 1, atol=1e-10)
    # The leapfrog keeps the pulse's energy, to its grid's dispersion of a pulse this short, and the helix returns
    # the hand that turns with it, all but the part of the pulse's spectrum outside the reflectance band.
    np.testing.assert_allclose(((pulse_reflected + pulse_transmitted) / before).kernel, 1, atol=5e-3)
    assert (pulse_reflected / before)[0] > 0.95 and (pulse_reflected / before)[1] < 0.01


if __name__ == "__main__":
    main()
