"""A matching film, a twisted birefringent reflector, and an axion interface."""

from __future__ import annotations

from itertools import islice

import numpy as np

from numga import stack
from examples.optics.coating import core

DESIGN_WAVELENGTH = 550.0                                      # nm
WAVELENGTH_SAMPLES = 401
WAVELENGTHS = np.linspace(450, 850, WAVELENGTH_SAMPLES)          # nm
INCIDENT_INDEX = 1.0
SUBSTRATE_INDEX = 1.5
COATING_NAMES = ("Bare glass", "Ideal matching layer", "MgF₂ layer")
COATING_INDICES = np.array([SUBSTRATE_INDEX, np.sqrt(INCIDENT_INDEX * SUBSTRATE_INDEX), 1.38])
OPTICAL_THICKNESSES = np.array([0.0, DESIGN_WAVELENGTH / 4, DESIGN_WAVELENGTH / 4])
COATING_THICKNESSES = OPTICAL_THICKNESSES / COATING_INDICES
ORDINARY_INDEX = 1.5
EXTRAORDINARY_INDEX = 1.8
SURROUNDING_INDEX = (ORDINARY_INDEX + EXTRAORDINARY_INDEX) / 2
PITCH = 400.0                                                 # nm per full director turn
TURNS = 6
SLICES_PER_TURN = 16
LAYERS = TURNS * SLICES_PER_TURN
THICKNESS = PITCH / SLICES_PER_TURN
DEPTHS = (np.arange(LAYERS) + 0.5) * THICKNESS                 # nm, slice centres
CRYSTAL_PERMITTIVITY = np.array([EXTRAORDINARY_INDEX ** 2, ORDINARY_INDEX ** 2, ORDINARY_INDEX ** 2])
TWIST_SIGNS = np.array([1.0, -1.0])
TWIST_NAMES = ("Positive twist", "Negative twist")
POLARIZATION_SIGNS = np.array([1.0, -1.0])
POLARIZATION_NAMES = ("x + i y", "x − i y")
BAND_WAVELENGTH = PITCH * SURROUNDING_INDEX                   # nm, the reflection band's centre
# The time domain: the band's pulse sent along a line of nodes through the same helix, light speed one.
SPACING = 20.0                                                # nm between nodes
CELLS = 2000
COURANT = 0.5                                                 # time step per node spacing
SLAB_START = 18000.0                                          # nm
PULSE_START = 10000.0                                         # nm, the pulse's centre
PULSE_WIDTH = 2000.0                                          # nm, its envelope's standard deviation
DURATION = 27000.0                                            # nm of light travel
FRAMES = 120
HANDS = np.array([1.0, -1.0])                                 # turning with and against the helix
AXION_SAMPLES = 241
AXION_JUMPS = np.linspace(-3, 3, AXION_SAMPLES)
AXION_NAMES = ("Parallel to incident", "Perpendicular to incident", "Total reflection")
ANGLE_SAMPLES = 81
ANGLES = np.linspace(0, np.deg2rad(70), ANGLE_SAMPLES)
MAP_WAVELENGTH_SAMPLES = 241
MAP_WAVELENGTHS = np.linspace(450, 850, MAP_WAVELENGTH_SAMPLES)


# --- math -----------------------------------------------------------------------------
def helix(depths: np.ndarray) -> tuple[core.Vector, core.Constitutive]:
    """The optic axis and the crystal's map at the given depths, for both senses of twist: [depths, helices]."""
    # Both helices use the same crystal; only the rotor transporting chi changes.
    angles = 2 * np.pi * depths[:, None] / PITCH * TWIST_SIGNS
    rotors = (core.mv.xy * (angles / 2)).exp()                 # [layers, helices] Rotor
    crystal = core.dielectric(CRYSTAL_PERMITTIVITY, 1.0)
    return rotors >> core.mv.x, rotors >> crystal(rotors << core.Bivector)


def surrounding(parallel: core.Vector) -> core.Ports:
    """The isotropic medium on both sides of the twisted stack."""
    chi = core.dielectric(np.full(3, SURROUNDING_INDEX ** 2), 1.0)
    return core.Ports.from_medium(core.Medium.from_chi(chi, parallel), SURROUNDING_INDEX, parallel)


def circular() -> core.Polarization:
    """The two circular inputs, x + i y and x - i y."""
    return (core.mv.x + core.mv.y * (1j * POLARIZATION_SIGNS)) / np.sqrt(2)


def antireflection() -> tuple[core.Scalar, core.Scalar]:
    incident_chi = core.dielectric(np.full(3, INCIDENT_INDEX ** 2), 1.0)
    substrate_chi = core.dielectric(np.full(3, SUBSTRATE_INDEX ** 2), 1.0)
    incident = core.Ports.from_medium(core.Medium.from_chi(incident_chi, core.mv.t), INCIDENT_INDEX, core.mv.t)
    substrate = core.Ports.from_medium(core.Medium.from_chi(substrate_chi, core.mv.t), SUBSTRATE_INDEX, core.mv.t)
    chi = core.dielectric(COATING_INDICES[:, None] ** 2 * np.ones(3), 1.0)
    medium = core.Medium.from_chi(chi, core.mv.t)
    distance = 2 * np.pi * COATING_THICKNESSES[:, None] / WAVELENGTHS
    coating = core.propagate(medium.generator[:, None], distance)
    result = core.scatter(coating(substrate.outgoing), incident)
    return result.powers(core.mv.x, incident, substrate)


def twisted() -> tuple[core.Vector, core.Vector, core.Scalar, core.Scalar]:
    directors, chi = helix(DEPTHS)
    medium = core.Medium.from_chi(chi, core.mv.t)
    distance = 2 * np.pi * THICKNESS / WAVELENGTHS
    layers = core.propagate(medium.generator[..., None], distance)
    propagation = core.compose(layers)                        # [helices, wavelengths] Boundary <- Boundary
    ports = surrounding(core.mv.t)
    result = core.scatter(propagation(ports.outgoing), ports)
    polarizations = circular()[None, :, None]
    incident_power = ports.power(polarizations)
    reflected = result.reflection[:, None, :](polarizations)
    transmitted = result.transmission[:, None, :](polarizations)
    centres = core.mv.z * DEPTHS                              # [layers] Vector
    return centres, directors, ports.power(reflected) / incident_power, ports.power(transmitted) / incident_power


def axion() -> core.Scalar:
    # Identical ordinary dielectric responses: all reflection comes from the axion jump.
    chi = core.dielectric(np.full(3, SURROUNDING_INDEX ** 2), 1.0)
    incident = surrounding(core.mv.t)
    axion_chi = chi + core.mv.scalar(AXION_JUMPS[:, None]) * core.Bivector
    substrate_medium = core.Medium.from_chi(axion_chi, core.mv.t)
    substrate = core.Ports.from_medium(substrate_medium, SURROUNDING_INDEX, core.mv.t)
    result = core.scatter(substrate.outgoing, incident)
    reflected = result.reflection(core.mv.x)                  # [jumps] Polarization
    parallel = -core.mv.x * (core.mv.x | reflected)
    perpendicular = -core.mv.y * (core.mv.y | reflected)
    return stack([incident.power(parallel), incident.power(perpendicular), incident.power(reflected)]) / incident.power(core.mv.x)


def angular() -> core.Scalar:
    _, chi = helix(DEPTHS)
    parallel = core.mv.t + core.mv.x * (SURROUNDING_INDEX * np.sin(ANGLES))
    medium = core.Medium.from_chi(chi[:, 0, None], parallel)
    distance = 2 * np.pi * THICKNESS / MAP_WAVELENGTHS
    # One layer at a time, each batched over angles and wavelengths.
    layers = (core.propagate(generator[:, None], distance) for generator in medium.generator)
    propagation = core.compose(layers)                        # [angles, wavelengths] Boundary <- Boundary
    ports = surrounding(parallel[:, None])
    scattering = core.scatter(propagation(ports.outgoing), ports)
    tilt = (core.mv.zx * (ANGLES / 2)).exp()
    field = tilt[None, :] >> circular()[:, None]
    tangential = -core.mv.z | (core.mv.z ^ field)
    return scattering.powers(tangential[..., None], ports, ports)[0]


def penetration() -> tuple[np.ndarray, core.Vector]:
    wavelength = BAND_WAVELENGTH
    samples_per_layer = 5
    fractions = np.linspace(0, 1, samples_per_layer)
    depths = (np.arange(LAYERS)[:, None] + fractions) * THICKNESS
    _, chi = helix(DEPTHS)
    medium = core.Medium.from_chi(chi[:, 0], core.mv.t)
    ports = surrounding(core.mv.t)
    layers = core.propagate(medium.generator, np.full(LAYERS, 2 * np.pi * THICKNESS / wavelength))
    scattering = core.scatter(core.compose(layers)(ports.outgoing), ports)
    exit_state = ports.outgoing(scattering.transmission(circular()))
    fields = tuple(medium.interior(np.full(LAYERS, THICKNESS), wavelength, exit_state, fractions))
    electric = core.mv.t | stack(fields[::-1], axis=1)
    return depths.reshape(-1), electric.reshape(len(POLARIZATION_SIGNS), -1)


def pulse(at: np.ndarray) -> core.Bivector:
    """The incoming pulse's electric planes at the given positions, [cases, positions]: the band's
    carrier turning with the helix, against it, and their average, which does not turn."""
    envelope = np.exp(-((at - PULSE_START) / PULSE_WIDTH) ** 2 / 2)
    phases = 2 * np.pi * SURROUNDING_INDEX * at / BAND_WAVELENGTH
    turned = (core.real.xy * (HANDS[:, None] * phases / 2)).exp() >> core.real.x     # [hands, positions] Vector
    cases = stack([turned[0], turned[1], turned.sum(axis=0) / 2])                    # [cases, positions] Vector
    return (envelope * cases) ^ core.real.t


def time_domain() -> tuple[core.Vector, core.Vector, core.Vector, core.Bivector, core.Scalar, core.Scalar, core.Scalar]:
    """The pulse through the positively twisted helix, stepped in time.

    Returns the nodes, the slab's nodes and optic axes, the electric planes at each frame
    [frames, cases, nodes], and per case the energy before, reflected out of the front and
    transmitted out of the back.
    """
    positions = np.arange(CELLS) * SPACING
    inside = ((positions >= SLAB_START) & (positions < SLAB_START + TURNS * PITCH)).astype(float)
    directors, chi = helix(positions - SLAB_START)
    surrounding_chi = core.dielectric(np.full(3, SURROUNDING_INDEX ** 2), 1.0)
    nodes = (inside * chi[:, 0] + (1 - inside) * surrounding_chi).real()            # [nodes] Antibivector <- Bivector
    line = core.Line.from_chi(nodes, surrounding_chi.real(), SPACING)

    # A pulse already travelling towards +z: its magnetic planes are its electric planes contracted
    # through z and t, times the index, half a node ahead and half a step earlier.
    interval = COURANT * SPACING
    electric = pulse(positions)
    magnetic = SURROUNDING_INDEX * (core.real.t | (core.real.z ^ pulse(positions + SPACING / 2 + interval / (2 * SURROUNDING_INDEX))))
    before = line.energy(electric, magnetic).sum(axis=-1)
    steps = round(DURATION / interval)
    stride = steps // FRAMES
    states = list(islice(line.evolve(electric, magnetic, interval, steps), stride - 1, None, stride))
    electric, magnetic = stack([e for e, _ in states]), stack([m for _, m in states])  # [frames, cases, nodes] Bivector

    energy = line.energy(electric[-1], magnetic[-1])                                  # [cases, nodes] Scalar
    reflected = (energy * (positions < SLAB_START)).sum(axis=-1)
    transmitted = (energy * (positions >= SLAB_START + TURNS * PITCH)).sum(axis=-1)
    slab = np.flatnonzero(inside)
    return (core.real.z * positions, core.real.z * positions[slab], directors[slab, 0].real(),
            electric, before, reflected, transmitted)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.optics.coating import render

    reflection, transmission = antireflection()
    centres, directors, twisted_reflection, twisted_transmission = twisted()
    axion_reflection = axion()
    angle_reflection = angular()
    depths, electric = penetration()
    save_figure(render.spectra(WAVELENGTHS, reflection, COATING_NAMES), "coating_chi_matching")
    save_figure(render.twist(centres, directors[:, 0]), "coating_twist")
    save_figure(render.handedness(WAVELENGTHS, twisted_reflection, TWIST_NAMES, POLARIZATION_NAMES),
                "coating_handedness")
    save_figure(render.axion(AXION_JUMPS, axion_reflection, AXION_NAMES), "coating_axion")
    save_figure(render.angular(MAP_WAVELENGTHS, ANGLES, angle_reflection, POLARIZATION_NAMES), "coating_polarization_map")
    save_figure(render.field(depths, electric, POLARIZATION_NAMES), "coating_internal_field")
    nodes, slab, directors, pulses, before, pulse_reflected, pulse_transmitted = time_domain()
    # The pulse that does not turn: one handedness comes back, the other passes.
    save_animation(render.pulse(nodes, slab, directors, pulses[:, 2]), "coating_pulse", 50)

    # --- checks
    np.testing.assert_allclose((reflection + transmission).kernel, 1, atol=1e-10)
    np.testing.assert_allclose((twisted_reflection + twisted_transmission).kernel, 1, atol=1e-10)
    # The leapfrog keeps the pulse's energy, and the helix returns the hand that turns with it.
    np.testing.assert_allclose(((pulse_reflected + pulse_transmitted) / before).kernel, 1, atol=1e-3)
    assert (pulse_reflected / before).kernel[0, 0] > 0.98 and (pulse_reflected / before).kernel[1, 0] < 0.01


if __name__ == "__main__":
    main()
