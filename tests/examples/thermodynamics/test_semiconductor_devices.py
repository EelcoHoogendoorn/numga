"""Exercises one solve step and one rendered frame for the p-n junction, the planar MOSFET, and the front-contact solar
cell."""

from functools import partial
from itertools import islice

import numpy as np

from examples.thermodynamics.semiconductor_devices import render, scenarios


def test_diode():
    """Solves one formation step and one zero-bias steady state on the p-n bar, checks non-negative entropy production,
    and renders one frame of each animation."""
    diode = scenarios.diode
    topology, vertices, dopants, ends, slab, wired_contacts = scenarios.junction()
    V, E = topology.n_elements
    volumes, gradient = diode.simplex_maps(topology, vertices, scenarios.Spatial)
    _, semiconductor = scenarios.silicon(topology)
    bands_of = partial(diode.band_energies, gap=scenarios.gap)
    densities_of = partial(diode.carrier_densities, intrinsic=scenarios.intrinsic)
    terminal_of = partial(diode.terminal_readings, topology, volumes, gradient, semiconductor, ends[:1], ends[1:],
                          intrinsic=scenarios.intrinsic, mobilities=scenarios.mobilities)
    formation_states = list(islice(scenarios.junction_formation(topology, vertices, dopants, ends, slab,
                                                                wired_contacts), 2))
    next(render.formation_animation(vertices, bands_of, densities_of, scenarios.intrinsic, scenarios.energy_window,
                                  formation_states[1:]))
    states = list(scenarios.biased_states(topology, vertices, dopants, ends, wired_contacts, formation_states[-1],
                                          np.zeros(1), np.zeros(1)))
    zero_generation = diode.mv.scalar(np.zeros((V, 1))).field()               # Scalar+[V]
    transport_s, recomb_s, absorb_s = diode.entropy_production(
        topology, volumes, gradient, semiconductor, states[-1], zero_generation, scenarios.intrinsic,
        scenarios.mobilities, scenarios.rate_constant, scenarios.gap, scenarios.sun)
    assert (transport_s.batch() >= -1e-12).all()
    assert (recomb_s.batch() >= -1e-12).all()
    assert (absorb_s.batch().abs() < 1e-12).all()
    next(render.sweep_animation(vertices, bands_of, densities_of, scenarios.intrinsic, terminal_of,
                              scenarios.energy_window, (scenarios.reverse_bias, scenarios.forward_bias),
                              scenarios.current_window, scenarios.volt, render.dark_colour, [], states[1:]))


def test_mosfet():
    """Solves the first MOSFET steady state and renders its cross-section beside its output-curve panel."""
    planar = scenarios.planar
    topology, vertices, permittivities, semiconductor, dopants, neutral, contacts = scenarios.transistor()
    gates, drains, curves = scenarios.gate_drain_schedule()
    areas, gradient = planar.simplex_maps(topology, vertices, scenarios.Upright)
    source, drain, _, _ = contacts
    terminal_of = partial(planar.terminal_readings, topology, areas, gradient, semiconductor, drain, source,
                          intrinsic=scenarios.intrinsic, mobilities=scenarios.mobilities)
    states = scenarios.transistor_states(topology, vertices, permittivities, semiconductor, dopants, neutral, contacts,
                                         gates[:1], drains[:1])
    next(render.transistor_animation(topology, vertices, semiconductor, contacts,
                                   partial(planar.carrier_densities, intrinsic=scenarios.intrinsic), terminal_of,
                                   scenarios.full_densities, curves[:1], scenarios.largest_drain,
                                   scenarios.drain_current_window, scenarios.volt, states))


def test_solar_cell():
    """Solves the dark zero-bias solar-cell period at the first finger spacing and renders one lighting-sweep frame."""
    planar = scenarios.planar
    mv = planar.mv
    finger_spacing = scenarios.finger_spacings[0]
    topology, vertices, dopants, neutral, contacts, generation = scenarios.solar_period(finger_spacing)
    V, E, F = topology.n_elements
    areas, gradient = planar.simplex_maps(topology, vertices, scenarios.Upright)
    all_silicon = mv.scalar(np.ones((F, 1))).field()                          # Scalar+[F]
    biases, shares = scenarios.lighting_biases()
    bias_extensor = mv.scalar(biases[:1, None])                               # [1] Scalar+
    states = scenarios.solar_steady(topology, dopants, neutral, contacts, areas, gradient, neutral, bias_extensor,
                                    shares[:1] * generation)
    dual_areas = topology.incidence[-1, 0].mean(axis=0).adjugate() * areas    # Scalar+[V]

    def holes_of(potentials):
        return planar.carrier_flows(topology, gradient, all_silicon, potentials, scenarios.intrinsic,
                                    scenarios.mobilities)[1]

    next(render.lighting_animation(topology, vertices, dopants, contacts, generation, holes_of,
                                 partial(scenarios.cell_terminal, topology, contacts, areas, gradient),
                                 scenarios.surface_generation * scenarios.absorption_length, finger_spacing,
                                 biases.max(), (dual_areas * generation).sites.sum(), scenarios.volt, shares[:1],
                                 states))
