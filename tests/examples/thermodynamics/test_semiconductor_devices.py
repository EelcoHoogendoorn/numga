"""The semiconductor devices run: the diode's bar, the MOSFET's cross-section and the solar cell's period, each solved
for one frame and that frame drawn."""

from functools import partial
from itertools import islice

import numpy as np

from examples.thermodynamics.semiconductor_devices import render, scenarios


def test_diode():
    """The bar's first frame as the junction forms, and from it the steady state at no bias in the dark, each drawn as
    its band diagram, the second beside its current."""
    diode = scenarios.diode
    topology, vertices, dopants, ends, slab, wired = scenarios.junction()
    volumes, driving, outflow = diode.simplices(topology, vertices, scenarios.Along)
    _, semiconductor = scenarios.silicon(topology)
    bands_of = partial(diode.bands, gap=scenarios.gap)
    densities_of = partial(diode.carrier_densities, intrinsic=scenarios.intrinsic)
    terminal_of = partial(diode.terminal, topology, volumes, driving, outflow, semiconductor, ends[:1], ends[1:],
                          intrinsic=scenarios.intrinsic, mobilities=scenarios.mobilities)
    formed = list(islice(scenarios.forming(topology, vertices, dopants, ends, slab, wired), 2))
    next(render.animate_forming(vertices, bands_of, densities_of, scenarios.intrinsic, scenarios.energy_window,
                                formed[1:]))
    states = list(scenarios.biased(topology, vertices, dopants, ends, wired, formed[-1], np.zeros(1), np.zeros(1)))
    next(render.animate_sweep(vertices, bands_of, densities_of, scenarios.intrinsic, terminal_of,
                              scenarios.energy_window, (scenarios.reverse_bias, scenarios.forward_bias),
                              scenarios.current_window, scenarios.volt, render.dark_colour, [], states[1:]))


def test_mosfet():
    """The MOSFET's steady state at the gate's first frame, drawn as its cross-section beside its output curves."""
    mosfet = scenarios.mosfet
    topology, vertices, permittivities, semiconductor, dopants, neutral, contacts = scenarios.transistor()
    gates, drains, curves = scenarios.schedule()
    volumes, driving, outflow = mosfet.simplices(topology, vertices, scenarios.Upright)
    source, drain, _, _ = contacts
    terminal_of = partial(mosfet.terminal, topology, volumes, driving, outflow, semiconductor, drain, source,
                          intrinsic=scenarios.intrinsic, mobilities=scenarios.mobilities)
    states = scenarios.switching(topology, vertices, permittivities, semiconductor, dopants, neutral, contacts,
                                 gates[:1], drains[:1])
    next(render.animate_switching(topology, vertices, semiconductor, contacts,
                                  partial(mosfet.carrier_densities, intrinsic=scenarios.intrinsic), terminal_of,
                                  scenarios.full_densities, curves[:1], scenarios.largest_drain,
                                  scenarios.drain_current_window, scenarios.volt, states))


def test_solar_cell():
    """The solar cell's steady state at the first spacing of its fingers, dark at no bias, drawn as its window of
    periods beside its current."""
    solar_cell = scenarios.solar_cell
    mv = solar_cell.mv
    finger_spacing = scenarios.finger_spacings[0]
    topology, vertices, dopants, neutral, contacts, generation = scenarios.solar_period(finger_spacing)
    maps = solar_cell.simplices(topology, vertices, scenarios.Upright)
    filled = mv.scalar(np.ones((topology.n_elements[-1], 1))).field()
    biases, shares = scenarios.lighting_biases()
    states = scenarios.solar_steady(topology, dopants, neutral, contacts, *maps, neutral, biases[:1],
                                    shares[:1] * generation)
    cells = topology.incidence[-1, 0].mean(axis=0).adjugate() * maps[0]

    def holes(potentials):
        return solar_cell.flows(topology, maps[1], filled, potentials, scenarios.intrinsic, scenarios.mobilities)[1]

    next(render.animate_lighting(topology, vertices, dopants, contacts, generation, holes,
                                 partial(scenarios.cell_terminal, topology, contacts, *maps),
                                 scenarios.surface_generation * scenarios.absorption_length, finger_spacing,
                                 biases.max(), (cells * generation).sites.sum(), scenarios.volt, shares[:1], states))
