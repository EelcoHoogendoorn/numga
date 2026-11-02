"""Electrokinetic transport on a mesh of triangles: a straight slit's response is reciprocal, dissipative and
Poiseuille's, the best efficiency of slits as generators is van der Heyden et al.'s, the double layer's osmotic
pressure and field balance across the channels, the channels pump as their charge's wave lies against their width's
and make exactly the battery's power into heat, and the scenes draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from examples.fluids.electrokinetics import core, render, scenarios

mv = core.mv


def test_straight_slit_is_reciprocal_dissipative_and_poiseuille():
    """Through a straight slit, each drive reads the other's flux alike, every drive spends power, and a unit pressure
    drop carries two thirds of the cubed half width, over the viscosity; the tolerance tests the cells across it."""
    half_width, n_columns, length = 2.0, 40, 4.0
    widths = np.full(n_columns, half_width)
    topology, coordinates, shifts, lower, upper = core.strip(n_columns, scenarios.n_across, np.array(length), widths)
    charge = core.wall_charge(scenarios.n_across, np.array(length), widths, np.full(n_columns, -2.5))
    _, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, stiffness, cells = scenarios.operators(topology, edges)
    potential = core.double_layer(stiffness, cells, mv.scalar(charge[..., None]).field(), scenarios.valences,
                                  scenarios.bulk, scenarios.n_newton_steps)
    *_, carried, _, _, _ = core.transport(topology, areas, sides, gradients, stiffness, cells, potential,
                                          scenarios.valences, scenarios.bulk, scenarios.viscosity, scenarios.mobilities,
                                          np.array(length), lower, upper, scenarios.experiments)
    table = (scenarios.experiments[:, None] & carried[None, :]).to_array().reshape(2, 2)
    np.testing.assert_allclose(table, table.T, rtol=1e-10)
    assert np.all(np.linalg.eigvalsh(table) > 0)
    np.testing.assert_allclose(table[0, 0], 2 * half_width ** 3 / (3 * scenarios.viscosity), rtol=1e-2)


def test_best_efficiency_is_van_der_heydens_low_salt_limit():
    """In slits much narrower than a Debye length, the best efficiency over wall charges is the peak of van der Heyden
    et al.'s analytic low-salt result, their equation 16, for the counter-ion's mobility; the tolerance tests the
    cells across the slit."""
    efficiency = scenarios.generator_efficiency().to_array().reshape(len(scenarios.half_widths),
                                                                     len(scenarios.surface_charges))
    # Their mobility parameter, mobility times viscosity times charge over twice the permittivity times thermal energy.
    parameter = scenarios.mobilities[0] / 2

    def their_efficiency(x: float) -> float:
        coupling = 3 * (quad(lambda y: y * np.tan(y), 0, x)[0] / x) ** 2 / (x * np.tan(x) * (1 + parameter) - x ** 2)
        return coupling / (2 - coupling + 2 * np.sqrt(1 - coupling))

    peak = -minimize_scalar(lambda x: -their_efficiency(x), bounds=(0.01, 1.565), method="bounded").fun
    np.testing.assert_allclose(efficiency[0].max(), peak, rtol=5e-2)


def test_channels_balance_across_pump_by_their_phase_and_make_the_power_spent_into_heat():
    """Across the channels the double layer's osmotic pressure and its field push the liquid equally and oppositely,
    up to the triangles' resolution of its edge; the channels pump along the field in step, against it half a wave
    behind, and all but nothing a quarter wave behind; and the heat their liquid makes is the battery's power."""
    topology, vertices, wall_charges, density, speeds, stream, heat, carried, osmotic, electric = scenarios.channel_flows()
    across_osmotic = (mv.y & osmotic).to_array()
    across_net = (mv.y & (osmotic + electric)).to_array()
    assert np.abs(across_net).max() < 0.1 * np.abs(across_osmotic).max()
    pumped = (scenarios.pump & carried).to_array().ravel()
    assert pumped[0] > 0 > pumped[2] and abs(pumped[1]) < 0.05 * pumped[0]
    _, coordinates, shifts, *_ = scenarios.channels()
    _, edges = core.placed(topology, coordinates, shifts)
    *_, cells = scenarios.operators(topology, edges)
    made = (heat * cells).sites.sum().to_array().ravel()
    spent = ((scenarios.battery & carried) * scenarios.wavelength).to_array().ravel()
    np.testing.assert_allclose(made, spent, rtol=1e-10)


def test_scenes_draw():
    topology, vertices, wall_charges, density, speeds, stream, heat, _, osmotic, electric = scenarios.channel_flows()
    plt.close(render.draw_mesh(topology, vertices, scenarios.n_across, scenarios.wavelength, wall_charges[0]))
    plt.close(render.draw_charge(topology, vertices, scenarios.n_across, scenarios.wavelength, density, wall_charges))
    plt.close(render.draw_balance(topology, vertices, scenarios.n_across, scenarios.wavelength, osmotic, electric,
                                  wall_charges[0]))
    plt.close(render.draw_flow(topology, vertices, scenarios.n_across, scenarios.wavelength, speeds, stream,
                               wall_charges))
    plt.close(render.draw_heat(topology, vertices, scenarios.n_across, scenarios.wavelength, heat, wall_charges))
