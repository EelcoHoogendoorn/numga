"""Electrokinetic transport on a mesh of triangles: a straight slit's response is reciprocal, dissipative and
Poiseuille's, the best efficiency of slits as generators is van der Heyden et al.'s, the double layer's osmotic
pressure and field balance across the channels, the channels pump as their charge's wave lies against their width's
and make exactly the battery's power into heat, a dye carried along a straight channel keeps its total and, charged,
parts by its charge at twice its drift under the field, and the scenes draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

from examples.fluids.electrokinetics import core, render, scenarios

mv = core.mv


@pytest.fixture(scope="module")
def dyes():
    """The straight capillary's strip, its vertices, its wall charges, its liquid's charge density, its drives and the liquid's
    velocity under each, at the vertices and in the triangles, and its dye's first and last frames."""
    topology, vertices, wall_charges, density, flows, drives, velocities, frames = scenarios.dyed(0.0)
    first = next(frames)
    *_, last = frames
    return topology, vertices, wall_charges, density, flows, drives, velocities, first, last


def test_straight_slit_is_reciprocal_dissipative_and_poiseuille():
    """Through a straight slit, each drive reads the other's flux alike, every drive spends power, and a unit pressure
    drop carries two thirds of the cubed half width, over the viscosity; the tolerance tests the cells across it."""
    half_width, n_columns, length = 2.0, 40, 4.0
    widths = np.full(n_columns, half_width)
    topology, coordinates, shifts, lower, upper = core.strip(n_columns, scenarios.n_across, np.array(length), widths)
    charge = core.wall_charge(scenarios.n_across, np.array(length), widths, np.full(n_columns, -2.5))
    _, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, outflow, circulation, cells = scenarios.operators(topology, edges)
    potential = core.double_layer(outflow, cells, mv.scalar(charge[..., None]).field(), scenarios.valences,
                                  scenarios.bulk, scenarios.n_newton_steps)
    *_, carried, _, _, _ = core.transport(topology, areas, sides, gradients, circulation, cells, potential,
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
    topology, vertices, wall_charges, density, speeds, heat, carried, osmotic, electric = scenarios.channel_flows()
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
    topology, vertices, wall_charges, density, speeds, heat, _, osmotic, electric = scenarios.channel_flows()
    plt.close(render.draw_mesh(topology, vertices, scenarios.n_across, scenarios.wavelength, wall_charges[0]))
    plt.close(render.draw_flow(topology, vertices, scenarios.n_across, scenarios.wavelength, density, speeds, wall_charges))
    plt.close(render.draw_heat(topology, vertices, scenarios.n_across, scenarios.wavelength, heat, wall_charges))


def test_dyes_keep_their_totals(dyes):
    """Each run's dye keeps its total over the cells, to round-off: what one cell gains, another loses."""
    topology, vertices, _, _, _, _, _, first, last = dyes
    _, coordinates, shifts, *_ = scenarios.capillary(0.0)
    _, edges = core.placed(topology, coordinates, shifts)
    *_, cells = scenarios.operators(topology, edges)
    np.testing.assert_allclose((cells * last).sites.sum().to_array(), (cells * first).sites.sum().to_array(), rtol=1e-12)


def test_charged_dyes_part_at_twice_their_drift(dyes):
    """The positive and the negative dye, carried alike by the electro-osmotic flow, part at twice the drift the field
    alone gives them, the dye's mobility times the field's push on a unit of charge; the tolerance tests the cells
    along the channel and the time steps."""
    topology, vertices, _, _, _, drives, _, first, last = dyes
    _, coordinates, shifts, *_ = scenarios.capillary(0.0)
    _, edges = core.placed(topology, coordinates, shifts)
    *_, cells = scenarios.operators(topology, edges)
    # How far each run's dye has moved along the channel on average, read by the plane x.
    travelled = (cells * (mv.x & vertices) * (last - first)).sites.sum() / (cells * first).sites.sum()   # [runs] Scalar+
    duration = scenarios.n_dye_frames * scenarios.dye_stride * scenarios.dye_dt
    drift = scenarios.dye_mobility * (mv.x & drives[1].commutator(core.time))                          # [] Scalar+
    np.testing.assert_allclose((travelled[2] - travelled[3]).to_array(), (2 * duration * drift).to_array(), rtol=1e-3)


def test_dyes_draw(dyes):
    topology, vertices, wall_charges, density, flows, _, velocities, _, last = dyes
    plt.close(render.draw_profiles(topology, vertices, scenarios.n_capillary_across, velocities))
    next(render.animate_dyes(topology, vertices, scenarios.n_capillary_across, scenarios.capillary_length, wall_charges,
                             density, flows, [last]))
