"""A p-n junction formed from two neutral slabs comes to rest with its Fermi level level across the bar and no current;
under a bias its current follows Shockley's form, growing exponentially forward and saturating in reverse; and the
carriers' transport, their recombination and the light's absorption never produce negative entropy. A MOSFET's gate
gathers a channel past a threshold, below which no current flows, the channel growing at the oxide's capacitance; its
drain's current is linear in a small drain voltage and saturates at a large one, as the channel pinches off near the
drain. A solar cell's fingers shade less the further apart they lie, while its top layer costs more voltage, and its
power peaks between; the light makes no pairs beneath the fingers, though holes rise to the junction there; and its
open-circuit voltage stays below the bound the light's temperature sets."""

import numpy as np
import pytest

from numga import stack
from examples.thermodynamics.semiconductor_devices import scenarios

diode = scenarios.diode


@pytest.fixture(scope="module")
def runs():
    """The bar, the junction formed from the slabs, and the dark and the lit sweep from it, each frame a state."""
    topology, vertices, dopants, ends, slab, wired = scenarios.junction()
    formed = list(scenarios.forming(topology, vertices, dopants, ends, slab, wired))
    biases = scenarios.sweep()
    dark = list(scenarios.biased(topology, vertices, dopants, ends, wired, formed[-1], biases, np.zeros(len(biases))))
    lit_biases, shares = scenarios.lit_sweep()
    generations = scenarios.generation_rate * shares
    lit = list(scenarios.biased(topology, vertices, dopants, ends, wired, dark[-1], lit_biases, generations))
    return topology, vertices, ends, formed, dark, lit, np.concatenate([[0.0], generations])


def readings(topology, vertices, ends, states):
    """The voltage and current across the bar at each state `[states]` each."""
    volumes, driving, outflow = diode.simplices(topology, vertices)
    _, semiconductor = scenarios.silicon(topology)
    return np.array([[float(reading.to_array()) for reading in diode.terminal(
        topology, volumes, driving, outflow, semiconductor, ends[:1], ends[1:], state, scenarios.intrinsic,
        scenarios.mobilities)] for state in states]).T


def test_equilibrium_is_level_and_still(runs):
    """At no bias in the dark, the electrons' and the holes' quasi-Fermi levels are one level across the whole bar,
    and no current flows."""
    topology, vertices, ends, formed, dark, _, _ = runs
    _, currents = readings(topology, vertices, ends, dark)
    for state in (dark[1], dark[-1]):
        fermi = np.asarray(diode.bands(state, scenarios.gap).batch().to_array())[2:]   # [2, vertices]
        assert np.abs(fermi - fermi[0, -1]).max() < 1e-9
    # The current at no bias, against the current the same bar carries a thermal volt forward.
    _, forward = readings(topology, vertices, ends, dark[3:4])
    assert abs(currents[1]) < 1e-3 * forward[0] and abs(currents[-1]) < 1e-3 * forward[0]
    # The junction formed: its Fermi level levels out by the end of the forming.
    fermi = np.asarray(diode.bands(formed[-1], scenarios.gap).batch().to_array())[2:]
    assert np.abs(fermi - fermi[0, -1]).max() < 0.05


def test_current_follows_shockley(runs):
    """In the dark the current grows by a factor e for each thermal volt forward, and in reverse it saturates, near
    the current that forward growth extrapolates to at no bias: Shockley's diode law, `saturation * (exp(bias) - 1)`,
    with a little more in reverse, where the widening depletion region makes more pairs."""
    topology, vertices, ends, _, dark, _, _ = runs
    voltages, currents = readings(topology, vertices, ends, dark[1:])
    forward = (voltages > 4) & (voltages < 14)
    slopes = np.diff(np.log(currents[forward])) / np.diff(voltages[forward])
    assert np.all((slopes > 0.9) & (slopes < 1.1))
    saturation = np.median(currents[forward] / np.expm1(voltages[forward]))
    reverse = -currents[voltages < -3]
    assert reverse.max() < 1.15 * reverse.min()
    assert saturation < reverse.min() and reverse.max() < 1.5 * saturation


def test_entropy_production_is_never_negative(runs):
    """At every state of the forming and of both sweeps, the carriers' transport across each edge, the recombination
    in each cell and the light's absorption in each cell produce entropy, or none."""
    topology, vertices, _, formed, dark, lit, generations = runs
    volumes, driving, outflow = diode.simplices(topology, vertices)
    _, semiconductor = scenarios.silicon(topology)
    cases = [(state, 0.0) for state in formed + dark] + list(zip(lit, generations))
    produced = np.array([[np.asarray(entropy.batch().to_array()) for entropy in diode.entropy_production(
        topology, volumes, driving, semiconductor, state, generation, scenarios.intrinsic, scenarios.mobilities,
        scenarios.rate_constant, scenarios.gap, scenarios.sun)] for state, generation in cases], dtype=object)
    # Each process's production at every state, against the largest it reaches over the runs, down to round-off.
    for process in produced.T:
        largest = max(entropy.max() for entropy in process)
        assert min(entropy.min() for entropy in process) > -1e-12 * largest
        assert largest > 0


mosfet = scenarios.mosfet


@pytest.fixture(scope="module")
def switched():
    """The MOSFET and its steady states as the gate ramps and the drain sweeps, with the gate's and the drain's
    voltage, the output curve each frame lies on, and the drain's voltage and current read off each state."""
    topology, vertices, permittivities, semiconductor, dopants, neutral, contacts = scenarios.transistor()
    gates, drains, curves = scenarios.schedule()
    states = list(scenarios.switching(topology, vertices, permittivities, semiconductor, dopants, neutral, contacts,
                                      gates, drains))
    volumes, driving, outflow = mosfet.simplices(topology, vertices)
    readings = np.array([[float(reading.to_array()) for reading in mosfet.terminal(
        topology, volumes, driving, outflow, semiconductor, contacts[1], contacts[0], state, scenarios.intrinsic,
        scenarios.mobilities)] for state in states]).T
    return topology, vertices, semiconductor, gates, curves, states, readings


def sheet(topology, vertices, semiconductor, state, left, right):
    """The electrons per area of the surface between the places across `left` and `right`, all of them beneath it:
    those of the cells of the vertices between, over the width."""
    volumes, _, _ = mosfet.simplices(topology, vertices)
    cells = ~topology.incidence[-1, 0].mean(axis=0) * (volumes * semiconductor)
    electrons = np.asarray((cells * mosfet.carrier_densities(state, scenarios.intrinsic)[0]).batch().to_array()).ravel()
    across = np.asarray((mosfet.mv.x & vertices.batch()).to_array()).ravel()
    return electrons[(across > left) & (across < right)].sum() / (right - left)


def test_channel_forms_above_threshold(switched):
    """As the gate ramps at a small drain voltage, the channel's electrons per area under the gate's middle rise
    linearly with the gate voltage past a threshold, at the oxide's capacitance per area, its permittivity over its
    thickness; below the threshold the drain's current is negligible."""
    topology, vertices, semiconductor, gates, curves, states, readings = switched
    ramp = np.arange(scenarios.n_ramp_frames)
    middle = np.array([sheet(topology, vertices, semiconductor, states[frame], -4.0, 4.0) for frame in ramp])
    capacitance = scenarios.oxide_permittivity / scenarios.oxide_thickness
    # The threshold, where the line through the ramp's last two frames meets nothing; past it by a few thermal volts,
    # the channel grows at the oxide's capacitance.
    threshold = gates[ramp[-1]] - middle[-1] * (gates[ramp[-1]] - gates[ramp[-2]]) / (middle[-1] - middle[-2])
    strong = ramp[gates[ramp] > threshold + 6]
    slopes = np.diff(middle[strong]) / np.diff(gates[strong])
    assert len(strong) > 2 and np.all(np.abs(slopes / capacitance - 1) < 0.2)
    # Well below the threshold, the drain's current against its current at the ramp's end.
    below = ramp[gates[ramp] < threshold - 10]
    assert len(below) > 0 and np.abs(readings[1, below]).max() < 1e-3 * readings[1, ramp[-1]]


def test_current_saturates(switched):
    """On each sweep of the drain, its current is linear in its voltage while that is small, and saturates once it is
    large: the current's slope at the largest drain voltages is less than a sixth of the channel's conductance at the
    smallest."""
    _, _, _, gates, curves, _, readings = switched
    voltages, currents = readings
    for curve in range(len(scenarios.gate_voltages)):
        on_curve = np.flatnonzero(curves == curve)
        order = on_curve[np.argsort(voltages[on_curve])]
        conductance = currents[order[:2]] / voltages[order[:2]]
        late = np.diff(currents[order[-3:]]) / np.diff(voltages[order[-3:]])
        assert abs(conductance[1] / conductance[0] - 1) < 0.15
        assert np.all(late < conductance[0] / 6)


def test_channel_pinches_off_near_drain(switched):
    """At the first gate voltage, the channel's electrons per area near the drain's end match those near the source's
    while the drain's voltage is small, and fall far below them once the current has saturated: the channel pinches
    off near the drain."""
    topology, vertices, semiconductor, gates, curves, states, readings = switched
    on_curve = np.flatnonzero(curves == 0)
    smallest, largest = on_curve[readings[0, on_curve].argmin()], on_curve[readings[0, on_curve].argmax()]
    ends = [[sheet(topology, vertices, semiconductor, states[frame], *window) for window in ((-11, -8), (8, 11))]
            for frame in (smallest, largest)]
    assert ends[0][1] > 0.8 * ends[0][0]
    assert ends[1][1] < 0.2 * ends[1][0]


solar_cell = scenarios.solar_cell


@pytest.fixture(scope="module")
def lit_cell():
    """The solar cell at its first spacing of the fingers as the light comes on and the bias rises to open circuit;
    and from its maximum power there, its fingers spreading over a few spacings, each at its maximum power."""
    topology, vertices, dopants, neutral, contacts, generation = scenarios.solar_period(scenarios.finger_spacings[0])
    biases, shares = scenarios.lighting_biases()
    maps = solar_cell.simplices(topology, vertices)
    lit = list(scenarios.solar_steady(topology, dopants, neutral, contacts, *maps, neutral, biases,
                                      shares * generation))
    powers = stack([scenarios.solar_power(topology, contacts, *maps, scenarios.finger_spacings[0], state)
                    for state in lit])
    best, bias = scenarios.peak(biases, powers)
    finger_spacings = np.geomspace(scenarios.finger_spacings[0], scenarios.finger_spacings[-1], 7)
    spread = list(scenarios.spreading(topology, dopants, neutral, contacts, generation, lit[best], bias,
                                      finger_spacings))
    return topology, vertices, contacts, generation, lit, finger_spacings, spread


def test_power_peaks_between_shade_and_resistance(lit_cell):
    """Further apart, the fingers shade less of the surface, so that the light makes more pairs per length of it, and
    the top layer costs the holes more voltage on their way to a finger, the spread of their quasi-Fermi level along
    the surface from the finger's edge to the middle of the gap growing a hundredfold over the sweep; the power each
    length of surface delivers at its maximum peaks between, well above its value at either end."""
    topology, vertices, contacts, generation, _, finger_spacings, spread = lit_cell
    cells = [~topology.incidence[-1, 0].mean(axis=0) * solar_cell.simplices(topology, vertices)[0]
             for vertices, _ in spread]
    made = np.array([float((cell * generation).sites.sum().to_array()) for cell in cells]) / finger_spacings
    assert np.all(np.diff(made) > 0)
    # The surface's vertices across, the last beneath a finger, and the one in the middle of the gap.
    places = scenarios.cell_places(finger_spacings[0])                         # [vertices, 2]
    surface = np.flatnonzero(places[:, 1] == 0)
    edge, middle = surface[len(contacts[0]) // 2 - 1], surface[len(surface) // 2]
    levels = np.array([np.asarray(solar_cell.bands(potentials, scenarios.gap)[3].batch().to_array()).ravel()
                       for _, potentials in spread])                          # [spacings, vertices]
    spreads = np.abs(levels[:, middle] - levels[:, edge])                     # [spacings]
    assert np.all(np.diff(spreads) > 0) and spreads[-1] > 100 * spreads[0]
    powers = np.array([float(scenarios.solar_power(topology, contacts, *solar_cell.simplices(topology, vertices),
                                                   finger_spacing, potentials).to_array())
                       for finger_spacing, (vertices, potentials) in zip(finger_spacings, spread)])
    best = powers.argmax()
    assert 0 < best < len(powers) - 1
    assert powers[0] < 0.9 * powers[best] and powers[-1] < 0.9 * powers[best]


def test_shadow_makes_no_pairs(lit_cell):
    """Beneath the fingers the light makes no pairs, and yet at short circuit holes rise to the junction there too,
    gathered sideways from the lit silicon on either side: the holes' upward flow just beneath the junction, in the
    middle of a finger's shadow, is more than half that in the middle of the gap."""
    topology, vertices, contacts, generation, lit, finger_spacings, _ = lit_cell
    _, _, _, shaded, _ = scenarios.period_grid(finger_spacings[0])
    assert np.all(np.asarray(generation.batch().to_array()).ravel()[shaded] == 0)
    mv = solar_cell.mv
    filled = mv.scalar(np.ones((topology.n_elements[-1], 1))).field()
    driving = solar_cell.simplices(topology, vertices)[1]
    short_circuit = lit[scenarios.n_light_frames]
    holes = solar_cell.flows(topology, driving, filled, short_circuit, scenarios.intrinsic, scenarios.mobilities)[1]
    rising = np.asarray((mv.y & holes.batch()).to_array()).ravel()           # [simplices]
    middles = np.asarray((stack((mv.x, mv.y)) & (topology.incidence[-1, 0].mean(axis=0) * vertices).batch()[..., None]
                          ).to_array()).reshape(-1, 2)                        # [simplices, 2]
    depth = scenarios.top_depth + 6
    beneath = np.abs(middles[:, 1] + depth) < 1.0
    in_shadow = beneath & (middles[:, 0] < scenarios.finger_width / 4)
    in_gap = beneath & (np.abs(middles[:, 0] - finger_spacings[0] / 2) < scenarios.finger_width / 4)
    assert rising[in_shadow].mean() > 0.5 * rising[in_gap].mean() > 0


def test_open_circuit_below_carnot(lit_cell):
    """At the end of the bias's rise the cell delivers about nothing, at open circuit, and its quasi-Fermi levels
    part everywhere by less than the gap times the Carnot factor the sun's temperature sets."""
    topology, vertices, contacts, _, lit, finger_spacings, _ = lit_cell
    maps = solar_cell.simplices(topology, vertices)
    currents = np.array([float(scenarios.cell_terminal(topology, contacts, *maps, state)[1].to_array())
                         for state in lit])
    assert abs(currents[-1]) < 0.1 * currents.max()
    bands = np.asarray(solar_cell.bands(lit[-1], scenarios.gap).batch().to_array()).reshape(4, -1)
    assert (bands[2] - bands[3]).max() < scenarios.gap * (1 - 1 / scenarios.sun)
