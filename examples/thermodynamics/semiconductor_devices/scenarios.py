"""A silicon p-n junction forming from two neutral slabs joined, a wire closing between their ends as they touch; its
current traced as a bias sweeps it forward and back into reverse; and the same sweep under light, once the light has
come on at no bias, the junction a solar cell. A MOSFET's cross-section, its gate ramping until a channel joins
source to drain, and its drain sweeping at a few gate voltages, its current saturating as the channel pinches off. And
a solar cell's cross-section between the fingers of its front contact: the light coming on and the bias rising to
open circuit, and the fingers spreading, the cell at its maximum power at each spacing, its power peaking where the
fingers' shade and the resistance of the thin top layer the holes run along to them balance."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from itertools import islice

import numpy as np

from numga import Algebra, stack
from numga.sparse import SparseExtensor
from examples import instantiate
from examples.util.topology import Topology

# The bar's line in spacetime, and the cross-section's plane, each joined with the electrons, the holes, the heat and
# the light; the solar cell's cross-section lies in the MOSFET's plane, and so shares its copy of the core.
diode = instantiate("examples.thermodynamics.semiconductor_devices.core", Algebra("x+t-") * Algebra("n0p0q0l0"))
mosfet = instantiate("examples.thermodynamics.semiconductor_devices.core", Algebra("x+y+t-") * Algebra("n0p0q0l0"))
solar_cell = instantiate("examples.thermodynamics.semiconductor_devices.core", Algebra("x+y+t-") * Algebra("n0p0q0l0"))
# The planes that read a place along the bar, and across and up the cross-section.
Along = diode.Plane.derive.within(Algebra("x+"))                              # Vector+
Upright = mosfet.Plane.derive.within(Algebra("x+y+"))                         # Vector+

# Silicon at room temperature: its intrinsic density, a millionth of the doping, the band gap in thermal units, 1.12 eV
# at 300 K, the electrons' and the holes' mobilities, and the permittivity.
intrinsic, gap = 1e-6, 43.6
mobilities = np.array([1.0, 1 / 3])
permittivity = 1.0
# The minority carriers' lifetime, one over the doping times the rate constant of recombination.
lifetime = 2.0
rate_constant = 1 / lifetime
# The bar: each side's length in Debye lengths, and its vertices' spacing.
side_length, spacing = 10.0, 0.25
# How far a Newton's step may change a chemical potential, in thermal units, and how little the last changes them.
limit, tolerance = 2.0, 1e-6
# The junction forming: the time between frames and the frames.
forming_dt, n_forming_frames = 0.7, 100
# The sweeps: the forward and the reverse bias, in thermal units, and the frames of the sweep from nothing forward,
# back into reverse and back to nothing, each frame a steady state.
forward_bias, reverse_bias, n_sweep_frames = 17.5, -8.0, 100
# The light: the pairs it makes per length and time, the frames over which it comes on, at no bias, and the sun's
# temperature, in units of the lattice's, 5800 K against 300 K.
generation_rate, n_lighting_frames = 1e-5, 15
sun = 5800 / 300
frame_duration_ms = 40
# The volt in thermal units, at 300 K; the band diagram's window of energy, in thermal units, and the current's, in
# units of the doping's charge times an electron's diffusion across a Debye length.
volt = 1 / 0.02585
energy_window, current_window = (-26.0, 60.0), (-1.35e-4, 1.35e-4)

# The MOSFET's cross-section, in Debye lengths of the substrate's doping: its half width and its depth below the oxide;
# the wells' inner edges' distance from the middle, their depth, and how far their edges spread; the oxide's thickness,
# and the gate's half length, reaching a little over the wells, the contacts on the wells beyond it.
half_width, depth = 20.0, 20.0
well_edge, well_depth, well_spread = 12.0, 4.0, 0.5
oxide_thickness, gate_edge = 1.0, 13.0
# The wells' donors, in units of the substrate's acceptors, and the oxide's permittivity, in silicon's.
well_doping, oxide_permittivity = 20.0, 1 / 3
# The mesh's spacing beneath the oxide, across the wells' edges and at the wells' bottoms; how fast it grows away from
# them, as a share of the distance, and how coarse it grows at most, across and down.
surface_spacing, edge_spacing, growth, coarsest_across, coarsest_down = 0.04, 0.4, 0.35, 2.0, 3.0
# The voltages, in thermal units: the drain's while the gate ramps, the gate's for each output curve, and the drain's
# at the sweeps' far end; the frames of the gate's ramp, of each sweep of the drain, and of each step of the gate.
small_drain, gate_voltages, largest_drain = 2.0, np.array([70.0, 85.0, 100.0]), 60.0
n_ramp_frames, n_drain_frames, n_step_frames = 30, 25, 6
# The cross-section's tints full at the wells' electrons and the substrate's holes; and the output curves' window of
# current, in units of the doping's charge times an electron's diffusion across a Debye length, per Debye length of the
# device's width.
full_densities, drain_current_window = np.array([well_doping, 1.0]), (-0.8, 17.0)

# The solar cell, in Debye lengths of its base's donors: the base's depth below the surface, the top layer's depth and
# how far its edge spreads, and a finger's width; the top layer's acceptors, in units of the base's donors.
cell_depth, top_depth, top_spread, finger_width = 160.0, 8.0, 0.5, 16.0
top_doping = 6.0
# The back's layer of more donors, which turns the holes back from the back contact: its depth above the back, and its
# donors, in units of the base's, so many that the holes in it are far fewer than in the base above it.
back_depth, back_doping = 4.0, 1000.0
# The holes' lifetime in the base, long enough for them to diffuse further than the base is deep.
cell_lifetime = 2e5
cell_rate_constant = 1 / cell_lifetime
# The light: the pairs it makes per volume and time just beneath the surface, and the depth over which that falls by a
# factor e.
surface_generation, absorption_length = 1.5e-4, 50.0
# The mesh: its spacing through the top layer and the junction beneath it, down to `junction_reach` below the surface,
# and at the fingers' edges; how coarse it grows at most down the base and under a finger; and the columns across each
# gap between two fingers, as many at every spacing of the fingers.
junction_spacing, junction_reach, finger_edge_spacing = 0.6, 15.0, 0.5
coarsest_cell_down, coarsest_finger = 8.0, 1.5
n_gap_columns = 24
# How little the last of Newton's steps changes the chemical potentials, for the solar cell.
cell_tolerance = 1e-3
# The fingers' spacings swept, from a few fingers' widths to three bases' depths, each the same share apart; at the
# first spacing, the frames over which the light comes on, the bias at open circuit, in thermal units, and the frames
# over which the bias rises to it; and how far apart the biases about each spacing's last maximum of power lie.
finger_spacings = np.geomspace(40.0, 480.0, 50)
n_light_frames, highest_bias, n_bias_frames, bias_step = 12, 26.15, 40, 1.5


# --- plumbing -------------------------------------------------------------------------
def bar() -> tuple[Topology, diode.Point, np.ndarray]:
    """The bar: its chain of edges, its vertices `Point-[vertices]`, and the doping at each `[vertices]`, the acceptors'
    negative charge on the p side, left, and the donors' positive one on the n side, right, in units of the doping."""
    n_vertices = int(round(2 * side_length / spacing))
    # The junction midway between two vertices.
    places = (np.arange(n_vertices) - (n_vertices - 1) / 2) * spacing
    vertices = diode.mv(Along, places[:, None]).dual().field()
    return Topology.chains([n_vertices]), vertices, np.sign(places)


def slabs(doping: np.ndarray) -> np.ndarray:
    """The chemical potentials of the electrons and the holes in neutral semiconductor `[carriers, ...]` of this doping
    `[...]`, its majority carriers balancing its dopants and its minority carriers in equilibrium with them: the
    logarithm of each density over the intrinsic one."""
    # In equilibrium the electrons and the holes are the intrinsic density times the exponentials of opposite chemical
    # potentials, and neutral, their difference, twice the intrinsic density times the sinh of the electrons', is the
    # doping.
    chemical = np.arcsinh(doping / (2 * intrinsic))                           # [...]
    return np.stack([chemical, -chemical])


def sweep() -> np.ndarray:
    """The bias at each frame of a sweep `[frames]`: from nothing forward, back through nothing into reverse, and back
    to nothing, at one speed."""
    turns = np.array([0.0, forward_bias, reverse_bias, 0.0])
    reach = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(turns)))])
    return np.interp(np.linspace(0, reach[-1], n_sweep_frames), reach, turns)


def graded(stop: float, fine_places: np.ndarray, finest: np.ndarray, coarsest: float) -> np.ndarray:
    """Places from nothing to `stop` `[places]`, as close as `finest` `[fine]` at each of the places `fine_places`
    `[fine]`, further apart away from them by `growth` of the distance, and never further than `coarsest`."""
    places = [0.0]
    while places[-1] < stop:
        here = places[-1]
        places.append(here + min(coarsest, (finest + growth * np.abs(here - fine_places)).min()))
    # Stretched to end on the stop.
    return np.array(places) * stop / places[-1]


def section() -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """The MOSFET's cross-section: a grid of triangles finest beneath the oxide and across the wells' edges, its
    columns mirrored about the middle, the silicon's rows below and the oxide's one row above; its vertices' places
    across and up `[vertices, 2]`; whether each triangle is oxide `[simplices]`; the net doping at each vertex
    `[vertices]`, the wells' donors less the substrate's acceptors, in the oxide the doping of the surface beneath; and
    the vertices of its contacts, the source, the drain, the body and the gate, `[contacts][held]`."""
    across = graded(half_width, np.array([well_edge, gate_edge]), np.array([edge_spacing, edge_spacing]),
                    coarsest_across)
    columns = np.concatenate([-across[:0:-1], across])                        # [columns]
    down = graded(depth, np.array([0.0, well_depth]), np.array([surface_spacing, edge_spacing]), coarsest_down)
    rows = np.concatenate([-down[::-1], [oxide_thickness]])                   # [rows]
    topology = Topology.grid(len(rows), len(columns))
    coordinates = np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)   # [vertices, 2]
    x, y = coordinates.T
    oxide = (y[topology.elements[-1]] > 0).any(axis=-1)                       # [simplices]
    # The wells, each a box with smooth edges reaching in from a side and down from the surface.
    lateral = (1 + np.tanh((np.abs(x) - well_edge) / well_spread)) / 2        # [vertices]
    vertical = (1 + np.tanh((np.minimum(y, 0) + well_depth) / well_spread)) / 2   # [vertices]
    doping = well_doping * lateral * vertical - 1                             # [vertices]
    # The source and the drain hold the surface and the oxide's row over the wells beyond the gate, the metal of their
    # contacts reaching down through the oxide; the body holds the bottom, and the gate the oxide's top between them.
    contacts = [np.flatnonzero((x <= -gate_edge) & (y >= 0)), np.flatnonzero((x >= gate_edge) & (y >= 0)),
                np.flatnonzero(y == -depth), np.flatnonzero((np.abs(x) < gate_edge) & (y > 0))]
    return topology, coordinates, oxide, doping, contacts


def schedule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The gate's and the drain's voltage at each frame `[frames]` each, in thermal units, and the output curve each
    frame lies on `[frames]`, minus one where it lies on none: the gate ramps to its first voltage at a small drain
    voltage; then the drain sweeps up, the gate steps to its next voltage, the drain sweeps back down, and so on."""
    # Each curve's gate ramps or steps from the one before, the first from nothing, while the drain waits at the end
    # of the sweep before, the first at the small voltage; then the drain sweeps to the other end.
    starts = np.concatenate([[0.0], gate_voltages[:-1]])                      # [curves]
    n_gate_frames = np.concatenate([[n_ramp_frames], np.full(len(gate_voltages) - 1, n_step_frames)])   # [curves]
    ends = np.array([small_drain, largest_drain])
    gates, drains, curves = [], [], []
    for curve, (start, gate, n_frames) in enumerate(zip(starts, gate_voltages, n_gate_frames)):
        near, far = ends[curve % 2], ends[1 - curve % 2]
        # The gate's last frame, at the curve's voltage, is the curve's first.
        on_curve = np.arange(n_frames) == n_frames - 1                        # [gate frames]
        gates += [np.linspace(start, gate, n_frames + 1)[1:], np.full(n_drain_frames, gate)]
        drains += [np.full(n_frames, near), np.linspace(near, far, n_drain_frames + 1)[1:]]
        curves += [np.where(on_curve, curve, -1), np.full(n_drain_frames, curve)]
    return np.concatenate(gates), np.concatenate(drains), np.concatenate(curves)


def cell_places(finger_spacing: float) -> np.ndarray:
    """The places across and up `[vertices, 2]` of one period of the solar cell, its fingers this far apart, from the
    middle of one finger to the middle of the next and from the back up to the surface, row by row: the rows as close as
    the junction's spacing through the top layer and the junction beneath it and further apart down the base, the same
    at every spacing; the columns closer towards each finger's edge beneath it, and as many across the gap at every
    spacing, closest at the fingers' edges, as a cosine's steps are."""
    # Fine through the top layer and the junction, and across the back layer's edge.
    fine = np.concatenate([np.arange(0.0, junction_reach, junction_spacing), [cell_depth - back_depth]])   # [fine]
    rows = -graded(cell_depth, fine, np.full(len(fine), junction_spacing), coarsest_cell_down)[::-1]
    half = finger_width / 2
    under = half - graded(half, np.array([0.0]), np.array([finger_edge_spacing]), coarsest_finger)[::-1]
    steps = (1 - np.cos(np.pi * np.linspace(0, 1, n_gap_columns + 1))) / 2   # [gap columns + 1]
    gap = half + (finger_spacing - finger_width) * steps                      # [gap columns + 1]
    columns = np.concatenate([under, gap[1:-1], finger_spacing - under[::-1]])   # [columns]
    return np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)


def period_grid(finger_spacing: float) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """One period of the solar cell, its fingers this far apart: a grid of triangles, the same at every spacing; its
    vertices' places across and up `[vertices, 2]`; the net doping at each vertex `[vertices]`, the donors of the base
    and of its back layer less the top layer's acceptors; whether each vertex lies in a finger's shadow `[vertices]`;
    and the vertices of its contacts, the fingers and the back, `[contacts][held]`."""
    coordinates = cell_places(finger_spacing)                                 # [vertices, 2]
    x, y = coordinates.T
    n_rows, n_columns = len(np.unique(y)), len(np.unique(x))
    topology = Topology.grid(n_rows, n_columns)
    # The top layer's acceptors reach down from the surface, and the back's donors up from the back, with smooth edges.
    top = (1 + np.tanh((y + top_depth) / top_spread)) / 2                      # [vertices]
    back = (1 - np.tanh((y + cell_depth - back_depth) / top_spread)) / 2       # [vertices]
    doping = 1 + back_doping * back - top_doping * top                        # [vertices]
    # The light comes straight down, and each finger shades the columns beneath it, from its middle to its edge.
    shaded = (x <= finger_width / 2) | (x >= finger_spacing - finger_width / 2)   # [vertices]
    contacts = [np.flatnonzero(shaded & (y == 0)), np.flatnonzero(y == -cell_depth)]
    return topology, coordinates, doping, shaded, contacts


def lighting_biases() -> tuple[np.ndarray, np.ndarray]:
    """The bias `[frames]` and the share of the light's full generation `[frames]` at each frame as the light comes on
    and the bias then rises: in the dark at no bias, the light coming on at no bias, and the bias rising to open
    circuit."""
    lit = np.linspace(1 / n_light_frames, 1, n_light_frames)                  # [lighting]
    # The bias rises ever more slowly towards the highest, where the current falls away fastest.
    rising = highest_bias * np.sin(np.pi / 2 * np.linspace(0, 1, n_bias_frames + 1)[1:])   # [bias frames]
    biases = np.concatenate([[0.0], 0 * lit, rising])                         # [frames]
    return biases, np.concatenate([[0.0], lit, np.ones(n_bias_frames)])


def peak(biases: np.ndarray, powers: solar_cell.Scalar) -> tuple[int, float]:
    """The frame of the highest of these powers `[frames] Scalar+` but the first and the last, one at each bias
    `[frames]`, and the bias of the peak of the parabola through it and its neighbours, the maximum power's, no further
    than a step beyond them: the power is concave in the bias, but nearly straight below its maximum, where a
    parabola's peak lies far off."""
    numbers = powers.to_array().reshape(-1)                                   # [frames]
    best = int(numbers[1:-1].argmax()) + 1
    low, middle, high = numbers[best - 1:best + 2]
    step = (biases[best + 1] - biases[best - 1]) / 2
    return best, biases[best] + step * np.clip((low - high) / (2 * (low - 2 * middle + high)), -1.5, 1.5)


# --- scenes ---------------------------------------------------------------------------
def junction() -> tuple[Topology, diode.Point, diode.Point, np.ndarray, diode.Potentials, diode.Potentials]:
    """The bar's chain, its vertices, its dopants `Point-[vertices]`, its two ends, the p end and the n end, the slabs'
    potentials at no electric potential `Potentials+[vertices]`, and the potentials at its ends once the wire has
    closed `Potentials+[ends]`."""
    mv = diode.mv
    topology, vertices, doping = bar()
    chemical = slabs(doping)                                                  # [carriers, vertices]
    slab = (0 * mv.t + mv.n * chemical[0] + mv.p * chemical[1]).field()       # Potentials+[vertices]
    # The wire brings the p end to the n end's Fermi level: its electric potential falls by the built-in voltage, the
    # sum of the majority carriers' chemical potentials at the two ends.
    built_in = chemical[1, 0] + chemical[0, -1]
    ends = np.array([0, topology.n_vertices - 1])
    wired = (mv.t * np.array([-built_in, 0.0]) + mv.n * chemical[0, ends] + mv.p * chemical[1, ends]).field()
    return topology, vertices, (diode.time * doping).field(), ends, slab, wired


def silicon(topology: Topology) -> tuple[diode.Scalar, diode.Scalar]:
    """The bar's simplices, all silicon: their permittivity, and their share of semiconductor, one each,
    `Scalar+[simplices]` each."""
    filled = diode.mv.scalar(np.ones((topology.n_elements[-1], 1))).field()   # Scalar+[simplices]
    return permittivity * filled, filled


def forming(topology: Topology, vertices: diode.Point, dopants: diode.Point, ends: np.ndarray,
            slab: diode.Potentials, wired: diode.Potentials) -> Iterator[diode.Potentials]:
    """The potentials at each frame as the junction forms, from the slabs as they touch, in the dark."""
    volumes, driving, outflow = diode.simplices(topology, vertices)
    permittivities, semiconductor = silicon(topology)
    dark = np.zeros(n_forming_frames) * diode.mv.scalar(np.ones((topology.n_vertices, 1))).field()
    return diode.run(topology, volumes, driving, outflow, permittivities, semiconductor, dopants, slab, ends,
                     np.ones(n_forming_frames) * wired, dark, intrinsic, mobilities, rate_constant, forming_dt, limit,
                     tolerance)


def biased(topology: Topology, vertices: diode.Point, dopants: diode.Point, ends: np.ndarray, wired: diode.Potentials,
           settled: diode.Potentials, biases: np.ndarray, generations: np.ndarray) -> Iterator[diode.Potentials]:
    """The steady potentials at each frame, from these settled ones, under each frame's bias `[frames]`, raising the p
    end's electric potential, and its light's generation `[frames]`, the same along the bar."""
    mv = diode.mv
    volumes, driving, outflow = diode.simplices(topology, vertices)
    permittivities, semiconductor = silicon(topology)
    contacts = wired + (mv.t * np.stack([biases, 0 * biases], axis=-1)).field()   # [frames] Potentials+[ends]
    lit = generations * mv.scalar(np.ones((topology.n_vertices, 1))).field()     # [frames] Scalar+[vertices]
    return diode.run(topology, volumes, driving, outflow, permittivities, semiconductor, dopants, settled, ends,
                     contacts, lit, intrinsic, mobilities, rate_constant, np.inf, limit, tolerance)


def transistor() -> tuple[Topology, mosfet.Point, mosfet.Scalar, mosfet.Scalar, mosfet.Point, mosfet.Potentials,
                          list[np.ndarray]]:
    """The MOSFET: its mesh, its vertices `Point-[vertices]`, its simplices' permittivities and share of
    semiconductor, one in the silicon and nothing in the oxide, `Scalar+[simplices]` each, its dopants
    `Point-[vertices]`, the potentials of neutral silicon at its doping with the Fermi level at nothing, the gate's
    vertices at the substrate's electric potential, `Potentials+[vertices]`, and the vertices of its contacts, the
    source, the drain, the body and the gate."""
    mv = mosfet.mv
    topology, coordinates, oxide, doping, contacts = section()
    vertices = mv(Upright, coordinates).dual().field()                       # Point-[vertices]
    permittivities = mv.scalar(np.where(oxide, oxide_permittivity, permittivity)[:, None]).field()
    semiconductor = mv.scalar(np.where(oxide, 0.0, 1.0)[:, None]).field()     # Scalar+[simplices]
    # With the Fermi level at nothing, each carrier's electrochemical potential is nothing, so that the electric
    # potential is the electrons' chemical potential. The gate's work function is the substrate's, its electric
    # potential the substrate's at no gate voltage, so that the bands lie flat; the carriers' potentials at its vertices
    # are idle.
    chemical = slabs(doping)                                                  # [carriers, vertices]
    substrate = slabs(np.array(-1.0))[0]
    gate = np.isin(np.arange(topology.n_vertices), contacts[3])               # [vertices]
    electric = np.where(gate, substrate, chemical[0])                         # [vertices]
    neutral = (mv.t * electric + mv.n * np.where(gate, 0.0, chemical[0])
               + mv.p * np.where(gate, 0.0, chemical[1])).field()            # Potentials+[vertices]
    return topology, vertices, permittivities, semiconductor, (mosfet.time * doping).field(), neutral, contacts


def switching(topology: Topology, vertices: mosfet.Point, permittivities: mosfet.Scalar, semiconductor: mosfet.Scalar,
              dopants: mosfet.Point, neutral: mosfet.Potentials, contacts: list[np.ndarray], gates: np.ndarray,
              drains: np.ndarray) -> Iterator[mosfet.Potentials]:
    """The steady potentials at each frame `Potentials+[vertices]`, the first found from neutral silicon, under each
    frame's gate and drain voltage `[frames]` each, raising the electric potential of the gate and of the drain, the
    source and the body at nothing."""
    mv = mosfet.mv
    volumes, driving, outflow = mosfet.simplices(topology, vertices)
    held = np.concatenate(contacts)                                           # [held]
    # Each held vertex's voltage at each frame: the drain's and the gate's raise its electric potential.
    source, drain, body, gate = contacts
    raised = np.concatenate([np.zeros((len(gates), len(source))), np.repeat(drains[:, None], len(drain), 1),
                             np.zeros((len(gates), len(body))), np.repeat(gates[:, None], len(gate), 1)],
                            axis=-1)                                          # [frames, held]
    holding = topology.reading(held, 0)                                       # Scalar+[held <- vertices]
    wired = holding * neutral + (mv.t * raised).field()                       # [frames] Potentials+[held]
    dark = np.zeros(len(gates)) * mv.scalar(np.ones((topology.n_vertices, 1))).field()   # [frames] Scalar+[vertices]
    states = mosfet.run(topology, volumes, driving, outflow, permittivities, semiconductor, dopants, neutral, held,
                        wired, dark, intrinsic, mobilities, rate_constant, np.inf, limit, tolerance)
    # The neutral silicon the run starts from is no steady state.
    return islice(states, 1, None)


def solar_period(finger_spacing: float) -> tuple[Topology, solar_cell.Point, solar_cell.Point, solar_cell.Potentials,
                                          list[np.ndarray], solar_cell.Scalar]:
    """The solar cell's period, its fingers this far apart: its mesh, its vertices `Point-[vertices]`, its dopants
    `Point-[vertices]`, the potentials of neutral silicon at its doping with the Fermi level at nothing
    `Potentials+[vertices]`, the vertices of its contacts, the fingers and the back, and the light's generation of
    pairs per volume and time at each vertex `Scalar+[vertices]`, fading with depth from the surface's and none in the
    fingers' shadow."""
    mv = solar_cell.mv
    topology, coordinates, doping, shaded, contacts = period_grid(finger_spacing)
    vertices = mv(Upright, coordinates).dual().field()                       # Point-[vertices]
    chemical = slabs(doping)                                                  # [carriers, vertices]
    neutral = (mv.t * chemical[0] + mv.n * chemical[0] + mv.p * chemical[1]).field()   # Potentials+[vertices]
    # The light falls by a factor e over each absorption length below the surface, where the plane y reads nothing.
    falling = surface_generation * ((mv.y & vertices) / absorption_length).exp()   # Scalar+[vertices]
    generation = mv.scalar(np.where(shaded, 0.0, 1.0)[:, None]).field() * falling   # Scalar+[vertices]
    return topology, vertices, (solar_cell.time * doping).field(), neutral, contacts, generation


def cell_contacts(topology: Topology, neutral: solar_cell.Potentials, contacts: list[np.ndarray],
                  biases: np.ndarray) -> solar_cell.Potentials:
    """The potentials the contacts hold at each frame's bias `[frames]` `[frames] Potentials+[held]`, the fingers' and
    the back's in turn: neutral silicon's at their doping, the fingers' electric potential raised by the bias."""
    finger, back = contacts
    raised = np.concatenate([np.repeat(biases[:, None], len(finger), 1), np.zeros((len(biases), len(back)))], axis=-1)
    return topology.reading(np.concatenate(contacts), 0) * neutral + (solar_cell.mv.t * raised).field()


def solar_steady(topology: Topology, dopants: solar_cell.Point, neutral: solar_cell.Potentials,
                 contacts: list[np.ndarray], volumes: solar_cell.Scalar, driving: SparseExtensor,
                 outflow: SparseExtensor, potentials: solar_cell.Potentials, biases: np.ndarray,
                 generations: solar_cell.Scalar) -> Iterator[solar_cell.Potentials]:
    """The solar cell's steady potentials at each frame `Potentials+[vertices]`, from these, with these volumes of its
    simplices, drives and outflows, under each frame's bias `[frames]` and its light's generation
    `[frames] Scalar+[vertices]`."""
    mv = solar_cell.mv
    filled = mv.scalar(np.ones((topology.n_elements[-1], 1))).field()        # Scalar+[simplices]
    states = solar_cell.run(topology, volumes, driving, outflow, permittivity * filled, filled, dopants, potentials,
                            np.concatenate(contacts), cell_contacts(topology, neutral, contacts, biases), generations,
                            intrinsic, mobilities, cell_rate_constant, np.inf, limit, cell_tolerance)
    return islice(states, 1, None)


def cell_terminal(topology: Topology, contacts: list[np.ndarray], volumes: solar_cell.Scalar,
                  driving: SparseExtensor, outflow: SparseExtensor,
                  potentials: solar_cell.Potentials) -> tuple[solar_cell.Scalar, solar_cell.Scalar]:
    """The solar cell's voltage at these potentials `[] Scalar+`, the rise of the electrons' quasi-Fermi level from the
    fingers to the back, and the current it delivers `[] Scalar+`, the charge its carriers' fluxes take out of the
    back's cells, with these volumes of its simplices, drives and outflows."""
    mv = solar_cell.mv
    filled = mv.scalar(np.ones((topology.n_elements[-1], 1))).field()        # Scalar+[simplices]
    voltage, current = solar_cell.terminal(topology, volumes, driving, outflow, filled, contacts[0], contacts[1],
                                           potentials, intrinsic, mobilities)
    return voltage, -current


def solar_power(topology: Topology, contacts: list[np.ndarray], volumes: solar_cell.Scalar, driving: SparseExtensor,
                outflow: SparseExtensor, finger_spacing: float,
                potentials: solar_cell.Potentials) -> solar_cell.Scalar:
    """The power the solar cell delivers per length of its surface at these potentials `[] Scalar+`, its fingers this
    far apart, with these volumes of its simplices, drives and outflows: its voltage times the current it delivers,
    over the period."""
    voltage, current = cell_terminal(topology, contacts, volumes, driving, outflow, potentials)
    return voltage * current / finger_spacing


def spreading(topology: Topology, dopants: solar_cell.Point, neutral: solar_cell.Potentials,
              contacts: list[np.ndarray], generation: solar_cell.Scalar, potentials: solar_cell.Potentials,
              bias: float, finger_spacings: np.ndarray) -> Iterator[tuple[solar_cell.Point, solar_cell.Potentials]]:
    """The solar cell at its maximum power at each of these spacings of its fingers in turn, its vertices and its
    steady potentials `Point-[vertices]`, `Potentials+[vertices]`, from these near the first spacing's maximum at this
    bias: at each spacing its steady states at the last maximum's bias and a step either side of it, the middle one
    the spacing's own, and the peak of the parabola through their powers the next spacing's bias."""
    mv = solar_cell.mv
    for finger_spacing in finger_spacings:
        vertices = mv(Upright, cell_places(finger_spacing)).dual().field()   # Point-[vertices]
        maps = solar_cell.simplices(topology, vertices)
        biases = bias + bias_step * np.array([-1.0, 0.0, 1.0])               # [trials]
        lower, potentials, upper = solar_steady(topology, dopants, neutral, contacts, *maps, potentials, biases,
                                                np.ones(3) * generation)
        powers = stack([solar_power(topology, contacts, *maps, finger_spacing, state)
                        for state in (lower, potentials, upper)])             # [trials] Scalar+
        _, bias = peak(biases, powers)
        yield vertices, potentials


def solar_animations() -> None:
    """The solar cell: its window of periods at the first spacing of its fingers; the light coming on and the bias
    rising to open circuit there; its fingers spreading from there, each spacing at its maximum power; its power
    against the spacing; and its period at the spacing of the most power."""
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.semiconductor_devices import render

    mv = solar_cell.mv
    topology, vertices, dopants, neutral, contacts, generation = solar_period(finger_spacings[0])
    filled = mv.scalar(np.ones((topology.n_elements[-1], 1))).field()        # Scalar+[simplices]
    save_figure(render.draw_cell(topology, vertices, dopants, contacts, generation, finger_spacings[0]), "solar_cell")
    # The streaks' scale, the holes' flow beneath a surface where the light makes all its pairs above.
    reference_flow = surface_generation * absorption_length

    def holes_of(vertices: solar_cell.Point, potentials: solar_cell.Potentials) -> solar_cell.Point:
        return solar_cell.flows(topology, solar_cell.simplices(topology, vertices)[1], filled, potentials, intrinsic,
                                mobilities)[1]

    def power_of(finger_spacing: float, vertices: solar_cell.Point,
                 potentials: solar_cell.Potentials) -> solar_cell.Scalar:
        return solar_power(topology, contacts, *solar_cell.simplices(topology, vertices), finger_spacing, potentials)

    biases, shares = lighting_biases()
    maps = solar_cell.simplices(topology, vertices)
    lit = list(solar_steady(topology, dopants, neutral, contacts, *maps, neutral, biases, shares * generation))
    # The current's scale, the pairs the light makes in the period each time, which the junction gathers nearly all of.
    cells = ~topology.incidence[-1, 0].mean(axis=0) * maps[0]                 # Scalar+[vertices]
    made = (cells * generation).sites.sum()                                   # [] Scalar+
    terminal_of = partial(cell_terminal, topology, contacts, *maps)
    save_animation(render.animate_lighting(topology, vertices, dopants, contacts, generation,
                                           partial(holes_of, vertices), terminal_of, reference_flow,
                                           finger_spacings[0], biases.max(), made, volt, shares, lit),
                   "solar_cell_lighting", frame_duration_ms)
    powers = stack([solar_power(topology, contacts, *maps, finger_spacings[0], state) for state in lit])
    best, bias = peak(biases, powers)
    # The most the first spacing's surface would deliver unshaded, its power per length of the surface the light falls
    # on, which the power chart reaches a little past.
    unshaded = powers[best] * finger_spacings[0] / (finger_spacings[0] - finger_width)   # [] Scalar+
    spread = list(spreading(topology, dopants, neutral, contacts, generation, lit[best], bias, finger_spacings))
    save_animation(render.animate_spreading(topology, dopants, contacts, generation, holes_of, power_of, reference_flow,
                                            finger_spacings, unshaded, spread),
                   "solar_cell_spacing", frame_duration_ms)
    spread_powers = stack([power_of(finger_spacing, *state) for finger_spacing, state in zip(finger_spacings, spread)])
    save_figure(render.draw_power(finger_spacings, spread_powers), "solar_cell_power")
    optimum = spread_powers.argmax()
    vertices, potentials = spread[optimum]
    save_figure(render.draw_period(topology, vertices, dopants, contacts, generation, holes_of(vertices, potentials),
                                   reference_flow, finger_spacings[optimum]), "solar_cell_period")


def diode_animations() -> None:
    """The bar's three animations: the junction forming, and the bias's sweep in the dark and under light."""
    from examples.util.animation import save_animation
    from examples.thermodynamics.semiconductor_devices import render

    topology, vertices, dopants, ends, slab, wired = junction()
    volumes, driving, outflow = diode.simplices(topology, vertices)
    _, semiconductor = silicon(topology)
    # The band diagram's energies, the carriers' densities, and the voltage and current across the bar, as functions
    # of the potentials.
    bands_of = partial(diode.bands, gap=gap)
    densities_of = partial(diode.carrier_densities, intrinsic=intrinsic)
    terminal_of = partial(diode.terminal, topology, volumes, driving, outflow, semiconductor, ends[:1], ends[1:],
                          intrinsic=intrinsic, mobilities=mobilities)
    formed = list(forming(topology, vertices, dopants, ends, slab, wired))
    save_animation(render.animate_forming(vertices, bands_of, densities_of, intrinsic, energy_window, formed),
                   "diode_forming", frame_duration_ms)
    biases = sweep()
    dark = list(biased(topology, vertices, dopants, ends, wired, formed[-1], biases, np.zeros(n_sweep_frames)))
    voltage_window = (reverse_bias, forward_bias)
    save_animation(render.animate_sweep(vertices, bands_of, densities_of, intrinsic, terminal_of, energy_window,
                                        voltage_window, current_window, volt, render.dark_colour, [], dark),
                   "diode_dark", frame_duration_ms)
    # The light comes on at no bias, and the bias sweeps again under it.
    lighting = np.linspace(1 / n_lighting_frames, 1, n_lighting_frames)
    generations = generation_rate * np.concatenate([lighting, np.ones(n_sweep_frames)])
    lit = biased(topology, vertices, dopants, ends, wired, dark[-1], np.concatenate([0 * lighting, biases]),
                 generations)
    save_animation(render.animate_sweep(vertices, bands_of, densities_of, intrinsic, terminal_of, energy_window,
                                        voltage_window, current_window, volt, render.lit_colour, [dark], lit),
                   "diode_lit", frame_duration_ms)


def mosfet_animations() -> None:
    """The MOSFET's cross-section, and its animation: the gate ramping and the drain sweeping, the output curves
    traced beside it."""
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.semiconductor_devices import render

    topology, vertices, permittivities, semiconductor, dopants, neutral, contacts = transistor()
    save_figure(render.draw_section(topology, vertices, semiconductor, dopants, contacts, full_densities),
                "mosfet_section")
    gates, drains, curves = schedule()
    volumes, driving, outflow = mosfet.simplices(topology, vertices)
    # The carriers' densities, and the drain's voltage and current, as functions of the potentials.
    densities_of = partial(mosfet.carrier_densities, intrinsic=intrinsic)
    source, drain, _, _ = contacts
    terminal_of = partial(mosfet.terminal, topology, volumes, driving, outflow, semiconductor, drain, source,
                          intrinsic=intrinsic, mobilities=mobilities)
    states = switching(topology, vertices, permittivities, semiconductor, dopants, neutral, contacts, gates, drains)
    save_animation(render.animate_switching(topology, vertices, semiconductor, contacts, densities_of, terminal_of,
                                            full_densities, curves, largest_drain, drain_current_window, volt, states),
                   "mosfet", frame_duration_ms)


def main() -> None:
    diode_animations()
    mosfet_animations()
    solar_animations()


if __name__ == "__main__":
    main()
