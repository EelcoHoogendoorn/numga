"""Device geometries, doping profiles, voltage schedules, and animation drivers configure the silicon p-n junction, the
planar MOSFET, and the front-contact solar cell."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from itertools import islice

import numpy as np

from examples import instantiate
from numga import Algebra, stack
from numga.sparse import SparseExtensor
from numga.topology import Topology

# One-dimensional bar spacetime and two-dimensional cross-section spacetime, each weighed by w0 and joined with null
# quantity directions for electrons, holes, heat, and light.
diode = instantiate("examples.thermodynamics.semiconductor_devices.core", Algebra("x+t-w0") * Algebra("n0p0q0l0"))
planar = instantiate("examples.thermodynamics.semiconductor_devices.core", Algebra("x+y+t-w0") * Algebra("n0p0q0l0"))
Spatial = diode.Plane.derive.within(Algebra("x+"))                            # Vector+
Upright = planar.Plane.derive.within(Algebra("x+y+"))                         # Vector+

# Room-temperature silicon parameters in Debye-length and thermal-voltage units (300 K, 1.12 eV band gap).
intrinsic, gap = 1e-6, 43.6
mobilities = np.array([1.0, 1 / 3])
permittivity = 1.0
lifetime = 2.0
rate_constant = 1 / lifetime
side_length, spacing = 10.0, 0.25
limit, tolerance = 6.0, 0.1
forming_dt, n_forming_frames = 0.7, 100
forward_bias, reverse_bias, n_sweep_frames = 17.5, -8.0, 100
generation_rate, n_lighting_frames = 1e-5, 15
sun = 5800 / 300
frame_duration_ms = 40
volt = 1 / 0.02585
energy_window, current_window = (-26.0, 60.0), (-1.35e-4, 1.35e-4)

# Planar MOSFET dimensions, well doping, oxide permittivity, grid grading, and gate/drain voltage schedule.
half_width, depth = 20.0, 20.0
well_edge, well_depth, well_spread = 12.0, 4.0, 0.5
oxide_thickness, gate_edge = 1.0, 13.0
well_doping, oxide_permittivity = 20.0, 1 / 3
surface_spacing, edge_spacing, growth, coarsest_across, coarsest_down = 0.1, 0.8, 0.6, 4.0, 5.0
small_drain, gate_voltages, largest_drain = 2.0, np.array([70.0, 85.0, 100.0]), 60.0
n_rise_frames, n_drain_frames, n_step_frames = 30, 25, 6
full_densities, drain_current_window = np.array([well_doping, 1.0]), (-0.8, 17.0)

# Solar-cell period dimensions, emitter and back-surface doping, carrier lifetime, optical absorption, and finger sweep.
cell_depth, top_depth, top_spread, finger_width = 160.0, 8.0, 0.5, 16.0
top_doping = 6.0
back_depth, back_doping = 4.0, 1000.0
cell_lifetime = 2e5
cell_rate_constant = 1 / cell_lifetime
surface_generation, absorption_length = 1.5e-4, 50.0
junction_spacing, junction_reach, finger_edge_spacing = 0.8, 12.0, 0.8
coarsest_cell_down, coarsest_finger = 12.0, 2.5
n_gap_columns = 16
finger_spacings = np.geomspace(40.0, 480.0, 50)
n_light_frames, highest_bias, n_bias_frames, bias_step = 12, 26.15, 40, 1.5


# --- plumbing -------------------------------------------------------------------------
def bar() -> tuple[Topology, diode.Point, np.ndarray, np.ndarray]:
    """Uniform spacing along the bar defines the one-dimensional edge chain, unit-weight vertex points (`Point-[V]`),
    step doping profile (`[V]`), and p-end and n-end vertex indices (`[2]`)."""
    n_vertices = int(round(2 * side_length / spacing))
    places = (np.arange(n_vertices) - (n_vertices - 1) / 2) * spacing
    vertices = (diode.mv(Spatial, places[:, None]) + diode.mv.w).dual().field()   # Point-[V]
    return Topology.chains([n_vertices]), vertices, np.sign(places), np.array([0, n_vertices - 1])


def neutral_potentials(doping: np.ndarray) -> np.ndarray:
    """Charge neutrality determines the equilibrium electron and hole chemical potentials (`[2, ...]`) at net `doping`
    (`[...]`)."""
    # Charge neutrality `n - p = 2 * intrinsic * sinh(mu_n) = doping` inverts via arcsinh.
    chemical = np.arcsinh(doping / (2 * intrinsic))                           # [...]
    return np.stack([chemical, -chemical])


def bias_sweep() -> np.ndarray:
    """Piecewise-linear interpolation sets a uniform-speed bias schedule (`[n_sweep_frames]`) from zero to forward
    bias, into reverse bias, and back to zero."""
    turns = np.array([0.0, forward_bias, reverse_bias, 0.0])
    reach = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(turns)))])
    return np.interp(np.linspace(0, reach[-1], n_sweep_frames), reach, turns)


def lit_bias_sweep() -> tuple[np.ndarray, np.ndarray]:
    """Short-circuit illumination turn-on followed by `bias_sweep` gives the bias and illumination schedules
    (`[n_lighting_frames + n_sweep_frames]` each)."""
    lit_fractions = np.linspace(1 / n_lighting_frames, 1, n_lighting_frames)  # [n_lighting_frames]
    return np.concatenate([0 * lit_fractions, bias_sweep()]), np.concatenate([lit_fractions, np.ones(n_sweep_frames)])


def bar_voltages(voltages: np.ndarray) -> np.ndarray:
    """Contact bias `voltages` (`[...]`) raise the p contact while the n contact stays grounded (`[..., 2]`)."""
    return np.stack([voltages, np.zeros_like(voltages)], axis=-1)


def graded_places(stop: float, fine_places: np.ndarray, finest: np.ndarray, coarsest: float) -> np.ndarray:
    """Distance-proportional step growth places coordinates monotonically from zero to `stop` with spacing `finest`
    near `fine_places` and at most `coarsest` far from them."""
    places = [0.0]
    while places[-1] < stop:
        position = places[-1]
        places.append(position + min(coarsest, (finest + growth * np.abs(position - fine_places)).min()))
    return np.array(places) * stop / places[-1]


def section() -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Graded rows and mirrored columns form the triangulated MOSFET cross-section, vertex coordinates (`[V, 2]`),
    oxide triangle mask (`[F]`), smooth net doping profile (`[V]`), and contact vertex index arrays."""
    half_columns = graded_places(half_width, np.array([well_edge, gate_edge]), np.array([edge_spacing, edge_spacing]),
                                 coarsest_across)
    columns = np.concatenate([-half_columns[:0:-1], half_columns])
    silicon_depths = graded_places(depth, np.array([0.0, well_depth]), np.array([surface_spacing, edge_spacing]),
                                   coarsest_down)
    rows = np.concatenate([-silicon_depths[::-1], [oxide_thickness]])
    topology = Topology.grid(len(rows), len(columns))
    V, E, F = topology.n_elements
    coordinates = np.stack(np.meshgrid(columns, rows), axis=-1).reshape(V, 2)   # [V, 2]
    x_coords, y_coords = coordinates.T
    oxide = (y_coords[topology.elements[-1]] > 0).any(axis=-1)                # [F]
    lateral = (1 + np.tanh((np.abs(x_coords) - well_edge) / well_spread)) / 2   # [V]
    vertical = (1 + np.tanh((np.minimum(y_coords, 0) + well_depth) / well_spread)) / 2   # [V]
    doping = well_doping * lateral * vertical - 1                             # [V]
    contacts = [np.flatnonzero((x_coords <= -gate_edge) & (y_coords >= 0)),
                np.flatnonzero((x_coords >= gate_edge) & (y_coords >= 0)),
                np.flatnonzero(y_coords == -depth),
                np.flatnonzero((np.abs(x_coords) < gate_edge) & (y_coords > 0))]
    return topology, coordinates, oxide, doping, contacts


def gate_drain_schedule() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Alternating gate steps and drain sweeps define the per-frame gate voltages, drain voltages, and output-curve
    indices (`[n_frames]` each, `-1` during gate transitions)."""
    n_curves = len(gate_voltages)
    starts = np.concatenate([[0.0], gate_voltages[:-1]])                      # [n_curves]
    n_gate_steps = np.concatenate([[n_rise_frames], np.full(n_curves - 1, n_step_frames)])   # [n_curves]
    ends = np.array([small_drain, largest_drain])
    gates, drains, curves = [], [], []
    for curve, (start, gate, n_frames) in enumerate(zip(starts, gate_voltages, n_gate_steps)):
        near, far = ends[curve % 2], ends[1 - curve % 2]
        on_curve = np.arange(n_frames) == n_frames - 1                        # [n_frames]
        gates += [np.linspace(start, gate, n_frames + 1)[1:], np.full(n_drain_frames, gate)]
        drains += [np.full(n_frames, near), np.linspace(near, far, n_drain_frames + 1)[1:]]
        curves += [np.where(on_curve, curve, -1), np.full(n_drain_frames, curve)]
    return np.concatenate(gates), np.concatenate(drains), np.concatenate(curves)


def transistor_voltages(contacts: list[np.ndarray], gates: np.ndarray, drains: np.ndarray) -> np.ndarray:
    """Gate and drain schedules (`[n_frames]` each) populate the held contact voltages (`[n_frames, V_held]`) across
    grounded source, biased drain, grounded body, and biased gate vertices."""
    n_frames = len(gates)
    source, drain, body, gate = contacts
    return np.concatenate([np.zeros((n_frames, len(source))), np.repeat(drains[:, None], len(drain), 1),
                           np.zeros((n_frames, len(body))), np.repeat(gates[:, None], len(gate), 1)], axis=-1)


def cell_places(finger_spacing: float) -> np.ndarray:
    """Graded junction rows and cosine-clustered gap columns position the vertex coordinates (`[V, 2]`) across one
    solar-cell period of width `finger_spacing`."""
    fine_depths = np.concatenate([np.arange(0.0, junction_reach, junction_spacing), [cell_depth - back_depth]])
    rows = -graded_places(cell_depth, fine_depths, np.full(len(fine_depths), junction_spacing),
                          coarsest_cell_down)[::-1]
    half = finger_width / 2
    finger_columns = half - graded_places(half, np.array([0.0]), np.array([finger_edge_spacing]),
                                          coarsest_finger)[::-1]
    steps = (1 - np.cos(np.pi * np.linspace(0, 1, n_gap_columns + 1))) / 2   # [n_gap_columns + 1]
    gap_columns = half + (finger_spacing - finger_width) * steps              # [n_gap_columns + 1]
    columns = np.concatenate([finger_columns, gap_columns[1:-1], finger_spacing - finger_columns[::-1]])
    return np.stack(np.meshgrid(columns, rows), axis=-1).reshape(-1, 2)


def period_grid(finger_spacing: float) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """One repeating solar-cell period at `finger_spacing` defines the grid topology, vertex coordinates (`[V, 2]`),
    net doping (`[V]`), finger-shadow mask (`[V]`), and front-finger and back-contact vertex indices."""
    coordinates = cell_places(finger_spacing)
    x_coords, y_coords = coordinates.T
    n_rows, n_columns = len(np.unique(y_coords)), len(np.unique(x_coords))
    topology = Topology.grid(n_rows, n_columns)
    V, E, F = topology.n_elements
    top = (1 + np.tanh((y_coords + top_depth) / top_spread)) / 2              # [V]
    back = (1 - np.tanh((y_coords + cell_depth - back_depth) / top_spread)) / 2   # [V]
    doping = 1 + back_doping * back - top_doping * top                        # [V]
    shadow_mask = (x_coords <= finger_width / 2) | (x_coords >= finger_spacing - finger_width / 2)   # [V]
    contacts = [np.flatnonzero(shadow_mask & (y_coords == 0)), np.flatnonzero(y_coords == -cell_depth)]
    return topology, coordinates, doping, shadow_mask, contacts


def lighting_biases() -> tuple[np.ndarray, np.ndarray]:
    """Short-circuit illumination turn-on followed by a sine-eased voltage rise gives the bias and illumination
    schedules (`[1 + n_light_frames + n_bias_frames]` each) up to open circuit."""
    lit_fractions = np.linspace(1 / n_light_frames, 1, n_light_frames)        # [n_light_frames]
    bias_rise = highest_bias * np.sin(np.pi / 2 * np.linspace(0, 1, n_bias_frames + 1)[1:])   # [n_bias_frames]
    biases = np.concatenate([[0.0], 0 * lit_fractions, bias_rise])
    return biases, np.concatenate([[0.0], lit_fractions, np.ones(n_bias_frames)])


def cell_voltages(contacts: list[np.ndarray], biases: planar.Scalar) -> planar.Scalar:
    """Front-finger `biases` (`[n_frames] Scalar+`) pair with a grounded back contact into held contact voltages."""
    fingers, back = contacts
    finger_mask = np.concatenate([np.ones(len(fingers)), np.zeros(len(back))])
    return biases[:, None] * finger_mask


def contact_partition(contacts: list[np.ndarray], n_vertices: int) -> tuple[np.ndarray, np.ndarray]:
    """Contact index arrays partition `n_vertices` into held contact vertices and complementary free vertices."""
    held_indices = np.concatenate(contacts)
    return held_indices, np.setdiff1d(np.arange(n_vertices), held_indices)


def power_peak(biases: planar.Scalar, powers: planar.Scalar) -> tuple[int, planar.Scalar]:
    """Three-point parabolic interpolation around the highest interior sample of `powers` (`[n_frames] Scalar+`)
    locates the maximum-power sample index and peak bias (`[] Scalar+`)."""
    best = int(powers[1:-1].argmax()) + 1
    low, middle, high = powers[best - 1], powers[best], powers[best + 1]
    step = (biases[best + 1] - biases[best - 1]) / 2
    return best, biases[best] + step * ((low - high) / (2 * (low - 2 * middle + high))).clip(-1.5, 1.5)


# --- scenes ---------------------------------------------------------------------------
def junction() -> tuple[Topology, diode.Point, diode.Point, np.ndarray, diode.Potentials, diode.Potentials]:
    """Neutral p-type and n-type halves define the bar topology, vertex points (`Point-[V]`), dopant charges
    (`Point-[V]`), end indices (`[2]`), initial slab potentials (`Potentials+[V]`), and wired end potentials
    (`Potentials+[2]`)."""
    mv = diode.mv
    topology, vertices, doping, ends = bar()
    V, E = topology.n_elements
    chemical = neutral_potentials(doping)                                     # [2, V]
    slab = (mv.n * chemical[0] + mv.p * chemical[1]).retyped(diode.Potentials).field()   # Potentials+[V]
    built_in = chemical[1, 0] + chemical[0, -1]
    wired_contacts = topology.reading(ends, 0) * slab + (mv.t * bar_voltages(-built_in)).field()   # Potentials+[2]
    return topology, vertices, (diode.time * doping).field(), ends, slab, wired_contacts


def silicon(topology: Topology) -> tuple[diode.Scalar, diode.Scalar]:
    """Uniform silicon fills every edge of `topology` with constant permittivity and unit semiconductor weight
    (`Scalar+[E]` each)."""
    V, E = topology.n_elements
    all_silicon = diode.mv.scalar(np.ones((E, 1))).field()                    # Scalar+[E]
    return permittivity * all_silicon, all_silicon


def junction_formation(topology: Topology, vertices: diode.Point, dopants: diode.Point, ends: np.ndarray,
                       slab: diode.Potentials, wired_contacts: diode.Potentials) -> Iterator[diode.Potentials]:
    """Yields the potential field (`Potentials+[V]`) as two neutral slabs relax into p-n junction equilibrium."""
    V, E = topology.n_elements
    volumes, gradient = diode.simplex_maps(topology, vertices, Spatial)
    permittivities, semiconductor = silicon(topology)
    held_indices, free_indices = contact_partition([ends], V)
    zero_generation = np.zeros(n_forming_frames) * diode.mv.scalar(np.ones((V, 1))).field()
    return diode.potential_frames(
        topology, volumes, gradient, permittivities, semiconductor, dopants, slab,
        topology.reading(held_indices, 0), topology.reading(free_indices, 0),
        np.ones(n_forming_frames) * wired_contacts, zero_generation, intrinsic, mobilities, rate_constant, forming_dt,
        limit, tolerance)


def biased_states(topology: Topology, vertices: diode.Point, dopants: diode.Point, ends: np.ndarray,
                  wired_contacts: diode.Potentials, equilibrium_state: diode.Potentials, biases: np.ndarray,
                  generations: np.ndarray) -> Iterator[diode.Potentials]:
    """Yields steady-state potential fields (`Potentials+[V]`) across `biases` (`[n_frames]`) and uniform optical
    `generations` (`[n_frames]`)."""
    V, E = topology.n_elements
    mv = diode.mv
    n_frames = len(biases)
    volumes, gradient = diode.simplex_maps(topology, vertices, Spatial)
    permittivities, semiconductor = silicon(topology)
    held_indices, free_indices = contact_partition([ends], V)
    contacts = wired_contacts + (mv.t * bar_voltages(biases)).field()         # [n_frames] Potentials+[2]
    generation_field = generations * mv.scalar(np.ones((V, 1))).field()       # [n_frames] Scalar+[V]
    return diode.potential_frames(
        topology, volumes, gradient, permittivities, semiconductor, dopants, equilibrium_state,
        topology.reading(held_indices, 0), topology.reading(free_indices, 0), contacts, generation_field, intrinsic,
        mobilities, rate_constant, np.inf, limit, tolerance)


def transistor() -> tuple[Topology, planar.Point, planar.Scalar, planar.Scalar, planar.Point, planar.Potentials,
                          list[np.ndarray]]:
    """Substrate and donor-well doping on `section` define the MOSFET triangulation, vertex points (`Point-[V]`),
    triangle permittivities and silicon mask (`Scalar+[F]` each), dopant charges (`Point-[V]`), flat-band reference
    potentials (`Potentials+[V]`), and contact index arrays."""
    mv = planar.mv
    topology, coordinates, oxide, doping, contacts = section()
    V, E, F = topology.n_elements
    vertices = (mv(Upright, coordinates) + mv.w).dual().field()               # Point-[V]
    permittivities = mv.scalar(np.where(oxide, oxide_permittivity, permittivity)[:, None]).field()   # Scalar+[F]
    semiconductor = mv.scalar(np.where(oxide, 0.0, 1.0)[:, None]).field()     # Scalar+[F]
    chemical = neutral_potentials(doping)                                     # [2, V]
    substrate = neutral_potentials(np.array(-1.0))[0]
    on_gate = np.isin(np.arange(V), contacts[3])                              # [V]
    electric = np.where(on_gate, substrate, chemical[0])                      # [V]
    neutral = (mv.t * electric + mv.n * np.where(on_gate, 0.0, chemical[0])
               + mv.p * np.where(on_gate, 0.0, chemical[1])).field()          # Potentials+[V]
    return topology, vertices, permittivities, semiconductor, (planar.time * doping).field(), neutral, contacts


def transistor_states(topology: Topology, vertices: planar.Point, permittivities: planar.Scalar,
                      semiconductor: planar.Scalar, dopants: planar.Point, neutral: planar.Potentials,
                      contacts: list[np.ndarray], gates: np.ndarray, drains: np.ndarray) -> Iterator[planar.Potentials]:
    """Yields steady-state MOSFET potential fields (`Potentials+[V]`) across per-frame `gates` and `drains` voltages
    (`[n_frames]` each)."""
    V, E, F = topology.n_elements
    mv = planar.mv
    n_frames = len(gates)
    areas, gradient = planar.simplex_maps(topology, vertices, Upright)
    held_indices, free_indices = contact_partition(contacts, V)
    held_reading = topology.reading(held_indices, 0)                          # Scalar+[V_held <- V]
    voltages = transistor_voltages(contacts, gates, drains)                   # [n_frames, V_held]
    contact_potentials = held_reading * neutral + (mv.t * voltages).field()   # [n_frames] Potentials+[V_held]
    zero_generation = np.zeros(n_frames) * mv.scalar(np.ones((V, 1))).field()   # [n_frames] Scalar+[V]
    states = planar.potential_frames(
        topology, areas, gradient, permittivities, semiconductor, dopants, neutral, held_reading,
        topology.reading(free_indices, 0), contact_potentials, zero_generation, intrinsic, mobilities, rate_constant,
        np.inf, limit, tolerance)
    return islice(states, 1, None)


def solar_period(finger_spacing: float) -> tuple[Topology, planar.Point, planar.Point, planar.Potentials,
                                                 list[np.ndarray], planar.Scalar]:
    """Emitter, base, and back-surface doping on `period_grid` define one solar-cell period at `finger_spacing`
    together with vertex points (`Point-[V]`), dopants (`Point-[V]`), neutral reference potentials (`Potentials+[V]`),
    contacts, and depth-attenuated optical pair generation (`Scalar+[V]`)."""
    mv = planar.mv
    topology, coordinates, doping, shadow_mask, contacts = period_grid(finger_spacing)
    V, E, F = topology.n_elements
    vertices = (mv(Upright, coordinates) + mv.w).dual().field()               # Point-[V]
    chemical = neutral_potentials(doping)                                     # [2, V]
    neutral = (mv.t * chemical[0] + mv.n * chemical[0] + mv.p * chemical[1]).field()   # Potentials+[V]
    depth_profile = surface_generation * ((mv.y & vertices) / absorption_length).exp()   # Scalar+[V]
    generation = mv.scalar(np.where(shadow_mask, 0.0, 1.0)[:, None]).field() * depth_profile   # Scalar+[V]
    return topology, vertices, (planar.time * doping).field(), neutral, contacts, generation


def solar_steady(topology: Topology, dopants: planar.Point, neutral: planar.Potentials, contacts: list[np.ndarray],
                 areas: planar.Scalar, gradient: SparseExtensor, potentials: planar.Potentials, biases: planar.Scalar,
                 generations: planar.Scalar) -> Iterator[planar.Potentials]:
    """Yields steady-state solar-cell potential fields (`Potentials+[V]`) across `biases` (`[n_frames] Scalar+`) and
    `generations` (`[n_frames] Scalar+[V]`)."""
    V, E, F = topology.n_elements
    mv = planar.mv
    all_silicon = mv.scalar(np.ones((F, 1))).field()                          # Scalar+[F]
    held_indices, free_indices = contact_partition(contacts, V)
    held_reading = topology.reading(held_indices, 0)                          # Scalar+[V_held <- V]
    voltages = cell_voltages(contacts, biases)                                # [n_frames, V_held] Scalar+
    held_potentials = held_reading * neutral + (mv.t * voltages).field()      # [n_frames] Potentials+[V_held]
    states = planar.potential_frames(
        topology, areas, gradient, permittivity * all_silicon, all_silicon, dopants, potentials, held_reading,
        topology.reading(free_indices, 0), held_potentials, generations, intrinsic, mobilities, cell_rate_constant,
        np.inf, limit, tolerance)
    return islice(states, 1, None)


def cell_terminal(topology: Topology, contacts: list[np.ndarray], areas: planar.Scalar, gradient: SparseExtensor,
                  potentials: planar.Potentials) -> tuple[planar.Scalar, planar.Scalar]:
    """Front-finger and back-contact readings give the solar cell's terminal voltage and delivered current
    (`[] Scalar+` each)."""
    V, E, F = topology.n_elements
    all_silicon = planar.mv.scalar(np.ones((F, 1))).field()                   # Scalar+[F]
    voltage, current = planar.terminal_readings(topology, areas, gradient, all_silicon, contacts[0], contacts[1],
                                                potentials, intrinsic, mobilities)
    return voltage, -current


def solar_power(topology: Topology, contacts: list[np.ndarray], areas: planar.Scalar, gradient: SparseExtensor,
                finger_spacing: float, potentials: planar.Potentials) -> planar.Scalar:
    """Terminal voltage times delivered current per period width `finger_spacing` gives the delivered power per unit
    front-surface length (`[] Scalar+`)."""
    voltage, current = cell_terminal(topology, contacts, areas, gradient, potentials)
    return voltage * current / finger_spacing


def finger_sweep(topology: Topology, dopants: planar.Point, neutral: planar.Potentials, contacts: list[np.ndarray],
                 generation: planar.Scalar, potentials: planar.Potentials, bias: planar.Scalar,
                 spacings: np.ndarray) -> Iterator[tuple[planar.Point, planar.Potentials]]:
    """Yields `(vertices, potentials)` (`Point-[V]`, `Potentials+[V]`) near the maximum-power point at each entry of
    `spacings` (`[n_spacings]`) by tracking the three-point parabolic power peak."""
    mv = planar.mv
    for finger_spacing in spacings:
        vertices = (mv(Upright, cell_places(finger_spacing)) + mv.w).dual().field()   # Point-[V]
        areas, gradient = planar.simplex_maps(topology, vertices, Upright)
        trial_biases = bias + bias_step * np.array([-1.0, 0.0, 1.0])          # [3] Scalar+
        lower, potentials, upper = solar_steady(topology, dopants, neutral, contacts, areas, gradient, potentials,
                                                trial_biases, np.ones(3) * generation)
        powers = stack([solar_power(topology, contacts, areas, gradient, finger_spacing, state)
                        for state in (lower, potentials, upper)])             # [3] Scalar+
        _, bias = power_peak(trial_biases, powers)
        yield vertices, potentials


def diode_animations() -> tuple[list[diode.Potentials], list[diode.Potentials], list[diode.Potentials]]:
    """Saves the p-n junction formation, dark I-V sweep, and illuminated I-V sweep animations and returns their states."""
    from examples.thermodynamics.semiconductor_devices import render
    from examples.util.animation import save_animation

    topology, vertices, dopants, ends, slab, wired_contacts = junction()
    volumes, gradient = diode.simplex_maps(topology, vertices, Spatial)
    _, semiconductor = silicon(topology)
    bands_of = partial(diode.band_energies, gap=gap)
    densities_of = partial(diode.carrier_densities, intrinsic=intrinsic)
    terminal_of = partial(diode.terminal_readings, topology, volumes, gradient, semiconductor, ends[:1], ends[1:],
                          intrinsic=intrinsic, mobilities=mobilities)
    formation_states = list(junction_formation(topology, vertices, dopants, ends, slab, wired_contacts))
    save_animation(render.formation_animation(vertices, bands_of, densities_of, intrinsic, energy_window,
                                            formation_states), "diode_forming", frame_duration_ms)
    biases = bias_sweep()
    dark_states = list(biased_states(topology, vertices, dopants, ends, wired_contacts, formation_states[-1], biases,
                                     np.zeros(n_sweep_frames)))
    voltage_window = (reverse_bias, forward_bias)
    save_animation(render.sweep_animation(vertices, bands_of, densities_of, intrinsic, terminal_of, energy_window,
                                        voltage_window, current_window, volt, render.dark_colour, [], dark_states),
                   "diode_dark", frame_duration_ms)
    lit_biases, shares = lit_bias_sweep()
    lit_states = list(biased_states(topology, vertices, dopants, ends, wired_contacts, dark_states[-1], lit_biases,
                                    generation_rate * shares))
    save_animation(render.sweep_animation(vertices, bands_of, densities_of, intrinsic, terminal_of, energy_window,
                                        voltage_window, current_window, volt, render.lit_colour, [dark_states],
                                        lit_states), "diode_lit", frame_duration_ms)
    return formation_states, dark_states, lit_states


def mosfet_animations() -> list[planar.Potentials]:
    """Saves the MOSFET cross-section still and switching animation and returns the solved steady states."""
    from examples.thermodynamics.semiconductor_devices import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, permittivities, semiconductor, dopants, neutral, contacts = transistor()
    save_figure(render.section_figure(topology, vertices, semiconductor, dopants, contacts, full_densities),
                "mosfet_section")
    gates, drains, curves = gate_drain_schedule()
    areas, gradient = planar.simplex_maps(topology, vertices, Upright)
    densities_of = partial(planar.carrier_densities, intrinsic=intrinsic)
    source, drain, _, _ = contacts
    terminal_of = partial(planar.terminal_readings, topology, areas, gradient, semiconductor, drain, source,
                          intrinsic=intrinsic, mobilities=mobilities)
    states = list(transistor_states(topology, vertices, permittivities, semiconductor, dopants, neutral, contacts,
                                    gates, drains))
    save_animation(render.transistor_animation(topology, vertices, semiconductor, contacts, densities_of, terminal_of,
                                             full_densities, curves, largest_drain, drain_current_window, volt, states),
                   "mosfet", frame_duration_ms)
    return states


def solar_animations() -> tuple[list[planar.Potentials], planar.Scalar]:
    """Saves the solar-cell geometry, lighting sweep, finger-spacing sweep, power curve, and optimal-period figures and
    returns the illuminated sweep states and spacing powers."""
    from examples.thermodynamics.semiconductor_devices import render
    from examples.util.animation import save_animation, save_figure

    mv = planar.mv
    topology, vertices, dopants, neutral, contacts, generation = solar_period(finger_spacings[0])
    V, E, F = topology.n_elements
    all_silicon = mv.scalar(np.ones((F, 1))).field()                          # Scalar+[F]
    save_figure(render.cell_figure(topology, vertices, dopants, contacts, generation, finger_spacings[0]), "solar_cell")
    reference_flow = surface_generation * absorption_length

    def holes_of(current_vertices: planar.Point, potentials: planar.Potentials) -> planar.Point:
        _, gradient = planar.simplex_maps(topology, current_vertices, Upright)
        return planar.carrier_flows(topology, gradient, all_silicon, potentials, intrinsic, mobilities)[1]

    def power_of(finger_spacing: float, current_vertices: planar.Point,
                 potentials: planar.Potentials) -> planar.Scalar:
        areas, gradient = planar.simplex_maps(topology, current_vertices, Upright)
        return solar_power(topology, contacts, areas, gradient, finger_spacing, potentials)

    biases, shares = lighting_biases()
    bias_extensor = mv.scalar(biases[:, None])                                # [n_frames] Scalar+
    areas, gradient = planar.simplex_maps(topology, vertices, Upright)
    lit_states = list(solar_steady(topology, dopants, neutral, contacts, areas, gradient, neutral, bias_extensor,
                                   shares * generation))
    dual_areas = topology.incidence[-1, 0].mean(axis=0).adjugate() * areas    # Scalar+[V]
    generated_current = (dual_areas * generation).sites.sum()                 # [] Scalar+
    terminal_of = partial(cell_terminal, topology, contacts, areas, gradient)
    save_animation(render.lighting_animation(topology, vertices, dopants, contacts, generation,
                                           partial(holes_of, vertices), terminal_of, reference_flow,
                                           finger_spacings[0], biases.max(), generated_current, volt, shares,
                                           lit_states), "solar_cell_lighting", frame_duration_ms)
    powers = stack([solar_power(topology, contacts, areas, gradient, finger_spacings[0], state)
                    for state in lit_states])
    best, best_bias = power_peak(bias_extensor, powers)
    unshaded_power = powers[best] * finger_spacings[0] / (finger_spacings[0] - finger_width)   # [] Scalar+
    spacing_states = list(finger_sweep(topology, dopants, neutral, contacts, generation, lit_states[best], best_bias,
                                       finger_spacings))
    save_animation(render.finger_sweep_animation(topology, dopants, contacts, generation, holes_of, power_of,
                                               reference_flow, finger_spacings, unshaded_power, spacing_states),
                   "solar_cell_spacing", frame_duration_ms)
    spacing_powers = stack([power_of(finger_spacing, *state)
                            for finger_spacing, state in zip(finger_spacings, spacing_states)])
    save_figure(render.power_figure(finger_spacings, spacing_powers), "solar_cell_power")
    optimum = int(spacing_powers.argmax())
    optimal_vertices, optimal_potentials = spacing_states[optimum]
    save_figure(render.period_figure(topology, optimal_vertices, dopants, contacts, generation,
                                   holes_of(optimal_vertices, optimal_potentials), reference_flow,
                                   finger_spacings[optimum]), "solar_cell_period")
    return lit_states, spacing_powers


def main() -> None:
    """Runs all three semiconductor device demonstrations, saves every animation and figure, and verifies physical
    invariants."""
    formation_states, dark_states, lit_states = diode_animations()
    mosfet_states = mosfet_animations()
    solar_lit_states, spacing_powers = solar_animations()

    # checks
    topology, vertices, _, ends, _, _ = junction()
    V, E = topology.n_elements
    volumes, gradient = diode.simplex_maps(topology, vertices, Spatial)
    _, semiconductor = silicon(topology)
    zero_field = diode.mv.scalar(np.zeros((V, 1))).field()                    # Scalar+[V]
    transport_s, recomb_s, _ = diode.entropy_production(
        topology, volumes, gradient, semiconductor, dark_states[len(dark_states) // 2], zero_field, intrinsic,
        mobilities, rate_constant, gap, sun)
    assert bool((transport_s.batch() >= -1e-12).all()) and bool((recomb_s.batch() >= -1e-12).all())
    lit_field = generation_rate * diode.mv.scalar(np.ones((V, 1))).field()    # Scalar+[V]
    _, _, absorb_s = diode.entropy_production(
        topology, volumes, gradient, semiconductor, lit_states[-1], lit_field, intrinsic, mobilities, rate_constant,
        gap, sun)
    assert bool((absorb_s.batch() >= -1e-12).all())
    assert len(mosfet_states) > 0 and len(solar_lit_states) > 0
    best = int(spacing_powers.argmax())
    assert 0 < best < len(finger_spacings) - 1
    assert bool(spacing_powers[0] < 0.9 * spacing_powers[best])
    assert bool(spacing_powers[-1] < 0.9 * spacing_powers[best])


if __name__ == "__main__":
    main()
