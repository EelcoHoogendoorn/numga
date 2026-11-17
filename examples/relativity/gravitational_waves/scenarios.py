"""Scenarios for the closed universe seen from inside its tetrahedral triangulation and from outside below its Hopf circles."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import product

import numpy as np
from numga import Algebra, stack
from numga.topology import Topology

from examples import instantiate
from examples.relativity.gravitational_waves import core, numerical
from examples.util import shapes

mv = core.mv
sphere_mesher = instantiate("examples.util.relaxed_sphere", core.ga)

# The 600-cell placed by a turn that keeps its vertices off the origin and its antipode.
cell_placement = (mv.xw * 0.21 + mv.yw * 0.13 + mv.zw * 0.08).exp()            # [] Motor+

n_guardrail_sides = 10
eye = (mv.xw * 0.37 + mv.yw * 0.11 + mv.zw * 0.23).exp()                        # [] Motor+
wave_swing = 0.12
squash_lags = np.zeros(1)                                                        # [1]
n_wave_frames = 75

# Three tetrahedral crystals ahead of the eye, with soft, tuned, and stiff springs.
crystal_distance, crystal_spacing, crystal_half_size = 0.55, 0.26, 0.05
tunings = np.array([1 / 16, 1, 16])                                             # [3]
critical_share = 0.25
crystal_step = 0.01
n_steps_per_swing = 40
n_settling_swings = 8
# The crystals' turning in the exact quadrupole: rates in the planes of space at the origin, sampled every few steps.
Turn = core.Rate.derive.within(Algebra("x+y+z+"))                               # Bivector+
n_steps_a_sample = 2
n_spin_periods, n_spun_periods, n_late_periods = 15, 8, 4
n_spin_frames_a_period = 20
n_trial_periods, n_trial_frames_a_period = 40, 8
# Trials: shares of the page's swing at a quarter of critical damping, then less damping at a third of the swing; and
# the crystals turned diagonal to the wave's axes.
trial_swing_shares = np.array([1 / 2, 2 / 5, 1 / 3, 1 / 3, 1 / 3])                # [5]
trial_critical_shares = np.array([1 / 4, 1 / 4, 1 / 4, 1 / 8, 1 / 16])          # [5]
diagonal_swing_shares = np.array([1, 1 / 2, 1 / 3])                              # [3]
diagonal_critical_shares = np.full(3, 1 / 4)                                     # [3]

# Relaxed irregular triangulation of the three-sphere under the binary icosahedral group.
n_seed_orbits = 5
orbit_seed = 7
n_relax_steps = 60
relax_pull = 0.5

duration_ms = 40
n_quadrupole_colours = 96
n_flight_frames = 90
n_pan_frames = 60
pan_distance = 1.02
pan_roll = 0.36

n_icosphere_levels = 3
n_circle_samples = 7
swing = 0.085
eighth = (mv.xy * (-np.pi / 8)).exp()                                            # [] Motor+
quadrupole_lags = np.array([0, np.pi / 2])                                       # [2]

gauge_share = 0.4
gauge_tuning = 1.4
gauge_damping = 1.0
n_arc_points = 12
# Trials of the triangle gauges from rest in the travelling quadrupole: shear tunings to the wave's frequency and
# shares of critical damping, one run each, the gauges tuned to the wave beside the gauges as set.
n_gauge_trial_periods = 60
n_gauge_samples_a_period = 20
n_gauge_frames_a_period = 5
gauge_trial_tunings = np.array([1.0, gauge_tuning])                             # [n_trials]
gauge_trial_shares = np.array([0.4, gauge_damping])                             # [n_trials]

light_source, light_target, light_spreading = mv.z.dual(), mv.y.dual(), mv.xy   # [] Point-, [] Point-, [] Rate+
n_rays = 7
light_spread = np.radians(18)
light_step = 0.02
n_light_steps = int(np.pi / 2 / light_step)


# --- math -----------------------------------------------------------------------------
def six_hundred_cell() -> tuple[Topology, core.Point]:
    """Centres the 600-cell's first vertex at the origin and places the triangulation by `cell_placement`."""
    topology, coordinates = shapes.cell_600()
    vertices = mv.vector(coordinates).dual()                                    # [V] Point-
    centred = core.frame(vertices[0]).inverse() >> vertices                     # [V] Point-
    return topology, cell_placement >> centred.field()


def circle_turn(vertices: core.Point, n_sides: int) -> core.Motor:
    """Finds the Hopf-circle rotor `[] Motor+` carrying the 600-cell's vertices onto themselves in `n_sides` steps."""
    unplaced = (cell_placement.inverse() >> vertices).batch()                   # [V] Point-
    # The vertex whose cosine from the origin matches one side of a great `n_sides`-gon generates the circle turn.
    target = unplaced[((core.equator & unplaced) - np.cos(2 * np.pi / n_sides)).abs().argmin()]   # [] Point-
    return cell_placement * core.frame(target) * cell_placement.inverse()


def guardrails(vertices: core.Point, n_sides: int) -> np.ndarray:
    """Groups the 600-cell's vertices into disjoint great `n_sides`-gon orbits of vertex indices."""
    polygons = numerical.vertex_orbits(vertices, circle_turn(vertices, n_sides)[None], n_sides)[0]   # [n_sides, V]
    _, firsts = np.unique(polygons.min(axis=0), return_index=True)
    return polygons[:, firsts].T


def great_circles(topology: Topology, vertices: core.Point) -> np.ndarray:
    """Traces the seventy-two great decagons of vertex indices running along the 600-cell's edges."""
    tails, heads = topology.incidence[1, 0] * vertices                          # Point-[E], Point-[E]
    turns = (core.frame(heads) * core.frame(tails).inverse()).batch()           # [E] Motor+
    tail_indices = topology.elements[1][:, 0]                                   # [E]
    circles = numerical.vertex_orbits(vertices, turns, n_guardrail_sides)[np.arange(len(tail_indices)), :, tail_indices]   # [E, n_guardrail_sides]
    _, firsts = np.unique(np.sort(circles, axis=-1), axis=0, return_index=True)
    return circles[firsts]


def crystal_places() -> core.Motor:
    """Places three crystal frames `[3] Motor+` side by side ahead of `eye`."""
    axis_rates = core.moving(stack((mv.x, mv.y, mv.z)).dual())                  # [3] Rate+
    offsets = crystal_spacing * np.arange(-1, 2)                                # [3]
    forward = (axis_rates[2] * (-crystal_distance / 2)).exp()                   # [] Motor+
    side_shifts = (axis_rates[0] * (-offsets / 2)).exp()                        # [3] Motor+
    return eye * forward * side_shifts


def diagonal_places() -> core.Motor:
    """Turns the crystal frames `[3] Motor+` at their places so that their cube's axes lie diagonal, an eighth turn in
    the wave's plane, to the axes the exact quadrupole stretches and squeezes: the wave then shears only the crystals'
    triply degenerate mode."""
    return core.frame(crystal_places() >> core.origin) * eighth


def crystals(frames: core.Motor) -> tuple[Topology, core.Point]:
    """Places a tetrahedral crystal of four bonded atoms `Point-[V]` around each motor in `frames`."""
    axis_rates = core.moving(stack((mv.x, mv.y, mv.z)).dual())                  # [3] Rate+
    cube = np.array(list(product([-1, 1], repeat=3)))                           # [8, 3]
    corners = cube[cube.prod(axis=-1) > 0]                                      # [4, 3]
    corner_moves = ((axis_rates * corners).sum(axis=-1) * (-crystal_half_size / 2)).exp()   # [4] Motor+
    atoms = ((frames[:, None] * corner_moves[None]) >> core.origin).reshape(-1)   # [V] Point-
    n_atoms = len(corners) * frames.shape[0]
    return Topology(n_atoms, np.arange(n_atoms).reshape(-1, len(corners))), atoms.field()


def squash(cell_topology: Topology, cell_vertices: core.Point, stretched: core.Plane,
           squeezed: core.Plane) -> core.Scalar:
    """Quadrupole stretching along `stretched` and squeezing along `squeezed` sets initial squared edge-length perturbations `[...] Scalar+[E]`."""
    steps = core.round_steps(cell_topology, cell_vertices)                      # Point-[E]
    chords = cell_topology.d[0] * cell_vertices                                 # Point-[E]
    shares = ((stretched & steps).squared() - (squeezed & steps).squared()) / (core.polar(steps) & steps)   # [...] Scalar+[E]
    return 2 * chords.scalar_norm_squared() * shares


def stepped_history(cell_topology: Topology, cell_vertices: core.Point, patterns: core.Scalar, lags: np.ndarray,
                    swing: float, reach: float) -> tuple[core.Waves, float]:
    """Integrates Regge gravitational waves from `patterns` over a full settling and ray-lookback span."""
    stiffness, masses, onto_waves = core.gravitational_waves(cell_topology, cell_vertices)
    wave_patterns = onto_waves(patterns)                                        # [...] Scalar+[E]
    # Rayleigh quotient of stiffness force over kinetic momentum sets the lattice wave frequency squared.
    squared = ((wave_patterns & stiffness(wave_patterns)).sites.sum().sum(axis=0)
               / (wave_patterns & masses(wave_patterns)).sites.sum().sum(axis=0))   # [] Scalar+
    frequency = np.sqrt(numerical.scalar_value(squared))
    period = 2 * np.pi / frequency
    step = period / n_steps_per_swing
    first = -(n_settling_swings + 1) * period - np.pi
    n_levels = int(np.ceil((period + reach - first) / step)) + 2
    phases = frequency * first - lags                                           # [...]
    strains = swing * (np.cos(phases) * wave_patterns).sum(axis=0)              # Scalar+[E]
    change_rates = -swing * frequency * (np.sin(phases) * wave_patterns).sum(axis=0)   # Scalar+[E]
    leapfrog = core.waves_stepped(stiffness, masses, onto_waves, strains, change_rates, step)
    levels = [state for _, state in zip(range(n_levels), leapfrog)]
    return core.Waves(cell_topology, cell_vertices, [changes for changes, _ in levels],
                      [change_rates for _, change_rates in levels], first, step), frequency


def crystal_states(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, frequency: float, bonds: Topology,
                   atoms: core.Point, times: np.ndarray) -> Iterator[tuple[core.Point, core.Scalar]]:
    """Steps soft, tuned, and stiff tetrahedral crystals through `waves` across `times` after `n_settling_swings`."""
    V, E, F, T = bonds.n_elements
    stiffnesses = frequency ** 2 / 2 * tunings                                  # [3]
    dampings = critical_share * np.sqrt(2 * stiffnesses)                        # [3]
    round_steps = core.round_steps(bonds, atoms)                                # Point-[E]
    rest = round_steps.norm()                                                   # Scalar+[E]
    n_bonds_each = E // len(tunings)
    stiffness = mv.scalar(np.repeat(stiffnesses, n_bonds_each)[:, None]).field()   # Scalar+[E]
    damping = mv.scalar(np.repeat(dampings, n_bonds_each)[:, None]).field()     # Scalar+[E]
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    at_rest = mv(core.Through, np.zeros((V, 3))).dual().field()                 # Point-[V]
    return core.massive(pick, waves, bonds, rest, stiffness, damping, atoms, at_rest, start, crystal_step, times)


def quadrupole_crystals(frames: core.Motor, swing: float, share: float, turns: Turn,
                        n_periods: int) -> tuple[np.ndarray, core.Point, core.Scalar]:
    """Steps the soft, tuned and stiff crystals placed by `frames` `[3] Motor+`, unstrained and turning rigidly at
    `turns` `[3] Bivector+`, through the exact quadrupole of `swing`, damped at `share` of critical, for `n_periods`;
    gives the sample times, each crystal's atoms read in the frame at its middle as proper offsets from it
    `[n_samples, 3, 4] Point-`, and the bond strains `[n_samples] Scalar+[E]`."""
    bonds, atoms = crystals(frames)
    V, E, F, T = bonds.n_elements
    # Each atom starts with the velocity of its step from the crystal's middle under the turn, carried to its own frame.
    places = atoms.batch().reshape(T, -1)                                       # [3, 4] Point-
    middles = core.frame(places.sum(axis=-1).normalized())[:, None]             # [3, 1] Motor+
    turned = (middles.inverse() >> places).cast(core.Step).commutator(turns[:, None])   # [3, 4] Point-
    velocities = ((core.frame(places).inverse() * middles) >> turned).cast(core.Step).reshape(-1).field()   # Point-[V]
    wave = core.Quadrupole(V, swing)
    stiffnesses = core.quadrupole_frequency ** 2 / 2 * tunings                  # [3]
    dampings = share * np.sqrt(2 * stiffnesses)                                 # [3]
    rest = core.round_steps(bonds, atoms).norm()                                # Scalar+[E]
    stiffness = mv.scalar(np.repeat(stiffnesses, E // T)[:, None]).field()      # Scalar+[E]
    damping = mv.scalar(np.repeat(dampings, E // T)[:, None]).field()           # Scalar+[E]
    period = 2 * np.pi / core.quadrupole_frequency
    sample_step = n_steps_a_sample * crystal_step
    times = sample_step * np.arange(int(n_periods * period / sample_step) + 1)  # [n_samples]
    states = core.massive(lambda places: None, wave, bonds, rest, stiffness, damping, atoms, velocities, 0.0,
                          crystal_step, times)
    offsets, strains = [], []
    for time, (positions, bond_strains) in zip(times, states):
        places = positions.batch().reshape(T, -1)                               # [3, 4] Point-
        middles = places.sum(axis=-1).normalized()                              # [3] Point-
        steps = (core.frame(middles)[:, None].inverse() >> places).cast(core.Step)   # [3, 4] Point-
        # The proper offset adds half the wave's stretch to the step, as the springs read their bonds.
        strain, _, _ = wave.strains(time)
        proper = steps + core.translating(strain(core.Step))(steps) / 2         # [3, 4] Point-
        offsets.append(proper - proper.sum(axis=-1)[:, None] / 4)
        strains.append(bond_strains.batch())
    return times, stack(offsets), stack(strains)


def turning_rates(offsets: core.Point, sample_step: float) -> Turn:
    """Each crystal's rigid turning `[n_samples - 2, 3] Bivector+`: its angular momentum, the offsets joined with their
    central-difference velocities, solved through its inertia."""
    middle = offsets[1:-1]                                                      # [n_samples - 2, 3, 4] Point-
    velocities = (offsets[2:] - offsets[:-2]) / (2 * sample_step)               # [n_samples - 2, 3, 4] Point-
    momenta = (middle & velocities).sum(axis=-1)                                # [n_samples - 2, 3] Antibivector-
    inertias = middle.regressive(middle.commutator(Turn)).sum(axis=-1)          # [n_samples - 2, 3] Antibivector- <- Bivector+
    return inertias.solve(momenta)


def tidal_turning(swing: float, times: np.ndarray, offsets: core.Point) -> Turn:
    """The turning `[n_samples - 2, 3] Bivector+` the wave's tidal pull alone gives each crystal from rest: the torque
    of half the strain's second rate on its proper offsets, summed over time and solved through its inertia."""
    _, _, second_rates = core.Quadrupole(1, swing).strains(times)               # [n_samples] Plane+ <- Point-
    tidal = core.translating(second_rates(core.Step)) / 2                       # [n_samples] Point- <- Point-
    torques = (offsets & tidal[:, None, None](offsets)).sum(axis=-1)            # [n_samples, 3] Antibivector-
    sample_step = times[1] - times[0]
    momenta = ((torques[1:-1] + torques[:-2]) * (sample_step / 2)).cumsum(axis=0)   # [n_samples - 2, 3] Antibivector-
    middle = offsets[1:-1]                                                      # [n_samples - 2, 3, 4] Point-
    inertias = middle.regressive(middle.commutator(Turn)).sum(axis=-1)          # [n_samples - 2, 3] Antibivector- <- Bivector+
    return inertias.solve(momenta)


def irregular_three_sphere() -> tuple[Topology, core.Point]:
    """Relaxes random binary-icosahedral vertex orbits around the 600-cell into an irregular three-sphere triangulation."""
    turns = sphere_mesher.binary_icosahedral_turns()                            # [...] Motor+
    seeds = mv.vector(np.random.default_rng(orbit_seed).normal(size=(n_seed_orbits, 4))).dual().normalized()   # [n_seed_orbits] Point-
    topology, vertices = list(sphere_mesher.relaxed(turns, turns >> core.origin, seeds, n_relax_steps, relax_pull))[-1]
    return topology, vertices


def decagon_bases(cell_vertices: core.Point, circling: core.Rate) -> core.Point:
    """Projects the twelve guardrail decagons down `circling` to icosahedron corners `[12] Point-` on the base sphere."""
    return core.base_points(cell_vertices.batch()[guardrails(cell_vertices, n_guardrail_sides)[:, 0]], circling)


def helix(cell_vertices: core.Point) -> tuple[np.ndarray, core.Point, core.Point, core.Point]:
    """Finds the three guardrail decagons, axis point, axis direction, and transverse step of a Boerdijk-Coxeter helix."""
    circling = circle_turn(cell_vertices, n_guardrail_sides).log()              # [] Rate+
    bases = decagon_bases(cell_vertices, circling)                              # [12] Point-
    icosahedron = numerical.base_hull(bases)
    corners = (icosahedron.incidence[2, 0] * bases.field()).batch()[:, 0]       # [3] Point-
    face_middle = corners.sum(axis=0).normalized()                              # [] Point-
    side_middle = (corners[0] + corners[1]).normalized()                        # [] Point-
    side_direction = (side_middle - face_middle * (core.polar(face_middle) & side_middle)).normalized()   # [] Point-
    rails = guardrails(cell_vertices, n_guardrail_sides)[icosahedron.elements[2][0]]   # [3, n_guardrail_sides]
    return (rails, core.lifts(face_middle, circling), face_middle,
            core.crossing(face_middle, side_direction).normalized())


def camera_aim(placing: core.Motor, forward: core.Point, up: core.Point) -> core.Motor:
    """Combines `placing` with local frame rotations aligning the forward axis to `forward` and vertical axis to `up`."""
    # Square root of the quotient between unit steps gives the half-angle rotor in their common plane.
    aiming = (forward / mv.z.dual()).square_root()                              # [] Motor+
    rolling = (up / (aiming >> mv.y.dual())).square_root()                      # [] Motor+
    return placing * rolling * aiming


def flight(start: core.Motor) -> core.Motor:
    """Advances `start` once around a great circle along its local forward axis over `[n_flight_frames] Motor+`."""
    forward_rate = core.moving(mv.z.dual())                                     # [] Rate+
    distances = np.linspace(0, 2 * np.pi, n_flight_frames, endpoint=False)      # [n_flight_frames]
    return start * (forward_rate * (-distances / 2)).exp()


def pan(axis_point: core.Point, axis_direction: core.Point, side_direction: core.Point) -> core.Motor:
    """Orbits a camera `[n_pan_frames] Motor+` once around a helix axis while looking inward from `pan_distance`."""
    station = core.frame(axis_point) * (core.moving(side_direction) * (-pan_distance / 2)).exp()   # [] Motor+
    start = camera_aim(station, -side_direction, axis_direction) * (mv.xy * (pan_roll / 2)).exp()   # [] Motor+
    axis_rate = core.frame(axis_point) >> core.moving(axis_direction)           # [] Rate+
    angles = np.linspace(0, 2 * np.pi, n_pan_frames, endpoint=False)            # [n_pan_frames]
    return (axis_rate * (-angles / 2)).exp() * start


def base_sphere() -> tuple[Topology, core.Point]:
    """Subdividing an icosahedron forms the base-sphere triangulation and its unit origin steps `Point-[V]`."""
    topology, coordinates = shapes.sphere(n_icosphere_levels)
    return topology, mv(core.Through, coordinates).dual().field()


def triangle_gauges(bases: core.Point, circling: core.Rate) -> tuple[Topology, core.Point]:
    """Lifts one horizontal triangle of bonded gauge atoms `Point-[V]` over each face of the icosahedral `bases` hull."""
    hull = numerical.base_hull(bases)
    V_hull, E_hull, F_hull = hull.n_elements
    corners = hull.incidence[2, 0] * bases.field()                              # [3] Point-[F]
    middles = corners.sum(axis=0).normalized()                                  # Point-[F]
    atom_bases = (middles + gauge_share * (corners - middles)).normalized()     # [3] Point-[F]
    # Horizontal displacement across the Hopf circles turns by half the base sphere's central angle.
    cosines = core.polar(middles) & atom_bases                                  # [3] Scalar+[F]
    directions = (atom_bases - middles * cosines).normalized()                  # [3] Point-[F]
    moves = (core.moving(core.crossing(middles, directions)) * (-cosines.arccos() / 4)).exp()   # [3] Motor+[F]
    face_atoms = (core.frame(core.lifts(middles, circling)) * moves) >> core.origin   # [3] Point-[F]
    atoms = stack(tuple(face_atoms.batch()), axis=-1).reshape(-1)               # [...] Point-
    return Topology.loops([3] * F_hull), atoms.field()


def bond_arcs(bonds: Topology, positions: core.Point, circling: core.Rate) -> core.Point:
    """Projects `n_arc_points` samples along each bond's great-circle arc onto the base sphere `[E, n_arc_points] Point-`."""
    tails, heads = bonds.incidence[1, 0] * positions                            # Point-[E], Point-[E]
    shares = np.linspace(0, 1, n_arc_points)                                    # [n_arc_points]
    arc_points = (tails.batch()[:, None] * (1 - shares) + heads.batch()[:, None] * shares).normalized()   # [E, n_arc_points] Point-
    return core.base_points(arc_points, circling)


def light_fan(circling: core.Rate) -> tuple[core.Point, core.Point]:
    """Lifts `n_rays` horizontal ray headings `[n_rays] Point-` fanned across `light_spread` from `light_source`."""
    start = core.lifts(light_source[None], circling)                            # [1] Point-
    aim = (light_target - light_source * (core.polar(light_source) & light_target)).normalized()   # [] Point-
    angles = np.linspace(-light_spread / 2, light_spread / 2, n_rays)           # [n_rays]
    fan = (light_spreading * (-angles / 2)).exp()                               # [n_rays] Motor+
    return start.broadcast_to((n_rays,)), core.crossing(light_source, fan >> aim).normalized()


def light_paths(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, circling: core.Rate, starts: core.Point,
                headings: core.Point, time: float) -> core.Point:
    """Traces null rays from `starts` along `headings` at `time` and projects their paths onto the base sphere."""
    places = stack([starts, *core.light(pick, waves, starts, headings, time, light_step, n_light_steps)],
                   axis=-1)                                                     # [n_rays, n_light_steps + 1] Point-
    return core.base_points(places, circling)


def gauge_states(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, frequency: float, tuning: float,
                 share: float, bonds: Topology, atoms: core.Point, start: float,
                 times: np.ndarray) -> Iterator[tuple[core.Point, core.Scalar]]:
    """Steps triangle gauges from rest at `start` through `waves` across `times`, their shear mode at `tuning` of the
    wave's `frequency` and damped at `share` of critical."""
    V, E = bonds.n_elements
    # Unit masses on an equilateral triangle shear at a squared frequency of three halves the bond stiffness.
    shear_frequency = tuning * frequency
    stiffness = 2 / 3 * shear_frequency ** 2
    damping = 2 / 3 * (2 * share * shear_frequency)
    round_steps = core.round_steps(bonds, atoms)                                # Point-[E]
    rest = round_steps.norm()                                                   # Scalar+[E]
    stiffnesses = mv.scalar(np.full((E, 1), stiffness)).field()                 # Scalar+[E]
    dampings = mv.scalar(np.full((E, 1), damping)).field()                      # Scalar+[E]
    at_rest = mv(core.Through, np.zeros((V, 3))).dual().field()                 # Point-[V]
    return core.massive(pick, waves, bonds, rest, stiffnesses, dampings, atoms, at_rest, start, crystal_step, times)


def gauge_readouts(states: Iterator[tuple[core.Point, core.Scalar]],
                   sample_step: float) -> tuple[core.Scalar, Turn, core.Scalar]:
    """Reads each triangle gauge across `states` sampled every `sample_step`: its bond strains `[n_samples, F, 3]
    Scalar+`, its rigid turning `[n_samples - 2, F] Bivector+` in its middle's frame, and its area over its area at
    rest `[n_samples, F] Scalar+`, which falls to zero where the triangle turns inside out."""
    offsets, strains = [], []
    for positions, bond_strains in states:
        places = positions.batch().reshape(-1, 3)                               # [F, 3] Point-
        middles = places.sum(axis=-1).normalized()                              # [F] Point-
        steps = (core.frame(middles)[:, None].inverse() >> places).cast(core.Step)   # [F, 3] Point-
        offsets.append(steps - steps.sum(axis=-1)[:, None] / 3)
        strains.append(bond_strains.batch().reshape(-1, 3))
    offsets = stack(offsets)                                                    # [n_samples, F, 3] Point-
    patches = (offsets[..., 1] - offsets[..., 0]) & (offsets[..., 2] - offsets[..., 0])   # [n_samples, F] Antibivector-
    # A triangle turns over freely, so only a patch that shrinks through nothing inverts it.
    areas = patches.norm() / patches[0].norm()                                  # [n_samples, F] Scalar+
    return stack(strains), turning_rates(offsets, sample_step), areas


def quadrupole_history(cell_topology: Topology, cell_vertices: core.Point, reach: float) -> tuple[core.Waves, float]:
    """Steps a circularly polarized travelling quadrupole wave on the 600-cell from two orthogonal squash patterns, from
    its settling to `reach` past one period."""
    patterns = squash(cell_topology, cell_vertices, stack((mv.x, eighth >> mv.x)),
                      stack((mv.y, eighth >> mv.y)))                            # [2] Scalar+[E]
    return stepped_history(cell_topology, cell_vertices, patterns, quadrupole_lags, swing, reach)


def quadrupole_frames(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, frequency: float,
                      turn: core.Motor, bases: core.Point, sphere_points: core.Point) -> Iterator[tuple]:
    """Yields one period of base-sphere fibre stretches, triangle-gauge states, and light paths in `waves`."""
    times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames    # [n_wave_frames]
    circling = turn.log()                                                       # [] Rate+
    bonds, atoms = triangle_gauges(bases, circling)
    starts, headings = light_fan(circling)
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    states = gauge_states(pick, waves, frequency, gauge_tuning, gauge_damping, bonds, atoms, start, times)
    for time, (positions, strains) in zip(times, states):
        yield (core.stretches(waves, pick, sphere_points, sphere_points, turn, n_guardrail_sides, n_circle_samples,
                              time),
               core.base_points(positions.batch(), circling), bond_arcs(bonds, positions, circling), strains.batch(),
               light_paths(pick, waves, circling, starts, headings, time))


# --- runs -----------------------------------------------------------------------------
def main() -> None:
    """Renders interior flights, exterior quadrupole, and resonant-crystal wave animations and verifies invariants."""
    from examples.relativity.gravitational_waves import render
    from examples.util.animation import save_animation

    cell_topology, cell_vertices = six_hundred_cell()
    pick = numerical.picking(cell_topology, cell_vertices)

    round_walk, cell_walk_values = core.round_walk_source, render.round_values(cell_topology, cell_vertices)
    rails, axis_point, axis_direction, side_direction = helix(cell_vertices)
    save_animation(render.polygon_view(round_walk, cell_walk_values, cell_topology, cell_vertices,
                                       flight(camera_aim(core.frame(axis_point), axis_direction, side_direction)),
                                       great_circles(cell_topology, cell_vertices)),
                   "gravitational_waves_flight", duration_ms)
    save_animation(render.helix_view(round_walk, cell_walk_values, cell_topology, cell_vertices,
                                     pan(axis_point, axis_direction, side_direction), rails),
                   "gravitational_waves_helix", duration_ms)

    irregular_topology, irregular_vertices = irregular_three_sphere()
    middle = (irregular_topology.incidence[3, 0].mean(axis=0) * irregular_vertices).batch()[0].normalized()   # [] Point-
    save_animation(render.plain_view(round_walk, render.round_values(irregular_topology, irregular_vertices),
                                     irregular_topology, irregular_vertices, flight(core.frame(middle))),
                   "gravitational_waves_irregular", duration_ms)

    quadrupole, quadrupole_frequency = quadrupole_history(cell_topology, cell_vertices, n_light_steps * light_step)
    turn = circle_turn(cell_vertices, n_guardrail_sides)                        # [] Motor+
    bases = decagon_bases(cell_vertices, turn.log())                            # [12] Point-
    sphere_topology, sphere_points = base_sphere()
    save_animation(render.base_views(sphere_topology, sphere_points, bases,
                                     quadrupole_frames(pick, quadrupole, quadrupole_frequency, turn, bases,
                                                       sphere_points)),
                   "gravitational_waves_quadrupole", duration_ms, colors=n_quadrupole_colours)

    squashing = squash(cell_topology, cell_vertices, stack((mv.x,)), stack((mv.y,)))   # [1] Scalar+[E]
    waves, frequency = stepped_history(cell_topology, cell_vertices, squashing, squash_lags, wave_swing, 0.0)
    eye_times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames   # [n_wave_frames]
    kept_times = np.arange(eye_times[0] - np.pi, eye_times[-1] + waves.level_step, waves.level_step)   # [...]
    bonds, atoms = crystals(crystal_places())
    states = crystal_states(pick, waves, frequency, bonds, atoms, kept_times)
    save_animation(render.waves_view(core.wave_walk_source, render.wave_values(cell_topology, waves), cell_topology,
                                     cell_vertices, eye, eye_times, guardrails(cell_vertices, n_guardrail_sides),
                                     bonds, kept_times, states, render.wave_ghost_share),
                   "gravitational_waves_inside", duration_ms)

    # checks
    assert 7.5 < quadrupole_frequency ** 2 < 8.0
    chords = irregular_topology.d[0] * irregular_vertices                       # Point-[E]
    deficits, _, _ = core.Space(irregular_topology, irregular_vertices).dynamics(chords.scalar_norm_squared())
    assert (deficits.abs() < 1e-12).all()
    at_one = core.stretches(quadrupole, pick, sphere_points, sphere_points, turn, n_guardrail_sides, 1, 0.0)
    averaged = core.stretches(quadrupole, pick, sphere_points, sphere_points, turn, n_guardrail_sides,
                              n_circle_samples, 0.0)
    assert ((at_one - averaged).abs() < 1e-6 * averaged.abs().max()).all()


def crystal_spin(name: str, frames: core.Motor, swing: float, share: float, turns: Turn, n_periods: int,
                 n_frames_a_period: int) -> tuple[core.Point, core.Scalar]:
    """Steps the three crystals placed by `frames`, turning at `turns`, through the exact quadrupole of `swing`, damped
    at `share` of critical; writes their animation and the chart of their turning, from their momentum and from the
    wave's tidal torque, and gives the proper offsets `[n_samples, 3, 4] Point-` and the bond strains
    `[n_samples] Scalar+[E]`."""
    from examples.relativity.gravitational_waves import render
    from examples.util.animation import save_animation, save_figure

    times, offsets, strains = quadrupole_crystals(frames, swing, share, turns, n_periods)
    turning = turning_rates(offsets, times[1] - times[0])                       # [n_samples - 2, 3] Bivector+
    tidal = tidal_turning(swing, times, offsets)                                # [n_samples - 2, 3] Bivector+
    readouts = render.spin_readouts(times, strains, turning, tidal)
    save_animation(render.spin_frames(readouts, n_frames_a_period), name, duration_ms)
    save_figure(render.spin_chart(readouts), f"{name}_torque")
    return offsets, strains


def spun_by_the_wave() -> None:
    """The three crystals from rest in the page's wave for `n_spin_periods`; then, with no wave, the unstrained tuned
    crystal started turning rigidly with the angular momentum it had over the last `n_late_periods`, beside the strains
    its bonds held over those periods in the wave."""
    from examples.relativity.gravitational_waves import render, web
    from examples.util.animation import save_figure

    at_rest = mv.xy * np.zeros(len(tunings))                                    # [3] Bivector+
    offsets, strains = crystal_spin("gravitational_waves_crystal_spin", crystal_places(), web.wave_swing,
                                    critical_share, at_rest, n_spin_periods, n_spin_frames_a_period)
    sample_step = n_steps_a_sample * crystal_step
    n_late = int(n_late_periods * 2 * np.pi / core.quadrupole_frequency / sample_step)
    late = offsets[-n_late - 1:-1]                                              # [n_late, 3, 4] Point-
    velocities = (offsets[-n_late:] - offsets[-n_late - 2:-2]) / (2 * sample_step)   # [n_late, 3, 4] Point-
    momenta = (late & velocities).sum(axis=-1).sum(axis=0) / n_late             # [3] Antibivector-
    unstrained = offsets[0].regressive(offsets[0].commutator(Turn)).sum(axis=-1)   # [3] Antibivector- <- Bivector+
    # Only the tuned crystal is set turning.
    turns = unstrained.solve(momenta) * np.array([0, 1, 0])                     # [3] Bivector+
    _, spun_strains = crystal_spin("gravitational_waves_crystal_spin_no_wave", crystal_places(), 0.0, critical_share,
                                   turns, n_spun_periods, n_spin_frames_a_period)
    save_figure(render.centrifugal_bars(strains[-n_late:].sum(axis=0) / n_late,
                                        spun_strains[-n_late:].sum(axis=0) / n_late),
                "gravitational_waves_crystal_spin_strains")


def swing_trials(name: str, frames: core.Motor, swing_shares: np.ndarray, critical_shares: np.ndarray) -> None:
    """The three crystals placed by `frames` from rest for `n_trial_periods` in the page's wave at each of
    `swing_shares` of its swing, damped at the matching `critical_shares` of critical, one animation and chart each."""
    from examples.relativity.gravitational_waves import web

    at_rest = mv.xy * np.zeros(len(tunings))                                    # [3] Bivector+
    for swing_share, share in zip(swing_shares, critical_shares):
        crystal_spin(f"{name}_swing_{swing_share:.2f}_damping_{share:.3f}", frames, swing_share * web.wave_swing,
                     share, at_rest, n_trial_periods, n_trial_frames_a_period)


def crystal_spin_trials() -> None:
    """Writes the crystals' turning in the exact quadrupole on the CPU: the page's wave, the same spin with no wave, and
    the trials of swing, damping and orientation."""
    spun_by_the_wave()
    swing_trials("gravitational_waves_crystal_spin", crystal_places(), trial_swing_shares, trial_critical_shares)
    swing_trials("gravitational_waves_crystal_spin_diagonal", diagonal_places(), diagonal_swing_shares,
                 diagonal_critical_shares)


def gauge_trials() -> None:
    """The Hopf view's triangle gauges from rest in its travelling quadrupole for `n_gauge_trial_periods`, at each of
    `gauge_trial_tunings` and `gauge_trial_shares`: one animation each of their strains, turning and areas, and one
    chart of all the runs."""
    from examples.relativity.gravitational_waves import render
    from examples.util.animation import save_animation, save_figure

    cell_topology, cell_vertices = six_hundred_cell()
    pick = numerical.picking(cell_topology, cell_vertices)
    _, frequency = quadrupole_history(cell_topology, cell_vertices, 0.0)
    period = 2 * np.pi / frequency
    start = -n_settling_swings * period
    waves, _ = quadrupole_history(cell_topology, cell_vertices, start + n_gauge_trial_periods * period)
    turn = circle_turn(cell_vertices, n_guardrail_sides)                        # [] Motor+
    bonds, atoms = triangle_gauges(decagon_bases(cell_vertices, turn.log()), turn.log())
    sample_step = period / n_gauge_samples_a_period
    times = start + sample_step * np.arange(n_gauge_trial_periods * n_gauge_samples_a_period + 1)   # [n_samples]
    runs = []
    for tuning, share in zip(gauge_trial_tunings, gauge_trial_shares):
        states = gauge_states(pick, waves, frequency, tuning, share, bonds, atoms, start, times)
        readouts = render.gauge_chart_readouts(times - start, period, frequency, *gauge_readouts(states, sample_step))
        save_animation(render.gauge_chart_frames(readouts, n_gauge_frames_a_period),
                       f"gravitational_waves_gauges_tuning_{tuning:.2f}_damping_{share:.2f}", duration_ms)
        runs.append(readouts)
    save_figure(render.gauge_comparison(runs, gauge_trial_tunings, gauge_trial_shares, n_settling_swings + 1),
                "gravitational_waves_gauge_trials")


def pinned() -> None:
    """Writes the repository's pinned animations for the exterior quadrupole and the interior 600-cell wave view."""
    from examples.relativity.gravitational_waves import render
    from examples.util.animation import pin_animation

    cell_topology, cell_vertices = six_hundred_cell()
    pick = numerical.picking(cell_topology, cell_vertices)
    quadrupole, quadrupole_frequency = quadrupole_history(cell_topology, cell_vertices, n_light_steps * light_step)
    turn = circle_turn(cell_vertices, n_guardrail_sides)                        # [] Motor+
    bases = decagon_bases(cell_vertices, turn.log())                            # [12] Point-
    sphere_topology, sphere_points = base_sphere()
    pin_animation(render.base_views(sphere_topology, sphere_points, bases,
                                    quadrupole_frames(pick, quadrupole, quadrupole_frequency, turn, bases,
                                                      sphere_points)),
                  "gravitational_waves_quadrupole", duration_ms, colors=n_quadrupole_colours)

    squashing = squash(cell_topology, cell_vertices, stack((mv.x,)), stack((mv.y,)))   # [1] Scalar+[E]
    waves, frequency = stepped_history(cell_topology, cell_vertices, squashing, squash_lags, wave_swing, 0.0)
    eye_times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames   # [n_wave_frames]
    kept_times = np.arange(eye_times[0] - np.pi, eye_times[-1] + waves.level_step, waves.level_step)   # [...]
    bonds, atoms = crystals(crystal_places())
    states = crystal_states(pick, waves, frequency, bonds, atoms, kept_times)
    pin_animation(render.waves_view(core.wave_walk_source, render.wave_values(cell_topology, waves), cell_topology,
                                    cell_vertices, eye, eye_times, guardrails(cell_vertices, n_guardrail_sides),
                                    bonds, kept_times, states, render.wave_ghost_share),
                  "gravitational_waves_inside", duration_ms)


if __name__ == "__main__":
    main()
