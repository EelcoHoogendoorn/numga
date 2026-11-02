"""The closed universe from inside and below its Hopf circles. Round, without waves: a flight down the axis of one of
the 600-cell's Boerdijk–Coxeter helices, every great circle of its edges in its own colour, the helix panned past, its
rails in colour, and a flight through an irregular mesh. With waves, on the 600-cell: from inside, its squash rings
past three crystals ahead of the eye, soft, tuned and stiff; and below the Hopf circles, a quadrupole travels round the
base, rung by triangle gauges held to their places and crossed by a bundle of light."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
from numga import stack

from examples import instantiate
from examples.relativity.closed_universe import core
from examples.util import shapes
from examples.util.topology import Topology

mv = core.mv
# Round meshes of the three-sphere relaxed under the binary icosahedral group, in the same algebra.
sphere_mesher = instantiate("examples.util.relaxed_sphere", core.ga)

# The 600-cell placed by a turn that keeps its vertices off the origin and its antipode.
cell_placement = (mv.xw * 0.21 + mv.yw * 0.13 + mv.zw * 0.08).exp()          # [] Motor+

# The guardrails' great polygons: decagons, whose sides are the 600-cell's edges.
n_guardrail_sides = 10
# The eye inside the 600-cell, the motor that carries the origin to it and turns its view; the stepped gravitational
# waves, the slowest of the closed universe started from its squash, swinging space along one invariant direction by
# this share and the opposite way along another, at the lattice's own frequency, with no lag; the frames over one
# swing.
eye = (mv.xw * 0.37 + mv.yw * 0.11 + mv.zw * 0.23).exp()                      # [] Motor+
wave_swing = 0.12
squash_lags = np.zeros(1)                                                      # [patterns]
n_wave_frames = 75
# Three little crystals ahead of the eye, side by side across its view: tetrahedra of four atoms of unit mass, bonded
# along their six edges, their bonds soft, tuned to the wave, and stiff: a sixteenth of the stiffness that puts the
# swing the squash drives, root two times the root of the stiffness, at the wave's, that one, and sixteen times it,
# with dashpots a quarter of critical. How far ahead, how far apart, how large; the time step, and the swings run
# before the frames so that the motion has settled.
crystal_distance, crystal_spacing, crystal_half_size = 0.55, 0.26, 0.05
tunings = np.array([1 / 16, 1, 16])                                           # [crystals]
critical_share = 0.25
crystal_step = 0.01
# The stepped waves' steps to a swing of the squash.
n_steps_per_swing = 40
n_settling_swings = 8
# An irregular three-sphere: the 600-cell's vertices and so many orbits of random seeds under the binary icosahedral
# group, from this random start, relaxed by so many optimal Delaunay steps, each turning a seed this share of the way.
n_mesh_orbits = 5
mesh_seed = 7
n_relax_steps = 60
relax_pull = 0.5

duration_ms = 40
# The colours of the quadrupole's palette, seen from outside.
n_quadrupole_colours = 96
# The round universe from inside. A Boerdijk–Coxeter helix of the 600-cell is a column of thirty tetrahedra stacked
# face to face round a great circle, its axis, its vertices on three of the decagons, those over one face of the
# icosahedron their base points make: flown along its axis once round in so many frames, every great circle of the
# 600-cell's edges in its own colour; and panned past once round its axis in so many frames, from this far from the
# axis, the view rolled in its own plane `xy` by this much, its rails in colour. The irregular mesh, flown through once
# round the same way.
n_flight_frames = 90
n_pan_frames = 60
pan_distance = 1.02
pan_roll = 0.36
# The base drawn as an icosphere, the icosahedron's faces split in four so many times; the points sampled round each
# circle for its average, so many that they fall between the tenth turns that carry the 600-cell onto itself; the
# swing of space.
n_icosphere_levels = 3
n_circle_samples = 7
swing = 0.085
# The squash stretches space along x and squeezes it along y in the frames; turned an eighth of a turn in the plane
# `xy`, it stretches and squeezes along the diagonals between them, and a quarter swing behind the squash, the two make
# a quadrupole that turns round in the plane `xy`, travelling round the base.
eighth = (mv.xy * (-np.pi / 8)).exp()                                          # [] Motor+
quadrupole_lags = np.array([0, np.pi / 2])                                     # [patterns]
# The triangle gauges, one in each face of the icosahedron the decagons' base points make, its corners this share of
# the way from the face's middle to its corners. Each atom is held to its place, a free particle at rest, by a spring
# swinging at this share of the waves' frequency, damped critically, so that a gauge keeps its place and its turn
# while it rings, rather than taking up the travelling waves' turning; its bonds put the triangle's shear swing at the
# waves' frequency, damped to this share of critical. The points sampled along each bond for its arc.
gauge_share = 0.4
holding_share = 1 / 4
gauge_damping = 0.4
n_arc_points = 12
# Light: a narrow bundle of so many rays set out across the circles from the circle above the base point along z,
# aimed about the direction toward the base point along y, which faces the viewer, and fanned in the plane `xy` square
# to the source over this angle, as each frame begins; traced in steps of this length for a quarter turn, the base's
# half from pole to pole.
light_source, light_target, light_spreading = mv.z.dual(), mv.y.dual(), mv.xy   # [] Point-, [] Point-, [] Rate+
n_rays = 7
light_spread = np.radians(18)
light_step = 0.02
n_light_steps = int(np.pi / 2 / light_step)
# The stepped squash's swings, read for its ringing.
n_ringing_swings = 4


# --- math -----------------------------------------------------------------------------
def six_hundred_cell() -> tuple[Topology, core.Point]:
    """The round three-sphere's mesh, the 600-cell, the stock shape carried by its first vertex's frame back
    so that the vertex is the origin, and placed by its turn. A 600-cell with a vertex at the origin is closed under
    the turns its vertices' frames make, so each turn that carries the origin to a vertex carries the
    vertices onto themselves."""
    topology, coordinates = shapes.cell_600()
    vertices = mv.vector(coordinates).dual()                                    # [vertices] Point-
    centred = core.frame(vertices[0]).inverse() >> vertices                     # [vertices] Point-
    return topology, cell_placement >> centred.field()


def circle_turn(vertices: core.Point, n_sides: int) -> core.Motor:
    """The turn along the Hopf circles that the 600-cell's vertices lie on in great polygons of so many sides, `[]
    Motor+`: the motor that carries the origin, before the placement, to the vertex that share of a whole
    turn from it, which carries the vertices onto themselves."""
    unplaced = (cell_placement.inverse() >> vertices).batch()                  # [vertices] Point-
    # The vertex whose reading by the origin's equator, the cosine of its angle from the origin, is that share's.
    target = unplaced[((core.equator & unplaced) - np.cos(2 * np.pi / n_sides)).abs().argmin()]   # [] Point-
    return cell_placement * core.frame(target) * cell_placement.inverse()


def guardrails(vertices: core.Point, n_sides: int) -> np.ndarray:
    """The 600-cell's vertices on great polygons of so many sides, `[polygons, sides]` vertex indices, great circles
    side by side, one Hopf fibration: the orbits of the vertices under `circle_turn`, each once. Ten sides give twelve
    decagons that the mesh's edges run along, six twenty hexagons and four thirty squares, over the icosahedron's
    vertices, faces and edges."""
    polygons = orbits(vertices, circle_turn(vertices, n_sides)[None], n_sides)[0]   # [sides, vertices]
    _, firsts = np.unique(polygons.min(axis=0), return_index=True)
    return polygons[:, firsts].T


def great_circles(mesh: Topology, vertices: core.Point) -> np.ndarray:
    """The great circles the 600-cell's edges run along, decagons of vertex indices `[circles, sides]`: each edge's the
    orbit of its tail under the turn that carries the tail to its head, the frame at the head by the inverse
    of the one at the tail, which carries the vertices onto themselves; each circle once."""
    tails, heads = mesh.incidence[1, 0] * vertices                              # Point-[edges], Point-[edges]
    turns = (core.frame(heads) * core.frame(tails).inverse()).batch()           # [edges] Motor+
    tail_indices = mesh.elements[1][:, 0]                                       # [edges]
    circles = orbits(vertices, turns, n_guardrail_sides)[np.arange(len(tail_indices)), :, tail_indices]   # [edges, sides]
    _, firsts = np.unique(np.sort(circles, axis=-1), axis=0, return_index=True)
    return circles[firsts]


def crystal_places() -> core.Motor:
    """The frames of the three crystals ahead of the eye, `[crystals] Motor+`: the eye carried ahead along its view
    and across it to each crystal's place, each by the rate that moves the origin along the eye's own axes."""
    axis_rates = core.moving(stack((mv.x, mv.y, mv.z)).dual())                  # [3] Rate+
    offsets = crystal_spacing * np.arange(-1, 2)                                # [crystals]
    forward = (axis_rates[2] * (-crystal_distance / 2)).exp()                   # [] Motor+
    side_shifts = (axis_rates[0] * (-offsets / 2)).exp()                        # [crystals] Motor+
    return eye * forward * side_shifts


def crystals(frames: core.Motor) -> tuple[Topology, core.Point]:
    """Crystals in the frames `[crystals] Motor+`, motors that carry the origin to each crystal's middle: a
    tetrahedron of atoms each, bonded along its edges, and the atoms, `Point-[atoms]`, at rest in the chart, each frame
    carried out to every other corner of a cube along its own axes."""
    from itertools import product

    axis_rates = core.moving(stack((mv.x, mv.y, mv.z)).dual())                  # [3] Rate+
    cube = np.array(list(product([-1, 1], repeat=3)))                          # [8, 3]
    corners = cube[cube.prod(axis=-1) > 0]                                      # [corners, 3]
    corner_moves = ((axis_rates * corners).sum(axis=-1) * (-crystal_half_size / 2)).exp()   # [corners] Motor+
    atoms = ((frames[:, None] * corner_moves[None]) >> core.origin).reshape(-1)   # [atoms] Point-
    n_atoms = len(corners) * frames.shape[0]
    return Topology(n_atoms, np.arange(n_atoms).reshape(-1, len(corners))), atoms.field()


def squash(cell_mesh: Topology, cell_vertices: core.Point, stretched: core.Plane, squeezed: core.Plane) -> core.Scalar:
    """The squash's change of each edge's squared length, per unit swing of space, `Scalar+[edges]`, for each pair of
    planes `[...] Plane+` that read the invariant directions it stretches and squeezes: its round step's share along
    the stretched direction less its share along the squeezed one, twice, since a squared length changes twice as fast
    as the length."""
    steps = core.round_steps(cell_mesh, cell_vertices)                          # Point-[edges]
    chords = cell_mesh.d[0] * cell_vertices                                     # Point-[edges]
    shares = ((stretched & steps).squared() - (squeezed & steps).squared()) / (core.polar(steps) & steps)   # [...] Scalar+[edges]
    return 2 * chords.dot(chords) * shares


def stepped_history(cell_mesh: Topology, cell_vertices: core.Point, patterns: core.Scalar, lags: np.ndarray,
                    swing: float, reach: float) -> tuple[core.Waves, float]:
    """The 600-cell's gravitational waves stepped from squash patterns `[patterns] Scalar+[edges]`, swinging space by
    `swing` as cosines at the lattice's frequency, each lagging by its share of `lags` `[patterns]`: from long enough
    before the first frame for the crystals to settle and the light that reaches the eye to set out, to `reach` past
    the swing of the frames, as far as light set out at a frame runs, as a smooth field over time; and that
    frequency."""
    stiffness, masses, onto_waves = core.gravitational_waves(cell_mesh, cell_vertices)
    wave_patterns = onto_waves(patterns)                                        # [patterns] Scalar+[edges]
    # The frequency squared: the stiffness's force over the momentum of a rate, each paired with the pattern.
    squared = ((stiffness(wave_patterns) & wave_patterns).sites.sum().sum(axis=0)
               / (masses(wave_patterns) & wave_patterns).sites.sum().sum(axis=0))   # [] Scalar+
    frequency = float(np.sqrt(squared.to_array()))
    period = 2 * np.pi / frequency
    step = period / n_steps_per_swing
    first = -(n_settling_swings + 1) * period - np.pi
    n_levels = int(np.ceil((period + reach - first) / step)) + 2
    phases = frequency * first - lags                                           # [patterns]
    strains = swing * (np.cos(phases) * wave_patterns).sum(axis=0)              # Scalar+[edges]
    change_rates = -swing * frequency * (np.sin(phases) * wave_patterns).sum(axis=0)   # Scalar+[edges]
    leapfrog = core.waves_stepped(stiffness, masses, onto_waves, strains, change_rates, step)
    levels = [state for _, state in zip(range(n_levels), leapfrog)]
    return core.Waves(cell_mesh, cell_vertices, [changes for changes, _ in levels],
                      [change_rates for _, change_rates in levels], first, step), frequency


def crystal_states(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, frequency: float, bonds: Topology,
                   atoms: core.Point, times: np.ndarray) -> Iterator[tuple[core.Point, core.Scalar]]:
    """The crystals' atoms and bonds' strains, `(Point-[atoms], Scalar+[bonds])`, at each of the times, after the motion
    has settled into the waves' swing at the frequency, the crystals tuned by `tunings`: the tuned crystal's stiffness
    puts the swing the squash drives, root two times the root of the stiffness, at the waves' frequency."""
    stiffnesses = frequency ** 2 / 2 * tunings                                  # [crystals]
    dampings = critical_share * np.sqrt(2 * stiffnesses)                        # [crystals]
    # The crystals' rest lengths: the lengths of their bonds' round steps.
    round_steps = core.round_steps(bonds, atoms)                                # Point-[bonds]
    rest = round_steps.dot(round_steps).square_root()                           # Scalar+[bonds]
    n_bonds_each = bonds.n_elements[1] // len(tunings)
    stiffness = mv.scalar(np.repeat(stiffnesses, n_bonds_each)[:, None]).field()   # Scalar+[bonds]
    damping = mv.scalar(np.repeat(dampings, n_bonds_each)[:, None]).field()        # Scalar+[bonds]
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    at_rest = mv(core.Step, np.zeros((bonds.n_vertices, 3))).field()           # Point-[atoms]
    return core.massive(pick, waves, bonds, rest, stiffness, damping, 0.0, atoms, at_rest, start, crystal_step, times)


def irregular_three_sphere() -> tuple[Topology, core.Point]:
    """An irregular mesh of the round three-sphere and its vertices, `Point-[vertices]`: the 600-cell's vertices, the
    origin's orbit under the binary icosahedral group, and the orbits of random seeds, relaxed until the tetrahedra
    between them are as round as the orbits allow."""
    turns = sphere_mesher.binary_icosahedral_turns()                            # [rotations] Motor+
    seeds = mv.vector(np.random.default_rng(mesh_seed).normal(size=(n_mesh_orbits, 4))).dual().normalized()
    mesh, vertices = list(sphere_mesher.relaxed(turns, turns >> core.origin, seeds, n_relax_steps, relax_pull))[-1]
    return mesh, vertices


def stepped_squash(cell_mesh: Topology, cell_vertices: core.Point, n_swings: int
                   ) -> tuple[core.Scalar, float, np.ndarray, core.Scalar]:
    """The 600-cell's gravitational waves stepped from the squash at rest: the spectrum of their frequencies squared,
    `Scalar+[edges]`, the squash's own, and the times and the squash's share of the stepped strains at each, `[steps]
    Scalar+`, over `n_swings` of it."""
    stiffness, masses, onto_waves = core.gravitational_waves(cell_mesh, cell_vertices)
    # The waves' stiffness and the inertia, dense on the 600-cell's few edges, as their pairings between every two unit
    # changes, the stiffness's kept to the waves: forms read as numbers, maps of plain scalars.
    unit_changes = mv.scalar(np.eye(cell_mesh.n_elements[1])[..., None]).field()   # [edges] Scalar+[edges]
    unit_waves = onto_waves(unit_changes)                                       # [edges] Scalar+[edges]
    stiffness_pairings = (stiffness(unit_waves)[None, :] & unit_waves[:, None]).sites.sum()   # [edges, edges] Scalar+
    mass_pairings = (masses(unit_changes)[None, :] & unit_changes[:, None]).sites.sum()       # [edges, edges] Scalar+
    spectrum, _ = (stiffness_pairings * core.Scalar).field(0, 1).eigh((mass_pairings * core.Scalar).field(0, 1))
    pattern = onto_waves(squash(cell_mesh, cell_vertices, mv.x, mv.y))          # Scalar+[edges]
    norm = (masses(pattern) & pattern).sites.sum()                              # [] Scalar+
    frequency_squared = float(((stiffness(pattern) & pattern).sites.sum() / norm).to_array())
    step = 2 * np.pi / np.sqrt(frequency_squared) / n_steps_per_swing
    n_steps = n_swings * n_steps_per_swing
    leapfrog = core.waves_stepped(stiffness, masses, onto_waves, wave_swing * pattern, 0 * pattern, step)
    shares = stack([(masses(strains) & pattern).sites.sum() / norm
                    for _, (strains, _) in zip(range(n_steps), leapfrog)])  # [steps] Scalar+
    return spectrum, frequency_squared, step * np.arange(n_steps), shares


def decagon_bases(cell_vertices: core.Point, circling: core.Rate) -> core.Point:
    """The base points of the guardrails' decagons, `[decagons] Point-`, below the circles the rate `circling` runs
    along: the corners of an icosahedron."""
    return core.base_points(cell_vertices.batch()[guardrails(cell_vertices, n_guardrail_sides)[:, 0]], circling)


def helix(cell_vertices: core.Point) -> tuple[np.ndarray, core.Point, core.Point, core.Point]:
    """A Boerdijk–Coxeter helix of the 600-cell: its rails, the decagons over the first face of the icosahedron their
    base points make, `[corners, sides]` vertex indices; a point on its axis, the circle above the face's middle, `[]
    Point-`; and at that point the axis's direction and a step across it toward the middle of the face's first side,
    unit steps of the origin in the frame there, `[] Point-` each."""
    circling = circle_turn(cell_vertices, n_guardrail_sides).log()             # [] Rate+
    bases = decagon_bases(cell_vertices, circling)                              # [decagons] Point-
    icosahedron = base_hull(bases)
    corners = (icosahedron.incidence[2, 0] * bases.field()).batch()[:, 0]       # [corners] Point-
    face_middle = corners.sum(axis=0).normalized()                              # [] Point-
    side_middle = (corners[0] + corners[1]).normalized()                        # [] Point-
    side_direction = (side_middle - face_middle * (core.polar(face_middle) & side_middle)).normalized()   # [] Point-
    rails = guardrails(cell_vertices, n_guardrail_sides)[icosahedron.elements[2][0]]   # [corners, sides]
    return (rails, core.lifts(face_middle, circling), face_middle,
            core.crossing(face_middle, side_direction).normalized())


def facing(placing: core.Motor, forward: core.Point, up: core.Point) -> core.Motor:
    """A camera, the motor `placing` that carries the origin to the eye followed by the turns that carry the origin's
    step along z to `forward` and then, in the plane square to it, the one along y to `up`, unit steps of the origin
    square to each other in the eye's frame, `[] Point-` each."""
    # The turn that carries one unit step to another in their plane is the square root of their quotient.
    aiming = (forward / mv.z.dual()).square_root()                              # [] Motor+
    rolling = (up / (aiming >> mv.y.dual())).square_root()                      # [] Motor+
    return placing * rolling * aiming


def flight(start: core.Motor) -> core.Motor:
    """The cameras flown once round from a start, `[frames] Motor+`, each carried ahead along its view."""
    forward_rate = core.moving(mv.z.dual())                                     # [] Rate+
    distances = np.linspace(0, 2 * np.pi, n_flight_frames, endpoint=False)    # [frames]
    return start * (forward_rate * (-distances / 2)).exp()


def pan(axis_point: core.Point, axis_direction: core.Point, side_direction: core.Point) -> core.Motor:
    """The cameras panned once round a helix's axis, `[frames] Motor+`: from the pan's distance across the axis
    from a point on it, toward the side, looking back at it with the axis up, rolled in its own plane `xy`, each
    carried along the axis by the turn in its plane, which keeps every point square to that plane."""
    station = core.frame(axis_point) * (core.moving(side_direction) * (-pan_distance / 2)).exp()       # [] Motor+
    start = facing(station, -side_direction, axis_direction) * (mv.xy * (pan_roll / 2)).exp()   # [] Motor+
    # The turn along the axis: the rate that moves the point on it along its direction, carried there.
    axis_rate = core.frame(axis_point) >> core.moving(axis_direction)           # [] Rate+
    angles = np.linspace(0, 2 * np.pi, n_pan_frames, endpoint=False)           # [frames]
    return (axis_rate * (-angles / 2)).exp() * start


def base_sphere() -> tuple[Topology, core.Point]:
    """The icosphere drawn as the base, and its points as unit steps of the three-sphere's origin, `Point-[points]`,
    the duals of the planes through the origin with the icosphere's coordinates."""
    topology, coordinates = shapes.sphere(n_icosphere_levels)
    return topology, mv(core.Through, coordinates).dual().field()


def triangle_gauges(bases: core.Point, circling: core.Rate) -> tuple[Topology, core.Point]:
    """The triangle gauges' bonds, loops of three over their atoms, and the atoms, `Point-[atoms]` on the three-sphere:
    in each face of the hull of the base points `[n] Point-`, the corners drawn toward the face's middle, lifted from
    the circle above the middle across the circles, so that the bonds run across them too."""
    hull = base_hull(bases)
    corners = hull.incidence[2, 0] * bases.field()                              # [corners] Point-[faces]
    middles = corners.sum(axis=0).normalized()                                  # Point-[faces]
    atom_bases = (middles + gauge_share * (corners - middles)).normalized()     # [corners] Point-[faces]
    # Each atom's direction along the base from its face's middle, and the cosine of the angle between them: the step
    # across the circles that the direction is the image of carries the middle's lift half that angle, which the Hopf
    # map doubles.
    cosines = core.polar(middles) & atom_bases                                  # [corners] Scalar+[faces]
    directions = (atom_bases - middles * cosines).normalized()                  # [corners] Point-[faces]
    moves = (core.moving(core.crossing(middles, directions)) * (-cosines.arccos() / 4)).exp()   # [corners] Motor+[faces]
    face_atoms = (core.frame(core.lifts(middles, circling)) * moves) >> core.origin       # [corners] Point-[faces]
    # The atoms three to a face, face after face.
    atoms = stack(tuple(face_atoms.batch()), axis=-1).reshape(-1)               # [atoms] Point-
    return Topology.loops([3] * hull.n_elements[2]), atoms.field()


def bond_arcs(bonds: Topology, positions: core.Point, circling: core.Rate) -> core.Point:
    """Each bond's great arc between its atoms on the three-sphere, `Point-[atoms]`, seen on the base: so many points
    along it, its chord's points moved out onto the sphere, and their base points, `[bonds, arc points] Point-`."""
    tails, heads = bonds.incidence[1, 0] * positions                            # Point-[bonds], Point-[bonds]
    shares = np.linspace(0, 1, n_arc_points)                                    # [arc points]
    arc_points = (tails.batch()[:, None] * (1 - shares) + heads.batch()[:, None] * shares).normalized()   # [bonds, arc points] Point-
    return core.base_points(arc_points, circling)


def light_fan(circling: core.Rate) -> tuple[core.Point, core.Point]:
    """The light's start on the circle above the base point `light_source`, `[rays] Point-`, and its rays' headings,
    unit steps across the circles, `[rays] Point-`: the steps whose images on the base fan out in the plane
    `light_spreading` about the direction toward the base point `light_target`."""
    start = core.lifts(light_source[None], circling)                            # [1] Point-
    aim = (light_target - light_source * (core.polar(light_source) & light_target)).normalized()   # [] Point-
    angles = np.linspace(-light_spread / 2, light_spread / 2, n_rays)           # [rays]
    fan = (light_spreading * (-angles / 2)).exp()                               # [rays] Motor+
    return start.broadcast_to((n_rays,)), core.crossing(light_source, fan >> aim).normalized()


def light_paths(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, circling: core.Rate, starts: core.Point,
                headings: core.Point, time: float) -> core.Point:
    """The light's paths set out at a time through the waves, seen on the base, `[rays, steps + 1] Point-`."""
    places = stack([starts, *core.light(pick, waves, starts, headings, time, light_step, n_light_steps)],
                   axis=-1)                                                     # [rays, steps + 1] Point-
    return core.base_points(places, circling)


def gauge_states(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, frequency: float, bonds: Topology,
                 atoms: core.Point, times: np.ndarray) -> Iterator[tuple[core.Point, core.Scalar]]:
    """The gauges' atoms and their bonds' strains, `(Point-[atoms], Scalar+[bonds])`, at each of the times, after their
    motion has settled into the waves' swing at the frequency."""
    holding = (holding_share * frequency) ** 2
    # A triangle of unit masses swings in shear at the root of three halves of its bonds' stiffness and its atoms'
    # holding; its bonds' dashpots, three halves of them, and the holding's damp the swing.
    stiffness = 2 / 3 * (frequency ** 2 - holding)
    damping = 2 / 3 * (2 * gauge_damping * frequency - 2 * np.sqrt(holding))
    round_steps = core.round_steps(bonds, atoms)                                # Point-[bonds]
    rest = round_steps.dot(round_steps).square_root()                           # Scalar+[bonds]
    n_bonds = bonds.n_elements[1]
    stiffnesses = mv.scalar(np.full((n_bonds, 1), stiffness)).field()          # Scalar+[bonds]
    dampings = mv.scalar(np.full((n_bonds, 1), damping)).field()               # Scalar+[bonds]
    at_rest = mv(core.Step, np.zeros((bonds.n_vertices, 3))).field()           # Point-[atoms]
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    return core.massive(pick, waves, bonds, rest, stiffnesses, dampings, holding, atoms, at_rest, start, crystal_step,
                        times)


def quadrupole_history(cell_mesh: Topology, cell_vertices: core.Point) -> tuple[core.Waves, float]:
    """The 600-cell's travelling quadrupole as a smooth field over time, as far past the frames' swing as the light
    runs, and its frequency: the squash, stretching space along x and squeezing it along y in the frames, and the
    squash turned an eighth of a turn in the plane `xy`, a quarter swing behind it."""
    patterns = squash(cell_mesh, cell_vertices, stack((mv.x, eighth >> mv.x)),
                      stack((mv.y, eighth >> mv.y)))                            # [patterns] Scalar+[edges]
    return stepped_history(cell_mesh, cell_vertices, patterns, quadrupole_lags, swing, n_light_steps * light_step)


def quadrupole_frames(pick: Callable[[core.Point], np.ndarray], waves: core.Waves, frequency: float,
                      turn: core.Motor, bases: core.Point, sphere_points: core.Point) -> Iterator[tuple]:
    """One swing of the 600-cell's travelling quadrupole seen on the base below the circles the turn runs along, at
    each frame: the circles' stretch at the base's points `[points] Point-`, `Scalar+[points]`; the gauges in the
    faces of the hull of the base points `bases`, their atoms `[atoms] Point-`, their bonds' arcs `[bonds, arc points]
    Point-` and strains `[bonds] Scalar+`; and the light's paths `[rays, steps + 1] Point-`."""
    times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames   # [frames]
    circling = turn.log()                                                       # [] Rate+
    bonds, atoms = triangle_gauges(bases, circling)
    starts, headings = light_fan(circling)
    states = gauge_states(pick, waves, frequency, bonds, atoms, times)
    for time, (positions, strains) in zip(times, states):
        yield (core.stretches(waves, pick, sphere_points, sphere_points, turn, n_guardrail_sides, n_circle_samples,
                              time),
               core.base_points(positions.batch(), circling), bond_arcs(bonds, positions, circling), strains.batch(),
               light_paths(pick, waves, circling, starts, headings, time))


# --- plumbing -------------------------------------------------------------------------
def orbits(vertices: core.Point, turns: core.Motor, n_sides: int) -> np.ndarray:
    """The orbits of the vertices `Point-[vertices]` under each of the turns `[turns] Motor+`, which carry the vertices
    onto themselves in so many steps, `[turns, sides, vertices]` vertex indices: each turned vertex found among the
    vertices by its coordinates, read by pairing with the planes across the four directions, outside the algebra."""
    from scipy.spatial import cKDTree

    coordinate_planes = mv.vector(np.eye(4))                                    # [4] Plane+
    tree = cKDTree((coordinate_planes & vertices.batch()[:, None]).to_array())
    turned_vertices = (turns[:, None] >> vertices.batch()[None]).reshape(-1)    # [turns * vertices] Point-
    following = tree.query((coordinate_planes & turned_vertices[:, None]).to_array())[1].reshape(turns.shape[0], -1)
    steps = [np.broadcast_to(np.arange(following.shape[1]), following.shape)]
    for _ in range(n_sides - 1):
        steps.append(np.take_along_axis(following, steps[-1], axis=-1))
    return np.stack(steps, axis=1)


def base_hull(steps: core.Point) -> Topology:
    """The hull of unit steps of the origin `[n] Point-`, a closed surface of triangles on them, found from their
    coordinates outside the algebra: for the decagons' base points, the icosahedron."""
    from scipy.spatial import ConvexHull

    coordinates = (stack((mv.x, mv.y, mv.z)) & steps[:, None]).to_array()      # [n, 3]
    return Topology.closed(ConvexHull(coordinates).simplices, coordinates)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.relativity.closed_universe import render

    cell_mesh, cell_vertices = six_hundred_cell()
    pick = core.picking(cell_mesh, cell_vertices)
    # The round universe from inside: the flight along a Boerdijk–Coxeter helix's axis, down the tunnel the helix's
    # tetrahedra make about it with the step across it up, every great circle of edges in its own colour; and the
    # helix panned past, its rails in colour.
    round_walk, cell_walk_values = core.round_walk_source, render.round_values(cell_mesh, cell_vertices)
    rails, axis_point, axis_direction, side_direction = helix(cell_vertices)
    save_animation(render.polygon_view(round_walk, cell_walk_values, cell_mesh, cell_vertices,
                                       flight(facing(core.frame(axis_point), axis_direction, side_direction)),
                                       great_circles(cell_mesh, cell_vertices)),
                   "closed_universe_flight", duration_ms)
    save_animation(render.helix_view(round_walk, cell_walk_values, cell_mesh, cell_vertices,
                                     pan(axis_point, axis_direction, side_direction), rails),
                   "closed_universe_helix", duration_ms)
    # The irregular mesh, flown through straight ahead from the middle of its first tetrahedron.
    irregular_mesh, irregular_vertices = irregular_three_sphere()
    middle = (irregular_mesh.incidence[3, 0].mean(axis=0) * irregular_vertices).batch()[0].normalized()   # [] Point-
    save_animation(render.plain_view(round_walk, render.round_values(irregular_mesh, irregular_vertices),
                                     irregular_mesh, irregular_vertices, flight(core.frame(middle))),
                   "closed_universe_irregular", duration_ms)

    # checks
    # The stepped waves have no growing mode, the squash is their slowest, a fiftieth short of the continuum's eight on
    # edges a fifth of a half turn long, and the squash, a wave of the 600-cell by its symmetry, rings at the
    # leapfrog's frequency for its own.
    spectrum, squash_squared, times, shares = stepped_squash(cell_mesh, cell_vertices, n_ringing_swings)
    eigenvalues = spectrum.to_array()
    assert eigenvalues.min() > -1e-6 * eigenvalues.max() and 7.5 < squash_squared < 8
    step = times[1]
    leapfrog_frequency = 2 / step * np.arcsin(np.sqrt(squash_squared) * step / 2)
    assert np.abs(shares.to_array() - wave_swing * np.cos(leapfrog_frequency * times)).max() < 1e-10 * wave_swing
    # The irregular mesh's round chords are at rest, every deficit of its tetrahedra, read as pieces of the round
    # sphere, nothing.
    chords = irregular_mesh.d[0] * irregular_vertices                           # Point-[edges]
    deficits, _, _ = core.Space(irregular_mesh, irregular_vertices).dynamics(chords.dot(chords))
    assert (deficits.abs() < 1e-12).all()
    # The quadrupole is the same all along each circle, between the tenth turns that carry the mesh onto itself too: a
    # circle's stretch read at one point of it is its average round it.
    quadrupole, _ = quadrupole_history(cell_mesh, cell_vertices)
    turn = circle_turn(cell_vertices, n_guardrail_sides)                        # [] Motor+
    _, sphere_points = base_sphere()
    at_one = core.stretches(quadrupole, pick, sphere_points, sphere_points, turn, n_guardrail_sides, 1, 0.0)
    averaged = core.stretches(quadrupole, pick, sphere_points, sphere_points, turn, n_guardrail_sides,
                              n_circle_samples, 0.0)
    assert ((at_one - averaged).abs() < 1e-6 * np.abs(averaged.to_array()).max()).all()


def pinned() -> None:
    """The pictures the repository tracks: the quadrupole travelling round the base below the Hopf circles, for the
    gallery, and the waves seen from inside, for the README."""
    from examples.util.animation import pin_animation
    from examples.relativity.closed_universe import render

    cell_mesh, cell_vertices = six_hundred_cell()
    pick = core.picking(cell_mesh, cell_vertices)
    quadrupole, quadrupole_frequency = quadrupole_history(cell_mesh, cell_vertices)
    turn = circle_turn(cell_vertices, n_guardrail_sides)                        # [] Motor+
    bases = decagon_bases(cell_vertices, turn.log())                            # [decagons] Point-
    sphere_mesh, sphere_points = base_sphere()
    pin_animation((render.base_view(sphere_mesh, sphere_points, bases, *frame)
                   for frame in quadrupole_frames(pick, quadrupole, quadrupole_frequency, turn, bases, sphere_points)),
                  "closed_universe_quadrupole", duration_ms, colors=n_quadrupole_colours)

    # The squash, stretching space along x and squeezing it along y in the frames, alone; seen from the eye, whose
    # light at each frame left the crystals their distance before.
    squashing = squash(cell_mesh, cell_vertices, stack((mv.x,)), stack((mv.y,)))   # [patterns] Scalar+[edges]
    waves, frequency = stepped_history(cell_mesh, cell_vertices, squashing, squash_lags, wave_swing, 0.0)
    eye_times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames   # [frames]
    bonds, atoms = crystals(crystal_places())
    states = crystal_states(pick, waves, frequency, bonds, atoms, eye_times - crystal_distance)
    pin_animation(render.waves_view(core.wave_walk_source, render.wave_values(cell_mesh, waves), cell_mesh,
                                    cell_vertices, eye, eye_times, guardrails(cell_vertices, n_guardrail_sides),
                                    bonds, atoms, states),
                  "closed_universe_inside", duration_ms)


if __name__ == "__main__":
    main()
