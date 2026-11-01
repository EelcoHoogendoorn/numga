"""The closed universe from inside and below its Hopf circles. Round, without waves: a flight down the axis of one of
the 600-cell's Boerdijk–Coxeter helices, every great circle of its edges in its own colour, the helix panned past, its
rails in colour, and a flight through an irregular mesh. With waves: inside the 600-cell, its squash rings past three
crystals ahead of the eye, soft, tuned and stiff; on the irregular mesh, a burst spreads from a source ahead of the
eye, seen through the decagons and through beacons across the universe; and below the Hopf circles, a quadrupole
travels round the base, rung by triangle gauges held to their places and crossed by a bundle of light."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numga import stack
from numga.algebras import VGA3D
from numga.sparse import SparseExtensor

from examples import instantiate
from examples.relativity.closed_universe import core

# Round meshes of the three-sphere relaxed under the binary icosahedral group, in the same algebra; and the algebra of
# space, which the base is drawn in.
sphere_mesher = instantiate("examples.util.annealed_sphere", core.ga)
flat = instantiate("examples.util.mesh", VGA3D)

# The 600-cell placed by a turn that keeps its vertices off the origin and its antipode.
cell_placement = (core.mv.xw * 0.21 + core.mv.yw * 0.13 + core.mv.zw * 0.08).exp()   # [] Motor+

# The guardrails' great polygons: decagons, whose sides are the 600-cell's edges.
guardrail_sides = 10
# The eye inside the 600-cell, the motor that carries the origin to it and turns its view; the stepped gravitational
# waves, the slowest of the closed universe started from its squash, swinging space along one invariant direction by
# this share and the opposite way along another, at the lattice's own frequency; the frames over one swing.
eye = (core.mv.xw * 0.37 + core.mv.yw * 0.11 + core.mv.zw * 0.23).exp()   # [] Motor+
wave_swing = 0.12
n_wave_frames = 75
# Three little crystals ahead of the eye, side by side across its view: tetrahedra of four atoms of unit mass, bonded
# along their six edges, their bonds soft, tuned to the wave, and stiff: a sixteenth of the stiffness that puts the
# swing the squash drives, root two times the root of the stiffness, at the wave's, that one, and sixteen times it,
# with dashpots a quarter of critical. How far ahead, how far apart, how large; the time step, and the swings run
# before the frames so that the motion has settled.
crystal_distance, crystal_spacing, crystal_half_size = 0.55, 0.26, 0.05
tunings = np.array([1 / 16, 1.0, 16.0])                                       # [crystals]
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
# A burst of gravitational waves on it: the squash's pattern held to a bump of this width about a source this far ahead
# of the eye, just past the crystals, swinging space by this share at its middle, kept to the waves and let go. It is
# stepped at this share of the fastest wave's limit, found by so many repetitions, and seen over a whole turn, the time
# light takes around the sphere, after which the shell closes in on its source again, in so many frames played round
# and round; the bonds' strain that saturates their colour.
burst_distance = 0.8
burst_width = 0.5
burst_swing = 0.15
stability_share = 0.5
n_power_steps = 60
n_burst_frames = 120
burst_saturating_strain = 0.03
# The decagons drawn as great circles through the burst, their tubes' radius.
decagon_radius = 0.016
# Beacons across the universe: a crystal tuned to the burst at each of the 600-cell's vertices, placed as the decagons
# are, so that the eye sits in a cell's middle; their states kept at this share of the frames' spacing, for each beacon
# to be seen when its light left it; the bonds' strain that saturates their colour.
beacon_tuning = 1.0
beacon_state_share = 0.5
beacon_saturating_strain = 0.02

duration_ms = 40
# The round universe from inside. A Boerdijk–Coxeter helix of the 600-cell is a column of thirty tetrahedra stacked
# face to face round a great circle, its axis, its vertices on three of the decagons, those over one face of the
# icosahedron their base points make: flown along its axis once round in so many frames, every great circle of the
# 600-cell's edges in its own colour; and panned past once round its axis in so many frames, from this far from the
# axis, the view tilted by this much, its rails in colour. The irregular mesh, flown through once round the same way.
n_flight_frames = 180
n_pan_frames = 60
pan_distance = 1.02
pan_tilt = 0.36
# The base drawn as an icosphere, the icosahedron's faces split in four so many times; the points sampled round each
# circle for its average, so many that they fall between the tenth turns that carry the 600-cell onto itself; the
# frames over one swing of the waves, as the view from inside has them; the swing of space.
n_icosphere_levels = 3
n_circle_samples = 7
n_frames = n_wave_frames
swing = 0.085
# The squash stretches space along x and squeezes it along y in the frames; turned an eighth of a turn about z, it
# stretches and squeezes along the diagonals between them, and a quarter swing behind the squash, the two make a
# quadrupole that turns round about z, travelling round the base.
eighth = (core.mv.xy * (-np.pi / 8)).exp()                              # [] Motor+
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
# aimed about the direction toward the base point along y, which faces the viewer, and spread over this angle, as
# each frame begins; traced in steps of this length for a quarter turn, the base's half from pole to pole.
n_rays = 7
light_spread = np.radians(18)
light_step = 0.02
n_light_steps = int(np.pi / 2 / light_step)
# What the gauges and the light are tuned to show: a bond's strain from one end of its swing to the other, and a ray's
# largest sway on the screen from its undisturbed path, as shares of the picture.
target_strain = 0.2
target_sways = (0.1, 0.2)



def no_circles() -> tuple[core.Point, core.Point, np.ndarray]:
    """No great circles for the inside view: none to start from, none a quarter turn on, no inks."""
    nothing = core.mv.vector(np.zeros((0, 4))).dual()                   # [0] Point-
    return nothing, nothing, np.zeros((0, 4))


def six_hundred_cell() -> tuple[core.Mesh, core.Point]:
    """The round three-sphere's mesh, the 600-cell: the hull of the 120 unit icosians, the units along the axes, the
    halves of every sign, and the even orderings of halves of the golden ratio, one and its inverse, of every sign,
    placed by its turn."""
    from itertools import permutations, product
    from scipy.spatial import ConvexHull

    golden = (1 + np.sqrt(5)) / 2
    axes = np.concatenate([np.eye(4), -np.eye(4)])
    halves = np.array(list(product([-0.5, 0.5], repeat=4)))
    even = [order for order in permutations(range(4))
            if sum(order[i] > order[j] for i in range(4) for j in range(i + 1, 4)) % 2 == 0]
    signed = np.array(list(product([-1, 1], repeat=3))) * np.array([golden, 1, 1 / golden]) / 2   # [signs, 3]
    golden_ones = np.concatenate([np.pad(signed, ((0, 0), (0, 1)))[:, np.argsort(order)] for order in even])
    coordinates = np.concatenate([axes, halves, golden_ones])                  # [vertices, 4]
    mesh = core.Mesh.closed(ConvexHull(coordinates).simplices, coordinates)
    return mesh, cell_placement >> core.mv.antivector(coordinates).field()


def circle_turn(vertices: core.Point, n_sides: int) -> core.Motor:
    """The turn along the Hopf circles that the 600-cell's vertices lie on in great polygons of so many sides, `[]
    Motor+`: the isoclinic motor that carries the origin, before the placement, to the vertex that share of a whole
    turn from it, which carries the vertices onto themselves."""
    unplaced = (cell_placement.inverse() >> vertices).batch()          # [vertices] Point-
    # Each vertex's reading by the origin's equator, the cosine of its angle from the origin.
    readings = (core.equator & unplaced).to_array()                    # [vertices]
    target = unplaced[np.argmin(np.abs(readings - np.cos(2 * np.pi / n_sides)))]   # [] Point-
    return cell_placement * core.isoclinic(target) * cell_placement.inverse()


def guardrails(vertices: core.Point, n_sides: int) -> np.ndarray:
    """The 600-cell's vertices on great polygons of so many sides, `[polygons, sides]` vertex indices, great circles
    side by side, one Hopf fibration: each the orbit of a vertex under `circle_turn`. Ten sides give twelve decagons
    that the mesh's edges run along, six twenty hexagons and four thirty squares, over the icosahedron's vertices,
    faces and edges."""
    from scipy.spatial import cKDTree

    turn = circle_turn(vertices, n_sides)                                       # [] Motor+
    readings = core.mesh_module.readings
    following = cKDTree(readings(vertices.batch())).query(readings((turn >> vertices).batch()))[1]   # [vertices]
    orbits = [np.arange(len(following))]
    for _ in range(n_sides - 1):
        orbits.append(following[orbits[-1]])
    orbits = np.stack(orbits)                                                   # [sides, vertices]
    _, firsts = np.unique(orbits.min(axis=0), return_index=True)
    return orbits[:, firsts].T


def crystals_ahead() -> core.Motor:
    """The frames of the three crystals ahead of the eye, `[crystals] Motor+`: the eye carried ahead along its view
    and across it to each crystal's place, each by the rate that moves the origin along the eye's own axes."""
    moving, _, _ = core.invariant_generators()                          # [3] Rate+
    offsets = crystal_spacing * np.arange(-1, 2)                                # [crystals]
    ahead = (moving[2] * (-crystal_distance / 2)).exp()                         # [] Motor+
    across_view = (moving[0] * (-offsets / 2)).exp()                            # [crystals] Motor+
    return eye * ahead * across_view


def crystals(frames: core.Motor) -> tuple[core.Mesh, core.Point]:
    """Crystals in the frames `[crystals] Motor+`, motors that carry the origin to each crystal's middle: their bonds,
    a graph over their atoms, and the atoms, `Point-[atoms]`, at rest in the chart, each frame carried out to every
    other corner of a cube along its own axes, and each two corners bonded."""
    from itertools import product

    moving, _, _ = core.invariant_generators()                          # [3] Rate+
    cube = np.array(list(product([-1.0, 1.0], repeat=3)))                      # [8, 3]
    corners = cube[cube.prod(axis=-1) > 0]                                      # [corners, 3]
    out = ((moving * corners).sum(axis=-1) * (-crystal_half_size / 2)).exp()  # [corners] Motor+
    atoms = ((frames[:, None] * out[None]) >> core.origin).reshape(-1)  # [atoms] Point-
    # Bonds join every two corners of each crystal, which differ along two of the cube's axes.
    differing = (corners[:, None] != corners[None]).sum(axis=-1)
    pairs = np.argwhere(np.triu((differing >= 1) & (differing <= 2)))           # [bonds per crystal, 2]
    bonds = (pairs[None] + len(corners) * np.arange(frames.shape[0])[:, None, None]).reshape(-1, 2)
    return core.Mesh(len(corners) * frames.shape[0], bonds), atoms.field()


def squash(cell_mesh: core.Mesh, cell_vertices: core.Point, cell_rates: core.Rate,
           stretched: core.Plane, squeezed: core.Plane) -> core.Scalar:
    """The squash's change of each edge's squared length, per unit swing of space, `Scalar+[edges]`, for each pair of
    planes `[...] Plane+` that read the invariant directions it stretches and squeezes: its step's share along the
    stretched direction less its share along the squeezed one, twice, since a squared length changes twice as fast as
    the length."""
    steps = core.origin.commutator(cell_rates).cast(core.Step)   # Antivector-[edges]
    chords = cell_mesh.d[0] * cell_vertices                                     # Point-[edges]
    shares = ((stretched & steps).squared() - (squeezed & steps).squared()) / steps.dot(steps)   # [...] Scalar+[edges]
    return 2 * chords.dot(chords) * shares


def stepped_history(cell_mesh: core.Mesh, cell_vertices: core.Point, patterns: core.Scalar,
                    lags: np.ndarray, swing: float) -> tuple[core.Waves, float]:
    """The 600-cell's gravitational waves stepped from squash patterns `[patterns] Scalar+[edges]`, swinging space by
    `swing` as cosines at the lattice's frequency, each lagging by its share of `lags` `[patterns]`, from long enough
    before the first frame for the crystals to settle and the light to arrive, to the last: as a smooth field over
    time, and that frequency."""
    stiffness, masses, onto_waves = core.gravitational_waves(cell_mesh, cell_vertices)
    kept = onto_waves(patterns)                                                 # [patterns] Scalar+[edges]
    squared = (kept * stiffness(kept)).sites.sum().sum(axis=0) / (kept * masses(kept)).sites.sum().sum(axis=0)   # [] Scalar+
    frequency = float(np.sqrt(squared.to_array()))
    period = 2 * np.pi / frequency
    step = period / n_steps_per_swing
    first = -(n_settling_swings + 1) * period - np.pi
    n_levels = int(np.ceil((period - first) / step)) + 2
    phases = frequency * first - lags                                           # [patterns]
    strains = swing * (np.cos(phases) * kept).sum(axis=0)                       # Scalar+[edges]
    change_rates = -swing * frequency * (np.sin(phases) * kept).sum(axis=0)     # Scalar+[edges]
    stepped = core.waves_stepped(stiffness, masses, onto_waves, strains, change_rates, step)
    levels = [state for _, state in zip(range(n_levels), stepped)]
    return core.Waves(cell_mesh, cell_vertices, [changes for changes, _ in levels],
                              [change_rates for _, change_rates in levels], first, step), frequency


def crystal_states(cell_mesh: core.Mesh, cell_vertices: core.Point, waves: core.Waves,
                   frequency: float, bonds: core.Mesh, atoms: core.Point, crystal_tunings: np.ndarray,
                   times: np.ndarray) -> list:
    """The crystals' atoms and bonds' strains, `(Point-[atoms], Scalar+[bonds])`, at each of the times, after the motion
    has settled into the waves' swing, for crystals of the bonds over the atoms tuned by `crystal_tunings`
    `[crystals]`: the tuned crystal's stiffness puts the swing the squash drives, root two times the root of the
    stiffness, at the waves' frequency."""
    stiffnesses = frequency ** 2 / 2 * crystal_tunings                          # [crystals]
    dampings = 0.25 * np.sqrt(2 * stiffnesses)                                  # [crystals]
    _, _, steps = core.invariant_generators()                           # [3] Antivector-
    # The crystals' rest lengths: the lengths of their bonds' round steps.
    round_steps = core.bond_steps(bonds, atoms)                         # Antivector-[bonds]
    rest = round_steps.dot(round_steps).square_root()                           # Scalar+[bonds]
    per_crystal = len(bonds.edges) // len(stiffnesses)
    stiffness = core.mesh_module.as_scalar(np.repeat(stiffnesses, per_crystal)).field()
    damping = core.mesh_module.as_scalar(np.repeat(dampings, per_crystal)).field()
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    at_rest = (steps[0] * np.zeros(atoms.batch().shape[0])).field()           # Antivector-[atoms]
    return list(core.massive(cell_mesh, cell_vertices, waves, bonds, rest, stiffness, damping, 0.0, atoms,
                                     at_rest, start, crystal_step, times))


def irregular_three_sphere() -> tuple[core.Mesh, core.Point]:
    """An irregular mesh of the round three-sphere and its vertices, `Point-[vertices]`: the 600-cell's vertices, the
    origin's orbit under the binary icosahedral group, and the orbits of random seeds, relaxed until the tetrahedra
    between them are as round as the orbits allow."""
    turns = sphere_mesher.binary_icosahedral_turns()                            # [rotations] Rotor+
    seeds = core.mv.vector(np.random.default_rng(mesh_seed).normal(size=(n_mesh_orbits, 4))).dual().normalized()
    mesh, vertices, _ = list(sphere_mesher.relaxed(turns, turns >> core.origin, seeds, n_relax_steps,
                                                   relax_pull))[-1]
    return mesh, vertices


def burst_source() -> core.Point:
    """The burst's source, `[] Point-`: the eye carried ahead along its view."""
    moving, _, _ = core.invariant_generators()                          # [3] Rate+
    return (eye * (moving[2] * (-burst_distance / 2)).exp()) >> core.origin


def burst(mesh: core.Mesh, vertices: core.Point, rates: core.Rate
          ) -> tuple[core.Waves, float]:
    """A burst of gravitational waves on a mesh of the three-sphere as a smooth field over time, from half a turn before
    its release to a whole turn after, and its frequency, the root of its mean square frequency by the stiffness over
    the masses: the squash's pattern held to a bump about the source, one there and falling off with each edge's
    middle's distance from it over the burst's width, kept to the waves and let go."""
    mv = core.mv
    stiffness, masses, onto_waves = core.gravitational_waves(mesh, vertices)
    middles = (mesh.incidence[1, 0] * vertices).sum(axis=0).normalized()        # Point-[edges]
    bump = ((middles.dot(burst_source()) - 1) * (2 / burst_width ** 2)).exp()  # Scalar+[edges]
    strains = onto_waves(burst_swing * bump * squash(mesh, vertices, rates, mv.x, mv.y))   # Scalar+[edges]
    step = stability_share * 2 / fastest_frequency(stiffness, masses, onto_waves, strains)
    squared = (strains * stiffness(strains)).sites.sum() / (strains * masses(strains)).sites.sum()   # [] Scalar+
    # Let go from rest, the waves are the same a time before and after: before, a shell closing in on the source.
    # The eye sees back by up to half a turn, the time light takes to the antipode.
    n_after, n_before = int(np.ceil(2 * np.pi / step)) + 2, int(np.ceil(np.pi / step)) + 1
    after = [state for _, state in zip(range(n_after), core.waves_stepped(stiffness, masses, onto_waves, strains,
                                                                                  0 * strains, step))]
    levels = [(changes, -change_rates) for changes, change_rates in after[n_before:0:-1]] + after
    waves = core.Waves(mesh, vertices, [changes for changes, _ in levels],
                               [change_rates for _, change_rates in levels], -n_before * step, step)
    return waves, float(np.sqrt(squared.to_array()))


def fastest_frequency(stiffness: SparseExtensor, masses: SparseExtensor, onto_waves: core.OntoWaves,
                      start: core.Scalar) -> float:
    """The frequency of the fastest wave under the stiffness and the masses: the change the stiffness drives, the
    masses solved for and kept to the waves, repeated from a start until only the fastest is left."""
    changes = start
    for _ in range(n_power_steps):
        changes = onto_waves(masses.solve(stiffness(changes)))
        changes = changes / float(changes.dot(changes).sites.sum().square_root().to_array())
    squared = (changes * stiffness(changes)).sites.sum() / (changes * masses(changes)).sites.sum()
    return float(np.sqrt(squared.to_array()))


def decagon_circles(placement: core.Motor) -> tuple[core.Point, core.Point, np.ndarray]:
    """The 600-cell's guardrail decagons as great circles, one Hopf fibration of its vertices, each through its first
    vertex and the point a quarter turn on toward its second, `[decagons] Point-` each, moved by the placement, with
    inks around the hues and the squared tube radius `[decagons, 4]`."""
    from matplotlib.colors import hsv_to_rgb

    _, vertices = six_hundred_cell()
    decagons = guardrails(vertices, guardrail_sides)                            # [decagons, sides]
    firsts, seconds = vertices.batch()[decagons[:, 0]], vertices.batch()[decagons[:, 1]]   # [decagons] Point-, Point-
    quarters = (seconds - firsts * firsts.dot(seconds)).normalized()           # [decagons] Point-
    hues = np.arange(len(decagons)) / len(decagons)
    inks = hsv_to_rgb(np.stack([hues, np.full_like(hues, 0.75), np.full_like(hues, 0.95)], axis=-1))
    return (placement >> firsts, placement >> quarters,
            np.concatenate([inks, np.full((len(decagons), 1), decagon_radius ** 2)], axis=-1))


def clear_placement() -> core.Motor:
    """The motor that places the decagons so that the eye sits where they leave the most room: at the 600-cell's cell
    centre farthest from all of them, its nearness to a circle the cosine of its distance, read by the two points that
    span the circle; the choice of the farthest is made outside the algebra."""
    mesh, vertices = six_hundred_cell()
    centres = core.coframe(mesh, vertices).sum(axis=0).dual().normalized().batch()   # [cells] Point-
    starts, quarters, _ = decagon_circles(core.mv.rotor())             # [decagons] Point-, Point-
    along_start, along_quarter = centres[:, None].dot(starts[None]), centres[:, None].dot(quarters[None])
    nearness = (along_start * along_start + along_quarter * along_quarter).to_array().reshape(len(centres), -1)
    farthest = centres[nearness.max(axis=-1).argmin()]                          # [] Point-
    return eye * core.isoclinic(farthest).inverse()


def burst_view(mesh: core.Mesh, vertices: core.Point, rates: core.Rate,
               waves: core.Waves, frequency: float) -> Iterator[np.ndarray]:
    """The burst seen from inside the mesh by the eye over a whole turn from its release: the decagons as great
    circles, fixed in the round sphere, through the light the waves bend, the tetrahedra's edges unseen, and the
    crystals ahead tuned to the burst's frequency, which the waves shake."""
    from examples.relativity.closed_universe import render

    eye_times = np.linspace(0, 2 * np.pi, n_burst_frames, endpoint=False)       # [frames]
    bonds, atoms = crystals(crystals_ahead())
    # The light from the crystals left them their distance before it reached the eye.
    states = crystal_states(mesh, vertices, waves, frequency, bonds, atoms, tunings,
                            eye_times - crystal_distance)
    unseen = np.zeros((len(mesh.tets) * 6, 4), dtype=np.float32)
    return render.inside(mesh, vertices, waves, eye.broadcast_to(eye_times.shape), eye_times, unseen,
                         decagon_circles(clear_placement()),
                         render.crystal_frames(mesh, vertices, bonds, states, burst_saturating_strain))


def beacon_view(mesh: core.Mesh, vertices: core.Point, rates: core.Rate,
                waves: core.Waves, frequency: float) -> Iterator[np.ndarray]:
    """The burst seen from inside the mesh by the eye over a whole turn from its release, through beacons at the
    600-cell's vertices: crystals tuned to the burst, each seen as it was when its light left it, its distance from
    the eye before."""
    from examples.relativity.closed_universe import render

    eye_times = np.linspace(0, 2 * np.pi, n_burst_frames, endpoint=False)       # [frames]
    _, cell_vertices = six_hundred_cell()
    beacons = clear_placement() >> cell_vertices.batch()                        # [beacons] Point-
    bonds, atoms = crystals(core.isoclinic(beacons))
    # Each beacon's distance from the eye, the time its light takes to arrive; its states over the times any frame
    # sees it at.
    distances = np.arccos(np.clip((eye >> core.origin).dot(beacons).to_array(), -1, 1))   # [beacons]
    spacing = (eye_times[1] - eye_times[0]) * beacon_state_share
    times = np.arange(eye_times[0] - distances.max(), eye_times[-1] + spacing, spacing)   # [states]
    states = crystal_states(mesh, vertices, waves, frequency, bonds, atoms,
                            np.full(beacons.shape[0], beacon_tuning), times)
    positions = stack([atoms.batch() for atoms, _ in states])                  # [states, atoms] Point-
    strains = stack([strains.batch() for _, strains in states])                # [states, bonds] Scalar+
    # The state each frame sees each beacon in, and each atom's and bond's beacon.
    seen = np.rint((eye_times[:, None] - distances[None] - times[0]) / spacing).astype(int)   # [frames, beacons]
    atom_beacons = np.arange(positions.shape[1]) // (positions.shape[1] // beacons.shape[0])   # [atoms]
    bond_beacons = np.arange(strains.shape[1]) // (strains.shape[1] // beacons.shape[0])      # [bonds]
    seen_states = [(positions[seen[frame, atom_beacons], np.arange(positions.shape[1])].field(),
                    strains[seen[frame, bond_beacons], np.arange(strains.shape[1])].field())
                   for frame in range(len(eye_times))]
    unseen = np.zeros((len(mesh.tets) * 6, 4), dtype=np.float32)
    return render.inside(mesh, vertices, waves, eye.broadcast_to(eye_times.shape), eye_times, unseen, no_circles(),
                         render.crystal_frames(mesh, vertices, bonds, seen_states, beacon_saturating_strain))


def stepped_squash(cell_mesh: core.Mesh, cell_vertices: core.Point, cell_rates: core.Rate,
                   n_swings: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """The 600-cell's gravitational waves stepped from the squash at rest: the spectrum of their frequencies squared,
    the squash's own, and the times and the squash's share of the stepped strains at each, over `n_swings` of it."""
    stiffness, masses, onto_waves = core.gravitational_waves(cell_mesh, cell_vertices)
    # The waves' own stiffness, dense on the 600-cell's few edges: the stiffness between every two unit changes kept
    # to the waves.
    kept = onto_waves(core.mesh_module.as_scalar(np.eye(len(cell_mesh.edges))).field())   # [edges] Scalar+[edges]
    pairings = (kept[:, None] * stiffness(kept)[None, :]).sites.sum()           # [edges, edges] Scalar+
    waves_stiffness = core.ctx.extensor(core.ga.gatype((core.Scalar, core.Scalar)),
                                                pairings.to_array()[..., None, None]).field(0, 1)   # (Scalar+ <- Scalar+)[edges <- edges]
    values, _ = waves_stiffness.eigh(masses.to_dense())
    mv = core.mv
    pattern = onto_waves(squash(cell_mesh, cell_vertices, cell_rates, mv.x, mv.y))   # Scalar+[edges]
    norm = (pattern * masses(pattern)).sites.sum()                              # [] Scalar+
    frequency_squared = float(((pattern * stiffness(pattern)).sites.sum() / norm).to_array())
    # Forty steps to the squash's swing, well inside the stiffest wave's limit, two over its frequency.
    step = 2 * np.pi / np.sqrt(frequency_squared) / n_steps_per_swing
    n_steps = n_swings * n_steps_per_swing
    stepped = core.waves_stepped(stiffness, masses, onto_waves, wave_swing * pattern, 0 * pattern, step)
    shares = np.array([float(((pattern * masses(strains)).sites.sum() / norm).to_array())
                       for _, (strains, _) in zip(range(n_steps), stepped)])  # [steps]
    return values.to_array(), frequency_squared, step * np.arange(n_steps), shares


def round_universe(mesh: core.Mesh, vertices: core.Point) -> core.Waves:
    """No waves on a mesh of the round three-sphere: two moments of no change, the same at every time."""
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    nothing = 0 * chords.dot(chords)                                            # Scalar+[edges]
    return core.Waves(mesh, vertices, [nothing, nothing], [nothing, nothing], 0.0, 1.0)


def helix(cell_vertices: core.Point) -> tuple[np.ndarray, core.Point, core.Point, core.Point]:
    """A Boerdijk–Coxeter helix of the 600-cell: its rails, the decagons over the first face of the icosahedron their
    base points make, `[3, sides]` vertex indices; a point on its axis, the circle above the face's middle, `[]
    Point-`; and at that point the axis's direction and a step across it toward the middle of the face's first side,
    unit steps of the origin in the isoclinic frame there, `[] Point-` each."""
    from scipy.spatial import ConvexHull

    circling = circle_turn(cell_vertices, guardrail_sides).log()               # [] Rate+
    decagons = guardrails(cell_vertices, guardrail_sides)                       # [decagons, sides]
    decagon_bases = core.base_points(cell_vertices.batch()[decagons[:, 0]], circling)   # [decagons] Point-
    coordinates = (core.mv.vector(np.eye(4))[:3] & decagon_bases[..., None]).to_array()   # [decagons, 3]
    face = ConvexHull(coordinates).simplices[0]                                 # [3]
    corners = decagon_bases[face]                                               # [3] Point-
    middle = corners.sum(axis=0).normalized()                                   # [] Point-
    side = (corners[0] + corners[1]).normalized()                               # [] Point-
    toward_side = (side - middle * (core.polar(middle) & side)).cast(core.Step).normalized()   # [] Point-
    return decagons[face], core.lifts(middle, circling), middle, core.across(middle, toward_side).normalized()


def facing(placing: core.Motor, forward: core.Point, up: core.Point) -> core.Motor:
    """A camera, the motor `placing` that carries the origin to the eye followed by the turns that carry the origin's
    step along z to `forward` and then, about it, the one along y to `up`, unit steps of the origin square to each
    other in the eye's frame, `[] Point-` each."""
    mv = core.mv
    aiming = (forward / mv.z.dual()).square_root()                              # [] Motor+
    rolling = (up / (aiming >> mv.y.dual())).square_root()                      # [] Motor+
    return placing * rolling * aiming


def flight(start: core.Motor) -> core.Motor:
    """The cameras flown once round from a start, `[frames] Motor+`, each carried ahead along its view."""
    ahead = core.moving(core.mv.z.dual().cast(core.Step))                       # [] Rate+
    distances = np.linspace(0, 2 * np.pi, n_flight_frames, endpoint=False)    # [frames]
    return start * (ahead * (-distances / 2)).exp()


def pan(on_axis: core.Point, along: core.Point, across_axis: core.Point) -> core.Motor:
    """The cameras panned once round a helix's axis, `[frames] Motor+`: from the pan's distance across the axis,
    looking back at it with the axis up, tilted about the view, each carried along the axis by the turn in its plane,
    which keeps every point square to that plane."""
    out = core.isoclinic(on_axis) * (core.moving(across_axis) * (-pan_distance / 2)).exp()   # [] Motor+
    tilted = (core.turning(across_axis) * (-pan_tilt / 2)).exp() >> along       # [] Point-
    start = facing(out, -across_axis, tilted)                                   # [] Motor+
    # The turn along the axis: the rate that moves the point on it along its direction, carried there.
    along_axis = core.isoclinic(on_axis) >> core.moving(along)                  # [] Rate+
    angles = np.linspace(0, 2 * np.pi, n_pan_frames, endpoint=False)           # [frames]
    return (along_axis * (-angles / 2)).exp() * start


def base_sphere() -> tuple[flat.Mesh, flat.Point, core.Point]:
    """The icosphere drawn as the base and its points `Point-[points]` in the algebra of space; and the same points as
    unit steps of the three-sphere's origin, `[points] Point-`."""
    mesh, points = flat.Mesh.icosphere(n_icosphere_levels)
    coordinates = (flat.mv.vector(np.eye(3)) & points.batch()[..., None]).to_array()   # [points, 3]
    bases = core.mv.vector(np.pad(coordinates, ((0, 0), (0, 1)))).dual().cast(core.Step)
    return mesh, points, bases


def triangle_gauges(decagon_bases: core.Point, circling: core.Rate
                    ) -> tuple[core.Mesh, core.Point]:
    """The triangle gauges' bonds, a graph over their atoms, and the atoms, `Point-[atoms]` on the three-sphere: in
    each face of the icosahedron the decagons' base points make, the corners drawn toward the face's middle, lifted
    from the circle above the middle across the circles, so that the bonds run across them too."""
    from scipy.spatial import ConvexHull

    coordinates = (core.mv.vector(np.eye(4))[:3] & decagon_bases[..., None]).to_array()   # [decagons, 3]
    faces = ConvexHull(coordinates).simplices                                   # [faces, 3]
    corners = decagon_bases[faces]                                              # [faces, 3] Point-
    middles = corners.sum(axis=-1).normalized()                                 # [faces] Point-
    atoms = (middles[:, None] + gauge_share * (corners - middles[:, None])).normalized()   # [faces, 3] Point-
    # Each atom's direction along the base from its face's middle, and the cosine of the angle between them: the step
    # across the circles that the direction is the image of carries the middle's lift half that angle, which the Hopf
    # map doubles.
    cosines = core.polar(middles[:, None]) & atoms                     # [faces, 3] Scalar+
    directions = (atoms - middles[:, None] * cosines).cast(core.Step).normalized()   # [faces, 3] Point-
    moves = (core.moving(core.across(middles[:, None], directions)) * (-cosines.arccos() / 4)).exp()   # [faces, 3] Motor+
    lifted = (core.isoclinic(core.lifts(middles, circling))[:, None] * moves) >> core.origin   # [faces, 3] Point-
    # Each triangle's three sides, by its atoms' places, three to a face.
    firsts = 3 * np.arange(len(faces))[:, None] + np.arange(3)[None]           # [faces, 3]
    bonds = np.stack([firsts, firsts - firsts % 3 + (firsts + 1) % 3], axis=-1).reshape(-1, 2)
    return core.Mesh(3 * len(faces), bonds), lifted.reshape(-1).field()


def bond_arcs(bonds: core.Mesh, positions: core.Point, circling: core.Rate
              ) -> core.Point:
    """Each bond's great arc between its atoms on the three-sphere, `Point-[atoms]`, seen on the base: so many points
    along it, its chord's points moved out onto the sphere, and their base points, `[bonds, arc points] Point-`."""
    tails, heads = bonds.incidence[1, 0] * positions                            # Point-[bonds], Point-[bonds]
    shares = np.linspace(0, 1, n_arc_points)                                    # [arc points]
    along = (tails.batch()[:, None] * (1 - shares) + heads.batch()[:, None] * shares).normalized()   # [bonds, arc points] Point-
    return core.base_points(along, circling)


def light_fan(circling: core.Rate, source: core.Point, toward: core.Point
              ) -> tuple[core.Point, core.Point]:
    """The light's start on the circle above the base point `source`, `[rays] Point-`, and its rays' headings, unit
    steps across the circles, `[rays] Antivector-`: the steps whose images on the base fan out about the direction
    toward the base point `toward`, turned about the source."""
    start = core.lifts(source[None], circling)                                  # [1] Point-
    aim = (toward - source * (core.polar(source) & toward)).cast(core.Step).normalized()   # [] Point-
    angles = np.linspace(-light_spread / 2, light_spread / 2, n_rays)           # [rays]
    fan = (core.turning(source) * (-angles / 2)).exp()                  # [rays] Motor+
    return start.broadcast_to((n_rays,)), core.across(source, fan >> aim).normalized()


def light_paths(mesh: core.Mesh, vertices: core.Point, waves: core.Waves,
                circling: core.Rate, starts: core.Point, headings: core.Point, time: float
                ) -> core.Point:
    """The light's paths set out at a time through the waves, seen on the base, `[rays, steps + 1] Point-`."""
    places = stack([starts, *core.light(mesh, vertices, waves, starts, headings, time, light_step,
                                                n_light_steps)], axis=-1)       # [rays, steps + 1] Point-
    return core.base_points(places, circling)


def undisturbed_paths(circling: core.Rate, starts: core.Point, headings: core.Point
                      ) -> core.Point:
    """The light's paths through the round sphere, seen on the base, `[rays, steps + 1] Point-`: each start rolled
    along its heading on its great circle, in its isoclinic frame."""
    lengths = light_step * np.arange(n_light_steps + 1)                         # [steps + 1]
    rolls = (core.moving(headings)[:, None] * (-lengths / 2)).exp()     # [rays, steps + 1] Motor+
    places = (core.isoclinic(starts)[:, None] * rolls) >> core.origin   # [rays, steps + 1] Point-
    return core.base_points(places, circling)


def gauge_states(cell_mesh: core.Mesh, cell_vertices: core.Point, waves: core.Waves,
                 frequency: float, bonds: core.Mesh, atoms: core.Point, times: np.ndarray) -> list:
    """The gauges' atoms and their bonds' strains, `(Point-[atoms], Scalar+[bonds])`, at each of the times, after their
    motion has settled into the waves' swing at the frequency."""
    holding = (holding_share * frequency) ** 2
    # A triangle of unit masses swings in shear at the root of three halves of its bonds' stiffness and its atoms'
    # holding; its bonds' dashpots, three halves of them, and the holding's damp the swing.
    stiffness = 2 / 3 * (frequency ** 2 - holding)
    damping = 2 / 3 * (2 * gauge_damping * frequency - 2 * np.sqrt(holding))
    round_steps = core.bond_steps(bonds, atoms)                         # Antivector-[bonds]
    rest = round_steps.dot(round_steps).square_root()                           # Scalar+[bonds]
    n_bonds = len(bonds.edges)
    stiffnesses = core.mesh_module.as_scalar(np.full(n_bonds, stiffness)).field()   # Scalar+[bonds]
    dampings = core.mesh_module.as_scalar(np.full(n_bonds, damping)).field()       # Scalar+[bonds]
    _, _, steps = core.invariant_generators()                           # [3] Antivector-
    at_rest = (steps[0] * np.zeros(bonds.n_vertices)).field()                  # Antivector-[atoms]
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    return list(core.massive(cell_mesh, cell_vertices, waves, bonds, rest, stiffnesses, dampings, holding,
                                     atoms, at_rest, start, crystal_step, times))


def quadrupole_history(cell_mesh: core.Mesh, cell_vertices: core.Point
                       ) -> tuple[core.Waves, float]:
    """The 600-cell's travelling quadrupole as a smooth field over time, and its frequency: the squash, stretching
    space along x and squeezing it along y in the frames, and the squash turned an eighth of a turn about z, a quarter
    swing behind it."""
    mv = core.mv
    cell_rates = core.round_rates(cell_mesh, core.isoclinic(cell_vertices))   # Rate+[edges]
    patterns = squash(cell_mesh, cell_vertices, cell_rates, stack((mv.x, eighth >> mv.x)),
                              stack((mv.y, eighth >> mv.y)))                     # [patterns] Scalar+[edges]
    return stepped_history(cell_mesh, cell_vertices, patterns, np.array([0, np.pi / 2]), swing)


def quadrupole_on_base() -> tuple[core.Point, core.Point, list]:
    """One swing of the 600-cell's travelling quadrupole seen on the base: the decagons' base points `[decagons]
    Point-`, the light's undisturbed paths `[rays, steps + 1] Point-`, and at each frame the circles' stretch at the
    icosphere's points `Scalar+[points]`, the gauges' atoms `[atoms] Point-`, their bonds' arcs `[bonds, arc points]
    Point-` and strains `[bonds] Scalar+`, and the light's paths `[rays, steps + 1] Point-`."""
    cell_mesh, cell_vertices = six_hundred_cell()
    waves, frequency = quadrupole_history(cell_mesh, cell_vertices)
    times = np.arange(n_frames) * 2 * np.pi / frequency / n_frames             # [frames]
    turn = circle_turn(cell_vertices, guardrail_sides)          # [] Motor+
    circling = turn.log()                                                       # [] Rate+
    decagons = guardrails(cell_vertices, guardrail_sides)        # [decagons, sides]
    decagon_bases = core.base_points(cell_vertices.batch()[decagons[:, 0]], circling)   # [decagons] Point-
    _, _, bases = base_sphere()
    bonds, atoms = triangle_gauges(decagon_bases, circling)
    states = gauge_states(cell_mesh, cell_vertices, waves, frequency, bonds, atoms, times)
    mv = core.mv
    starts, headings = light_fan(circling, mv.z.dual().cast(core.Step), mv.y.dual().cast(core.Step))
    frames = [(core.stretches(waves, cell_mesh, cell_vertices, bases, bases, turn, guardrail_sides,
                              n_circle_samples, time),
               core.base_points(positions.batch(), circling), bond_arcs(bonds, positions, circling), strains.batch(),
               light_paths(cell_mesh, cell_vertices, waves, circling, starts, headings, time))
              for time, (positions, strains) in zip(times, states)]
    return decagon_bases, undisturbed_paths(circling, starts, headings), frames


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.closed_universe import render

    cell_mesh, cell_vertices = six_hundred_cell()
    cell_rates = core.round_rates(cell_mesh, core.isoclinic(cell_vertices))   # Rate+[edges]
    # The squash, stretching space along x and squeezing it along y in the frames, alone.
    squashing = squash(cell_mesh, cell_vertices, cell_rates, stack((core.mv.x,)), stack((core.mv.y,)))
    waves, frequency = stepped_history(cell_mesh, cell_vertices, squashing, np.zeros(1), wave_swing)
    eye_times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames   # [frames]
    bonds, atoms = crystals(crystals_ahead())
    # The light from the crystals left them their distance before it reached the eye.
    states = crystal_states(cell_mesh, cell_vertices, waves, frequency, bonds, atoms, tunings,
                            eye_times - crystal_distance)
    rails = render.polygon_struts(guardrails(cell_vertices, guardrail_sides))
    frames = render.inside(cell_mesh, cell_vertices, waves, eye.broadcast_to(eye_times.shape), eye_times,
                           render.edge_styles(cell_mesh, *rails), no_circles(),
                           render.crystal_frames(cell_mesh, cell_vertices, bonds, states, render.saturating_strain))
    save_animation(frames, "closed_universe_600_cell", duration_ms)
    spectrum, squash_squared, times, shares = stepped_squash(cell_mesh, cell_vertices, cell_rates, 4)

    # A burst of waves on an irregular mesh of the three-sphere, seen through the decagons and through beacons across
    # the universe.
    burst_mesh, burst_vertices = irregular_three_sphere()
    burst_rates = core.round_rates(burst_mesh, core.isoclinic(burst_vertices))   # Rate+[edges]
    burst_waves, burst_frequency = burst(burst_mesh, burst_vertices, burst_rates)
    no_struts = np.zeros((0, 2), dtype=int), np.zeros((0, 3))
    # From the middle of the irregular mesh's first tetrahedron, straight ahead.
    middle = (burst_mesh.incidence[3, 0].mean(axis=0) * burst_vertices).batch()[0].normalized()   # [] Point-
    save_animation(render.inside(burst_mesh, burst_vertices, round_universe(burst_mesh, burst_vertices),
                                 flight(core.isoclinic(middle)), np.zeros(n_flight_frames),
                                 render.edge_styles(burst_mesh, *no_struts), no_circles(),
                                 render.no_crystals(burst_mesh)), "closed_universe_irregular", duration_ms)
    save_animation(burst_view(burst_mesh, burst_vertices, burst_rates, burst_waves, burst_frequency),
                   "closed_universe_burst", duration_ms)
    save_animation(beacon_view(burst_mesh, burst_vertices, burst_rates, burst_waves, burst_frequency),
                   "closed_universe_beacons", duration_ms)

    # The round universe from inside: the flight along a Boerdijk–Coxeter helix's axis, every great circle of edges in
    # its own colour; and the helix panned past, its rails in colour.
    still = round_universe(cell_mesh, cell_vertices)
    rails, on_axis, along, across_axis = helix(cell_vertices)
    circles = render.edge_styles(cell_mesh, *render.great_polygon_struts(cell_mesh, cell_vertices))
    # Down the axis, the tunnel the helix's tetrahedra make about it, with the step across it up.
    save_animation(render.inside(cell_mesh, cell_vertices, still, flight(facing(core.isoclinic(on_axis), along,
                                                                                across_axis)),
                                 np.zeros(n_flight_frames), circles, no_circles(), render.no_crystals(cell_mesh)),
                   "closed_universe_flight", duration_ms)
    save_animation(render.inside(cell_mesh, cell_vertices, still, pan(on_axis, along, across_axis),
                                 np.zeros(n_pan_frames), render.edge_styles(cell_mesh, *render.helix_struts(cell_mesh, rails)),
                                 no_circles(), render.no_crystals(cell_mesh)),
                   "closed_universe_helix", duration_ms)

    # The travelling quadrupole below the Hopf circles, and how its gauges and light respond.
    sphere_mesh, sphere_points, _ = base_sphere()
    decagon_bases, undisturbed, base_frames = quadrupole_on_base()
    save_animation((render.base_view(sphere_mesh, sphere_points, stretches, decagon_bases, atoms, arcs, strains, light,
                                     wave_swing)
                    for stretches, atoms, arcs, strains, light in base_frames), "closed_universe_quadrupole",
                   duration_ms)
    save_figure(render.response(stack([strains for *_, strains, _ in base_frames]),
                                stack([light for *_, light in base_frames]), undisturbed, target_strain, target_sways),
                "closed_universe_quadrupole_response")

    # checks
    # Over the wave's swing, the soft crystal carries the wave's strain, the tuned one rings at about twice it, the
    # most a quarter of critical damping allows, and the stiff one keeps its shape.
    swings = np.abs(np.stack([strains.to_array() for _, strains in states])).reshape(len(states), len(tunings), -1)
    soft, tuned, stiff = swings.max(axis=(0, 2))
    assert 0.5 * wave_swing < soft < 1.5 * wave_swing and tuned > 1.5 * soft and stiff < 0.2 * wave_swing
    # The stepped waves have no growing mode, the squash is their slowest, a fiftieth short of the continuum's eight on
    # edges a fifth of a half turn long, and the stepped squash rings at its own frequency.
    assert spectrum.min() > -1e-6 * spectrum.max() and 7.5 < squash_squared < 8
    assert np.abs(shares - wave_swing * np.cos(np.sqrt(squash_squared) * times)).max() < 0.03 * wave_swing
    # The irregular mesh's round chords are at rest, every deficit of its tetrahedra, read as pieces of the round
    # sphere, nothing.
    chords = burst_mesh.d[0] * burst_vertices                                   # Point-[edges]
    deficits, _, _ = core.Space(burst_mesh, burst_vertices).dynamics(chords.dot(chords))
    assert np.abs(deficits.to_array()).max() < 1e-12
    # The quadrupole is the same all along each circle, between the tenth turns that carry the mesh onto itself too: a
    # circle's stretch read at one point of it is its average round it.
    quadrupole, _ = quadrupole_history(cell_mesh, cell_vertices)
    turn = circle_turn(cell_vertices, guardrail_sides)                          # [] Motor+
    _, _, bases = base_sphere()
    at_one = core.stretches(quadrupole, cell_mesh, cell_vertices, bases, bases, turn, guardrail_sides, 1, 0.0)
    averaged = core.stretches(quadrupole, cell_mesh, cell_vertices, bases, bases, turn, guardrail_sides,
                              n_circle_samples, 0.0)
    assert np.abs((at_one - averaged).to_array()).max() < 1e-6 * np.abs(averaged.to_array()).max()


if __name__ == "__main__":
    main()
