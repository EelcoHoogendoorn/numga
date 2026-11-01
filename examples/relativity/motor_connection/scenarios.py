"""Free paths rolled through flat meshes that carry a curved space's motors on their edges, by the same lines in the
plane and in spacetime: a fan of geodesics on the unit sphere, which set out from one point and meet again at its
antipode, and a fan of particles thrown from one event in anti-de Sitter spacetime at every speed, which meet again
after the same proper time, with two rays of light running along the chart's light cone."""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType

import numpy as np

from numga import Algebra, Extensor, concatenate, stack
from numga.algebras import PGA2D, ProjectiveMultivectorFactory
from numga.sparse import SparseExtensor
from examples import instantiate

plane = instantiate("examples.relativity.motor_connection.core", PGA2D)
spacetime = instantiate("examples.relativity.motor_connection.core",
                        Algebra("x+t-w0", multivector_factory=ProjectiveMultivectorFactory))
sphere = instantiate("examples.relativity.motor_connection.core", Algebra("x+y+w+"))
three_sphere = instantiate("examples.relativity.motor_connection.core", Algebra("x+y+z+w+"))
# Round meshes of the three-sphere relaxed under the binary icosahedral group, in the same algebra.
sphere_mesher = instantiate("examples.surfaces.annealed_sphere.core", three_sphere.ga)

# The sphere's chart, a disc out to this radius, of about this many triangles across its radius.
chart_radius = 2.6
n_divisions = 16
# The paths' start in the chart, its antipode on the sphere, on the far side of the chart's centre at the inverse of
# the start's distance from it, and the headings of the fan about the direction toward it.
source_distance = 0.6
source = plane.mv.point([-source_distance, 0.0])                                # [] Point-
antipode = plane.mv.point([1 / source_distance, 0.0])                           # [] Point-
n_paths = 13
spread = np.radians(75)
turns = (plane.mv.xy * (-np.linspace(-spread, spread, n_paths) / 2)).exp()      # [paths] Motor+
headings = turns >> plane.mv.direction([1.0, 0.0])                              # [paths] Antivector-

# Anti-de Sitter's strip, across to this distance from its middle, in radii of the space, short of its boundary at a
# quarter turn, and over these times, of nearly equilateral triangles this wide.
strip_half_width = 1.35
strip_times = np.array([-0.15, 1.15 * np.pi + 0.15])
lattice_spacing = 0.12
# One unit along time, the direction across the slice `mv.t`, and one unit across the strip; the event the particles
# are thrown from, and where they meet again, half a turn of the radius later; their rapidities in the rolled
# spacetime, and the two rays of light.
time = -spacetime.mv.t.dual()                                                   # [] Antivector-
sideways = spacetime.mv.x.dual()                                                # [] Antivector-
event = spacetime.origin                                                        # [] Point-
meeting = spacetime.origin + time * np.pi                                       # [] Point-
rapidities = np.linspace(-1.5, 1.5, 9)
thrown = (spacetime.mv.xt * (-rapidities / 2)).exp() >> time                    # [particles] Antivector-
rays = concatenate([(time - sideways)[None], thrown, (time + sideways)[None]])  # [paths] Antivector-

# The round spheres' meshes, placed by a turn that keeps every vertex off the antipode of the origin, where the paths
# start; the paths' headings about the origin, in a fan on the sphere and over the turns of the three-sphere.
n_sphere_levels = 3
sphere_placement = (sphere.mv.xw * 0.21 + sphere.mv.yw * 0.13).exp()           # [] Motor+
sphere_headings = (sphere.mv.xy * (-np.linspace(0, 2 * np.pi, n_paths, endpoint=False) / 2)).exp() >> sphere.mv.x.dual()   # [paths] Antivector-
three_sphere_placement = (three_sphere.mv.xw * 0.21 + three_sphere.mv.yw * 0.13 + three_sphere.mv.zw * 0.08).exp()   # [] Motor+
spins = (three_sphere.mv.xy * (-np.linspace(0, 2 * np.pi, n_paths, endpoint=False)[:, None] / 2)).exp()   # [paths, 1] Motor+
tilts = (three_sphere.mv.zx * (-np.radians([30.0, 90.0, 150.0]) / 2)).exp()   # [3] Motor+
three_sphere_headings = (spins * tilts >> three_sphere.mv.z.dual()).reshape(-1)   # [paths * 3] Antivector-

# The guardrails' great polygons: decagons, whose sides are the 600-cell's edges.
guardrail_sides = 10
# The eye inside the 600-cell, the motor that carries the origin to it and turns its view; the stepped gravitational
# waves, the slowest of the closed universe started from its squash, swinging space along one invariant direction by
# this share and the opposite way along another, at the lattice's own frequency; the frames over one swing.
eye = (three_sphere.mv.xw * 0.37 + three_sphere.mv.yw * 0.11 + three_sphere.mv.zw * 0.23).exp()   # [] Motor+
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

# The step of length in the rolled space, the proper time of a particle, and the steps to half a turn and beyond.
step = 0.02
n_steps = int(1.15 * np.pi / step)
n_half_turn = int(round(np.pi / step))
duration_ms = 40


def chart() -> tuple[plane.Mesh, plane.Point]:
    """The sphere's chart, a disc of triangles, its vertices placed about the origin."""
    mesh, offsets = plane.Mesh.concentric_disk(np.array([chart_radius]), n_divisions)
    return mesh, plane.origin + offsets


def strip() -> tuple[spacetime.Mesh, spacetime.Point]:
    """Anti-de Sitter's strip, triangulated from a lattice of nearly equilateral triangles, every other row shifted by
    half a step across."""
    times = np.arange(strip_times[0], strip_times[1], lattice_spacing * np.sqrt(3) / 2)   # [rows]
    columns = np.arange(-strip_half_width, strip_half_width + lattice_spacing, lattice_spacing)   # [columns]
    shifts = (np.arange(len(times)) % 2)[:, None] * lattice_spacing / 2       # [rows, 1]
    offsets = np.clip(columns[None, :] + shifts, -strip_half_width, strip_half_width)   # [rows, columns]
    lattice = np.unique(np.stack(np.broadcast_arrays(offsets, times[:, None]), axis=-1).reshape(-1, 2), axis=0)
    vertices = (spacetime.origin + sideways * lattice[:, 0] + time * lattice[:, 1]).field()   # Point-[vertices]
    return spacetime.Mesh.triangulate(vertices), vertices


def icosphere() -> tuple[sphere.Mesh, sphere.Point]:
    """The round sphere's mesh, an icosahedron split in four a few times, placed by its turn."""
    mesh, vertices = sphere.Mesh.icosphere(n_sphere_levels)
    return mesh, sphere_placement >> vertices


def no_circles() -> tuple[three_sphere.Point, three_sphere.Point, np.ndarray]:
    """No great circles for the inside view: none to start from, none a quarter turn on, no inks."""
    nothing = three_sphere.mv.vector(np.zeros((0, 4))).dual()                   # [0] Point-
    return nothing, nothing, np.zeros((0, 4))


def six_hundred_cell() -> tuple[three_sphere.Mesh, three_sphere.Point]:
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
    mesh = three_sphere.Mesh.closed(ConvexHull(coordinates).simplices, coordinates)
    return mesh, three_sphere_placement >> three_sphere.mv.antivector(coordinates).field()


def great_circle_frames(vertices: sphere.Point) -> sphere.Motor:
    """Frames for the round sphere, `Motor+[vertices]`: the motors that carry the origin to each vertex along a great
    circle, smooth everywhere but at the origin's antipode."""
    return (vertices.normalized() / sphere.origin).square_root()


def circle_turn(vertices: three_sphere.Point, n_sides: int) -> three_sphere.Motor:
    """The turn along the Hopf circles that the 600-cell's vertices lie on in great polygons of so many sides, `[]
    Motor+`: the isoclinic motor that carries the origin, before the placement, to the vertex that share of a whole
    turn from it, which carries the vertices onto themselves."""
    unplaced = (three_sphere_placement.inverse() >> vertices).batch()          # [vertices] Point-
    # Each vertex's reading by the origin's equator, the cosine of its angle from the origin.
    readings = (three_sphere.infinity & unplaced).to_array()                    # [vertices]
    target = unplaced[np.argmin(np.abs(readings - np.cos(2 * np.pi / n_sides)))]   # [] Point-
    return three_sphere_placement * three_sphere.isoclinic(target) * three_sphere_placement.inverse()


def guardrails(vertices: three_sphere.Point, n_sides: int) -> np.ndarray:
    """The 600-cell's vertices on great polygons of so many sides, `[polygons, sides]` vertex indices, great circles
    side by side, one Hopf fibration: each the orbit of a vertex under `circle_turn`. Ten sides give twelve decagons
    that the mesh's edges run along, six twenty hexagons and four thirty squares, over the icosahedron's vertices,
    faces and edges."""
    from scipy.spatial import cKDTree

    turn = circle_turn(vertices, n_sides)                                       # [] Motor+
    readings = three_sphere.mesh_module.readings
    following = cKDTree(readings(vertices.batch())).query(readings((turn >> vertices).batch()))[1]   # [vertices]
    orbits = [np.arange(len(following))]
    for _ in range(n_sides - 1):
        orbits.append(following[orbits[-1]])
    orbits = np.stack(orbits)                                                   # [sides, vertices]
    _, firsts = np.unique(orbits.min(axis=0), return_index=True)
    return orbits[:, firsts].T


def crystals_ahead() -> three_sphere.Motor:
    """The frames of the three crystals ahead of the eye, `[crystals] Motor+`: the eye carried ahead along its view
    and across it to each crystal's place, each by the rate that moves the origin along the eye's own axes."""
    moving, _, _ = three_sphere.invariant_generators()                          # [3] Rate+
    offsets = crystal_spacing * np.arange(-1, 2)                                # [crystals]
    ahead = (moving[2] * (-crystal_distance / 2)).exp()                         # [] Motor+
    across_view = (moving[0] * (-offsets / 2)).exp()                            # [crystals] Motor+
    return eye * ahead * across_view


def crystals(frames: three_sphere.Motor) -> tuple[three_sphere.Mesh, three_sphere.Point]:
    """Crystals in the frames `[crystals] Motor+`, motors that carry the origin to each crystal's middle: their bonds,
    a graph over their atoms, and the atoms, `Point-[atoms]`, at rest in the chart, each frame carried out to every
    other corner of a cube along its own axes, and each two corners bonded."""
    from itertools import product

    moving, _, _ = three_sphere.invariant_generators()                          # [3] Rate+
    cube = np.array(list(product([-1.0, 1.0], repeat=3)))                      # [8, 3]
    corners = cube[cube.prod(axis=-1) > 0]                                      # [corners, 3]
    out = ((moving * corners).sum(axis=-1) * (-crystal_half_size / 2)).exp()  # [corners] Motor+
    atoms = ((frames[:, None] * out[None]) >> three_sphere.origin).reshape(-1)  # [atoms] Point-
    # Bonds join every two corners of each crystal, which differ along two of the cube's axes.
    differing = (corners[:, None] != corners[None]).sum(axis=-1)
    pairs = np.argwhere(np.triu((differing >= 1) & (differing <= 2)))           # [bonds per crystal, 2]
    bonds = (pairs[None] + len(corners) * np.arange(frames.shape[0])[:, None, None]).reshape(-1, 2)
    return three_sphere.Mesh(len(corners) * frames.shape[0], bonds), atoms.field()


def squash(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point, cell_rates: three_sphere.Rate,
           stretched: three_sphere.Plane, squeezed: three_sphere.Plane) -> three_sphere.Scalar:
    """The squash's change of each edge's squared length, per unit swing of space, `Scalar+[edges]`, for each pair of
    planes `[...] Plane+` that read the invariant directions it stretches and squeezes: its step's share along the
    stretched direction less its share along the squeezed one, twice, since a squared length changes twice as fast as
    the length."""
    steps = three_sphere.origin.commutator(cell_rates).cast(three_sphere.Step)   # Antivector-[edges]
    chords = cell_mesh.d[0] * cell_vertices                                     # Point-[edges]
    shares = ((stretched & steps).squared() - (squeezed & steps).squared()) / steps.dot(steps)   # [...] Scalar+[edges]
    return 2 * chords.dot(chords) * shares


def stepped_history(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point, patterns: three_sphere.Scalar,
                    lags: np.ndarray, swing: float) -> tuple[three_sphere.Waves, float]:
    """The 600-cell's gravitational waves stepped from squash patterns `[patterns] Scalar+[edges]`, swinging space by
    `swing` as cosines at the lattice's frequency, each lagging by its share of `lags` `[patterns]`, from long enough
    before the first frame for the crystals to settle and the light to arrive, to the last: as a smooth field over
    time, and that frequency."""
    stiffness, masses, onto_waves = three_sphere.gravitational_waves(cell_mesh, cell_vertices)
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
    stepped = three_sphere.waves_stepped(stiffness, masses, onto_waves, strains, change_rates, step)
    levels = [state for _, state in zip(range(n_levels), stepped)]
    return three_sphere.Waves(cell_mesh, cell_vertices, [changes for changes, _ in levels],
                              [change_rates for _, change_rates in levels], first, step), frequency


def crystal_states(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point, waves: three_sphere.Waves,
                   frequency: float, bonds: three_sphere.Mesh, atoms: three_sphere.Point, crystal_tunings: np.ndarray,
                   times: np.ndarray) -> list:
    """The crystals' atoms and bonds' strains, `(Point-[atoms], Scalar+[bonds])`, at each of the times, after the motion
    has settled into the waves' swing, for crystals of the bonds over the atoms tuned by `crystal_tunings`
    `[crystals]`: the tuned crystal's stiffness puts the swing the squash drives, root two times the root of the
    stiffness, at the waves' frequency."""
    stiffnesses = frequency ** 2 / 2 * crystal_tunings                          # [crystals]
    dampings = 0.25 * np.sqrt(2 * stiffnesses)                                  # [crystals]
    _, _, steps = three_sphere.invariant_generators()                           # [3] Antivector-
    # The crystals' rest lengths: the lengths of their bonds' round steps.
    round_steps = three_sphere.bond_steps(bonds, atoms)                         # Antivector-[bonds]
    rest = round_steps.dot(round_steps).square_root()                           # Scalar+[bonds]
    per_crystal = len(bonds.edges) // len(stiffnesses)
    stiffness = three_sphere.mesh_module.as_scalar(np.repeat(stiffnesses, per_crystal)).field()
    damping = three_sphere.mesh_module.as_scalar(np.repeat(dampings, per_crystal)).field()
    start = times[0] - n_settling_swings * 2 * np.pi / frequency
    at_rest = (steps[0] * np.zeros(atoms.batch().shape[0])).field()           # Antivector-[atoms]
    return list(three_sphere.massive(cell_mesh, cell_vertices, waves, bonds, rest, stiffness, damping, 0.0, atoms,
                                     at_rest, start, crystal_step, times))


def irregular_three_sphere() -> tuple[three_sphere.Mesh, three_sphere.Point]:
    """An irregular mesh of the round three-sphere and its vertices, `Point-[vertices]`: the 600-cell's vertices, the
    origin's orbit under the binary icosahedral group, and the orbits of random seeds, relaxed until the tetrahedra
    between them are as round as the orbits allow."""
    turns = sphere_mesher.binary_icosahedral_turns()                            # [rotations] Rotor+
    seeds = three_sphere.mv.vector(np.random.default_rng(mesh_seed).normal(size=(n_mesh_orbits, 4))).dual().normalized()
    mesh, vertices, _ = list(sphere_mesher.relaxed(turns, turns >> three_sphere.origin, seeds, n_relax_steps,
                                                   relax_pull))[-1]
    return mesh, vertices


def burst_source() -> three_sphere.Point:
    """The burst's source, `[] Point-`: the eye carried ahead along its view."""
    moving, _, _ = three_sphere.invariant_generators()                          # [3] Rate+
    return (eye * (moving[2] * (-burst_distance / 2)).exp()) >> three_sphere.origin


def burst(mesh: three_sphere.Mesh, vertices: three_sphere.Point, rates: three_sphere.Rate
          ) -> tuple[three_sphere.Waves, float]:
    """A burst of gravitational waves on a mesh of the three-sphere as a smooth field over time, from half a turn before
    its release to a whole turn after, and its frequency, the root of its mean square frequency by the stiffness over
    the masses: the squash's pattern held to a bump about the source, one there and falling off with each edge's
    middle's distance from it over the burst's width, kept to the waves and let go."""
    mv = three_sphere.mv
    stiffness, masses, onto_waves = three_sphere.gravitational_waves(mesh, vertices)
    middles = (mesh.incidence[1, 0] * vertices).sum(axis=0).normalized()        # Point-[edges]
    bump = ((middles.dot(burst_source()) - 1) * (2 / burst_width ** 2)).exp()  # Scalar+[edges]
    strains = onto_waves(burst_swing * bump * squash(mesh, vertices, rates, mv.x, mv.y))   # Scalar+[edges]
    step = stability_share * 2 / fastest_frequency(stiffness, masses, onto_waves, strains)
    squared = (strains * stiffness(strains)).sites.sum() / (strains * masses(strains)).sites.sum()   # [] Scalar+
    # Let go from rest, the waves are the same a time before and after: before, a shell closing in on the source.
    # The eye sees back by up to half a turn, the time light takes to the antipode.
    n_after, n_before = int(np.ceil(2 * np.pi / step)) + 2, int(np.ceil(np.pi / step)) + 1
    after = [state for _, state in zip(range(n_after), three_sphere.waves_stepped(stiffness, masses, onto_waves, strains,
                                                                                  0 * strains, step))]
    levels = [(changes, -change_rates) for changes, change_rates in after[n_before:0:-1]] + after
    waves = three_sphere.Waves(mesh, vertices, [changes for changes, _ in levels],
                               [change_rates for _, change_rates in levels], -n_before * step, step)
    return waves, float(np.sqrt(squared.to_array()))


def fastest_frequency(stiffness: SparseExtensor, masses: SparseExtensor, onto_waves: three_sphere.OntoWaves,
                      start: three_sphere.Scalar) -> float:
    """The frequency of the fastest wave under the stiffness and the masses: the change the stiffness drives, the
    masses solved for and kept to the waves, repeated from a start until only the fastest is left."""
    changes = start
    for _ in range(n_power_steps):
        changes = onto_waves(masses.solve(stiffness(changes)))
        changes = changes / float(changes.dot(changes).sites.sum().square_root().to_array())
    squared = (changes * stiffness(changes)).sites.sum() / (changes * masses(changes)).sites.sum()
    return float(np.sqrt(squared.to_array()))


def decagon_circles(placement: three_sphere.Motor) -> tuple[three_sphere.Point, three_sphere.Point, np.ndarray]:
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


def clear_placement() -> three_sphere.Motor:
    """The motor that places the decagons so that the eye sits where they leave the most room: at the 600-cell's cell
    centre farthest from all of them, its nearness to a circle the cosine of its distance, read by the two points that
    span the circle; the choice of the farthest is made outside the algebra."""
    mesh, vertices = six_hundred_cell()
    centres = three_sphere.coframe(mesh, vertices).sum(axis=0).dual().normalized().batch()   # [cells] Point-
    starts, quarters, _ = decagon_circles(three_sphere.mv.rotor())             # [decagons] Point-, Point-
    along_start, along_quarter = centres[:, None].dot(starts[None]), centres[:, None].dot(quarters[None])
    nearness = (along_start * along_start + along_quarter * along_quarter).to_array().reshape(len(centres), -1)
    farthest = centres[nearness.max(axis=-1).argmin()]                          # [] Point-
    return eye * three_sphere.isoclinic(farthest).inverse()


def burst_view(mesh: three_sphere.Mesh, vertices: three_sphere.Point, rates: three_sphere.Rate,
               waves: three_sphere.Waves, frequency: float) -> Iterator[np.ndarray]:
    """The burst seen from inside the mesh by the eye over a whole turn from its release: the decagons as great
    circles, fixed in the round sphere, through the light the waves bend, the tetrahedra's edges unseen, and the
    crystals ahead tuned to the burst's frequency, which the waves shake."""
    from examples.relativity.motor_connection import render

    eye_times = np.linspace(0, 2 * np.pi, n_burst_frames, endpoint=False)       # [frames]
    bonds, atoms = crystals(crystals_ahead())
    # The light from the crystals left them their distance before it reached the eye.
    states = crystal_states(mesh, vertices, waves, frequency, bonds, atoms, tunings,
                            eye_times - crystal_distance)
    unseen = np.zeros((len(mesh.tets) * 6, 4), dtype=np.float32)
    return render.inside(three_sphere, mesh, vertices, eye, waves, eye_times, unseen,
                         decagon_circles(clear_placement()), bonds, states, burst_saturating_strain)


def beacon_view(mesh: three_sphere.Mesh, vertices: three_sphere.Point, rates: three_sphere.Rate,
                waves: three_sphere.Waves, frequency: float) -> tuple[Iterator[np.ndarray], three_sphere.Point, list]:
    """The burst seen from inside the mesh by the eye over a whole turn from its release, through beacons at the
    600-cell's vertices: crystals tuned to the burst, each seen as it was when its light left it, its distance from
    the eye before; and the beacons' middles `[beacons] Point-` and their atoms and bonds' strains as each frame sees
    them, `(Point-[atoms], Scalar+[bonds])`."""
    from examples.relativity.motor_connection import render

    eye_times = np.linspace(0, 2 * np.pi, n_burst_frames, endpoint=False)       # [frames]
    _, cell_vertices = six_hundred_cell()
    beacons = clear_placement() >> cell_vertices.batch()                        # [beacons] Point-
    bonds, atoms = crystals(three_sphere.isoclinic(beacons))
    # Each beacon's distance from the eye, the time its light takes to arrive; its states over the times any frame
    # sees it at.
    distances = np.arccos(np.clip((eye >> three_sphere.origin).dot(beacons).to_array(), -1, 1))   # [beacons]
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
    frames = render.inside(three_sphere, mesh, vertices, eye, waves, eye_times, unseen, no_circles(),
                           bonds, seen_states, beacon_saturating_strain)
    return frames, beacons, seen_states


def stepped_squash(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point, cell_rates: three_sphere.Rate,
                   n_swings: int) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """The 600-cell's gravitational waves stepped from the squash at rest: the spectrum of their frequencies squared,
    the squash's own, and the times and the squash's share of the stepped strains at each, over `n_swings` of it."""
    stiffness, masses, onto_waves = three_sphere.gravitational_waves(cell_mesh, cell_vertices)
    # The waves' own stiffness, dense on the 600-cell's few edges: the stiffness between every two unit changes kept
    # to the waves.
    kept = onto_waves(three_sphere.mesh_module.as_scalar(np.eye(len(cell_mesh.edges))).field())   # [edges] Scalar+[edges]
    pairings = (kept[:, None] * stiffness(kept)[None, :]).sites.sum()           # [edges, edges] Scalar+
    waves_stiffness = three_sphere.ctx.extensor(three_sphere.ga.gatype((three_sphere.Scalar, three_sphere.Scalar)),
                                                pairings.to_array()[..., None, None]).field(0, 1)   # (Scalar+ <- Scalar+)[edges <- edges]
    values, _ = waves_stiffness.eigh(masses.to_dense())
    mv = three_sphere.mv
    pattern = onto_waves(squash(cell_mesh, cell_vertices, cell_rates, mv.x, mv.y))   # Scalar+[edges]
    norm = (pattern * masses(pattern)).sites.sum()                              # [] Scalar+
    frequency_squared = float(((pattern * stiffness(pattern)).sites.sum() / norm).to_array())
    # Forty steps to the squash's swing, well inside the stiffest wave's limit, two over its frequency.
    step = 2 * np.pi / np.sqrt(frequency_squared) / n_steps_per_swing
    n_steps = n_swings * n_steps_per_swing
    stepped = three_sphere.waves_stepped(stiffness, masses, onto_waves, wave_swing * pattern, 0 * pattern, step)
    shares = np.array([float(((pattern * masses(strains)).sites.sum() / norm).to_array())
                       for _, (strains, _) in zip(range(n_steps), stepped)])  # [steps]
    return values.to_array(), frequency_squared, step * np.arange(n_steps), shares


def rolled(core: ModuleType, mesh, vertices: Extensor, rates: Extensor, starts: Extensor, directions: Extensor
           ) -> Extensor:
    """The paths from the starts along the directions, `[paths, steps + 1] Point-`, through the connection of the
    edges' rates, in the algebra of the core's instance."""
    ramps = core.coframe(mesh, vertices)                                        # [corners] Plane+[faces]
    connection = core.connection(mesh, ramps, rates)                            # (Rate+ <- Line-)[faces]
    return stack(list(core.geodesics(mesh, vertices, ramps, connection, starts, directions, step, n_steps)), axis=-1)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.relativity.motor_connection import render

    mesh, vertices = chart()
    paths = rolled(plane, mesh, vertices, plane.sphere_rates(mesh, vertices), source[None].broadcast_to((n_paths,)),
                   headings)                                                    # [paths, steps + 1] Point-
    extent = (-chart_radius, chart_radius, -chart_radius, chart_radius)
    save_animation(render.animate(mesh, vertices, paths, extent), "motor_connection_sphere", duration_ms)

    strip_mesh, strip_vertices = strip()
    worldlines = rolled(spacetime, strip_mesh, strip_vertices, spacetime.anti_de_sitter_rates(strip_mesh, strip_vertices),
                        event[None].broadcast_to(rays.shape), rays)             # [paths, steps + 1] Point-
    strip_extent = (-strip_half_width, strip_half_width, *strip_times)
    save_animation(render.animate(strip_mesh, strip_vertices, worldlines, strip_extent), "motor_connection_anti_de_sitter",
                   duration_ms)

    cell_mesh, cell_vertices = six_hundred_cell()
    cell_rates = three_sphere.round_rates(cell_mesh, three_sphere.isoclinic(cell_vertices))   # Rate+[edges]
    # The squash, stretching space along x and squeezing it along y in the frames, alone.
    squashing = squash(cell_mesh, cell_vertices, cell_rates, stack((three_sphere.mv.x,)), stack((three_sphere.mv.y,)))
    waves, frequency = stepped_history(cell_mesh, cell_vertices, squashing, np.zeros(1), wave_swing)
    eye_times = np.arange(n_wave_frames) * 2 * np.pi / frequency / n_wave_frames   # [frames]
    bonds, atoms = crystals(crystals_ahead())
    # The light from the crystals left them their distance before it reached the eye.
    states = crystal_states(cell_mesh, cell_vertices, waves, frequency, bonds, atoms, tunings,
                            eye_times - crystal_distance)
    frames = render.inside(three_sphere, cell_mesh, cell_vertices, eye, waves, eye_times,
                           render.edge_styles(three_sphere, cell_mesh, guardrails(cell_vertices, guardrail_sides)),
                           no_circles(),
                           bonds, states, render.saturating_strain)
    save_animation(frames, "motor_connection_600_cell", duration_ms)

    # The gate for stepping the motors: the Chern–Simons form on the edges' rates, and the stiffness it leaves on their
    # lengths, against their expected null directions, gauge alone: the vertices' motors less the six of the whole, and
    # the vertices' translations less the six isometries.
    chern_simons = (three_sphere.chern_simons(cell_mesh, cell_vertices) * three_sphere.Rate).to_dense()
    stiffness = three_sphere.length_stiffness(cell_mesh, cell_vertices, chern_simons)
    n_vertices = cell_mesh.n_vertices
    save_figure(render.gate([(render.spectrum(chern_simons, three_sphere.Rate), 6 * (n_vertices - 1)),
                             (render.spectrum(stiffness, three_sphere.Scalar), 3 * n_vertices - 6)]),
                "motor_connection_stepping_gate")
    spectrum, squash_squared, times, shares = stepped_squash(cell_mesh, cell_vertices, cell_rates, 4)
    save_figure(render.waves(spectrum, squash_squared, times, shares, wave_swing), "motor_connection_stepped_waves")

    # A burst of waves on an irregular mesh of the three-sphere, seen through the decagons and through beacons across
    # the universe, and each beacon's largest strain against its distance from the source.
    burst_mesh, burst_vertices = irregular_three_sphere()
    burst_rates = three_sphere.round_rates(burst_mesh, three_sphere.isoclinic(burst_vertices))   # Rate+[edges]
    burst_waves, burst_frequency = burst(burst_mesh, burst_vertices, burst_rates)
    save_animation(burst_view(burst_mesh, burst_vertices, burst_rates, burst_waves, burst_frequency),
                   "motor_connection_burst", duration_ms)
    beacon_frames, beacons, seen_states = beacon_view(burst_mesh, burst_vertices, burst_rates, burst_waves,
                                                      burst_frequency)
    save_animation(beacon_frames, "motor_connection_beacons", duration_ms)
    save_figure(render.beacon_strains(burst_source().dot(beacons), seen_states), "motor_connection_beacon_strains")

    # checks
    # Half a turn along, every path on the sphere is at the antipode.
    misses = plane.polar((paths[:, n_half_turn] - antipode).cast(plane.Step)).squared().square_root()   # [paths] Scalar+
    assert misses.to_array().max() < 0.05
    # Every particle thrown is back at the middle of the strip after the same proper time, half a turn of the radius,
    # read across the strip and along time.
    missed = (worldlines[1:-1, n_half_turn] - meeting).cast(spacetime.Step)  # [particles] Antivector-
    readings = stack((spacetime.mv.x, spacetime.mv.t))                          # [2] Plane+
    assert np.abs((readings[:, None] & missed[None]).to_array()).max() < 0.03
    # Light keeps to the chart's own light cone from the event: as far across as along time.
    offsets = (worldlines[[0, -1]] - event).cast(spacetime.Step)          # [2, steps + 1] Antivector-
    assert np.abs(((spacetime.mv.x & offsets).abs() - (spacetime.mv.t & offsets)).to_array()).max() < 0.02
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
    deficits, _, _ = three_sphere.Space(burst_mesh, burst_vertices).dynamics(chords.dot(chords))
    assert np.abs(deficits.to_array()).max() < 1e-12


if __name__ == "__main__":
    main()
