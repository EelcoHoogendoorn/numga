"""Free paths rolled through flat meshes that carry a curved space's motors on their edges, by the same lines in the
plane and in spacetime: a fan of geodesics on the unit sphere, which set out from one point and meet again at its
antipode, and a fan of particles thrown from one event in anti-de Sitter spacetime at every speed, which meet again
after the same proper time, with two rays of light running along the chart's light cone."""

from __future__ import annotations

from types import ModuleType

import numpy as np

from numga import Algebra, Extensor, concatenate, stack
from numga.algebras import PGA2D, ProjectiveMultivectorFactory
from examples import instantiate

plane = instantiate("examples.relativity.motor_connection.core", PGA2D)
spacetime = instantiate("examples.relativity.motor_connection.core",
                        Algebra("x+t-w0", multivector_factory=ProjectiveMultivectorFactory))
sphere = instantiate("examples.relativity.motor_connection.core", Algebra("x+y+w+"))
three_sphere = instantiate("examples.relativity.motor_connection.core", Algebra("x+y+z+w+"))

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
across = spacetime.mv.x.dual()                                                  # [] Antivector-
event = spacetime.origin                                                        # [] Point-
meeting = spacetime.origin + time * np.pi                                       # [] Point-
rapidities = np.linspace(-1.5, 1.5, 9)
thrown = (spacetime.mv.xt * (-rapidities / 2)).exp() >> time                    # [particles] Antivector-
rays = concatenate([(time - across)[None], thrown, (time + across)[None]])      # [paths] Antivector-

# The round spheres' meshes, placed by a turn that keeps every vertex off the antipode of the origin, where the paths
# start; the paths' headings about the origin, in a fan on the sphere and over the turns of the three-sphere.
n_sphere_levels = 3
sphere_placement = (sphere.mv.xw * 0.21 + sphere.mv.yw * 0.13).exp()           # [] Motor+
sphere_headings = (sphere.mv.xy * (-np.linspace(0, 2 * np.pi, n_paths, endpoint=False) / 2)).exp() >> sphere.mv.x.dual()   # [paths] Antivector-
three_sphere_placement = (three_sphere.mv.xw * 0.21 + three_sphere.mv.yw * 0.13 + three_sphere.mv.zw * 0.08).exp()   # [] Motor+
spins = (three_sphere.mv.xy * (-np.linspace(0, 2 * np.pi, n_paths, endpoint=False)[:, None] / 2)).exp()   # [paths, 1] Motor+
tilts = (three_sphere.mv.zx * (-np.radians([30.0, 90.0, 150.0]) / 2)).exp()   # [3] Motor+
three_sphere_headings = (spins * tilts >> three_sphere.mv.z.dual()).reshape(-1)   # [paths * 3] Antivector-

# The eye inside the 600-cell, the motor that carries the origin to it and turns its view; the slowest gravitational
# wave of the closed universe, its swing and its frequency, the root of eight light crossings of the radius; the eye's
# times over one swing.
eye = (three_sphere.mv.xw * 0.37 + three_sphere.mv.yw * 0.11 + three_sphere.mv.zw * 0.23).exp()   # [] Motor+
wave_swing = 0.12
wave_frequency = np.sqrt(8.0)
n_wave_frames = 75
eye_times = np.arange(n_wave_frames) * 2 * np.pi / wave_frequency / n_wave_frames   # [frames]
# Three little crystals ahead of the eye, side by side across its view: tetrahedra of four atoms of unit mass, bonded
# along their six edges, their bonds soft, tuned to the wave, and stiff: a sixteenth of the stiffness that puts the
# swing the squash drives, root two times the root of the stiffness, at the wave's, that one, and sixteen times it,
# with dashpots a quarter of critical. How far ahead, how far apart, how large; the time step, and the swings run
# before the frames so that the motion has settled.
crystal_distance, crystal_spacing, crystal_half_size = 0.55, 0.26, 0.05
tuned_stiffness = wave_frequency ** 2 / 2
stiffnesses = tuned_stiffness * np.array([1 / 16, 1.0, 16.0])                   # [crystals]
dampings = 0.25 * np.sqrt(2 * stiffnesses)                                      # [crystals]
crystal_step = 0.01
n_settling_swings = 8

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
    vertices = (spacetime.origin + across * lattice[:, 0] + time * lattice[:, 1]).field()   # Point-[vertices]
    return spacetime.Mesh.triangulate(vertices), vertices


def icosphere() -> tuple[sphere.Mesh, sphere.Point]:
    """The round sphere's mesh, an icosahedron split in four a few times, placed by its turn."""
    mesh, vertices = sphere.Mesh.icosphere(n_sphere_levels)
    return mesh, sphere_placement >> vertices


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


def isoclinic_frames(vertices: three_sphere.Point) -> three_sphere.Motor:
    """Frames for the round three-sphere, `Motor+[vertices]`, smooth everywhere: the motors that carry the origin to
    each vertex by turning alike in the plane of the great circle between them and in the plane square to it, the
    half of the quotient of the vertex by the origin that the extrinsic pseudoscalar keeps, with one in the other
    half."""
    pseudoscalar = three_sphere.mv.xyzw                                         # [] Antiscalar+
    return ((1 + pseudoscalar) * (vertices.normalized() / three_sphere.origin) + (1 - pseudoscalar)) / 2


def guardrails(vertices: three_sphere.Point) -> np.ndarray:
    """The 600-cell's vertices in twelve decagons, `[decagons, 10]` vertex indices, great circles side by side that
    the mesh's edges run along: each the orbit of a vertex under the isoclinic motor that carries the origin to its
    nearest vertex, before the placement, and so the vertices onto themselves."""
    from scipy.spatial import cKDTree

    unplaced = (three_sphere_placement.inverse() >> vertices).batch()          # [vertices] Point-
    weights = (three_sphere.infinity & unplaced).to_array()                     # [vertices]
    nearest = unplaced[np.argmax(np.where(weights < 1 - 1e-9, weights, -2))]    # [] Point-
    turn = three_sphere_placement * isoclinic_frames(nearest) * three_sphere_placement.inverse()   # [] Motor+
    readings = three_sphere.mesh_module.readings
    following = cKDTree(readings(vertices.batch())).query(readings((turn >> vertices).batch()))[1]   # [vertices]
    orbits = [np.arange(len(following))]
    for _ in range(9):
        orbits.append(following[orbits[-1]])
    orbits = np.stack(orbits)                                                   # [10, vertices]
    _, firsts = np.unique(orbits.min(axis=0), return_index=True)
    return orbits[:, firsts].T


def crystals() -> tuple[three_sphere.Mesh, three_sphere.Point]:
    """The crystals' bonds, a graph over their atoms, and the atoms, `Point-[atoms]`, at rest in the chart: the eye
    carried ahead along its view, across it to each crystal's place, and out to each corner of a cube, each by the
    rate that moves the origin along the eye's own axes; each crystal's corners are every other corner of a cube."""
    from itertools import product

    mv = three_sphere.mv
    steps = stack((mv.x, mv.y, mv.z)).dual()                                    # [3] Antivector-
    moving = three_sphere.infinity ^ three_sphere.across(steps.cast(three_sphere.Step))   # [3] Rate+
    cube = np.array(list(product([-1.0, 1.0], repeat=3)))                      # [8, 3]
    corners = cube[cube.prod(axis=-1) > 0]                                      # [corners, 3]
    offsets = crystal_spacing * np.arange(-1, 2)                                # [crystals]
    ahead = (moving[2] * (-crystal_distance / 2)).exp()                         # [] Motor+
    across_view = (moving[0] * (-offsets[:, None] / 2)).exp()                   # [crystals, 1] Motor+
    out = ((moving * corners).sum(axis=-1) * (-crystal_half_size / 2)).exp()  # [corners] Motor+
    atoms = ((eye * ahead * across_view * out) >> three_sphere.origin).reshape(-1)   # [atoms] Point-
    # Bonds join every two corners of each crystal, which differ along two of the cube's axes.
    differing = (corners[:, None] != corners[None]).sum(axis=-1)
    pairs = np.argwhere(np.triu((differing >= 1) & (differing <= 2)))           # [bonds per crystal, 2]
    bonds = (pairs[None] + len(corners) * np.arange(len(offsets))[:, None, None]).reshape(-1, 2)
    return three_sphere.Mesh(len(corners) * len(offsets), bonds), atoms.field()


def crystal_states(cell_mesh: three_sphere.Mesh, cell_vertices: three_sphere.Point, cell_rates: three_sphere.Rate
                   ) -> tuple[three_sphere.Mesh, list]:
    """The crystals' bonds, and their atoms and bonds' strains, `(Point-[atoms], Scalar+[bonds])`, as the eye sees them
    at each of its times: when their light left them, after the motion has settled into the wave's swing."""
    ramps = three_sphere.coframe(cell_mesh, cell_vertices)                      # [corners] Plane+[tets]
    coframes = three_sphere.invariant_coframe(cell_mesh, ramps, cell_rates)     # [3] Bivector+[tets]
    mv = three_sphere.mv
    steps = stack((mv.x, mv.y, mv.z)).dual()                                    # [3] Antivector-
    moving = three_sphere.infinity ^ three_sphere.across(steps.cast(three_sphere.Step))   # [3] Rate+
    turning = moving * mv.xyzw                                                  # [3] Rate+

    def generators(time: float):
        """The wave's generators and stretch rates at a time."""
        return three_sphere.wave(moving, turning, wave_swing, wave_frequency, time)

    bonds, atoms = crystals()
    pick = cell_mesh.picking(cell_vertices)
    faces = pick(atoms)
    unit = ramps.batch()[:, faces].sum(axis=0).field()
    placed = atoms / (unit & atoms)
    # The crystals' rest lengths: their bonds' proper lengths on the round sphere.
    round_local = ((moving + turning)[:, None] * (coframes.batch()[:, faces] & three_sphere.Line)).sum(axis=0).field()
    rest = three_sphere.proper_separations(round_local, bonds, placed)
    rest = rest.dot(rest).square_root()                                         # Scalar+[bonds]
    per_crystal = len(bonds.edges) // len(stiffnesses)
    stiffness = three_sphere.mesh_module.as_scalar(np.repeat(stiffnesses, per_crystal)).field()
    damping = three_sphere.mesh_module.as_scalar(np.repeat(dampings, per_crystal)).field()
    # The light from the crystals left them their distance before it reached the eye.
    seen_at = eye_times - crystal_distance                                      # [frames]
    start = seen_at[0] - n_settling_swings * 2 * np.pi / wave_frequency
    at_rest = (steps[0] * np.zeros(atoms.batch().shape[0])).field()           # Antivector-[atoms]
    states = list(three_sphere.massive(cell_mesh, cell_vertices, ramps, coframes, generators, bonds, rest, stiffness,
                                       damping, placed, at_rest, start, crystal_step, seen_at))
    return bonds, states


def rolled(core: ModuleType, mesh, vertices: Extensor, rates: Extensor, starts: Extensor, directions: Extensor
           ) -> Extensor:
    """The paths from the starts along the directions, `[paths, steps + 1] Point-`, through the connection of the
    edges' rates, in the algebra of the core's instance."""
    ramps = core.coframe(mesh, vertices)                                        # [corners] Plane+[faces]
    connection = core.connection(mesh, ramps, rates)                            # (Rate+ <- Line-)[faces]
    return stack(list(core.geodesics(mesh, vertices, ramps, connection, starts, directions, step, n_steps)), axis=-1)


def main() -> None:
    from examples.util.animation import save_animation
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
    cell_rates = three_sphere.round_rates(cell_mesh, isoclinic_frames(cell_vertices))   # Rate+[edges]
    bonds, states = crystal_states(cell_mesh, cell_vertices, cell_rates)
    frames = render.inside(three_sphere, cell_mesh, cell_vertices, cell_rates, eye, isoclinic_frames, wave_swing,
                           wave_frequency, eye_times, guardrails(cell_vertices), bonds, states)
    save_animation(frames, "motor_connection_600_cell", duration_ms)

    # checks
    # Half a turn along, every path on the sphere is at the antipode.
    misses = plane.across((paths[:, n_half_turn] - antipode).cast(plane.Step)).squared().square_root()   # [paths] Scalar+
    assert misses.to_array().max() < 0.05
    # Every particle thrown is back at the middle of the strip after the same proper time, half a turn of the radius,
    # read across the strip and along time.
    missed = (worldlines[1:-1, n_half_turn] - meeting).cast(spacetime.Step)  # [particles] Antivector-
    readings = stack((spacetime.mv.x, spacetime.mv.t))                          # [2] Plane+
    assert np.abs((readings[:, None] & missed[None]).to_array()).max() < 0.03
    # Light keeps to the chart's own light cone from the event: as far across as along time.
    offsets = (worldlines[[0, -1]] - event).cast(spacetime.Step)          # [2, steps + 1] Antivector-
    assert np.abs(((spacetime.mv.x & offsets).abs() - (spacetime.mv.t & offsets)).to_array()).max() < 0.02
    # Over the wave's swing, the soft crystal carries the wave's strain, the tuned one more, and the stiff one keeps
    # its shape.
    swings = np.abs(np.stack([strains.to_array() for _, strains in states])).reshape(len(states), len(stiffnesses), -1)
    soft, tuned, stiff = swings.max(axis=(0, 2))
    assert 0.5 * wave_swing < soft < 1.5 * wave_swing and tuned > 2 * wave_swing and stiff < 0.2 * wave_swing


if __name__ == "__main__":
    main()
