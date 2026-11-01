"""Geodesics through a connection of motors: one motor on each edge of a flat mesh, blended across each triangle by
the triangle's Whitney forms, makes the mesh a curved surface or a curved spacetime, and free paths roll through it.

Cartan described a curved space by how a flat one rolls along it. Rolled from one end of an edge to the other, the
flat space moves by a motor: a step, how far the edge reaches in the curved space, and a turn about the point of
contact on the way, which in spacetime is a boost. A mesh whose edges carry such motors is curved, whatever its own
flat shape, which serves only as a chart. Inside each triangle the motors' generators blend into one connection, a map
from the line a path runs along to the rate at which the rolled space moves. Along an edge it gives back exactly that
edge's motor, and two triangles' connections agree along the side they share, so the rolling goes on across a side
without a jump and the curvature is spread through the triangles, not gathered at their corners: paths passing on
either side of a corner are turned alike. A free particle, or a ray of light, keeps one direction in the space being
rolled; its path on the mesh is the one along which the rolled space's point of contact runs straight.

In a projective algebra each corner of a triangle has a ramp, the plane through the side across from it over the join
of the corners, `coframe & point` the point's barycentric weights. An edge's Whitney form inside the triangle is the
meet of its two ends' ramps, `coframe[tails] ^ coframe[heads]`, here the corner across from the edge, and it reads a
path's tangent line, `position & velocity`, by the line's moment about that corner: one along its own edge from tail
to head, nothing along the other two. The connection is a constant map from lines to rates in each triangle, the
edges' rates weighted by those readings. The rolled space's point of contact is its origin; under a rate it moves by
`origin.commutator(rate)`, the rate's step, while the rest of the rate turns the space about it. A free path's
velocity through the mesh is the one whose step is its fixed direction in the rolled space, carried into the space's
frame of the moment, and the motor of the frame grows by each rate along the way.

A metric that is the chart's own scaled by a conformal factor has every edge's rate in closed form: the rolled space
steps along the edge by the factor, and turns about the meet of the plane across the motion with the plane
`conformal * infinity + gradient`, the gradient being that of the factor's logarithm. The plane across a direction is
the one whose meet with the plane at infinity steps the origin along it, so the chart's metric enters only through
the algebra's own squares, and the same lines serve the plane and spacetime. Taken at an edge's midpoint along the
whole edge, that rate stands for the edge's motor. The unit sphere through its stereographic chart has the factor
`2 / (1 + r * r)` at distance `r` from the chart's centre, and every geodesic from a point meets the others again at
the antipode, half a turn along. Anti-de Sitter spacetime through its strip has the factor one over the cosine of the
distance from the strip's middle, and every free particle thrown from an event, at whatever speed, meets the others
again after the same proper time, half a turn of the space's radius, while light runs along the chart's own light
cone.

The rolled space need not be flat. With `w` squaring to one instead of nothing, the same lines roll a round sphere on
itself, in any dimension: `infinity` is then the equator of the origin, and the rates that moved the origin along a
plane now carry it along a great circle. A round sphere's edges carry the motors between frames at their ends, and
its connection is flat; a free path keeps the one rate that carries it along its great circle. What departs from
roundness, a lens or a wave, is what the edges' motors depart from those.

In the notation of Cartan geometry, the connection is a Cartan connection valued in the Lie algebra of the
Euclidean, the Poincaré or the rotation group, its translation part the coframe $e$ and its rotation part the
Levi-Civita connection $\\omega$, and geodesics are the curves whose development is straight; the edge motors are a
lattice gauge field of that group, and the blend is the lowest-order Whitney interpolation of a connection, as in the
simplicial gauge theory of Christiansen and Halvorsen (2012).
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from operator import and_

import numpy as np

from numga import Algebra, stack
from numga.backend.context import Context
from examples import instantiate

# Supplied by examples.instantiate: an algebra of the Cayley-Klein kind, whose generator `w` squares to nothing for a
# flat space, as the plane's or a spacetime's projective algebra, or to one for a round sphere, of any dimension.
ga: Algebra
ctx: Context
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                         # Scalar+
# The corners' ramps, gradients and the planes across directions: vectors, extrinsic; those through the origin.
Plane = ga.gatype.vector()                          # Vector+
Through = Plane.derive.within(" ".join(name for name in ga.basis_names if name != "w"))   # Vector+
# Positions in the chart and the origin's steps, and the lines paths run along: antivectors and antibivectors.
Point = ga.gatype.antivector()                      # Antivector-
Step = Point.derive.containing("w")                 # Antivector-
Line = ga.gatype.antibivector()                     # Antibivector-
# The rolled space's rates and the motors they generate; the Whitney forms, meets of the ramps.
Rate = ga.gatype.bivector()                         # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Connection = ga.gatype((Rate, Line))                # Rate+ <- Line-
Translating = ga.gatype((Step, Through))            # Antivector- <- Vector+
Stepping = ga.gatype((Point, Point))                # Antivector- <- Antivector-

# The plane `w`, at infinity in a flat space and the equator on a sphere, and the rolled space's point of contact, its
# origin, the plane's pole.
infinity = mv.w                                     # [] Plane+
origin = infinity.dual()                            # [] Point-
# The step of the origin under the motion across a plane through it, a step by its value, cast so; its inverse, the
# plane across each step, is the chart's metric as a correlation.
translating = origin.commutator(infinity ^ Through).cast(Translating)   # [] Antivector- <- Vector+
across = translating.inverse()                                          # [] Vector+ <- Antivector-


# --- math -----------------------------------------------------------------------------
def coframe(mesh: Mesh, vertices: Point) -> Plane:
    """Each corner's ramp across its simplex, `[corners] Plane+[simplices]`: the plane through the facet across from
    the corner, the join of the facet's corners, scaled to read one at the corner and so nothing on the facet."""
    corners = mesh.incidence[-1, 0] * vertices                                  # [corners] Point-[simplices]
    facets = reduce(and_, corners[mesh.corners[-1, -2]])                        # [corners] Antivector-[simplices]
    return facets / (facets & corners)


def whitney(mesh: Mesh, ramps: Plane) -> Rate:
    """The Whitney form of each edge of each simplex, `[edges] Bivector+[simplices]`, in the simplex's own order of its
    edges: the meet of the edge's two ramps, in a triangle the corner across from the edge and in a tetrahedron the
    edge across from it, which reads a line by its moment about that."""
    tails, heads = mesh.corners[-1, 1]
    return ramps[tails] ^ ramps[heads]


def connection(mesh: Mesh, ramps: Plane, rates: Rate) -> Connection:
    """The connection in each simplex, `(Rate+ <- Line-)[simplices]`, from the rates of the edges, `Rate+[edges]`:
    each edge's rate, run as the simplex runs the edge, weighted by the edge's Whitney form's reading of the line."""
    return ((mesh.oriented[-1, 1] * rates) * (whitney(mesh, ramps) & Line)).sum(axis=0)


def conformal_rates(edges: Point, conformal: Scalar, gradient: Plane) -> Rate:
    """The rate of each edge, `Rate+[edges]`, where the chart's metric is scaled by a conformal factor: the rolled
    space steps along the edge by the factor, and turns about where the plane across the motion meets
    `conformal * infinity + gradient`, the gradient of the factor's logarithm, both taken at the edge's midpoint."""
    # In a flat chart the edges, differences of points of weight one, are steps by value, cast so for the metric.
    return (conformal * infinity + gradient) ^ across(edges.cast(Step))


def sphere_rates(mesh: Mesh, vertices: Point) -> Rate:
    """The rate of each edge, `Rate+[edges]`, for the unit sphere seen through the stereographic chart, whose
    conformal factor is `2 / (1 + r * r)` at distance `r` from the chart's centre."""
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices                    # Point-[edges]
    # The plane across each midpoint's radius, as long as the radius: half the gradient of the radius squared.
    radial = across((midpoints - origin).cast(Step))                           # Plane+[edges]
    conformal = 2 / (1 + radial.squared())                                      # Scalar+[edges]
    return conformal_rates(mesh.d[0] * vertices, conformal, -conformal * radial)


def anti_de_sitter_rates(mesh: Mesh, vertices: Point) -> Rate:
    """The rate of each edge, `Rate+[edges]`, for anti-de Sitter spacetime of unit radius seen through its strip,
    whose conformal factor is one over the cosine of the distance across the strip from its middle, read by `mv.x`."""
    midpoints = mesh.incidence[1, 0].mean(axis=0) * vertices                    # Point-[edges]
    across_strip = mv.x & midpoints                                             # Scalar+[edges]
    return conformal_rates(mesh.d[0] * vertices, 1 / across_strip.cos(), mv.x * across_strip.tan())


def round_rates(mesh: Mesh, frames: Motor) -> Rate:
    """The rate of each edge, `Rate+[edges]`, for a round sphere of unit radius rolled on itself, from frames at the
    vertices, `Motor+[vertices]`, motors that carry the origin to each vertex: the tail's inverse times the head's, so
    that around any loop they cancel and the connection is flat. Any frames will do, and the rates are smallest where
    neighbouring frames differ least; a sphere of two dimensions has no frames that are smooth everywhere, while the
    three-sphere has the isoclinic ones."""
    tails, heads = mesh.incidence[1, 0] * frames                                # Motor+[edges], Motor+[edges]
    return -2 * (tails.inverse() * heads).log()


def geodesics(mesh: Mesh, vertices: Point, ramps: Plane, connection: Connection, starts: Point, headings: Point,
              step: float, n_steps: int) -> Iterator[Point]:
    """Free paths from `starts` `[paths] Point-`, setting out along `headings` `[paths] Antivector-`, steps of the
    origin, and stepped by `step` of their length in the rolled space, the proper time along a unit timelike
    direction: their positions in the chart at each step, each held where it leaves the mesh. A free path keeps one
    rate in the rolled space, the one that moves the origin along its heading, and so runs along a straight line of
    the rolled space, or a great circle of a rolled sphere; its velocity is the step that rate gives the point of
    contact, seen from the frame of the moment."""
    pick = mesh.picking(vertices)
    kept = infinity ^ across(headings)                                          # [paths] Rate+
    positions = starts                                                          # [paths] Point-
    frames = mv.rotor().broadcast_to(starts.shape)                              # [paths] Motor+
    yield positions
    for _ in range(n_steps):
        faces = pick(positions)                                                 # [paths]
        corner_ramps = ramps.batch()[:, faces]                                  # [corners, paths] Plane+
        # The plane on which the simplex's barycentric weights add up to one, and the position scaled onto it.
        unit = corner_ramps.sum(axis=0)                                         # [paths] Plane+
        positions = positions / (unit & positions)                              # [paths] Point-
        # The simplex picked holds the point unless the point has left the mesh, past a margin for round-off.
        inside = ((corner_ramps & positions) >= -margin).all(axis=0)            # [paths]
        local = (mesh.reading(faces, mesh.n_dim) * connection).batch()         # [paths] Rate+ <- Line-
        # How the rolled space's point of contact moves per velocity through the mesh, the step of the rate of the
        # line through the position along that velocity, and with it, in the origin's own part, the weight the
        # simplex reads in the velocity, to be nothing, so that the position stays on the simplex's plane.
        # A commutator with a rate keeps the grade, which the type does not carry in four dimensions: cast to points.
        stepping = (origin.commutator(local(positions & Point)) + origin * (unit & Point)).cast(Stepping)   # [paths] Point- <- Point-
        velocities = stepping.solve(origin.commutator(frames << kept).cast(Point))   # [paths] Point-
        rates = local(positions & velocities)                                   # [paths] Rate+
        held = step * inside                                                    # [paths]
        frames = frames * (rates * (-held / 2)).exp()                           # [paths] Motor+
        positions = positions + velocities * held                               # [paths] Point-
        yield positions


def invariant_coframe(mesh: Mesh, ramps: Plane, rates: Rate) -> Rate:
    """The round three-sphere's coframe along its three invariant directions, `[3] Bivector+[tetrahedra]`, from rates
    `Rate+[edges]` built on isoclinic frames: each edge's step read along `x`, `y` and `z`, weighting the edge's
    Whitney form. Each edge's rate is its steps times the isoclinic generators, `P + P * pseudoscalar`, of `P`, the
    rate that carries the origin along each direction; a squashed sphere stretches those steps and turns its own way."""
    steps = (mesh.oriented[-1, 1] * origin.commutator(rates)).cast(Step)       # [edges] Antivector-[tetrahedra]
    readings = stack((mv.x, mv.y, mv.z))                                        # [3] Plane+
    return (whitney(mesh, ramps)[None] * (readings[:, None] & steps[None])).sum(axis=1)


def wave(moving: Rate, turning: Rate, swing: float, frequency: float, time: float) -> tuple[Rate, np.ndarray]:
    """The slowest gravitational wave of a closed universe at a time, a squash of all of space alike: the generators
    along the three invariant directions, `[3] Rate+`, each moving the origin along its direction `moving` stretched as
    space is stretched along it, `1 + swing * cos(frequency * time)`, the second as far squeezed and the third still,
    and turning square to it, `turning`, by Milnor's mu, which keeps the squashed sphere free of torsion and is one on
    the round sphere; and the slices' stretch rates along each, `[3]`."""
    swung = swing * np.cos(frequency * time)
    stretches = np.array([1 + swung, 1 - swung, 1.0])                           # [3]
    lambdas = 2 * stretches / (np.roll(stretches, -1) * np.roll(stretches, -2))
    mu = lambdas.sum() / 2 - lambdas                                            # [3]
    stretch_rates = np.array([1.0, -1.0, 0.0]) * (-swing * frequency * np.sin(frequency * time)) / stretches
    return stretches * (moving + mu * turning), stretch_rates


def proper_separations(local: Connection, bonds: Mesh, positions: Point) -> Point:
    """Each bond's proper separation, `Antivector-[bonds]`, a step of the origin in the frames: the step the connection
    at the bond's tail, `(Rate+ <- Line-)[particles]`, gives the chart separation of its ends `Point-[particles]`."""
    tails = bonds.incidence[1, 0][0]                                            # Scalar+[bonds <- particles]
    separations = bonds.d[0] * positions                                        # Point-[bonds]
    return origin.commutator((tails * local)((tails * positions) & separations)).cast(Step)


def massive(mesh: Mesh, vertices: Point, ramps: Plane, coframes: Rate, generators, bonds: Mesh, rest: Scalar,
            stiffness: Scalar, damping: Scalar, starts: Point, velocities: Point, start: float, step: float,
            times: np.ndarray) -> Iterator[tuple[Point, Scalar]]:
    """Massive particles through a three-sphere whose connection at a time is the invariant coframe `[3]
    Bivector+[tetrahedra]` times `generators(time)`, `([3] Rate+, [3])`, held by springs along `bonds`, a graph `Mesh`
    over the particles, of rest proper lengths `rest`, stiffness `stiffness` and damping `damping` over unit masses,
    `Scalar+[bonds]` each: from `starts` `Point-[particles]` with velocities `velocities`, steps of the origin
    `Antivector-[particles]` in the frames, at time `start`, stepped by `step` of time with symplectic Euler, their
    positions and the bonds' proper strains `Scalar+[bonds]` at each of the `times`. A particle's velocity turns with
    the connection along its motion, the slices' stretch squeezes it, and its bonds pull it along their proper
    separations."""
    pick = mesh.picking(vertices)
    readings = stack((mv.x, mv.y, mv.z))                                        # [3] Plane+
    axes = stack((mv.x, mv.y, mv.z)).dual()                                     # [3] Antivector-
    positions, velocities, lengths, time = starts.batch(), velocities.batch(), rest, start
    for target in times:
        while time < target - step / 2:
            faces = pick(positions)                                             # [particles]
            unit = ramps.batch()[:, faces].sum(axis=0)                          # [particles] Plane+
            positions = positions / (unit & positions)                          # [particles] Point-
            generated, stretch_rates = generators(time)                         # [3] Rate+, [3]
            local = (generated[:, None] * (coframes.batch()[:, faces] & Line)).sum(axis=0)   # [particles] Rate+ <- Line-
            proper = proper_separations(local.field(), bonds, positions.field())   # Antivector-[bonds]
            previous, lengths = lengths, proper.dot(proper).square_root()       # Scalar+[bonds], Scalar+[bonds]
            # Springs and dashpots along the proper separations, pulling a stretched bond's ends together.
            pulls = (stiffness * (lengths - rest) + damping * (lengths - previous) / step) / lengths   # Scalar+[bonds]
            forces = -(~bonds.d[0] * (pulls * proper)).batch()                  # [particles] Antivector-
            # The velocity turned by the connection along the motion, kept a step of the origin, squeezed by the
            # slices' stretch and pushed by the springs.
            stepping = (origin.commutator(local(positions & Point)) + origin * (unit & Point)).cast(Stepping)
            motion = stepping.solve(velocities)                                 # [particles] Point-
            turned = velocities - velocities.commutator(local(positions & motion)).cast(Point) * step
            kept = turned - origin * (infinity & turned)                        # [particles] Antivector-
            squeeze = (stretch_rates[:, None] * (readings[:, None] & kept[None]) * axes[:, None]).sum(axis=0)
            velocities = (kept - squeeze * step + forces * step).cast(Step)     # [particles] Antivector-
            positions = positions + stepping.solve(velocities) * step           # [particles] Point-
            time += step
        yield positions.field(), lengths / rest - 1


# A ray's step on the GPU, in numga's dialect, through a three-sphere meshed by tetrahedra and squashed by the slowest
# gravitational wave of a closed universe: each tetrahedron's corner ramps, its neighbours across each facet and the
# facet each enters by, and its invariant coframe; the rates that carry the origin along `x`, `y` and `z` and their
# square turns, the planes that read a step along each, and the steps themselves; the wave's swing and frequency.
geodesic_step_source = """
var<storage, read> ramps: array<Plane>;
var<storage, read> neighbours: array<u32>;
var<storage, read> entries: array<u32>;
var<storage, read> coframes: array<Rate>;
var<uniform> origin: Point;
var<uniform> infinity: Plane;
var<uniform> basis_0: Point;
var<uniform> basis_1: Point;
var<uniform> basis_2: Point;
var<uniform> basis_3: Point;
var<uniform> moving_x: Rate;
var<uniform> moving_y: Rate;
var<uniform> moving_z: Rate;
var<uniform> turning_x: Rate;
var<uniform> turning_y: Rate;
var<uniform> turning_z: Rate;
var<uniform> reading_x: Plane;
var<uniform> reading_y: Plane;
var<uniform> reading_z: Plane;
var<uniform> step_x: Point;
var<uniform> step_y: Point;
var<uniform> step_z: Point;
var<uniform> swing: f32;
var<uniform> frequency: f32;

// How far space is stretched along each invariant direction at a time: one swinging out, one in, one still.
fn stretches(time: f32) -> vec3<f32> {
    let out = swing * cos(frequency * time);
    return vec3<f32>(1.0 + out, 1.0 - out, 1.0);
}

// The rate at which the slices stretch along each direction, over their stretch.
fn stretch_rates(time: f32) -> vec3<f32> {
    let a = stretches(time);
    let rate = -swing * frequency * sin(frequency * time);
    return vec3<f32>(rate / a.x, -rate / a.y, 0.0);
}

// The connection in a tetrahedron at a time: the line's steps along the invariant directions, stretched, each moving
// the origin along its direction and turning square to it by Milnor's mu, which keeps the squashed sphere free of
// torsion and is one on the round sphere.
fn connection(tet: u32, line: Line, time: f32) -> Rate {
    let a = stretches(time);
    let lambda = 2.0 * vec3<f32>(a.x / (a.y * a.z), a.y / (a.z * a.x), a.z / (a.x * a.y));
    let mu = vec3<f32>((lambda.x + lambda.y + lambda.z) / 2.0) - lambda;
    let base = tet * 3u;
    let c = vec3<f32>(coframes[base] & line, coframes[base + 1u] & line, coframes[base + 2u] & line) * a;
    return moving_x * c.x + moving_y * c.y + moving_z * c.z
           + turning_x * (c.x * mu.x) + turning_y * (c.y * mu.y) + turning_z * (c.z * mu.z);
}

// The plane on which a tetrahedron's barycentric weights add up to one.
fn unit_plane(tet: u32) -> Plane {
    let base = tet * 4u;
    return ramps[base] + ramps[base + 1u] + ramps[base + 2u] + ramps[base + 3u];
}

// The origin's step under the connection of the line through the place along a velocity, and in the origin's own
// part the weight the tetrahedron reads in the velocity.
fn stepping(tet: u32, place: Point, unit: Plane, velocity: Point, time: f32) -> Point {
    return (origin.commutator(connection(tet, place & velocity, time)) + origin * (unit & velocity)).cast(Point);
}

// The velocity whose stepping is the given step, by Cramer's rule: each spanning point's share is the join of the
// steppings with the step in its place, over the join of them all.
fn velocity(tet: u32, place: Point, unit: Plane, step: Point, time: f32) -> Point {
    let s0 = stepping(tet, place, unit, basis_0, time);
    let s1 = stepping(tet, place, unit, basis_1, time);
    let s2 = stepping(tet, place, unit, basis_2, time);
    let s3 = stepping(tet, place, unit, basis_3, time);
    let volume = s0 & s1 & s2 & s3;
    return (basis_0 * (step & s1 & s2 & s3) + basis_1 * (s0 & step & s2 & s3) + basis_2 * (s0 & s1 & step & s3)
            + basis_3 * (s0 & s1 & s2 & step)) / volume;
}

// A ray's heading, a unit step of the origin in the frames, after a step of length `h` traced back in time: turned by
// the connection's rate, kept a step of the origin, and stretched as the slices change, back in time.
fn headed(heading: Point, rate: Rate, time: f32, h: f32) -> Point {
    let turned = heading - heading.commutator(rate).cast(Point) * h;
    let kept = turned - origin * (infinity & turned);
    let k = stretch_rates(time);
    let w = vec3<f32>(reading_x & kept, reading_y & kept, reading_z & kept);
    let squeeze = k.x * w.x * w.x + k.y * w.y * w.y + k.z * w.z * w.z;
    let stretched = kept + (step_x * (k.x * w.x) + step_y * (k.y * w.y) + step_z * (k.z * w.z) - kept * squeeze) * h;
    return stretched.normalized();
}
"""


# --- plumbing -------------------------------------------------------------------------
# How far outside a side its ramp may read a point, by round-off, while the point still counts as inside.
margin = 1e-9
