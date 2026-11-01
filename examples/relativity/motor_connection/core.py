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
from numga.sparse import SparseExtensor, spdiag
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
Metric = ga.gatype((Plane, Point))                  # Plane+ <- Point-
InverseMetric = ga.gatype((Point, Plane))           # Point- <- Plane+
# A strain, the change of the metric on the origin's steps, the plane that reads each step's stretch; a chart, the
# directions in a tetrahedron's chart that take each step; and a turning, the rate a step adds to a frame's turn.
Strain = ga.gatype((Through, Step))                 # Vector+ <- Antivector-
Chart = ga.gatype((Point, Step))                    # Antivector- <- Antivector-
Turning = ga.gatype((Rate, Step))                   # Bivector+ <- Antivector-
Across = ga.gatype((Through, Step))                 # Vector+ <- Antivector-

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


# Each edge's Whitney form against each face's, integrated over a tetrahedron of positive sense, in its own order of
# its edges, (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), and of its faces, each facing a corner: a twelfth, signed
# by their senses, where the edge and the face share one corner, and nothing where they share two.
cup = np.array([[1, 1, 0, 0], [1, 0, -1, 0], [1, 0, 0, 1], [0, -1, -1, 0], [0, -1, 0, 1], [0, 0, 1, 1]]) / 12


def chern_simons(mesh: Mesh, vertices: Point) -> SparseExtensor:
    """The Chern–Simons form of a connection on the edges of a mesh of tetrahedra, linearized about a flat one, as the
    coupling of edge fields it pairs, `Scalar-[edges <- edges]`: in each tetrahedron, its edges' values against the
    circulations of its faces, `mesh.d[1]`, weighted by `cup` and signed by the tetrahedron's sense in the embedding,
    each pairing written once each way."""
    corners = mesh.incidence[3, 0] * vertices                                   # [corners] Point-[tets]
    senses = corners[0] & corners[1] & corners[2] & corners[3]                  # Scalar-[tets]
    local = (cup[:, :, None] * senses.normalized().batch()[None, None]).field()   # [edges, faces] Scalar-[tets]
    edges = mesh.oriented[3, 1]                                                 # [edges] Scalar+[tets <- edges]
    faces = mesh.oriented[3, 2] * mesh.d[1]                                     # [faces] Scalar+[tets <- edges]
    once = (~edges[:, None] * spdiag(local) * faces[None, :]).sum(axis=0).sum(axis=0)
    return (once + ~once) / 2


def length_reading(mesh: Mesh, vertices: Point) -> Rate:
    """How each edge's squared length changes with its motor's departure from round, `(Scalar+ <- Rate+)[edges]`:
    twice the chord read along the step the departure gives the chord's midpoint, on the sphere."""
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    midpoints = (mesh.incidence[1, 0].mean(axis=0) * vertices).normalized()     # Point-[edges]
    return 2 * chords.dot(midpoints.commutator(Rate).cast(Point))


def edge_turns(mesh: Mesh, vertices: Point) -> Rate:
    """Each edge's turn along itself, `Rate+[edges]`: the rate in the plane of its great circle that lengthens it by
    one in its squared length."""
    tails, heads = mesh.incidence[1, 0] * vertices                              # Point-[edges], Point-[edges]
    planes = (tails & heads).dual()                                             # Rate+[edges]
    return planes / length_reading(mesh, vertices)(planes)


def length_stiffness(mesh: Mesh, vertices: Point, form: Rate) -> Rate:
    """The stiffness the Chern–Simons form `(Rate- <- Rate+)[edges <- edges]`, dense, leaves on the edges' squared
    lengths, `(Scalar- <- Scalar+)[edges <- edges]`: each edge's motor lengthened along itself, and its turns and
    sideways steps, which change no length, slaved by the form's own stationarity, least squares leaving its null
    directions at nothing."""
    reading = length_reading(mesh, vertices)                                    # (Scalar+ <- Rate+)[edges]
    along = edge_turns(mesh, vertices)                                          # Rate+[edges]
    sideways = (Rate - along * reading).on_diagonal()                           # (Rate+ <- Rate+)[edges <- edges]
    lengthening = (along * Scalar).on_diagonal()                                # (Rate+ <- Scalar+)[edges <- edges]
    held = sideways.adjugate()(form(sideways))                                  # (Rate- <- Rate+)[edges <- edges]
    lifting = lengthening + sideways(held.lstsq(-sideways.adjugate()(form(lengthening)), rcond=1e-10))
    return lifting.adjugate()(form(lifting))


class Space:
    """A triangulated three-sphere as its flat tetrahedra know it, for stepping its edges' squared lengths by Regge's
    action: their corner gradients, `[corners] Plane+[cells]`, hyperplanes and the map from squared edge lengths to
    their metrics."""

    def __init__(self, mesh: Mesh, vertices: Point) -> None:
        corners = mesh.incidence[3, 0] * vertices                               # [corners] Point-[cells]
        ramps = coframe(mesh, vertices)                                         # [corners] Plane+[cells]
        # The tetrahedron's hyperplane, one on each of its points; less a quarter of it, the ramps vanish on the
        # centroid and still read the edges.
        self.mesh, self.radial = mesh, ramps.sum(axis=0)                        # Plane+[cells]
        self.gradients = ramps - self.radial / 4                                # [corners] Plane+[cells]
        # Along its own edge two corner gradients pair to -1 and +1 and the others vanish, so that their symmetric
        # product reads that edge's squared length and nothing of the other five.
        tails, heads = mesh.corners[3, 1]
        first, second = self.gradients[tails], self.gradients[heads]            # [cell_edges] Plane+[cells], [cell_edges] Plane+[cells]
        basis = -(first * (second & Point) + second * (first & Point)) / 2      # [cell_edges] (Plane+ <- Point-)[cells]
        self.metric_map = (spdiag(basis) * mesh.incidence[3, 1]).sum(axis=0)    # (Plane+ <- Point-)[cells <- edges]
        self.edges = stack((corners[1] - corners[0], corners[2] - corners[0], corners[3] - corners[0]))   # [3] Point-[cells]
        self.joins = corners[0] & corners[1] & corners[2] & corners[3]          # Scalar-[cells]

    def dynamics(self, squared_lengths: Scalar, cosmological: float) -> tuple[Scalar, Scalar, SparseExtensor]:
        """The deficit angle at each edge, the slope of the potential, minus twice the Regge action plus twice the
        cosmological constant times the volume, with respect to each squared edge length, and the mass map of
        relativity's kinetic term, a quarter of each tetrahedron's volume times the square of its inverse metric's
        reading of the metric's rate, for the edges' squared lengths at rest."""
        mesh = self.mesh
        _, inverses = self.metrics(squared_lengths)
        # The pairings of the tetrahedron's corner gradients through its inverse metric: on an edge's own corners
        # they give its volume's change, on the corners off it the angle between the faces that meet there.
        pairings = self.gradients[:, None] & inverses(self.gradients[None, :])   # [corners, corners] Scalar+[cells]
        tails, heads = mesh.corners[3, 1]
        on_edges = pairings[tails, heads]                                       # [cell_edges] Scalar+[cells]
        off_edges = pairings[tails[::-1], heads[::-1]]                          # [cell_edges] Scalar+[cells]
        sides = pairings[tails[::-1], tails[::-1]] * pairings[heads[::-1], heads[::-1]]   # [cell_edges] Scalar+[cells]
        angles = (-off_edges / sides.square_root()).arccos()                    # [cell_edges] Scalar+[cells]
        deficits = 2 * np.pi - (~mesh.incidence[3, 1] * angles).sum(axis=-1)    # Scalar+[edges]
        volumes = self.volumes(squared_lengths)                                 # Scalar+[cells]
        # By Schlafli the action's change is the deficits' alone; the volume's change through the metric is minus half
        # the volume times the pairing on the edge's own corners.
        volume_slopes = (~mesh.incidence[3, 1] * (volumes * on_edges)).sum(axis=-1)   # Scalar+[edges]
        slopes = -deficits / squared_lengths.square_root() - cosmological * volume_slopes   # Scalar+[edges]
        # Relativity's kinetic term on a tetrahedron, between the rates of two edges' squared lengths: a quarter of
        # the volume times the pairings of their corners across, both ways.
        local = volumes / 4 * (pairings[tails[:, None], tails[None, :]] * pairings[heads[:, None], heads[None, :]]
                               + pairings[tails[:, None], heads[None, :]] * pairings[heads[:, None], tails[None, :]])   # [cell_edges, cell_edges] Scalar+[cells]
        edges = mesh.incidence[3, 1]                                            # [cell_edges] Scalar+[cells <- edges]
        masses = (~edges[:, None] * spdiag(local) * edges[None, :]).sum(axis=0).sum(axis=0) * Scalar   # (Scalar+ <- Scalar+)[edges <- edges]
        return deficits, slopes, masses

    def volumes(self, squared_lengths: Scalar) -> Scalar:
        """Each tetrahedron's volume from the squared lengths: the root of the determinant of its metric over the
        corner gradients' frame, a sixth of it."""
        metrics, _ = self.metrics(squared_lengths)
        first, second, third = self.edges                                       # Point-[cells], Point-[cells], Point-[cells]
        return ((metrics(first) ^ metrics(second) ^ metrics(third) ^ self.radial) & self.joins).abs().square_root() / 6

    def metrics(self, squared_lengths: Scalar) -> tuple[Metric, InverseMetric]:
        """Each tetrahedron's metric from the squared lengths of the edges, the hyperplane taking the centroid, which
        the gradients leave out, so that it can be inverted; and its inverse."""
        metrics = self.radial * (self.radial & Point) + self.metric_map * squared_lengths   # (Plane+ <- Point-)[cells]
        return metrics, metrics.batch().inverse().field()


def gravitational_waves(mesh: Mesh, vertices: Point) -> tuple[Scalar, Scalar, float]:
    """The linear gravitational waves of a triangulated three-sphere at rest, as the stiffness and the masses on its
    edges' squared lengths, `(Scalar+ <- Scalar+)[edges <- edges]` each, and the cosmological constant that holds the
    round sphere at rest against its deficits. The stiffness is that of Regge's potential with that constant, kept
    to the lengths no move of the vertices makes: the tangent moves are the mesh's own relabelling of the sphere, and
    the radial ones change each edge's squared length by `s * (r_tail + r_head)`, the conformal stretches, whose
    breathing is the universe at rest's own instability. What is left are the transverse traceless strains, the waves."""
    space = Space(mesh, vertices)
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    rest = chords.dot(chords)                                                   # Scalar+[edges]
    deficits, unbalanced, masses = space.dynamics(rest, 0.0)
    # The constant that cancels the deficits' pull, the same on every edge of the round 600-cell by its symmetry.
    _, scaled, _ = space.dynamics(rest, 1.0)
    cosmological = float((unbalanced / (unbalanced - scaled)).to_array().mean())
    stiffness = jacobian(lambda lengths: space.dynamics(lengths, cosmological)[1], rest, 1e-6)
    moves = (spdiag(2 * chords.dot(Point)) * mesh.d[0]).to_dense()            # (Scalar+ <- Point-)[edges <- vertices]
    masses = masses.to_dense()
    kept = transverse(masses, moves)                                            # (Scalar+ <- Scalar+)[edges <- edges]
    return kept.adjoint()(stiffness(kept)), masses, cosmological


def waves_stepped(stiffness: Scalar, masses: Scalar, strains: Scalar, rates: Scalar, step: float
                  ) -> Iterator[tuple[Scalar, Scalar]]:
    """The squared lengths' changes and their rates, `Scalar+[edges]` each, at every step of a leapfrog under the
    stiffness and the masses: half a kick by the stiffness, a drift by the rates the masses read off the momenta, and
    the other half kick."""
    inverse_masses = masses.inverse()
    momenta = masses(rates)                                                     # Scalar+[edges]
    while True:
        yield strains, inverse_masses(momenta)
        momenta = momenta - (step / 2) * stiffness(strains)
        strains = strains + step * inverse_masses(momenta)
        momenta = momenta - (step / 2) * stiffness(strains)


def invariant_generators() -> tuple[Rate, Rate, Point]:
    """The round three-sphere's generators along its three invariant directions `x`, `y`, `z` at the origin, `[3]`
    each: the rates that move the origin along them, the turns square to them, each rate times the extrinsic
    pseudoscalar, and the unit steps themselves."""
    steps = stack((mv.x, mv.y, mv.z)).dual()                                    # [3] Antivector-
    moving = infinity ^ across(steps.cast(Step))                                # [3] Rate+
    return moving, moving * mv.xyzw, steps


def charts(mesh: Mesh, vertices: Point, ramps: Plane, coframes: Rate) -> Chart:
    """Each tetrahedron's chart at its centroid, `(Point- <- Antivector-)[tets]`: the velocity through the chart whose
    stepping under the round connection is each step of the origin."""
    moving, turning, _ = invariant_generators()
    centroids = (mesh.incidence[3, 0].mean(axis=0) * vertices).batch()        # [tets] Point-
    unit = ramps.batch().sum(axis=0)                                            # [tets] Plane+
    centroids = centroids / (unit & centroids)
    local = ((moving + turning)[:, None] * (coframes.batch() & Line)).sum(axis=0)   # [tets] Rate+ <- Line-
    stepping = (origin.commutator(local(centroids & Point)) + origin * (unit & Point)).cast(Stepping)
    return stepping.solve(Step).field()


def strain_field(mesh: Mesh, space: Space, ramps: Plane, chart: Chart, changes: Scalar) -> tuple[Strain, Turning]:
    """The strain of a change of the squared edge lengths, `Scalar+[edges]`, as a smooth field on the origin's steps:
    at each vertex the average of its tetrahedra's strains, each its metric's change pulled back through its chart,
    `Strain[vertices]`; and in each tetrahedron the turn the strain's change from corner to corner adds, half the
    curl of the strain, `Turning[tets]`."""
    # Each tetrahedron's own pullback, its chart's adjugate after its metric's change after its chart.
    pullback = chart.batch().adjugate()                                         # [tets] Vector+ <- Plane+
    strains = pullback((space.metric_map * changes)(chart).batch()).field()     # (Vector+ <- Antivector-)[tets]
    shares = ~mesh.incidence[3, 0]                                              # [corners] Scalar+[vertices <- tets]
    counts = (shares * mesh_module.as_scalar(np.ones(len(mesh.tets))).field()).sum(axis=0)   # Scalar+[vertices]
    at_vertices = (shares * strains).sum(axis=0) / counts                       # (Vector+ <- Antivector-)[vertices]
    # Each corner's ramp read on the steps, and the strain at each corner: their wedge, summed over the corners, is
    # the strain's change across the tetrahedron turned into the plane it turns in.
    corner_planes = pullback(ramps.batch()).field()                             # [corners] Vector+[tets]
    corner_strains = mesh.incidence[3, 0] * at_vertices                         # [corners] (Vector+ <- Antivector-)[tets]
    return at_vertices, (corner_planes ^ corner_strains).sum(axis=0) / 2


def moving(steps: Point) -> Rate:
    """The rates that move the origin along steps of it, `Rate+`: the meet of the plane at infinity with the plane
    across each step."""
    return infinity ^ across(steps)


def turning(steps: Point) -> Rate:
    """The rates that turn the frame square to steps of the origin, `Rate+`: each moving rate times the extrinsic
    pseudoscalar, the plane square to it."""
    return moving(steps) * mv.xyzw


class Waves:
    """Stepped gravitational waves of a three-sphere triangulated by `mesh` on `vertices`, round under `rates`, as a
    smooth field over its tetrahedra and over time: at each of the stored moments, from `first_time` every `level_step`,
    the strain of the squared edge lengths' changes `changes` at the vertices, its rate from `change_rates`, its trace,
    and each tetrahedron's turn from the strain's change across it. Read at a place and a time, a tetrahedron's corners
    blend by the place's barycentric weights and the two nearest moments by their nearness."""

    def __init__(self, mesh: Mesh, vertices: Point, rates: Rate, changes: list, change_rates: list, first_time: float,
                 level_step: float) -> None:
        self.mesh, self.first_time, self.level_step = mesh, first_time, level_step
        self.ramps = coframe(mesh, vertices)                                    # [corners] Plane+[tets]
        self.coframes = invariant_coframe(mesh, self.ramps, rates)              # [3] Bivector+[tets]
        space, chart = Space(mesh, vertices), charts(mesh, vertices, self.ramps, self.coframes)
        fields = [strain_field(mesh, space, self.ramps, chart, change) for change in changes]
        self.strains = stack([strain.batch() for strain, _ in fields])           # [levels, vertices] Vector+ <- Antivector-
        self.turns = stack([turn.batch() for _, turn in fields])                 # [levels, tets] Bivector+ <- Antivector-
        self.strain_rates = stack([strain_field(mesh, space, self.ramps, chart, rate)[0].batch() for rate in change_rates])
        self.traces = translating(self.strains).trace()                          # [levels, vertices] Scalar+

    def at(self, faces: np.ndarray, places: Point, time: float) -> tuple[Connection, Stepping]:
        """The connection at the places `[places] Point-` in the tetrahedra `faces` at a time, `[places] Rate+ <- Line-`,
        and the squeeze of a step there, half the strain's rate read back as a step, `[places] Antivector- <-
        Antivector-`."""
        level = np.clip((time - self.first_time) / self.level_step, 0, len(self.traces) - 1.000001)
        early, share = int(level), level - int(level)
        corners = self.mesh.tets[faces].T                                       # [corners, places]
        weights = self.ramps.batch()[:, faces] & places[None]                   # [corners, places] Scalar+

        def blended(values):
            """A value at the vertices, `[levels, vertices]`, at the places and the time."""
            between = values[early] * (1 - share) + values[early + 1] * share   # [vertices]
            return (weights * between[corners]).sum(axis=0)                     # [places]

        strain, strain_rate, trace = blended(self.strains), blended(self.strain_rates), blended(self.traces)
        turn = self.turns[early, faces] * (1 - share) + self.turns[early + 1, faces] * share   # [places] Bivector+ <- Antivector-
        _, _, axes = invariant_generators()
        # The round step of a line, its reading by the invariant coframe along each direction.
        steps = (axes[:, None] * (self.coframes.batch()[:, faces] & Line)).sum(axis=0)   # [places] Antivector- <- Line-
        stretch = translating(strain(steps))                                    # [places] Antivector- <- Line-
        local = moving(steps + stretch / 2) + turning(steps - 2 * stretch + steps * (trace / 2)) + turn(steps)
        return local, translating(strain_rate) / 2


def proper_separations(local: Connection, bonds: Mesh, positions: Point) -> Point:
    """Each bond's proper separation, `Antivector-[bonds]`, a step of the origin in the frames: the step the connection
    at the bond's tail, `(Rate+ <- Line-)[particles]`, gives the chart separation of its ends `Point-[particles]`."""
    tails = bonds.incidence[1, 0][0]                                            # Scalar+[bonds <- particles]
    separations = bonds.d[0] * positions                                        # Point-[bonds]
    return origin.commutator((tails * local)((tails * positions) & separations)).cast(Step)


def massive(mesh: Mesh, vertices: Point, ramps: Plane, waves: Waves, bonds: Mesh, rest: Scalar, stiffness: Scalar,
            damping: Scalar, starts: Point, velocities: Point, start: float, step: float, times: np.ndarray
            ) -> Iterator[tuple[Point, Scalar]]:
    """Massive particles through a three-sphere carrying the gravitational waves `waves`, held by springs along
    `bonds`, a graph `Mesh` over the particles, of rest proper lengths `rest`, stiffness `stiffness` and damping
    `damping` over unit masses, `Scalar+[bonds]` each: from `starts` `Point-[particles]` with velocities `velocities`,
    steps of the origin `Antivector-[particles]` in the frames, at time `start`, stepped by `step` of time with
    symplectic Euler, their positions and the bonds' proper strains `Scalar+[bonds]` at each of the `times`. A
    particle's velocity turns with the connection along its motion, the slices' stretch squeezes it, and its bonds pull
    it along their proper separations."""
    pick = mesh.picking(vertices)
    positions, velocities, lengths, time = starts.batch(), velocities.batch().cast(Step), rest, start
    for target in times:
        while time < target - step / 2:
            faces = pick(positions)                                             # [particles]
            unit = ramps.batch()[:, faces].sum(axis=0)                          # [particles] Plane+
            positions = positions / (unit & positions)                          # [particles] Point-
            local, squeeze = waves.at(faces, positions, time)                   # [particles] Rate+ <- Line-, Antivector- <- Antivector-
            proper = proper_separations(local.field(), bonds, positions.field())   # Antivector-[bonds]
            previous, lengths = lengths, proper.dot(proper).square_root()       # Scalar+[bonds], Scalar+[bonds]
            # Springs and dashpots along the proper separations, pulling a stretched bond's ends together.
            pulls = (stiffness * (lengths - rest) + damping * (lengths - previous) / step) / lengths   # Scalar+[bonds]
            forces = -(~bonds.d[0] * (pulls * proper)).batch()                  # [particles] Antivector-
            # The velocity turned by the connection along the motion, squeezed by the slices' stretch and pushed by the
            # springs.
            stepping = (origin.commutator(local(positions & Point)) + origin * (unit & Point)).cast(Stepping)
            motion = stepping.solve(velocities)                                 # [particles] Point-
            # The turn's step part only, cast so: what the turn moves toward the origin's own part a step drops.
            turned = velocities - velocities.commutator(local(positions & motion)).cast(Step) * step
            velocities = (turned - squeeze(turned) * step + forces * step).cast(Step)   # [particles] Antivector-
            positions = positions + stepping.solve(velocities) * step           # [particles] Point-
            time += step
        yield positions.field(), lengths / rest - 1


# A ray's step on the GPU, in numga's dialect, through a three-sphere meshed by tetrahedra and carrying stepped
# gravitational waves, by the lines of `Waves.at` and `geodesics`: each tetrahedron's corner ramps, its neighbours
# across each facet and the facet each enters by, its invariant coframe and its corners' vertices; at each stored
# moment the strains at the vertices, their rates and traces, and the tetrahedra's turns; the origin, the plane at
# infinity, four points that span the rest for the solve, and the invariant steps.
geodesic_step_source = """
var<storage, read> ramps: array<Plane>;
var<storage, read> neighbours: array<u32>;
var<storage, read> entries: array<u32>;
var<storage, read> coframes: array<Rate>;
var<storage, read> tet_vertices: array<u32>;
var<storage, read> strains: array<Strain>;
var<storage, read> strain_rates: array<Strain>;
var<storage, read> traces: array<f32>;
var<storage, read> turns: array<Turning>;
var<uniform> n_vertices: u32;
var<uniform> n_tets: u32;
var<uniform> n_levels: u32;
var<uniform> first_time: f32;
var<uniform> level_step: f32;
var<uniform> origin: Point;
var<uniform> infinity: Plane;
var<uniform> basis_0: Point;
var<uniform> basis_1: Point;
var<uniform> basis_2: Point;
var<uniform> basis_3: Point;
var<uniform> step_x: Step;
var<uniform> step_y: Step;
var<uniform> step_z: Step;
var<uniform> across: Across;
var<uniform> translating: Translating;

// The waves at a place in a tetrahedron, its corners' barycentric weights `l`, at a time: the strain, its rate and
// trace blended over the corners and the two nearest moments, and the tetrahedron's turn.
struct Local {
    strain: Strain,
    strain_rate: Strain,
    trace: f32,
    turn: Turning,
}

fn local_waves(tet: u32, l: array<f32, 4>, time: f32) -> Local {
    let level = clamp((time - first_time) / level_step, 0.0, f32(n_levels) - 1.001);
    let early = u32(floor(level));
    let share = level - floor(level);
    let now = early * n_vertices;
    let next = now + n_vertices;
    // A variable starts at nothing.
    var waves: Local;
    for (var corner = 0u; corner < 4u; corner++) {
        let vertex = tet_vertices[tet * 4u + corner];
        let early_weight = l[corner] * (1.0 - share);
        let late_weight = l[corner] * share;
        waves.strain = waves.strain + strains[now + vertex] * early_weight + strains[next + vertex] * late_weight;
        waves.strain_rate = waves.strain_rate + strain_rates[now + vertex] * early_weight
                            + strain_rates[next + vertex] * late_weight;
        waves.trace = waves.trace + traces[now + vertex] * early_weight + traces[next + vertex] * late_weight;
    }
    waves.turn = turns[early * n_tets + tet] * (1.0 - share) + turns[(early + 1u) * n_tets + tet] * share;
    return waves;
}

// The connection in a tetrahedron: the round step of the line, stretched by the strain, moves the origin; turned
// less twice the strain plus half its trace, it turns the frame square to it; and the strain's change adds its turn.
fn connection(tet: u32, line: Line, waves: Local) -> Rate {
    let base = tet * 3u;
    let steps = step_x * (coframes[base] & line) + step_y * (coframes[base + 1u] & line)
                + step_z * (coframes[base + 2u] & line);
    let stretch = translating(waves.strain(steps));
    let moved = steps + stretch * 0.5;
    let turned = steps - stretch * 2.0 + steps * (waves.trace * 0.5);
    return (infinity ^ across(moved)) + (infinity ^ across(turned)) * mv.xyzw + waves.turn(steps);
}

// The plane on which a tetrahedron's barycentric weights add up to one.
fn unit_plane(tet: u32) -> Plane {
    let base = tet * 4u;
    return ramps[base] + ramps[base + 1u] + ramps[base + 2u] + ramps[base + 3u];
}

// The origin's step under the connection of the line through the place along a velocity, and in the origin's own
// part the weight the tetrahedron reads in the velocity.
fn stepping(tet: u32, place: Point, unit: Plane, motion: Point, waves: Local) -> Point {
    return (origin.commutator(connection(tet, place & motion, waves)) + origin * (unit & motion)).cast(Point);
}

// The velocity whose stepping is the given step, by Cramer's rule: each spanning point's share is the join of the
// steppings with the step in its place, over the join of them all.
fn velocity(tet: u32, place: Point, unit: Plane, target: Point, waves: Local) -> Point {
    let s0 = stepping(tet, place, unit, basis_0, waves);
    let s1 = stepping(tet, place, unit, basis_1, waves);
    let s2 = stepping(tet, place, unit, basis_2, waves);
    let s3 = stepping(tet, place, unit, basis_3, waves);
    let volume = s0 & s1 & s2 & s3;
    return (basis_0 * (target & s1 & s2 & s3) + basis_1 * (s0 & target & s2 & s3) + basis_2 * (s0 & s1 & target & s3)
            + basis_3 * (s0 & s1 & s2 & target)) / volume;
}

// A ray's heading, a unit step of the origin in the frames, after a step of length `h` traced back in time: turned by
// the connection's rate, the turn kept a step of the origin, cast so, and stretched as the slices change, back in
// time, by half the strain's rate.
fn headed(heading: Step, rate: Rate, waves: Local, h: f32) -> Step {
    let turned = heading - heading.commutator(rate).cast(Step) * h;
    let reading = waves.strain_rate(turned);
    let stretched = turned + (translating(reading) - turned * (reading & turned)) * (0.5 * h);
    return stretched.normalized();
}
"""


# --- plumbing -------------------------------------------------------------------------
def jacobian(slopes, rest: Scalar, nudge: float) -> Scalar:
    """The change of `slopes(lengths)`, `Scalar+[edges]`, with each of the lengths about `rest`, by a forward difference
    of relative size `nudge` along each in turn, as a dense map `(Scalar+ <- Scalar+)[edges <- edges]`: a numeric
    excursion, its columns gathered outside the algebra."""
    at_rest = slopes(rest).to_array()
    sizes = nudge * rest.to_array()
    units = np.eye(len(sizes))
    columns = np.stack([(slopes(rest + mesh_module.as_scalar(units[i] * sizes[i]).field()).to_array() - at_rest)
                        / sizes[i] for i in range(len(sizes))], axis=-1)        # [edges, edges]
    return ctx.extensor(ga.gatype((Scalar, Scalar)), columns[..., None, None]).field(0, 1)


def transverse(masses: Scalar, moves: Scalar) -> Scalar:
    """The projection `(Scalar+ <- Scalar+)[edges <- edges]` that keeps of a field of changes on the edges what is
    square, by the masses, to every change the `moves` `(Scalar+ <- Point-)[edges <- vertices]` make: a numeric
    excursion, by a pseudo-inverse outside the algebra."""
    mass = np.asarray(masses.kernel)[..., 0, 0]                                 # [edges, edges]
    families = np.asarray(moves.kernel)[:, :, 0, :].reshape(mass.shape[0], -1)  # [edges, vertices * blades]
    weighted = families.T @ mass
    kept = np.eye(mass.shape[0]) - families @ np.linalg.pinv(weighted @ families, rcond=1e-10) @ weighted
    return ctx.extensor(ga.gatype((Scalar, Scalar)), kept[..., None, None]).field(0, 1)


# How far outside a side its ramp may read a point, by round-off, while the point still counts as inside.
margin = 1e-9
