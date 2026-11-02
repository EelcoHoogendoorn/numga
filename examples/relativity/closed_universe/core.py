"""Gravitational waves of a closed universe, a round three-sphere meshed by tetrahedra: the waves stepped on the
edges' squared lengths, a smooth field of strain over the mesh, light and massive particles rolled through it, and the
sphere below its Hopf circles, on which waves that are the same all along the circles are seen without loss.

Regge calculus holds a curved space in the lengths of a mesh's edges. Read as pieces of the round three-sphere, each
tetrahedron's lengths give its angles, and the deficit about each edge, how far its tetrahedra's angles fall short of
a whole turn, is the curvature there; the round sphere's own chords leave no deficit, on any mesh. Gravitational
waves are small changes of the squared lengths: the deficits they cause pull them back, a stiffness over the masses
of the lengths' motion, and what moving the vertices alone would change is no wave and is kept out. The slowest, the
squash, stretches the whole sphere along one direction and squeezes it along another.

The algebra is that of four directions squaring to one, its points antivectors on the unit sphere and the origin the
pole of its equator, the plane `w`. Each point carries its isoclinic frame, `isoclinic(place)`, the motor that carries
the origin there by turning alike in two square planes, smooth everywhere, and a step of the origin read in these
frames is a direction anywhere. A strain is the change of the metric on such steps, a map from a step to the plane
that reads its stretch, blended across each tetrahedron from its corners. A ray rolls its frame along its step: the
step, stretched by half the strain, moves the origin, and the strain turns the frame; in the round sphere a ray's
heading keeps still along its great circle, so only the waves turn it. Massive particles roll the same way, pulled by
springs along their bonds' proper separations. On the GPU a ray crosses the mesh in legs, each straight across a
tetrahedron's chart, the plane through its corners.

The Hopf circles are the orbits of the turns that act on the frames' own side. Read in its frames, a circle's
direction is one and the same step of the origin all along it, a point of the sphere of such steps below the whole
circle, the base. Read along a circle, a wave's strain says how much the circle stretches; read across it, turned a
quarter in the plane square to its base point, it is the strain of the base itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import reduce
from operator import and_

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag
from examples.util.topology import Topology, lookup_simplices

# The algebra of the round three-sphere: four directions squaring to one, `w` the origin's own.
ga = Algebra("x+y+z+w+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# What is integrated over a volume, the action and the momenta of the squared lengths among them.
Amount = ga.gatype.antiscalar()                     # Antiscalar-
# The corners' ramps, gradients and the polar planes of directions: vectors, extrinsic; those through the origin.
Plane = ga.gatype.vector()                          # Vector+
Through = Plane.derive.within("x y z")              # Vector+
# Points of the sphere and the origin's steps: antivectors.
Point = ga.gatype.antivector()                      # Antivector-
Step = Point.derive.containing("w")                 # Antivector-
# The rates that move and turn the frames, and the motors they generate.
Rate = ga.gatype.bivector()                         # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Translating = ga.gatype((Step, Through))            # Antivector- <- Vector+
Metric = ga.gatype((Plane, Point))                  # Vector+ <- Antivector-
InverseMetric = ga.gatype((Point, Plane))           # Antivector- <- Vector+
# A strain, the change of the metric on the origin's steps, the plane that reads each step's stretch; a turning, the
# rate a step adds to a frame's turn; a rolling, the rate the frames roll by along each round step; and a stretching,
# the steps a map of them gives.
Strain = ga.gatype((Through, Step))                 # Vector+ <- Antivector-
Turning = ga.gatype((Rate, Step))                   # Bivector+ <- Antivector-
Rolling = ga.gatype((Rate, Step))                   # Bivector+ <- Antivector-
Stretching = ga.gatype((Step, Step))                # Antivector- <- Antivector-
# The rates that move the origin along its steps, in the planes through it.
Moving = ga.gatype((Rate.derive.containing("w"), Step))   # Bivector+ <- Antivector-

# The plane `w`, the origin's equator, and the origin, its pole.
equator = mv.w                                      # [] Plane+
origin = equator.dual()                             # [] Point-
# The step of the origin under the motion across a plane through it, a step by its value, cast so; its inverse, each
# step's polar plane, is the round sphere's metric on the origin's steps as a correlation.
translating = origin.commutator(equator ^ Through).cast(Translating)   # [] Point- <- Plane+
polar = translating.inverse()                                           # [] Plane+ <- Point-
# The rate that moves the origin along each of its steps, the meet of the equator with the step's polar plane.
moving = equator ^ polar(Step)                                          # [] Rate+ <- Point-
# The extrinsic pseudoscalar, which takes a rate to the rate in the plane square to it: the halves of the rates it
# keeps and the halves it flips turn alike in two square planes, the first as the isoclinic frames do.
pseudoscalar = mv.xyzw                                                  # [] Antiscalar+
# The rate that turns a frame in the plane square to each step's moving rate.
turning = moving * pseudoscalar                                         # [] Rate+ <- Point-
# The share of the vertex moves' largest square that settles the moves of the whole sphere, and the relative size of
# the nudges to the squared lengths that find the stiffness.
settling = 1e-10
nudge = 1e-6


# --- math -----------------------------------------------------------------------------
def coframe(mesh: Topology, vertices: Point) -> Plane:
    """Each corner's ramp across its simplex, `[corners] Plane+[simplices]`: the plane through the facet across from
    the corner, the join of the facet's corners, scaled to read one at the corner and so nothing on the facet."""
    corners = mesh.incidence[-1, 0] * vertices                                  # [corners] Point-[simplices]
    facets = reduce(and_, corners[mesh.corners[-1, -2]])                        # [corners] Plane-[simplices]
    return facets / (facets & corners)


class Space:
    """A triangulated three-sphere as its tetrahedra know it, for stepping its edges' squared lengths by Regge's
    action: each tetrahedron a piece of the round sphere for its angles, and flat for its masses. It holds their
    ramps and centred corner gradients, `[corners] Plane+[cells]`, their hyperplanes, and the maps from squared edge
    lengths to the metrics of the cone over each from the centre and of the flat tetrahedron."""

    def __init__(self, mesh: Topology, vertices: Point) -> None:
        corners = mesh.incidence[3, 0] * vertices                               # [corners] Point-[cells]
        ramps = coframe(mesh, vertices)                                         # [corners] Plane+[cells]
        # The tetrahedron's hyperplane, one on each of its points; less a quarter of it, the ramps vanish on the
        # centroid and still read the edges.
        self.mesh, self.ramps = mesh, ramps
        self.radial = ramps.sum(axis=0)                                         # Plane+[cells]
        self.gradients = ramps - self.radial / 4                                # [corners] Plane+[cells]
        # Along its own edge two corner gradients pair to -1 and +1 and the others vanish, so that their symmetric
        # product reads that edge's squared length and nothing of the other five.
        tails, heads = mesh.corners[3, 1]
        first, second = self.gradients[tails], self.gradients[heads]            # [cell_edges] Plane+[cells], [cell_edges] Plane+[cells]
        basis = -(first * (second & Point) + second * (first & Point)) / 2      # [cell_edges] (Plane+ <- Point-)[cells]
        self.metric_map = (spdiag(basis) * mesh.incidence[3, 1]).sum(axis=0)    # (Plane+ <- Point-)[cells <- edges]
        # The same products of the ramps themselves give the cone from the centre: its corners pair to one, less half
        # the squared chord between them.
        tail_ramps, head_ramps = ramps[tails], ramps[heads]                     # [cell_edges] Plane+[cells], [cell_edges] Plane+[cells]
        cone_basis = -(tail_ramps * (head_ramps & Point) + head_ramps * (tail_ramps & Point)) / 2   # [cell_edges] (Plane+ <- Point-)[cells]
        self.cone_map = (spdiag(cone_basis) * mesh.incidence[3, 1]).sum(axis=0)   # (Plane+ <- Point-)[cells <- edges]
        self.spans = corners[1:] - corners[0]                                   # [spans] Point-[cells]
        self.joins = corners[0] & corners[1] & corners[2] & corners[3]          # Scalar-[cells]

    def dynamics(self, squared_lengths: Scalar) -> tuple[Scalar, Scalar, SparseExtensor]:
        """The deficit angle at each edge of the tetrahedra read as pieces of the round sphere; the slope of the
        potential, minus twice the Regge action of those spherical tetrahedra less four times their volume, with
        respect to each squared edge length; and the mass map of relativity's kinetic term, a quarter of each flat
        tetrahedron's volume times the square of its inverse metric's reading of the metric's rate, for the edges'
        squared lengths at rest. Squared chords of points on the round sphere tile it with no deficit, at rest."""
        mesh = self.mesh
        tails, heads = mesh.corners[3, 1]
        # A spherical tetrahedron's faces lie on planes through the centre, its ramps; through the inverse of the
        # cone's metric they pair to the cosines of the angles between its faces, on the corners off each edge.
        cone = self.radial * (self.radial & Point) + self.cone_map * squared_lengths   # (Plane+ <- Point-)[cells]
        cone_pairings = self.ramps[:, None] & cone.batch().inverse().field()(self.ramps[None, :])   # [corners, corners] Scalar+[cells]
        off_tails, off_heads = mesh.complements[3, 1]
        off_edges = cone_pairings[off_tails, off_heads]                         # [cell_edges] Scalar+[cells]
        sides = cone_pairings[off_tails, off_tails] * cone_pairings[off_heads, off_heads]   # [cell_edges] Scalar+[cells]
        angles = (-off_edges / sides.square_root()).arccos()                    # [cell_edges] Scalar+[cells]
        deficits = 2 * np.pi - (~mesh.incidence[3, 1] * angles).sum(axis=-1)    # Scalar+[edges]
        # By Schlafli a spherical tetrahedron's volume changes by half its arcs times its angles' changes, so four
        # times it cancels the angles' part of the action's change, and the slope is the deficits' alone, times the
        # arc's change with the squared chord: a half over the chord and the cosine of half the arc. The action is
        # an integral over the volume, so its slopes are amounts.
        slopes = (-deficits / (squared_lengths * (1 - squared_lengths / 4)).square_root()).dual()   # Amount-[edges]
        _, inverses = self.metrics(squared_lengths)
        pairings = self.gradients[:, None] & inverses(self.gradients[None, :])   # [corners, corners] Scalar+[cells]
        volumes = self.volumes(squared_lengths)                                 # Scalar+[cells]
        # Relativity's kinetic term on a tetrahedron, between the rates of two edges' squared lengths: a quarter of
        # the volume times the pairings of their corners across, both ways; integrated over the volume, it is the
        # inertia of space, taking the rates to their momenta, amounts.
        local = volumes / 4 * (pairings[tails[:, None], tails[None, :]] * pairings[heads[:, None], heads[None, :]]
                               + pairings[tails[:, None], heads[None, :]] * pairings[heads[:, None], tails[None, :]])   # [cell_edges, cell_edges] Scalar+[cells]
        edges = mesh.incidence[3, 1]                                            # [cell_edges] numbers[cells <- edges]
        masses = (~edges[:, None] * spdiag(local * Scalar.dual()) * edges[None, :]).sum(axis=0).sum(axis=0)   # (Amount- <- Scalar+)[edges <- edges]
        return deficits, slopes, masses

    def volumes(self, squared_lengths: Scalar) -> Scalar:
        """Each flat tetrahedron's volume from the squared lengths: the root of the determinant of its metric over the
        corner gradients' frame, a sixth of it."""
        metrics, _ = self.metrics(squared_lengths)
        first, second, third = self.spans                                       # Point-[cells], Point-[cells], Point-[cells]
        return ((metrics(first) ^ metrics(second) ^ metrics(third) ^ self.radial) & self.joins).abs().square_root() / 6

    def metrics(self, squared_lengths: Scalar) -> tuple[Metric, InverseMetric]:
        """Each flat tetrahedron's metric from the squared lengths of the edges, the hyperplane taking the centroid,
        which the gradients leave out, so that it can be inverted; and its inverse."""
        metrics = self.radial * (self.radial & Point) + self.metric_map * squared_lengths   # (Plane+ <- Point-)[cells]
        return metrics, metrics.batch().inverse().field()


def gravitational_waves(mesh: Topology, vertices: Point) -> tuple[SparseExtensor, SparseExtensor, OntoWaves]:
    """The linear gravitational waves of a triangulated three-sphere at rest, as the stiffness and the masses on its
    edges' squared lengths, sparse maps `(Amount- <- Scalar+)[edges <- edges]` from the lengths' changes and rates to
    the forces and momenta, amounts, coupling the edges that share a tetrahedron, and the projection onto the waves. The stiffness is that of Regge's potential with each tetrahedron
    read as a piece of the round sphere, at rest on any mesh of it. The waves are the lengths no move of the vertices
    makes: the tangent moves slide the vertices over the sphere, which the spherical tetrahedra still tile, and the
    radial ones change each edge's squared length in proportion to the sum of its ends' moves, the conformal
    stretches, whose breathing is the universe at rest's own instability. What is left are the transverse traceless
    strains."""
    space = Space(mesh, vertices)
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    rest = chords.dot(chords)                                                   # Scalar+[edges]
    _, _, masses = space.dynamics(rest)
    stiffness = jacobian(lambda lengths: space.dynamics(lengths)[1], rest, nudge, mesh)
    moves = spdiag(2 * chords.dot(Point)) * mesh.d[0]                          # (Scalar+ <- Point-)[edges <- vertices]
    return stiffness, masses, OntoWaves(masses, moves)


class OntoWaves:
    """The projection of the changes of the squared edge lengths, `Scalar+[edges]`, onto the waves: the changes less
    the change the vertex moves make that is nearest them by the masses, the moves found by a solve over the vertices,
    the masses pulled back to the moves by their adjugate. The moves of the whole sphere change no length, and a tiny
    share of the isotropic form on each vertex's moves, the algebra's own dual, settles them at nothing."""

    def __init__(self, masses: SparseExtensor, moves: SparseExtensor) -> None:
        self.masses, self.moves = masses, moves
        reach = moves.adjugate()(masses(moves))                                 # (Plane+ <- Point-)[vertices <- vertices]
        self.reach = reach + settling * largest(reach) * Point.dual()

    def __call__(self, changes: Scalar) -> Scalar:
        shares = self.reach.solve(self.moves.adjugate()(self.masses(changes)))  # Point-[vertices]
        return changes - self.moves(shares)


def waves_stepped(stiffness: SparseExtensor, masses: SparseExtensor, onto_waves: OntoWaves, strains: Scalar,
                  rates: Scalar, step: float) -> Iterator[tuple[Scalar, Scalar]]:
    """The squared lengths' changes and their rates, `Scalar+[edges]` each, at every step of a leapfrog under the
    stiffness and the masses, kept to the waves."""
    while True:
        yield strains, rates
        rates = rates - (step / 2) * onto_waves(masses.solve(stiffness(strains)))
        strains = strains + step * rates
        rates = rates - (step / 2) * onto_waves(masses.solve(stiffness(strains)))


def isoclinic(places: Point) -> Motor:
    """The isoclinic frames at places on the round three-sphere, `Motor+`: the motors that carry the origin to each
    by turning alike in the plane of the great circle between them and in the plane square to it, the half of the
    quotient of the place by the origin that the pseudoscalar keeps, with one in the other half. They are smooth
    everywhere, and in them a free path's heading keeps still along its great circle."""
    return ((1 + pseudoscalar) * (places.normalized() / origin) + (1 - pseudoscalar)) / 2


def round_steps(graph: Topology, places: Point) -> Point:
    """Each edge's round step, `Point-[edges]`, a step of the origin: its head carried back by its tail's isoclinic
    frame, less the origin it is beside."""
    tails, heads = graph.incidence[1, 0] * places                               # Point-[edges], Point-[edges]
    return (isoclinic(tails).inverse() >> heads).cast(Step)


def strain_field(mesh: Topology, space: Space, ramps: Plane, frames: Motor, changes: Scalar) -> tuple[Strain, Turning]:
    """The strain of a change of the squared edge lengths, `Scalar+[edges]`, as a smooth field on the origin's steps:
    at each vertex the average of its tetrahedra's strains, `Strain[vertices]`, each the change of the metric of the
    cone over it from the centre, read on the steps its isoclinic frame at its middle, `Motor+[tets]`, carries out to
    it, and carried back; and in each tetrahedron the turn the strain's change from corner to corner adds, half the
    curl of the strain, `Turning[tets]`."""
    frame_steps = frames >> Step                                                # (Point- <- Point-)[tets]
    # The strain read in the frames: what reads nothing on the steps, along the origin itself, cast away.
    strains = (frames.inverse() >> (space.cone_map * changes)(frame_steps)).cast(Strain)   # (Plane+ <- Point-)[tets]
    shares = ~mesh.incidence[3, 0]                                              # [corners] numbers[vertices <- tets]
    counts = (shares * mv.scalar(np.ones((mesh.n_elements[3], 1))).field()).sum(axis=0)   # Scalar+[vertices]
    at_vertices = (shares * strains).sum(axis=0) / counts                       # (Plane+ <- Point-)[vertices]
    # Each corner's ramp read on the carried steps, and the strain at each corner: their wedge, summed over the
    # corners, is the strain's change across the tetrahedron turned into the plane it turns in.
    corner_planes = (frames.inverse() >> ramps).cast(Through)                   # [corners] Plane+[tets]
    corner_strains = mesh.incidence[3, 0] * at_vertices                         # [corners] (Plane+ <- Point-)[tets]
    return at_vertices, (corner_planes ^ corner_strains).sum(axis=0) / 2


class Waves:
    """Stepped gravitational waves of a round three-sphere triangulated by `mesh` on `vertices`, as a smooth field over
    its tetrahedra and over time, read in the isoclinic frames: at each of the stored moments, from `first_time` every
    `level_step`, the strain of the squared edge lengths' changes `changes` at the vertices, its rate from
    `change_rates`, its trace, and each tetrahedron's turn from the strain's change across it. Read at a place and a
    time, a tetrahedron's corners blend by the place's barycentric weights and the two nearest moments by their
    nearness."""

    def __init__(self, mesh: Topology, vertices: Point, changes: list, change_rates: list, first_time: float,
                 level_step: float) -> None:
        self.mesh, self.first_time, self.level_step = mesh, first_time, level_step
        self.ramps = coframe(mesh, vertices)                                    # [corners] Plane+[tets]
        frames = isoclinic(mesh.incidence[3, 0].mean(axis=0) * vertices)      # Motor+[tets]
        space = Space(mesh, vertices)
        fields = [strain_field(mesh, space, self.ramps, frames, change) for change in changes]
        self.strains = stack([strain.batch() for strain, _ in fields])           # [levels, vertices] Plane+ <- Point-
        self.turns = stack([turn.batch() for _, turn in fields])                 # [levels, tets] Rate+ <- Point-
        self.strain_rates = stack([strain_field(mesh, space, self.ramps, frames, rate)[0].batch()
                                   for rate in change_rates])                   # [levels, vertices] Plane+ <- Point-
        self.traces = translating(self.strains).trace()                          # [levels, vertices] Scalar+

    def fields(self, cells: np.ndarray, places: Point, time: float) -> tuple[Strain, Strain, Scalar, Turning]:
        """The waves' fields at the places `[...] Point-` on the sphere, in the tetrahedra `cells` `[...]`, at a time:
        the strain, its rate and its trace, blended over the corners and the two nearest moments, and the tetrahedra's
        turn, each `[...]`."""
        level = (time - self.first_time) / self.level_step
        early, share = int(level), level - int(level)
        corners = np.moveaxis(self.mesh.elements[3][cells], -1, 0)              # [corners, ...]
        ramps = self.ramps.batch()[:, cells]                                    # [corners, ...] Plane+
        weights = (ramps & places[None]) / (ramps.sum(axis=0) & places)         # [corners, ...] Scalar+

        def blended(values):
            """A value at the vertices, `[levels, vertices]`, at the places and the time."""
            between = values[early] * (1 - share) + values[early + 1] * share   # [vertices]
            return (weights * between[corners]).sum(axis=0)                     # [...]

        turn = self.turns[early, cells] * (1 - share) + self.turns[early + 1, cells] * share   # [...] Rate+ <- Point-
        return blended(self.strains), blended(self.strain_rates), blended(self.traces), turn

    def at(self, cells: np.ndarray, places: Point, time: float) -> tuple[Rolling, Stretching]:
        """The connection at the places `[places] Point-` on the sphere, in the tetrahedra `cells`, at a time: the rate
        the frames roll by along each round step there, `[places] Rate+ <- Point-`; and the squeeze of a step there,
        half the strain's rate read back as a step, `[places] Point- <- Point-`."""
        strain, strain_rate, trace, turn = self.fields(cells, places, time)
        # A round step, stretched by the strain, moves the origin; turned less twice the strain plus half its trace, it
        # turns the frame square to it; and the strain's change adds its turn.
        stretch = translating(strain(Step))                                     # [places] Point- <- Point-
        local = moving(Step + stretch / 2) + turning(Step - 2 * stretch + Step * (trace / 2)) + turn(Step)
        return local, translating(strain_rate) / 2


def proper_separations(local: Rolling, bonds: Topology, positions: Point) -> Point:
    """Each bond's proper separation, `Point-[bonds]`, a step of the origin in the frames: the step the connection at
    the bond's tail, `(Rate+ <- Point-)[particles]`, moves the origin by along the bond's round step."""
    tails = bonds.incidence[1, 0][0]                                            # numbers[bonds <- particles]
    return origin.commutator((tails * local)(round_steps(bonds, positions))).cast(Step)


def massive(pick: Callable[[Point], np.ndarray], waves: Waves, bonds: Topology, rest: Scalar, stiffness: Scalar,
            damping: Scalar, holding: float, starts: Point, velocities: Point, start: float, step: float,
            times: np.ndarray) -> Iterator[tuple[Point, Scalar]]:
    """Massive particles through a round three-sphere carrying the gravitational waves `waves`, the tetrahedra holding
    places found by `pick`, held by springs along
    the edges of `bonds`, over the particles, of rest proper lengths `rest`, stiffness `stiffness` and damping
    `damping` over unit masses, `Scalar+[bonds]` each, and each held to its start by a spring of stiffness `holding`,
    damped critically: from `starts` `Point-[particles]` on the sphere with velocities `velocities`, steps of the
    origin `Point-[particles]` in the isoclinic frames, at time `start`, stepped by `step` of time with symplectic
    Euler, their positions and the bonds' proper strains `Scalar+[bonds]` at each of the `times`. A particle's velocity
    turns with the connection along its motion, the slices' stretch squeezes it, and its bonds pull it along their
    proper separations; it rolls along its round step on its great circle."""
    homes = starts.batch()                                                      # [particles] Point-
    positions, velocities, lengths, time = homes, velocities.batch(), rest, start
    for target in times:
        while time < target - step / 2:
            cells = pick(positions)                                             # [particles]
            local, squeeze = waves.at(cells, positions, time)                   # [particles] Rate+ <- Point-, Point- <- Point-
            proper = proper_separations(local.field(), bonds, positions.field())   # Point-[bonds]
            previous_lengths, lengths = lengths, proper.dot(proper).square_root()   # Scalar+[bonds], Scalar+[bonds]
            # Springs and dashpots along the proper separations, pulling a stretched bond's ends together.
            pulls = (stiffness * (lengths - rest) + damping * (lengths - previous_lengths) / step) / lengths   # Scalar+[bonds]
            forces = -(~bonds.d[0] * (pulls * proper)).batch()                  # [particles] Point-
            # Each particle pulled toward its home, a free particle at rest where it started, and its motion damped.
            home_steps = (isoclinic(positions).inverse() >> homes).cast(Step)  # [particles] Point-
            forces = forces + holding * home_steps - 2 * np.sqrt(holding) * velocities
            # The step each round step moves the origin by: a velocity's round step is the one it moves the origin by.
            stepping = origin.commutator(local(Step)).cast(Stretching)          # [particles] Point- <- Point-
            motion = stepping.solve(velocities)                                 # [particles] Point-
            # The velocity turned by the connection along the motion, squeezed by the slices' stretch and pushed by the
            # springs; the turn's step part only, cast so.
            carried_velocities = velocities - velocities.commutator(local(motion)).cast(Step) * step
            velocities = carried_velocities - squeeze(carried_velocities) * step + forces * step   # [particles] Point-
            # Rolled along the round step of its velocity, on its great circle from the frame at its place.
            rolls = isoclinic(positions) * (moving(stepping.solve(velocities)) * (-step / 2)).exp()   # [particles] Motor+
            positions = rolls >> origin                                         # [particles] Point-
            time += step
        yield positions.field(), lengths / rest - 1


def light(pick: Callable[[Point], np.ndarray], waves: Waves, starts: Point, headings: Point, start: float,
          step: float, n_steps: int) -> Iterator[Point]:
    """Rays of light through a round three-sphere carrying the waves `waves`, the tetrahedra holding places found by
    `pick`, from `starts` `[rays] Point-` on the sphere along `headings`, unit steps of the origin `[rays] Point-` in
    the isoclinic frames, set out at time `start`: their places `[rays] Point-` after each step of length `step`, the
    light's time running with it. As the GPU's rays do, a ray's heading turns with the connection along its round step
    and the slices' stretch squeezes it, and it rolls along that step on its great circle, each step read at its
    midpoint."""

    def turning_and_rolling(places: Point, heading: Point, time: float) -> tuple[Point, Point]:
        """The heading's change per length at the places and the time, and the round step it rolls along."""
        local, squeeze = waves.at(pick(places), places, time)                   # [rays] Rate+ <- Point-, Point- <- Point-
        stepping = origin.commutator(local(Step)).cast(Stretching)              # [rays] Point- <- Point-
        rolling = stepping.solve(heading)                                       # [rays] Point-
        return -heading.commutator(local(rolling)).cast(Step) - squeeze(heading), rolling

    positions, time = starts, start
    for _ in range(n_steps):
        start_change, start_rolling = turning_and_rolling(positions, headings, time)
        # Half a step on: the heading turned by the waves at the start, and the place rolled there along its round step.
        half_headings = (headings + start_change * (step / 2)).normalized()     # [rays] Point-
        half_rolls = isoclinic(positions) * (moving(start_rolling) * (-step / 4)).exp()   # [rays] Motor+
        half_positions = half_rolls >> origin                                   # [rays] Point-
        change, rolling = turning_and_rolling(half_positions, half_headings, time + step / 2)
        headings = (headings + change * step).normalized()                      # [rays] Point-
        rolls = isoclinic(positions) * (moving(rolling) * (-step / 2)).exp()    # [rays] Motor+
        positions = rolls >> origin                                             # [rays] Point-
        time += step
        yield positions


def base_points(places: Point, circling: Rate) -> Point:
    """The base point below each place, `Point-`: the direction its circle runs in under the rate `circling`, read in
    the isoclinic frame at the place, a unit step of the origin, the same all along the circle."""
    circle_directions = places.commutator(circling)                            # Point-
    return (isoclinic(places).inverse() >> circle_directions).cast(Step).normalized()


def crossing(bases: Point, directions: Point) -> Point:
    """Steps along the base, `directions` at the base points, turned a quarter in the plane square to the base point,
    `Point-`: the steps across the circles that a base direction is the image of."""
    return turning(bases).commutator(directions)


def lifts(bases: Point, circling: Rate) -> Point:
    """A point on the circle above each base point, `Point-`: the origin moved across the circles, along the step that
    the direction from the origin's own base point toward each is the image of, by half the angle between them, which
    the Hopf map doubles."""
    origin_base = base_points(origin, circling)                                # [] Point-
    cosines = polar(origin_base) & bases                                        # Scalar+
    directions = (bases - origin_base * cosines).normalized()                   # Point-
    lifting = (moving(crossing(origin_base, directions)) * (-cosines.arccos() / 4)).exp()   # Motor+
    return lifting >> origin


def round_circles(bases: Point, turn: Motor, n_turns: int, n_samples: int) -> Point:
    """So many points evenly round the circle above each base point, `[samples, bases] Point-`: the lifts carried by
    powers of the turn along the circles, `n_turns` of which close them."""
    circling = turn.log()                                                       # [] Rate+
    shares = np.arange(n_samples) * n_turns / n_samples                         # [samples]
    turns = (circling * shares[:, None]).exp()                                 # [samples] Motor+
    return turns >> lifts(bases, circling)[None]


def stretches(waves: Waves, pick: Callable[[Point], np.ndarray], bases: Point, directions: Point, turn: Motor,
              n_turns: int, n_samples: int, time: float) -> Scalar:
    """How much lengths stretch along steps of the origin `directions` in the frames, at the circles above the base
    points, averaged round each, at a time, `Scalar+[bases]`: half the strain's reading of the step on itself."""
    places = round_circles(bases, turn, n_turns, n_samples)                    # [samples, bases] Point-
    strain, _, _, _ = waves.fields(pick(places), places, time)                 # [samples, bases] Plane+ <- Point-
    # The strain read against the metric it changes, each the steps' correlation to the planes that read them.
    readings = (strain(directions) & directions) / (2 * (polar(directions) & directions))   # [samples, bases] Scalar+
    return readings.sum(axis=0) / n_samples


# A ray's legs on the GPU, in numga's dialect, through a round three-sphere meshed by tetrahedra: from a place on the
# sphere along a heading, a unit step of the origin in the isoclinic frame there, across the tetrahedron the place is
# in, having entered it by a face, four where it starts inside, at a time. On a tetrahedron's chart, the plane through
# its corners, a great circle runs straight and the barycentric weights change evenly along it: each leg ends where the
# weight of the corner across from a face runs out first, never the face it entered by, and the next leg starts in the
# neighbour across that face. Through the round sphere a heading keeps still in the frames and a leg runs straight to
# the face, its reach past any tetrahedron's far face and short of its chart's horizon. Each walk defines its `Leg`:
# the chart's plane, where the leg starts on it, the straight segment it runs, the arc it covers, the heading at its
# end, and the tetrahedron and face the next leg starts in.
round_reach = 0.6

round_walk_source = """
var<storage, read> ramps: array<Plane>;
var<storage, read> neighbours: array<u32>;
var<storage, read> entries: array<u32>;
var<uniform> n_tets: u32;
var<uniform> origin: Point;
var<uniform> moving: Moving;
var<uniform> reach: f32;

struct Leg {
    plane: Plane,
    chart: Point,
    segment: Point,
    length: f32,
    heading: Step,
    tet: u32,
    entry: u32,
}

// The isoclinic frame at a place on the round sphere: the motor that carries the origin to it, turning alike in the
// plane of the great circle between them and in the plane square to it, the half of the quotient of the place by the
// origin that the pseudoscalar keeps, with one in the other half.
fn frame(place: Point) -> Motor {
    return ((1.0 + mv.xyzw) * (place / origin) + (1.0 - mv.xyzw)) * 0.5;
}

fn next_leg(place: Point, heading: Step, tet: u32, entry: u32, time: f32) -> Leg {
    // The tetrahedron's chart, the plane that reads one at each corner; the place on it, and the straight line on it
    // to where the ray's great circle is a reach on.
    let unit = ramps[tet] + ramps[n_tets + tet] + ramps[2u * n_tets + tet] + ramps[3u * n_tets + tet];
    let chart = place / (unit & place);
    let reach_end = (frame(place) * (moving(heading) * (-0.5 * reach)).exp()) >> origin;
    let segment = reach_end / (unit & reach_end) - chart;
    // The share of the line where the first corner's weight runs out; a place a hair outside the face it is on, by
    // round-off, leaves at once.
    var share = 1.0;
    var exit = 4u;
    for (var corner = 0u; corner < 4u; corner++) {
        let change = ramps[corner * n_tets + tet] & segment;
        if (corner != entry && change < 0.0) {
            let run_out = max(-(ramps[corner * n_tets + tet] & chart) / change, 0.0);
            if (run_out < share) {
                share = run_out;
                exit = corner;
            }
        }
    }
    let end = (chart + segment * share).normalized();
    var this_leg = Leg(unit, chart, segment * share, acos(clamp(place.dot(end), -1.0, 1.0)), heading, tet, 4u);
    if (exit < 4u) {
        this_leg.tet = neighbours[tet * 4u + exit];
        this_leg.entry = entries[tet * 4u + exit];
    }
    return this_leg;
}
"""

# Through the waves a leg is at most a step long, read at its midpoint: the heading half a step on, turned by the
# waves at the start, and the place there, at the light's time there, running back from the eye's. The waves at a
# place in a tetrahedron, at a time, blend its corners' strains, their rates and traces by the place's barycentric
# weights and the two nearest stored moments by their nearness, with the tetrahedron's turn. The connection along a
# round step and a heading's change follow `Waves.at` and `light`.
wave_step = 0.02

wave_walk_source = """
var<storage, read> ramps: array<Plane>;
var<storage, read> neighbours: array<u32>;
var<storage, read> entries: array<u32>;
var<storage, read> tet_vertices: array<u32>;
var<storage, read> strains: array<Strain>;
var<storage, read> strain_rates: array<Strain>;
var<storage, read> traces: array<f32>;
var<storage, read> turns: array<Turning>;
var<uniform> n_vertices: u32;
var<uniform> n_tets: u32;
var<uniform> first_time: f32;
var<uniform> level_step: f32;
var<uniform> origin: Point;
var<uniform> translating: Translating;
var<uniform> moving: Moving;
var<uniform> turning: Rolling;
var<uniform> step: f32;

struct Leg {
    plane: Plane,
    chart: Point,
    segment: Point,
    length: f32,
    heading: Step,
    tet: u32,
    entry: u32,
}

struct Local {
    strain: Strain,
    strain_rate: Strain,
    trace: f32,
    turn: Turning,
}

fn local_waves(tet: u32, place: Point, time: f32) -> Local {
    let level = (time - first_time) / level_step;
    let early = u32(floor(level));
    let share = level - floor(level);
    let now = early * n_vertices;
    let later = now + n_vertices;
    let unit = ramps[tet] + ramps[n_tets + tet] + ramps[2u * n_tets + tet] + ramps[3u * n_tets + tet];
    // A variable starts at nothing.
    var waves: Local;
    for (var corner = 0u; corner < 4u; corner++) {
        let vertex = tet_vertices[tet * 4u + corner];
        let weight = (ramps[corner * n_tets + tet] & place) / (unit & place);
        let early_weight = weight * (1.0 - share);
        let late_weight = weight * share;
        waves.strain = waves.strain + strains[now + vertex] * early_weight + strains[later + vertex] * late_weight;
        waves.strain_rate = waves.strain_rate + strain_rates[now + vertex] * early_weight
                            + strain_rates[later + vertex] * late_weight;
        waves.trace = waves.trace + traces[now + vertex] * early_weight + traces[later + vertex] * late_weight;
    }
    waves.turn = turns[early * n_tets + tet] * (1.0 - share) + turns[(early + 1u) * n_tets + tet] * share;
    return waves;
}

// The connection along a round step of the origin: stretched by the strain, the step moves the origin; turned less
// twice the strain plus half its trace, it turns the frame square to it; and the strain's change adds its turn.
fn connection(steps: Step, waves: Local) -> Rate {
    let stretch = translating(waves.strain(steps));
    return moving(steps + stretch * 0.5) + turning(steps - stretch * 2.0 + steps * (waves.trace * 0.5))
           + waves.turn(steps);
}

// The round step a heading takes, less half the waves' stretch of it.
fn round_step(heading: Step, waves: Local) -> Step {
    return heading - translating(waves.strain(heading)) * 0.5;
}

// How a heading changes per length traced back in time: turned by the connection along its round step, the turn kept
// a step of the origin, cast so, and stretched as the slices change, back in time, by half the strain's rate.
fn heading_change(heading: Step, waves: Local) -> Step {
    let reading = waves.strain_rate(heading);
    return -heading.commutator(connection(round_step(heading, waves), waves)).cast(Step)
           + (translating(reading) - heading * (reading & heading)) * 0.5;
}

fn frame(place: Point) -> Motor {
    return ((1.0 + mv.xyzw) * (place / origin) + (1.0 - mv.xyzw)) * 0.5;
}

fn next_leg(place: Point, heading: Step, tet: u32, entry: u32, time: f32) -> Leg {
    let unit = ramps[tet] + ramps[n_tets + tet] + ramps[2u * n_tets + tet] + ramps[3u * n_tets + tet];
    let chart = place / (unit & place);
    let start_waves = local_waves(tet, place, time);
    // A step along the round step at the start, and the share of it where the first corner's weight runs out.
    let step_end = (frame(place) * (moving(round_step(heading, start_waves)) * (-0.5 * step)).exp()) >> origin;
    let step_segment = step_end / (unit & step_end) - chart;
    var share = 1.0;
    var exit = 4u;
    for (var corner = 0u; corner < 4u; corner++) {
        let change = ramps[corner * n_tets + tet] & step_segment;
        if (corner != entry && change < 0.0) {
            let run_out = max(-(ramps[corner * n_tets + tet] & chart) / change, 0.0);
            if (run_out < share) {
                share = run_out;
                exit = corner;
            }
        }
    }
    let length = acos(clamp(place.dot((chart + step_segment * share).normalized()), -1.0, 1.0));
    // The heading and the place half the leg on, and the waves there.
    let half_heading = (heading + heading_change(heading, start_waves) * (0.5 * length)).normalized();
    let half_place = (frame(place) * (moving(round_step(heading, start_waves)) * (-0.25 * length)).exp()) >> origin;
    let half_waves = local_waves(tet, half_place, time - 0.5 * length);
    // The leg rolled its length along the round step half way, from the frame at the start.
    let leg_end = (frame(place) * (moving(round_step(half_heading, half_waves)) * (-0.5 * length)).exp()) >> origin;
    var this_leg = Leg(unit, chart, leg_end / (unit & leg_end) - chart, length,
                       (heading + heading_change(half_heading, half_waves) * length).normalized(), tet, 4u);
    if (exit < 4u) {
        this_leg.tet = neighbours[tet * 4u + exit];
        this_leg.entry = entries[tet * 4u + exit];
    }
    return this_leg;
}
"""


# --- plumbing -------------------------------------------------------------------------
def jacobian(slopes: Callable[[Scalar], Amount], rest: Scalar, nudge: float, mesh: Topology) -> SparseExtensor:
    """The change of `slopes(lengths)`, `Amount-[edges]`, with each of the lengths about `rest`, by forward differences
    of relative size `nudge`, as a sparse map `(Amount- <- Scalar+)[edges <- edges]`. An edge's length reaches only the
    slopes of the edges that share a tetrahedron with it, so a group of edges no two of which reach a common edge is
    nudged at once, and each slope's change is told apart by the one edge of the group that reaches it: a numeric
    excursion, its columns gathered outside the algebra."""
    from scipy.sparse import csr_matrix

    n_edges = mesh.n_elements[1]
    # Each tetrahedron's six edges, found by their ends.
    cell_edges = lookup_simplices(mesh.elements[1], np.sort(mesh.elements[3][:, mesh.corners[3, 1].T], axis=-1))   # [tets, 6]
    cells = csr_matrix((np.ones(cell_edges.size), (np.repeat(np.arange(len(cell_edges)), 6), cell_edges.reshape(-1))),
                       shape=(len(cell_edges), n_edges))
    reaching = ((cells.T @ cells) > 0).astype(float).tocsr()                   # [edges, edges]
    clashing = ((reaching @ reaching) > 0).tocsr()                              # [edges, edges]
    # Each edge in turn takes the first group none of the edges it clashes with has taken.
    groups = np.full(n_edges, -1)
    for edge in range(n_edges):
        taken = groups[clashing.indices[clashing.indptr[edge]:clashing.indptr[edge + 1]]]
        groups[edge] = np.setdiff1d(np.arange(len(taken) + 1), taken)[0]
    # The slopes, amounts, read as numbers by their duals.
    at_rest = slopes(rest).dual().to_array()
    sizes = nudge * rest.to_array()
    rows, columns, values = [], [], []
    for group in range(groups.max() + 1):
        nudged = np.where(groups == group, sizes, 0.0)
        changes = (slopes(rest + mv.scalar(nudged[:, None]).field()).dual().to_array() - at_rest)
        reached, members = reaching[:, groups == group].nonzero()
        edges = np.flatnonzero(groups == group)[members]
        rows.append(reached), columns.append(edges), values.append(changes[reached] / sizes[edges])
    cells = ctx.extensor(ga.gatype((Amount, Scalar)), np.concatenate(values)[:, None, None])
    return SparseExtensor.from_indices(cells, np.concatenate(rows), np.concatenate(columns), (n_edges, n_edges))


def picking(mesh: Topology, vertices: Point) -> Callable[[Point], np.ndarray]:
    """`[...]`: the tetrahedron holding each point of the sphere `[...] Point-`, by a query of a k-d tree built once.
    The ray from the centre to a point crosses the tetrahedron whose plane through its corners, the sum of its ramps,
    reads the point the most. Lifted out by the root of how much each plane's squared size falls short of the
    largest, the planes lie nearest a point in the order they read it, so one query finds the tetrahedron. Planes are
    read by pairing with the unit steps along the four directions, points with the planes across them."""
    from scipy.spatial import cKDTree

    coordinate_planes = mv.vector(np.eye(4))                                    # [4] Plane+
    radial = coframe(mesh, vertices).sum(axis=0).batch()                        # [tets] Plane+
    planes = (radial[:, None] & coordinate_planes.dual()[None]).to_array()      # [tets, 4]
    sizes = (planes ** 2).sum(axis=-1)                                          # [tets]
    tree = cKDTree(np.concatenate([planes, np.sqrt(sizes.max() - sizes)[:, None]], axis=-1))
    def located(points: Point) -> np.ndarray:
        coordinates = (coordinate_planes & points.batch()[..., None]).to_array()   # [..., 4]
        return tree.query(np.pad(coordinates, [(0, 0)] * (coordinates.ndim - 1) + [(0, 1)]))[1]

    return located


def largest(value: SparseExtensor) -> float:
    """The largest of a sparse map's coefficients in size, a scale read outside the algebra."""
    return float(np.abs(np.asarray(value.cells.kernel)).max())

