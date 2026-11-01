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
springs along their bonds' proper separations.

The Hopf circles are the orbits of the turns that act on the frames' own side. Read in its frames, a circle's
direction is one and the same step of the origin all along it, a point of the sphere of such steps below the whole
circle, the base. Read along a circle, a wave's strain says how much the circle stretches; read across it, turned a
quarter about its base point, it is the strain of the base itself.
"""

from __future__ import annotations

from collections.abc import Iterator
from functools import reduce
from operator import and_

import numpy as np

from numga import Algebra, stack
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

# The algebra of the round three-sphere: four directions squaring to one, `w` the origin's own.
ga = Algebra("x+y+z+w+")
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv, ctx = mesh_module.Mesh, mesh_module.mv, mesh_module.ctx
Scalar = ga.gatype.scalar()                         # Scalar+
# The corners' ramps, gradients and the polar planes of directions: vectors, extrinsic; those through the origin.
Plane = ga.gatype.vector()                          # Vector+
Through = Plane.derive.within(" ".join(name for name in ga.basis_names if name != "w"))   # Vector+
# Points of the sphere and the origin's steps: antivectors.
Point = ga.gatype.antivector()                      # Antivector-
Step = Point.derive.containing("w")                 # Antivector-
# The rates that move and turn the frames, and the motors they generate.
Rate = ga.gatype.bivector()                         # Bivector+
Motor = ga.gatype.rotor()                           # Rotor+
Translating = ga.gatype((Step, Through))            # Antivector- <- Vector+
Metric = ga.gatype((Plane, Point))                  # Plane+ <- Point-
InverseMetric = ga.gatype((Point, Plane))           # Point- <- Plane+
# A strain, the change of the metric on the origin's steps, the plane that reads each step's stretch; a turning, the
# rate a step adds to a frame's turn; a rolling, the rate the frames roll by along each round step; and a stretching,
# the steps a map of them gives.
Strain = ga.gatype((Through, Step))                 # Vector+ <- Antivector-
Turning = ga.gatype((Rate, Step))                   # Bivector+ <- Antivector-
Rolling = ga.gatype((Rate, Step))                   # Bivector+ <- Antivector-
Stretching = ga.gatype((Step, Step))                # Antivector- <- Antivector-
Polarity = ga.gatype((Through, Step))               # Vector+ <- Antivector-

# The plane `w`, the origin's equator, and the origin, its pole.
equator = mv.w                                      # [] Plane+
origin = equator.dual()                             # [] Point-
# The step of the origin under the motion across a plane through it, a step by its value, cast so; its inverse, each
# step's polar plane, is the metric on the origin's steps as a correlation.
translating = origin.commutator(equator ^ Through).cast(Translating)   # [] Antivector- <- Vector+
polar = translating.inverse()                                           # [] Vector+ <- Antivector-


# --- math -----------------------------------------------------------------------------
def coframe(mesh: Mesh, vertices: Point) -> Plane:
    """Each corner's ramp across its simplex, `[corners] Plane+[simplices]`: the plane through the facet across from
    the corner, the join of the facet's corners, scaled to read one at the corner and so nothing on the facet."""
    corners = mesh.incidence[-1, 0] * vertices                                  # [corners] Point-[simplices]
    facets = reduce(and_, corners[mesh.corners[-1, -2]])                        # [corners] Antivector-[simplices]
    return facets / (facets & corners)


def round_rates(mesh: Mesh, frames: Motor) -> Rate:
    """The rate of each edge, `Rate+[edges]`, for a round sphere of unit radius rolled on itself, from frames at the
    vertices, `Motor+[vertices]`, motors that carry the origin to each vertex: the tail's inverse times the head's, so
    that around any loop they cancel and the connection is flat. Any frames will do, and the rates are smallest where
    neighbouring frames differ least; a sphere of two dimensions has no frames that are smooth everywhere, while the
    three-sphere has the isoclinic ones."""
    tails, heads = mesh.incidence[1, 0] * frames                                # Motor+[edges], Motor+[edges]
    return -2 * (tails.inverse() * heads).log()


class Space:
    """A triangulated three-sphere as its tetrahedra know it, for stepping its edges' squared lengths by Regge's
    action: each tetrahedron a piece of the round sphere for its angles, and flat for its masses. It holds their
    ramps and centred corner gradients, `[corners] Plane+[cells]`, their hyperplanes, and the maps from squared edge
    lengths to the metrics of the cone over each from the centre and of the flat tetrahedron."""

    def __init__(self, mesh: Mesh, vertices: Point) -> None:
        corners = mesh.incidence[3, 0] * vertices                               # [corners] Point-[cells]
        ramps = coframe(mesh, vertices)                                         # [corners] Plane+[cells]
        # The tetrahedron's hyperplane, one on each of its points; less a quarter of it, the ramps vanish on the
        # centroid and still read the edges.
        self.mesh, self.ramps, self.radial = mesh, ramps, ramps.sum(axis=0)     # Plane+[cells]
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
        self.edges = stack((corners[1] - corners[0], corners[2] - corners[0], corners[3] - corners[0]))   # [3] Point-[cells]
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
        off_edges = cone_pairings[tails[::-1], heads[::-1]]                     # [cell_edges] Scalar+[cells]
        sides = cone_pairings[tails[::-1], tails[::-1]] * cone_pairings[heads[::-1], heads[::-1]]   # [cell_edges] Scalar+[cells]
        angles = (-off_edges / sides.square_root()).arccos()                    # [cell_edges] Scalar+[cells]
        deficits = 2 * np.pi - (~mesh.incidence[3, 1] * angles).sum(axis=-1)    # Scalar+[edges]
        # By Schlafli a spherical tetrahedron's volume changes by half its arcs times its angles' changes, so four
        # times it cancels the angles' part of the action's change, and the slope is the deficits' alone, times the
        # arc's change with the squared chord: a half over the chord and the cosine of half the arc.
        slopes = -deficits / (squared_lengths * (1 - squared_lengths / 4)).square_root()   # Scalar+[edges]
        _, inverses = self.metrics(squared_lengths)
        pairings = self.gradients[:, None] & inverses(self.gradients[None, :])   # [corners, corners] Scalar+[cells]
        volumes = self.volumes(squared_lengths)                                 # Scalar+[cells]
        # Relativity's kinetic term on a tetrahedron, between the rates of two edges' squared lengths: a quarter of
        # the volume times the pairings of their corners across, both ways.
        local = volumes / 4 * (pairings[tails[:, None], tails[None, :]] * pairings[heads[:, None], heads[None, :]]
                               + pairings[tails[:, None], heads[None, :]] * pairings[heads[:, None], tails[None, :]])   # [cell_edges, cell_edges] Scalar+[cells]
        edges = mesh.incidence[3, 1]                                            # [cell_edges] Scalar+[cells <- edges]
        masses = (~edges[:, None] * spdiag(local) * edges[None, :]).sum(axis=0).sum(axis=0) * Scalar   # (Scalar+ <- Scalar+)[edges <- edges]
        return deficits, slopes, masses

    def volumes(self, squared_lengths: Scalar) -> Scalar:
        """Each flat tetrahedron's volume from the squared lengths: the root of the determinant of its metric over the
        corner gradients' frame, a sixth of it."""
        metrics, _ = self.metrics(squared_lengths)
        first, second, third = self.edges                                       # Point-[cells], Point-[cells], Point-[cells]
        return ((metrics(first) ^ metrics(second) ^ metrics(third) ^ self.radial) & self.joins).abs().square_root() / 6

    def metrics(self, squared_lengths: Scalar) -> tuple[Metric, InverseMetric]:
        """Each flat tetrahedron's metric from the squared lengths of the edges, the hyperplane taking the centroid,
        which the gradients leave out, so that it can be inverted; and its inverse."""
        metrics = self.radial * (self.radial & Point) + self.metric_map * squared_lengths   # (Plane+ <- Point-)[cells]
        return metrics, metrics.batch().inverse().field()


def gravitational_waves(mesh: Mesh, vertices: Point) -> tuple[SparseExtensor, SparseExtensor, OntoWaves]:
    """The linear gravitational waves of a triangulated three-sphere at rest, as the stiffness and the masses on its
    edges' squared lengths, sparse maps `(Scalar+ <- Scalar+)[edges <- edges]` coupling the edges that share a
    tetrahedron, and the projection onto the waves. The stiffness is that of Regge's potential with each tetrahedron
    read as a piece of the round sphere, at rest on any mesh of it. The waves are the lengths no move of the vertices
    makes: the tangent moves slide the vertices over the sphere, which the spherical tetrahedra still tile, and the
    radial ones change each edge's squared length by `s * (r_tail + r_head)`, the conformal stretches, whose breathing
    is the universe at rest's own instability. What is left are the transverse traceless strains."""
    space = Space(mesh, vertices)
    chords = mesh.d[0] * vertices                                               # Point-[edges]
    rest = chords.dot(chords)                                                   # Scalar+[edges]
    _, _, masses = space.dynamics(rest)
    stiffness = jacobian(lambda lengths: space.dynamics(lengths)[1], rest, 1e-6, mesh)
    moves = spdiag(2 * chords.dot(Point)) * mesh.d[0]                          # (Scalar+ <- Point-)[edges <- vertices]
    return stiffness, masses, OntoWaves(masses, moves)


class OntoWaves:
    """The projection of the changes of the squared edge lengths, `Scalar+[edges]`, onto the waves: the changes less
    the change the vertex moves make that is nearest them by the masses, the moves found by a solve over the
    vertices. The moves of the whole sphere change no length, and a tiny share of each move's own square added to
    their reach settles them at nothing."""

    def __init__(self, masses: SparseExtensor, moves: SparseExtensor) -> None:
        self.masses, self.moves = masses, moves
        reach = moves.adjoint()(masses(moves))                                  # (Point- <- Point-)[vertices <- vertices]
        self.reach = reach + settling * largest(reach) * Point

    def __call__(self, changes: Scalar) -> Scalar:
        shares = self.reach.solve(self.moves.adjoint()(self.masses(changes)))   # Point-[vertices]
        return changes - self.moves(shares)


def waves_stepped(stiffness: SparseExtensor, masses: SparseExtensor, onto_waves: OntoWaves, strains: Scalar,
                  rates: Scalar, step: float) -> Iterator[tuple[Scalar, Scalar]]:
    """The squared lengths' changes and their rates, `Scalar+[edges]` each, at every step of a leapfrog under the
    stiffness and the masses, kept to the waves: half a kick by the stiffness, the rates' change the masses give it
    and the waves keep, a drift, and the other half kick."""
    while True:
        yield strains, rates
        rates = rates - (step / 2) * onto_waves(masses.solve(stiffness(strains)))
        strains = strains + step * rates
        rates = rates - (step / 2) * onto_waves(masses.solve(stiffness(strains)))


def invariant_generators() -> tuple[Rate, Rate, Point]:
    """The round three-sphere's generators along its three invariant directions `x`, `y`, `z` at the origin, `[3]`
    each: the rates that move the origin along them, the turns square to them, each rate times the extrinsic
    pseudoscalar, and the unit steps themselves."""
    steps = stack((mv.x, mv.y, mv.z)).dual()                                    # [3] Antivector-
    moving = equator ^ polar(steps.cast(Step))                                 # [3] Rate+
    return moving, moving * mv.xyzw, steps


def isoclinic(places: Point) -> Motor:
    """The isoclinic frames at places on the round three-sphere, `Motor+`: the motors that carry the origin to each
    by turning alike in the plane of the great circle between them and in the plane square to it, the half of the
    quotient of the place by the origin that the extrinsic pseudoscalar keeps, with one in the other half. They are
    smooth everywhere, and in them a free path's heading keeps still along its great circle."""
    pseudoscalar = mv.xyzw                                                      # [] Antiscalar+
    return ((1 + pseudoscalar) * (places.normalized() / origin) + (1 - pseudoscalar)) / 2


def strain_field(mesh: Mesh, space: Space, ramps: Plane, frames: Motor, changes: Scalar) -> tuple[Strain, Turning]:
    """The strain of a change of the squared edge lengths, `Scalar+[edges]`, as a smooth field on the origin's steps:
    at each vertex the average of its tetrahedra's strains, `Strain[vertices]`, each the change of the metric of the
    cone over it from the centre, read on the steps its isoclinic frame at its middle, `Motor+[tets]`, carries out to
    it, and carried back; and in each tetrahedron the turn the strain's change from corner to corner adds, half the
    curl of the strain, `Turning[tets]`. What reads nothing on the steps, along the origin itself, is cast away."""
    carried = frames >> Step                                                    # (Point- <- Antivector-)[tets]
    strains = (frames.inverse() >> (space.cone_map * changes)(carried)).cast(Strain)   # (Vector+ <- Antivector-)[tets]
    shares = ~mesh.incidence[3, 0]                                              # [corners] Scalar+[vertices <- tets]
    counts = (shares * mesh_module.as_scalar(np.ones(len(mesh.tets))).field()).sum(axis=0)   # Scalar+[vertices]
    at_vertices = (shares * strains).sum(axis=0) / counts                       # (Vector+ <- Antivector-)[vertices]
    # Each corner's ramp read on the carried steps, and the strain at each corner: their wedge, summed over the
    # corners, is the strain's change across the tetrahedron turned into the plane it turns in.
    corner_planes = (frames.inverse() >> ramps).cast(Through)                   # [corners] Vector+[tets]
    corner_strains = mesh.incidence[3, 0] * at_vertices                         # [corners] (Vector+ <- Antivector-)[tets]
    return at_vertices, (corner_planes ^ corner_strains).sum(axis=0) / 2


def moving(steps: Point) -> Rate:
    """The rates that move the origin along steps of it, `Rate+`: the meet of the origin's equator with each step's
    polar plane."""
    return equator ^ polar(steps)


def turning(steps: Point) -> Rate:
    """The rates that turn the frame square to steps of the origin, `Rate+`: each moving rate times the extrinsic
    pseudoscalar, the plane square to it."""
    return moving(steps) * mv.xyzw


class Waves:
    """Stepped gravitational waves of a round three-sphere triangulated by `mesh` on `vertices`, as a smooth field over
    its tetrahedra and over time, read in the isoclinic frames: at each of the stored moments, from `first_time` every
    `level_step`, the strain of the squared edge lengths' changes `changes` at the vertices, its rate from
    `change_rates`, its trace, and each tetrahedron's turn from the strain's change across it. Read at a place and a
    time, a tetrahedron's corners blend by the place's barycentric weights and the two nearest moments by their
    nearness."""

    def __init__(self, mesh: Mesh, vertices: Point, changes: list, change_rates: list, first_time: float,
                 level_step: float) -> None:
        self.mesh, self.first_time, self.level_step = mesh, first_time, level_step
        self.ramps = coframe(mesh, vertices)                                    # [corners] Plane+[tets]
        frames = isoclinic(mesh.incidence[3, 0].mean(axis=0) * vertices)      # Motor+[tets]
        space = Space(mesh, vertices)
        fields = [strain_field(mesh, space, self.ramps, frames, change) for change in changes]
        self.strains = stack([strain.batch() for strain, _ in fields])           # [levels, vertices] Vector+ <- Antivector-
        self.turns = stack([turn.batch() for _, turn in fields])                 # [levels, tets] Bivector+ <- Antivector-
        self.strain_rates = stack([strain_field(mesh, space, self.ramps, frames, rate)[0].batch() for rate in change_rates])
        self.traces = translating(self.strains).trace()                          # [levels, vertices] Scalar+

    def fields(self, faces: np.ndarray, places: Point, time: float) -> tuple[Strain, Strain, Scalar, Turning]:
        """The waves' fields at the places `[places] Point-` on the sphere, in the tetrahedra `faces`, at a time: the
        strain, its rate and its trace, blended over the corners and the two nearest moments, and the tetrahedra's
        turn, each `[places]`."""
        level = np.clip((time - self.first_time) / self.level_step, 0, len(self.traces) - 1.000001)
        early, share = int(level), level - int(level)
        corners = self.mesh.tets[faces].T                                       # [corners, places]
        ramps = self.ramps.batch()[:, faces]                                    # [corners, places] Plane+
        weights = (ramps & places[None]) / (ramps.sum(axis=0) & places)         # [corners, places] Scalar+

        def blended(values):
            """A value at the vertices, `[levels, vertices]`, at the places and the time."""
            between = values[early] * (1 - share) + values[early + 1] * share   # [vertices]
            return (weights * between[corners]).sum(axis=0)                     # [places]

        turn = self.turns[early, faces] * (1 - share) + self.turns[early + 1, faces] * share   # [places] Bivector+ <- Antivector-
        return blended(self.strains), blended(self.strain_rates), blended(self.traces), turn

    def at(self, faces: np.ndarray, places: Point, time: float) -> tuple[Rolling, Stretching]:
        """The connection at the places `[places] Point-` on the sphere, in the tetrahedra `faces`, at a time: the rate
        the frames roll by along each round step there, `[places] Rate+ <- Antivector-`; and the squeeze of a step
        there, half the strain's rate read back as a step, `[places] Antivector- <- Antivector-`."""
        strain, strain_rate, trace, turn = self.fields(faces, places, time)
        # A round step, stretched by the strain, moves the origin; turned less twice the strain plus half its trace, it
        # turns the frame square to it; and the strain's change adds its turn.
        stretch = translating(strain(Step))                                     # [places] Antivector- <- Antivector-
        local = moving(Step + stretch / 2) + turning(Step - 2 * stretch + Step * (trace / 2)) + turn(Step)
        return local, translating(strain_rate) / 2


def bond_steps(bonds: Mesh, positions: Point) -> Point:
    """Each bond's round step, `Antivector-[bonds]`, a step of the origin: its head carried back by its tail's
    isoclinic frame, less the origin it is beside."""
    tails, heads = bonds.incidence[1, 0] * positions                            # Point-[bonds], Point-[bonds]
    return (isoclinic(tails).inverse() >> heads).cast(Step)


def proper_separations(local: Rolling, bonds: Mesh, positions: Point) -> Point:
    """Each bond's proper separation, `Antivector-[bonds]`, a step of the origin in the frames: the step the connection
    at the bond's tail, `(Rate+ <- Antivector-)[particles]`, moves the origin by along the bond's round step."""
    tails = bonds.incidence[1, 0][0]                                            # Scalar+[bonds <- particles]
    return origin.commutator((tails * local)(bond_steps(bonds, positions))).cast(Step)


def massive(mesh: Mesh, vertices: Point, waves: Waves, bonds: Mesh, rest: Scalar, stiffness: Scalar, damping: Scalar,
            holding: float, starts: Point, velocities: Point, start: float, step: float, times: np.ndarray
            ) -> Iterator[tuple[Point, Scalar]]:
    """Massive particles through a round three-sphere carrying the gravitational waves `waves`, held by springs along
    `bonds`, a graph `Mesh` over the particles, of rest proper lengths `rest`, stiffness `stiffness` and damping
    `damping` over unit masses, `Scalar+[bonds]` each, and each held to its start by a spring of stiffness `holding`,
    damped critically: from `starts` `Point-[particles]` on the sphere with velocities `velocities`, steps of the
    origin `Antivector-[particles]` in the isoclinic frames, at time `start`, stepped by `step` of time with symplectic
    Euler, their positions and the bonds' proper strains `Scalar+[bonds]` at each of the `times`. A particle's velocity
    turns with the connection along its motion, the slices' stretch squeezes it, and its bonds pull it along their
    proper separations; it rolls along its round step on its great circle."""
    pick = mesh.picking(vertices)
    homes = starts.batch().normalized()                                         # [particles] Point-
    positions, velocities, lengths, time = homes, velocities.batch().cast(Step), rest, start
    for target in times:
        while time < target - step / 2:
            faces = pick(positions)                                             # [particles]
            local, squeeze = waves.at(faces, positions, time)                   # [particles] Rate+ <- Antivector-, Antivector- <- Antivector-
            proper = proper_separations(local.field(), bonds, positions.field())   # Antivector-[bonds]
            previous, lengths = lengths, proper.dot(proper).square_root()       # Scalar+[bonds], Scalar+[bonds]
            # Springs and dashpots along the proper separations, pulling a stretched bond's ends together.
            pulls = (stiffness * (lengths - rest) + damping * (lengths - previous) / step) / lengths   # Scalar+[bonds]
            forces = -(~bonds.d[0] * (pulls * proper)).batch()                  # [particles] Antivector-
            # Each particle pulled toward its home, a free particle at rest where it started, and its motion damped.
            homeward = (isoclinic(positions).inverse() >> homes).cast(Step)    # [particles] Antivector-
            forces = forces + holding * homeward - 2 * np.sqrt(holding) * velocities
            # The step each round step moves the origin by: a velocity's round step is the one it moves the origin by.
            stepping = origin.commutator(local(Step)).cast(Stretching)          # [particles] Antivector- <- Antivector-
            motion = stepping.solve(velocities)                                 # [particles] Antivector-
            # The velocity turned by the connection along the motion, squeezed by the slices' stretch and pushed by the
            # springs; the turn's step part only, cast so.
            turned = velocities - velocities.commutator(local(motion)).cast(Step) * step
            velocities = (turned - squeeze(turned) * step + forces * step).cast(Step)   # [particles] Antivector-
            # Rolled along the round step of its velocity, on its great circle from the frame at its place.
            rolled = isoclinic(positions) * (moving(stepping.solve(velocities)) * (-step / 2)).exp()   # [particles] Motor+
            positions = rolled >> origin                                        # [particles] Point-
            time += step
        yield positions.field(), lengths / rest - 1


def light(mesh: Mesh, vertices: Point, waves: Waves, starts: Point, headings: Point, start: float, step: float,
          n_steps: int) -> Iterator[Point]:
    """Rays of light through a round three-sphere carrying the waves `waves`, from `starts` `[rays] Point-` on the
    sphere along `headings`, unit steps of the origin `[rays] Antivector-` in the isoclinic frames, set out at time
    `start`: their places `[rays] Point-` after each step of length `step`, the light's time running with it. As the
    GPU's rays do, a ray's heading turns with the connection along its round step and the slices' stretch squeezes
    it, and it rolls along that step on its great circle."""
    pick = mesh.picking(vertices)
    positions, headings, time = starts.normalized(), headings.cast(Step), start
    for _ in range(n_steps):
        local, squeeze = waves.at(pick(positions), positions, time)            # [rays] Rate+ <- Antivector-, Antivector- <- Antivector-
        stepping = origin.commutator(local(Step)).cast(Stretching)              # [rays] Antivector- <- Antivector-
        turned = headings - headings.commutator(local(stepping.solve(headings))).cast(Step) * step
        headings = (turned - squeeze(turned) * step).cast(Step).normalized()   # [rays] Antivector-
        rolled = isoclinic(positions) * (moving(stepping.solve(headings)) * (-step / 2)).exp()   # [rays] Motor+
        positions = rolled >> origin                                            # [rays] Point-
        time += step
        yield positions


def base_points(places: Point, circling: Rate) -> Point:
    """The base point below each place, `Point-`: the direction its circle runs in under the rate `circling`, read in
    the isoclinic frame at the place, a unit step of the origin, the same all along the circle."""
    along = places.commutator(circling)                                        # Point-
    return (isoclinic(places).inverse() >> along).cast(Step).normalized()


def lifts(bases: Point, circling: Rate) -> Point:
    """A point on the circle above each base point, `Point-`: the origin moved square to its own base point and the
    base point, by half the angle between them, which the Hopf map doubles."""
    below_origin = base_points(origin, circling)                               # [] Point-
    # The plane the two base points turn in, times the pseudoscalar: the rate that moves the origin square to both.
    lifting = ((polar(bases) ^ polar(below_origin)) * mv.xyzw).normalized()    # Rate+
    angles = (polar(bases) & below_origin).arccos()                            # Scalar+
    return (lifting * (-angles / 4)).exp() >> origin


def round_circles(bases: Point, turn: Motor, n_turns: int, n_samples: int) -> Point:
    """So many points evenly round the circle above each base point, `[samples, bases] Point-`: the lifts carried by
    powers of the turn along the circles, `n_turns` of which close them."""
    circling = turn.log()                                                       # [] Rate+
    shares = np.arange(n_samples) * n_turns / n_samples                         # [samples]
    return (circling * shares[:, None]).exp() >> lifts(bases, circling)[None]


def stretches(waves: Waves, mesh: Mesh, vertices: Point, bases: Point, directions: Point, turn: Motor, n_turns: int,
              n_samples: int, time: float) -> Scalar:
    """How much lengths stretch along steps of the origin `directions` in the frames, at the circles above the base
    points, averaged round each, at a time, `Scalar+[bases]`: half the strain's reading of the step on itself."""
    places = round_circles(bases, turn, n_turns, n_samples)                    # [samples, bases] Point-
    faces = mesh.picking(vertices)(places.reshape(-1))                         # [samples * bases]
    strain, _, _, _ = waves.fields(faces, places.reshape(-1), time)            # [samples * bases] Vector+ <- Antivector-
    along = directions[None].broadcast_to(places.shape).reshape(-1)            # [samples * bases] Antivector-
    readings = (strain(along) & along) / (2 * along.dot(along))                 # [samples * bases] Scalar+
    return readings.reshape(places.shape).sum(axis=0) / n_samples


def across(bases: Point, directions: Point) -> Point:
    """Steps along the base, `directions` at the base points, turned a quarter about them, `Point-`: the steps across
    the circles that a base direction is the image of."""
    return turning(bases).commutator(directions).cast(Step)


# A ray's step on the GPU, in numga's dialect, through a round three-sphere meshed by tetrahedra and carrying stepped
# gravitational waves, by the lines of `Waves.at`: each tetrahedron's corner ramps, its neighbours across each facet
# and the facet each enters by, and its corners' vertices; at each stored moment the strains at the vertices, their
# rates and traces, and the tetrahedra's turns; the origin and the origin's equator. The round sphere needs no mesh:
# its isoclinic frames are smooth everywhere, and in them a ray's heading keeps still along its great circle, so only
# the waves turn it.
geodesic_step_source = """
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
var<uniform> n_levels: u32;
var<uniform> first_time: f32;
var<uniform> level_step: f32;
var<uniform> origin: Point;
var<uniform> equator: Plane;
var<uniform> polar: Polarity;
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

// The connection along a round step of the origin: stretched by the strain, the step moves the origin; turned less
// twice the strain plus half its trace, it turns the frame square to it; and the strain's change adds its turn.
fn connection(steps: Step, waves: Local) -> Rate {
    let stretch = translating(waves.strain(steps));
    let moved = steps + stretch * 0.5;
    let turned = steps - stretch * 2.0 + steps * (waves.trace * 0.5);
    return (equator ^ polar(moved)) + (equator ^ polar(turned)) * mv.xyzw + waves.turn(steps);
}

// The plane on which a tetrahedron's barycentric weights add up to one.
fn unit_plane(tet: u32) -> Plane {
    let base = tet * 4u;
    return ramps[base] + ramps[base + 1u] + ramps[base + 2u] + ramps[base + 3u];
}

// The isoclinic frame at a place on the round sphere: the motor that carries the origin to it, turning alike in the
// plane of the great circle between them and in the plane square to it, the half of the quotient of the place by the
// origin that the pseudoscalar keeps, with one in the other half.
fn frame(place: Point) -> Motor {
    return ((1.0 + mv.xyzw) * (place / origin) + (1.0 - mv.xyzw)) * 0.5;
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
def jacobian(slopes, rest: Scalar, nudge: float, mesh: Mesh) -> Scalar:
    """The change of `slopes(lengths)`, `Scalar+[edges]`, with each of the lengths about `rest`, by forward differences
    of relative size `nudge`, as a sparse map `(Scalar+ <- Scalar+)[edges <- edges]`. An edge's length reaches only the
    slopes of the edges that share a tetrahedron with it, so a group of edges no two of which reach a common edge is
    nudged at once, and each slope's change is told apart by the one edge of the group that reaches it: a numeric
    excursion, its columns gathered outside the algebra."""
    from scipy.sparse import csr_matrix

    n_edges = mesh.n_elements[1]
    edge_ids = mesh_module.as_scalar(np.arange(n_edges)).field()
    cell_edges = np.rint((mesh.incidence[3, 1] * edge_ids).to_array()).astype(int).reshape(6, -1).T   # [tets, 6]
    cells = csr_matrix((np.ones(cell_edges.size), (np.repeat(np.arange(len(cell_edges)), 6), cell_edges.reshape(-1))),
                       shape=(len(cell_edges), n_edges))
    reaching = ((cells.T @ cells) > 0).astype(float).tocsr()                   # [edges, edges]
    clashing = ((reaching @ reaching) > 0).tocsr()                              # [edges, edges]
    # Each edge in turn takes the first group none of the edges it clashes with has taken.
    groups = np.full(n_edges, -1)
    for edge in range(n_edges):
        taken = groups[clashing.indices[clashing.indptr[edge]:clashing.indptr[edge + 1]]]
        groups[edge] = np.setdiff1d(np.arange(len(taken) + 1), taken)[0]
    at_rest = slopes(rest).to_array()
    sizes = nudge * rest.to_array()
    rows, columns, values = [], [], []
    for group in range(groups.max() + 1):
        nudged = np.where(groups == group, sizes, 0.0)
        changes = (slopes(rest + mesh_module.as_scalar(nudged).field()).to_array() - at_rest)
        reached, members = reaching[:, groups == group].nonzero()
        edges = np.flatnonzero(groups == group)[members]
        rows.append(reached), columns.append(edges), values.append(changes[reached] / sizes[edges])
    cells = ctx.extensor(ga.gatype((Scalar, Scalar)), np.concatenate(values)[:, None, None])
    return SparseExtensor.from_indices(cells, np.concatenate(rows), np.concatenate(columns), (n_edges, n_edges))


def largest(value: SparseExtensor) -> float:
    """The largest of a sparse map's coefficients in size, a scale read outside the algebra."""
    return float(np.abs(np.asarray(value.cells.kernel)).max())


# The share of the vertex moves' largest square that settles the moves of the whole sphere.
settling = 1e-10
