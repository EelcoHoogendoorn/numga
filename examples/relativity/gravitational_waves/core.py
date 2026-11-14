"""Gravitational waves of a closed universe, a round three-sphere triangulated by tetrahedra, evolve on the squared
edge lengths, interpolate to a smooth strain field across the tetrahedra, carry light and massive particles along the
perturbed connection, and project onto the two-sphere below the Hopf circles.

Regge calculus represents a curved three-space by the edge lengths of a tetrahedral triangulation. Reading each
tetrahedron as a piece of the round three-sphere determines its dihedral angles from its squared chord lengths, and the
deficit by which the angles around an edge fall short of a full turn measures the curvature concentrated on that edge;
the round sphere's own chords leave zero deficit on every triangulation. Linear gravitational waves perturb those
squared lengths: the deficits restore them against relativity's kinetic inertia, while projecting out vertex motions
leaves the physical transverse-traceless waves. The slowest mode squashes the sphere along one invariant direction and
stretches it along an orthogonal direction.

Four positive-square basis planes represent the ambient four-space, with points on the unit three-sphere as antivectors
and the origin as the pole of the equatorial plane. Each place carries a smooth frame motor that carries the origin
there by turning equally in two orthogonal planes, identifying tangent steps everywhere with steps at the origin. A
strain maps each step at the origin to the plane that reads its stretch and blends across each tetrahedron by the
corner weights read by its simplex gradients. Light rays and spring-bound particles roll their frames along great
circles while the wave connection turns and squeezes their headings and velocities.

Hopf circles arise as the orbits of turns acting on the same side as the frames. Read in the frames, a circle's tangent
direction is a single constant step of the origin all along the circle, defining its point on the base two-sphere; the
strain along the circle measures how the fibre stretches, while the strain across the circle measures the strain on the
base.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

from examples.relativity.gravitational_waves import numerical

# Ambient four-space of the unit three-sphere, with `w` as the equatorial plane of the origin.
ga = Algebra("x+y+z+w+")
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                         # Scalar+
# Volume-integrated quantities, including the Regge action and momenta conjugate to squared edge lengths.
Amount = ga.gatype.antiscalar()                                     # Antiscalar-
# Simplex gradients across tetrahedra and polar planes through the origin.
Plane = ga.gatype.vector()                                          # Vector+
Through = Plane.derive.within(Algebra("x+y+z+"))                    # Vector+
# Points of the three-sphere and tangent steps at the origin.
Point = ga.gatype.antivector()                                      # Antivector-
Step = Through.derive.dual                                          # Antivector-
# Bivector rates that move and turn frames, and the motors they generate.
Rate = ga.gatype.bivector()                                         # Bivector+
Motor = ga.gatype.rotor()                                           # Rotor+
Translating = ga.gatype((Step, Through))                            # Point- <- Plane+
Metric = ga.gatype((Plane, Point))                                  # Plane+ <- Point-
InverseMetric = ga.gatype((Point, Plane))                           # Point- <- Plane+
# Strain, frame-turning, frame-rolling, and step-stretching maps on steps at the origin.
Strain = ga.gatype((Through, Step))                                 # Plane+ <- Point-
Turning = ga.gatype((Rate, Step))                                   # Rate+ <- Point-
Rolling = ga.gatype((Rate, Step))                                   # Rate+ <- Point-
Stretching = ga.gatype((Step, Step))                                # Point- <- Point-
# Rates in planes through the origin that translate the origin along its steps.
Moving = ga.gatype((Rate.derive.touching(Algebra("w+")), Step))     # Rate+ <- Point-

# Equatorial plane `w` and the origin, its pole on the three-sphere.
equator = mv.w                                                      # [] Plane+
origin = equator.dual()                                             # [] Point-
# Step of the origin across a plane through it; its inverse sends each step to its polar plane as a duality map.
translating = origin.commutator(equator ^ Through).cast(Translating)   # [] Point- <- Plane+
polar = translating.inverse()                                           # [] Plane+ <- Point-
# Rate that translates the origin along a step, formed by meeting the equator with the step's polar plane.
moving = equator ^ polar(Step)                                          # [] Rate+ <- Point-
# Extrinsic pseudoscalar mapping each rate to the rate in the orthogonal plane, splitting left- and right-isoclinic turns.
pseudoscalar = mv.xyzw                                                  # [] Pseudoscalar+
# Rate that turns a frame in the plane orthogonal to each step's translating rate.
turning = moving * pseudoscalar                                         # [] Rate+ <- Point-
# Regularization share settling rigid motions of the whole sphere during gauge projection.
settling = 1e-10


# --- math -----------------------------------------------------------------------------
def simplex_gradients(topology: Topology, vertices: Point) -> Plane:
    """Simplex gradients `[4] Plane+[T]` pair to one at each tetrahedron corner and vanish on the opposite facet."""
    corners = topology.incidence[3, 0] * vertices                               # [4] Point-[T]
    facing = topology.off[3, 0] * vertices                                      # [3, 4] Point-[T]
    facets = facing[0] & facing[1] & facing[2]                                  # [4] Plane-[T]
    return facets / (facets & corners)


class Space:
    """Tetrahedral geometry of a triangulated three-sphere maps squared edge lengths to spherical cone metrics for
    Regge dihedral angles and flat simplex metrics for kinetic masses."""

    def __init__(self, topology: Topology, vertices: Point) -> None:
        corners = topology.incidence[3, 0] * vertices                           # [4] Point-[T]
        gradients = simplex_gradients(topology, vertices)                       # [4] Plane+[T]
        # Subtracting a quarter of the chart hyperplane centres the simplex gradients on the centroid while preserving
        # their differences along every edge.
        self.topology, self.gradients = topology, gradients
        self.radial = gradients.sum(axis=0)                                     # Plane+[T]
        self.centred_gradients = gradients - self.radial / 4                    # [4] Plane+[T]
        # Along its own edge, two centred gradients pair to -1 and +1 while the other two vanish, isolating that edge's
        # squared length.
        tails, heads = topology.corners[3, 1]
        first, second = self.centred_gradients[tails], self.centred_gradients[heads]   # [6] Plane+[T], [6] Plane+[T]
        basis = -(first * (second & Point) + second * (first & Point)) / 2      # [6] (Plane+ <- Point-)[T]
        self.metric_map = (spdiag(basis) * topology.incidence[3, 1]).sum(axis=0)   # (Plane+ <- Point-)[T <- E]
        # Symmetric products of the uncentred simplex gradients build the cone metric from the sphere's centre.
        tail_gradients, head_gradients = gradients[tails], gradients[heads]     # [6] Plane+[T], [6] Plane+[T]
        cone_basis = -(tail_gradients * (head_gradients & Point) + head_gradients * (tail_gradients & Point)) / 2   # [6] (Plane+ <- Point-)[T]
        self.cone_map = (spdiag(cone_basis) * topology.incidence[3, 1]).sum(axis=0)   # (Plane+ <- Point-)[T <- E]
        self.spans = corners[1:] - corners[0]                                   # [3] Point-[T]
        self.joins = corners[0] & corners[1] & corners[2] & corners[3]          # Amount-[T]

    def dynamics(self, squared_lengths: Scalar) -> tuple[Scalar, Amount, SparseExtensor]:
        """Spherical dihedral angles determine the edge deficit angles `Scalar+[E]` and Regge action slopes
        `Amount-[E]`, while the flat tetrahedron volumes and inverse metrics assemble the kinetic mass map
        `(Amount- <- Scalar+)[E <- E]`."""
        topology = self.topology
        tails, heads = topology.corners[3, 1]
        # Through the inverse cone metric, the simplex gradients off each edge pair to the cosine of the dihedral angle
        # between the two faces meeting at that edge.
        cone = self.radial * (self.radial & Point) + self.cone_map * squared_lengths   # (Plane+ <- Point-)[T]
        cone_pairings = self.gradients[:, None] & cone.batch().inverse().field()(self.gradients[None, :])   # [4, 4] Scalar+[T]
        off_tails, off_heads = topology.complements[3, 1]
        off_edges = cone_pairings[off_tails, off_heads]                         # [6] Scalar+[T]
        sides = cone_pairings[off_tails, off_tails] * cone_pairings[off_heads, off_heads]   # [6] Scalar+[T]
        angles = (-off_edges / sides.square_root()).arccos()                    # [6] Scalar+[T]
        deficits = 2 * np.pi - (topology.incidence[3, 1].adjugate() * angles).sum(axis=-1)   # Scalar+[E]
        # Schläfli's identity cancels the dihedral-angle derivatives against the cosmological volume term, leaving each
        # deficit weighted by the derivative of arc length with respect to squared chord length.
        slopes = (-deficits / (squared_lengths * (1 - squared_lengths / 4)).square_root()).dual()   # Amount-[E]
        _, inverses = self.metrics(squared_lengths)
        pairings = self.centred_gradients[:, None] & inverses(self.centred_gradients[None, :])   # [4, 4] Scalar+[T]
        volumes = self.volumes(squared_lengths)                                 # Scalar+[T]
        # Relativity's kinetic supermetric couples edge-length rates across each tetrahedron's inverse metric.
        tet_masses = volumes / 4 * (pairings[tails[:, None], tails[None, :]] * pairings[heads[:, None], heads[None, :]]
                                    + pairings[tails[:, None], heads[None, :]] * pairings[heads[:, None], tails[None, :]])   # [6, 6] Scalar+[T]
        edges = topology.incidence[3, 1]                                        # [6] Scalar+[T <- E]
        masses = (edges[:, None].adjugate() * spdiag(tet_masses * Scalar.dual()) * edges[None, :]).sum(axis=0).sum(axis=0)   # (Amount- <- Scalar+)[E <- E]
        return deficits, slopes, masses

    def stiffness(self, rest: Scalar) -> SparseExtensor:
        """Variations of the spherical dihedral angles through the inverse cone metric at zero-deficit `rest` chords
        assemble the linearized Regge stiffness `(Amount- <- Scalar+)[E <- E]`."""
        topology = self.topology
        tails, heads = topology.corners[3, 1]
        off_tails, off_heads = topology.complements[3, 1]
        cone = self.radial * (self.radial & Point) + self.cone_map * rest       # (Plane+ <- Point-)[T]
        pairings = self.gradients[:, None] & cone.batch().inverse().field()(self.gradients[None, :])   # [4, 4] Scalar+[T]
        # Corners opposite each dihedral edge run down the rows; endpoints of each varied edge run across the columns.
        first_off, second_off = off_tails[:, None], off_heads[:, None]
        tail_ends, head_ends = tails[None, :], heads[None, :]
        # Variations of the cross-face pairing and of each face's self-pairing through the inverse cone metric.
        cross_pairings = (pairings[first_off, tail_ends] * pairings[second_off, head_ends]
                          + pairings[first_off, head_ends] * pairings[second_off, tail_ends]) / 2   # [6, 6] Scalar+[T]
        self_pairings = (pairings[first_off, tail_ends] * pairings[first_off, head_ends] / pairings[first_off, first_off]
                         + pairings[second_off, tail_ends] * pairings[second_off, head_ends] / pairings[second_off, second_off])
        edge_pairings = pairings[first_off, second_off]                         # [6, 1] Scalar+[T]
        sines = (pairings[first_off, first_off] * pairings[second_off, second_off] - edge_pairings.squared()).square_root()   # [6, 1] Scalar+[T]
        angle_changes = (cross_pairings - edge_pairings * self_pairings / 2) / sines   # [6, 6] Scalar+[T]
        edges = topology.incidence[3, 1]                                        # [6] Scalar+[T <- E]
        roots = edges * (rest * (1 - rest / 4)).square_root()                   # [6] Scalar+[T]
        tet_stiffness = angle_changes / roots[:, None]                          # [6, 6] Scalar+[T]
        return (edges[:, None].adjugate() * spdiag(tet_stiffness * Scalar.dual()) * edges[None, :]).sum(axis=0).sum(axis=0)

    def volumes(self, squared_lengths: Scalar) -> Scalar:
        """Each flat tetrahedron's volume `Scalar+[T]` equals one sixth the root of its metric determinant on its edge spans."""
        metrics, _ = self.metrics(squared_lengths)
        first, second, third = self.spans                                       # Point-[T], Point-[T], Point-[T]
        return ((metrics(first) ^ metrics(second) ^ metrics(third) ^ self.radial) & self.joins).abs().square_root() / 6

    def metrics(self, squared_lengths: Scalar) -> tuple[Metric, InverseMetric]:
        """Each flat tetrahedron's metric `(Plane+ <- Point-)[T]` and inverse `(Point- <- Plane+)[T]` combine the chart
        hyperplane on the centroid with the edge-length metric on the tangent space."""
        metrics = self.radial * (self.radial & Point) + self.metric_map * squared_lengths   # (Plane+ <- Point-)[T]
        return metrics, metrics.batch().inverse().field()


def gravitational_waves(topology: Topology, vertices: Point) -> tuple[SparseExtensor, SparseExtensor, WaveProjection]:
    """Linearized Regge stiffness and kinetic mass maps `(Amount- <- Scalar+)[E <- E]` on the rest chord lengths of a
    triangulated three-sphere pair with the mass-orthogonal `WaveProjection` removing tangent and radial vertex motions."""
    space = Space(topology, vertices)
    chords = topology.d[0] * vertices                                           # Point-[E]
    rest = chords.scalar_norm_squared()                                         # Scalar+[E]
    _, _, masses = space.dynamics(rest)
    stiffness = space.stiffness(rest)
    moves = spdiag(2 * chords.dot(Point)) * topology.d[0]                       # (Scalar+ <- Point-)[E <- V]
    return stiffness, masses, WaveProjection(masses, moves)


class WaveProjection:
    """Mass-orthogonal projection subtracts from squared-length perturbations `Scalar+[E]` the nearest perturbation
    generated by four-space vertex motions."""

    def __init__(self, masses: SparseExtensor, moves: SparseExtensor) -> None:
        self.masses, self.moves = masses, moves
        reach = moves.adjugate()(masses(moves))                                 # (Plane+ <- Point-)[V <- V]
        self.reach = reach + settling * numerical.largest(reach) * Point.dual()

    def __call__(self, changes: Scalar) -> Scalar:
        shares = self.reach.solve(self.moves.adjugate()(self.masses(changes)))  # Point-[V]
        return changes - self.moves(shares)


def waves_stepped(stiffness: SparseExtensor, masses: SparseExtensor, wave_projection: WaveProjection, strains: Scalar,
                  rates: Scalar, step: float) -> Iterator[tuple[Scalar, Scalar]]:
    """Yields squared edge-length perturbations and synchronized rates `(Scalar+[E], Scalar+[E])` across projected
    leapfrog steps."""
    restoring = wave_projection(masses.solve(stiffness(strains)))               # Scalar+[E]
    prior_rates = rates + (step / 2) * restoring                                # Scalar+[E]
    while True:
        next_rates = prior_rates - step * restoring                             # Scalar+[E]
        yield strains, (prior_rates + next_rates) / 2
        strains, prior_rates = strains + step * next_rates, next_rates
        restoring = wave_projection(masses.solve(stiffness(strains)))


def frame(places: Point) -> Motor:
    """Smooth left-isoclinic frame motors `Motor+` carry the origin to `places` by turning equally in the great-circle
    plane and its orthogonal complement."""
    return ((1 + pseudoscalar) * (places.normalized() / origin) + (1 - pseudoscalar)) / 2


def round_steps(graph: Topology, places: Point) -> Point:
    """Pulling each edge's head back into its tail's frame gives the edge's tangent step `Point-[E]` at the origin."""
    tails, heads = graph.incidence[1, 0] * places                               # Point-[E], Point-[E]
    return (frame(tails).inverse() >> heads).cast(Step)


def strain_field(topology: Topology, space: Space, gradients: Plane, frames: Motor,
                 changes: Scalar) -> tuple[Strain, Turning]:
    """Pulling each tetrahedron's cone-metric perturbation into its midpoint frame and averaging to vertices gives the
    vertex strain `(Plane+ <- Point-)[V]` and the tetrahedral curl turn `(Rate+ <- Point-)[T]`."""
    V, E, F, T = topology.n_elements
    frame_steps = frames >> Step                                                # (Point- <- Point-)[T]
    # Projecting onto `Strain` discards radial components orthogonal to tangent steps at the origin.
    strains = (frames.inverse() >> (space.cone_map * changes)(frame_steps)).cast(Strain)   # (Plane+ <- Point-)[T]
    shares = topology.incidence[3, 0].adjugate()                                # [4] Scalar+[V <- T]
    counts = (shares * mv.scalar(np.ones((T, 1))).field()).sum(axis=0)          # Scalar+[V]
    at_vertices = (shares * strains).sum(axis=0) / counts                       # (Plane+ <- Point-)[V]
    # Wedging the pulled-back simplex gradients with the corner strains forms half the spatial curl of the strain.
    corner_planes = (frames.inverse() >> gradients).cast(Through)               # [4] Plane+[T]
    corner_strains = topology.incidence[3, 0] * at_vertices                     # [4] (Plane+ <- Point-)[T]
    return at_vertices, (corner_planes ^ corner_strains).sum(axis=0) / 2


class Waves:
    """Stored wave history interpolates vertex strains, strain rates, traces, and tetrahedral curl turns across space
    and time in the left-isoclinic frames."""

    def __init__(self, topology: Topology, vertices: Point, changes: list, change_rates: list, first_time: float,
                 level_step: float) -> None:
        V, E, F, T = topology.n_elements
        n_levels = len(changes)
        self.topology, self.first_time, self.level_step = topology, first_time, level_step
        self.gradients = simplex_gradients(topology, vertices)                  # [4] Plane+[T]
        frames = frame(topology.incidence[3, 0].mean(axis=0) * vertices)        # Motor+[T]
        space = Space(topology, vertices)
        fields = [strain_field(topology, space, self.gradients, frames, change) for change in changes]
        self.strains = stack([strain.batch() for strain, _ in fields])          # [n_levels, V] Plane+ <- Point-
        self.turns = stack([turn.batch() for _, turn in fields])                # [n_levels, T] Rate+ <- Point-
        self.strain_rates = stack([strain_field(topology, space, self.gradients, frames, rate)[0].batch()
                                   for rate in change_rates])                   # [n_levels, V] Plane+ <- Point-
        self.traces = translating(self.strains).trace()                         # [n_levels, V] Scalar+

    def fields(self, tetrahedra: np.ndarray, places: Point, time: float) -> tuple[Strain, Strain, Scalar, Turning]:
        """Simplex-gradient corner weights and linear time weights interpolate the strain, strain rate, trace, and
        tetrahedral turn at `places` `[...] Point-[V_sites]` inside `tetrahedra`."""
        level = (time - self.first_time) / self.level_step
        early_index, share = int(level), level - int(level)
        # Pairing each place with its tetrahedron's simplex gradients reads its normalized corner weights.
        tet_reading = self.topology.reading(tetrahedra, 3)[..., None]           # [..., 1] Scalar+[V_sites <- T]
        place_gradients = tet_reading * self.gradients                          # [..., 4] Plane+[V_sites]
        weights = (place_gradients & places[..., None]) / (place_gradients.sum(axis=-1) & places)[..., None]   # [..., 4] Scalar+[V_sites]

        def blended_field(values):
            """Linear time and corner interpolation evaluates vertex tables `[n_levels, V]` at `places`."""
            time_slice = (values[early_index] * (1 - share) + values[early_index + 1] * share).field()   # [V]
            return (weights * (tet_reading * (self.topology.incidence[3, 0] * time_slice))).sum(axis=-1)

        turns = (self.turns[early_index] * (1 - share) + self.turns[early_index + 1] * share).field()   # (Rate+ <- Point-)[T]
        return (blended_field(self.strains), blended_field(self.strain_rates),
                blended_field(self.traces), tet_reading[..., 0] * turns)

    def at(self, tetrahedra: np.ndarray, places: Point, time: float) -> tuple[Rolling, Stretching, Stretching]:
        """Wave fields at `places` `[...] Point-[V_sites]` assemble the perturbed frame-rolling connection
        `[...] (Rate+ <- Point-)[V_sites]`, spatial stretch, and slice-rate squeeze `[...] (Point- <- Point-)[V_sites]`."""
        strain, strain_rate, trace, turn = self.fields(tetrahedra, places, time)
        # Strain stretches the translating step, bends the orthogonal frame turn, and adds its spatial curl.
        stretch = translating(strain(Step))                                     # [...] (Point- <- Point-)[V_sites]
        connection_rate = moving(Step + stretch / 2) + turning(Step - 2 * stretch + Step * (trace / 2)) + turn(Step)
        return connection_rate, stretch, translating(strain_rate) / 2


def proper_separations(connection_rate: Rolling, bonds: Topology, positions: Point) -> Point:
    """Applying each tail particle's connection `(Rate+ <- Point-)[V]` to its bond's round step gives the bond's proper
    separation `Point-[E]`."""
    tails = bonds.incidence[1, 0][0]                                            # Scalar+[E <- V]
    return origin.commutator((tails * connection_rate)(round_steps(bonds, positions))).cast(Step)


def massive(pick: Callable[[Point], np.ndarray], waves: Waves, bonds: Topology, rest: Scalar, stiffness: Scalar,
            damping: Scalar, holding: float, starts: Point, velocities: Point, start: float, step: float,
            times: np.ndarray) -> Iterator[tuple[Point, Scalar]]:
    """Yields particle positions `Point-[V]` and bond proper strains `Scalar+[E]` at `times` as spring-bound unit-mass
    particles step through `waves` by symplectic Euler."""
    homes = starts.batch()                                                      # [V] Point-
    positions, velocities, lengths, time = homes, velocities.batch(), rest, start
    for target in times:
        while time < target - step / 2:
            places = positions.field()                                          # Point-[V]
            connection_rate, stretch, squeeze = waves.at(pick(positions), places, time)   # (Rate+ <- Point-)[V], (Point- <- Point-)[V], (Point- <- Point-)[V]
            proper = proper_separations(connection_rate, bonds, places)         # Point-[E]
            previous_lengths, lengths = lengths, proper.norm()                  # Scalar+[E], Scalar+[E]
            # Springs and dashpots act along proper separations to pull stretched bond endpoints together.
            pulls = (stiffness * (lengths - rest) + damping * (lengths - previous_lengths) / step) / lengths   # Scalar+[E]
            forces = -(bonds.d[0].adjugate() * (pulls * proper)).batch()        # [V] Point-
            # Critically damped holding springs tether gauge particles gently to their free-fall home positions.
            home_steps = (frame(positions).inverse() >> homes).cast(Step)       # [V] Point-
            forces = forces + holding * home_steps - 2 * np.sqrt(holding) * velocities
            connection_rate, stretch, squeeze = connection_rate.batch(), stretch.batch(), squeeze.batch()   # [V] Rate+ <- Point-, Point- <- Point-, Point- <- Point-
            motion = velocities - stretch(velocities) / 2                       # [V] Point-
            carried_velocities = velocities - velocities.commutator(connection_rate(motion)).cast(Step) * step
            velocities = carried_velocities - squeeze(carried_velocities) * step + forces * step   # [V] Point-
            rolling = velocities - stretch(velocities) / 2                      # [V] Point-
            rolls = frame(positions) * (moving(rolling) * (-step / 2)).exp()    # [V] Motor+
            positions = rolls >> origin                                         # [V] Point-
            time += step
        yield positions.field(), lengths / rest - 1


def light(pick: Callable[[Point], np.ndarray], waves: Waves, starts: Point, headings: Point, start: float,
          step: float, n_steps: int) -> Iterator[Point]:
    """Yields ray positions `[...] Point-` across `n_steps` midpoint geodesic steps of arc length `step` through `waves`."""

    def turning_and_rolling(places: Point, heading: Point, time: float) -> tuple[Point, Point]:
        """Wave connection and slice stretch at `places` give the heading derivative and round rolling step `[...] Point-`."""
        connection_rate, stretch, squeeze = waves.at(pick(places), places.field(), time)   # (Rate+ <- Point-)[V_rays], (Point- <- Point-)[V_rays], (Point- <- Point-)[V_rays]
        connection_rate, stretch, squeeze = connection_rate.batch(), stretch.batch(), squeeze.batch()   # [...] Rate+ <- Point-, Point- <- Point-, Point- <- Point-
        rolling = heading - stretch(heading) / 2                                # [...] Point-
        return -heading.commutator(connection_rate(rolling)).cast(Step) - squeeze(heading), rolling

    positions, time = starts, start
    for _ in range(n_steps):
        start_change, start_rolling = turning_and_rolling(positions, headings, time)
        half_headings = (headings + start_change * (step / 2)).normalized()     # [...] Point-
        half_rolls = frame(positions) * (moving(start_rolling) * (-step / 4)).exp()   # [...] Motor+
        half_positions = half_rolls >> origin                                   # [...] Point-
        change, rolling = turning_and_rolling(half_positions, half_headings, time + step / 2)
        headings = (headings + change * step).normalized()                      # [...] Point-
        rolls = frame(positions) * (moving(rolling) * (-step / 2)).exp()        # [...] Motor+
        positions = rolls >> origin                                             # [...] Point-
        time += step
        yield positions


def base_points(places: Point, circling: Rate) -> Point:
    """Pulling the fibre velocity under `circling` into the frame at `places` gives the invariant Hopf base point `Point-`."""
    circle_directions = places.commutator(circling)                             # Point-
    return (frame(places).inverse() >> circle_directions).cast(Step).normalized()


def crossing(bases: Point, directions: Point) -> Point:
    """Quarter-turns of base tangent `directions` in the plane orthogonal to `bases` give horizontal steps `Point-` across
    the Hopf circles."""
    return turning(bases).commutator(directions)


def lifts(bases: Point, circling: Rate) -> Point:
    """Moving the origin horizontally across the Hopf circles by half the base angle lifts `bases` to points `Point-` on
    their fibres."""
    origin_base = base_points(origin, circling)                                 # [] Point-
    cosines = polar(origin_base) & bases                                        # Scalar+
    directions = (bases - origin_base * cosines).normalized()                   # Point-
    lifting = (moving(crossing(origin_base, directions)) * (-cosines.arccos() / 4)).exp()   # Motor+
    return lifting >> origin


def round_circles(bases: Point, turn: Motor, n_turns: int, n_samples: int) -> Point:
    """Powers of `turn` carry lifted points around `n_samples` evenly spaced fibre points `[n_samples] Point-[V_bases]`."""
    circling = turn.log()                                                       # [] Rate+
    shares = np.arange(n_samples) * n_turns / n_samples                         # [n_samples]
    turns = (circling * shares[:, None]).exp()                                  # [n_samples] Motor+
    return turns >> lifts(bases, circling)[None]


def stretches(waves: Waves, pick: Callable[[Point], np.ndarray], bases: Point, directions: Point, turn: Motor,
              n_turns: int, n_samples: int, time: float) -> Scalar:
    """Averaging the relative metric stretch along `directions` around `n_samples` fibre points gives the circle-averaged
    stretch `Scalar+[V_bases]`."""
    places = round_circles(bases, turn, n_turns, n_samples)                     # [n_samples] Point-[V_bases]
    strain, _, _, _ = waves.fields(pick(places), places, time)                  # [n_samples] (Plane+ <- Point-)[V_bases]
    # Strain paired against the round duality map measures fractional length stretch along `directions`.
    readings = (strain(directions) & directions) / (2 * (polar(directions) & directions))   # [n_samples] Scalar+[V_bases]
    return readings.sum(axis=0) / n_samples


# GPU ray legs across tetrahedral charts of the three-sphere in numga's WGSL dialect. On each tetrahedron's chart
# hyperplane, great circles project to straight segments and corner weights vary linearly, so each leg exits through the
# first facet whose opposite corner weight reaches zero.
round_reach = 0.6

tet_struct_source = '''
struct Tet {
    neighbours: vec4<u32>,
    entries: vec4<u32>,
    gradients: array<Plane, 4>,
}
'''
tets_source = tet_struct_source + '''
var<storage, read> tets: array<Tet>;
'''

round_walk_source = tets_source + '''
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

fn frame(place: Point) -> Motor {
    return ((1.0 + mv.xyzw) * (place / origin) + (1.0 - mv.xyzw)) * 0.5;
}

fn next_leg(place: Point, heading: Step, tet: u32, entry: u32, time: f32) -> Leg {
    // Chart hyperplane reading one at all four corners, together with the straight segment toward a step of `reach`.
    let unit = tets[tet].gradients[0] + tets[tet].gradients[1] + tets[tet].gradients[2] + tets[tet].gradients[3];
    let chart = place / (unit & place);
    let reach_end = (frame(place) * (moving(heading) * (-0.5 * reach)).exp()) >> origin;
    let segment = reach_end / (unit & reach_end) - chart;
    // Share of the segment where the first corner weight reaches zero, skipping the entry facet.
    var share = 1.0;
    var exit = 4u;
    for (var corner = 0u; corner < 4u; corner++) {
        let change = tets[tet].gradients[corner] & segment;
        if (corner != entry && change < 0.0) {
            let run_out = max(-(tets[tet].gradients[corner] & chart) / change, 0.0);
            if (run_out < share) {
                share = run_out;
                exit = corner;
            }
        }
    }
    let end = (chart + segment * share).normalized();
    var this_leg = Leg(unit, chart, segment * share, acos(clamp(place.dot(end), -1.0, 1.0)), heading, tet, 4u);
    if (exit < 4u) {
        this_leg.tet = tets[tet].neighbours[exit];
        this_leg.entry = tets[tet].entries[exit];
    }
    return this_leg;
}
'''

# Wave-perturbed ray legs subdivide each tetrahedral crossing into at most `n_rolls` midpoint substeps while tracing
# backward in light travel time from the eye.
wave_step = 0.16
n_rolls = 8

wave_tables_source = '''
struct Tet {
    corners: vec4<u32>,
    neighbours: vec4<u32>,
    entries: vec4<u32>,
    gradients: array<Plane, 4>,
}

var<storage, read> tets: array<Tet>;
var<storage, read> strains: array<Strain>;
var<storage, read> strain_rates: array<Strain>;
var<storage, read> traces: array<f32>;
var<storage, read> turns: array<Turning>;
var<uniform> n_vertices: u32;
var<uniform> n_tets: u32;
var<uniform> first_time: f32;
var<uniform> level_step: f32;

fn local_waves(tet: u32, place: Point, time: f32) -> Local {
    let level = (time - first_time) / level_step;
    let early = u32(floor(level));
    let share = level - floor(level);
    let now = early * n_vertices;
    let later = now + n_vertices;
    let unit = tets[tet].gradients[0] + tets[tet].gradients[1] + tets[tet].gradients[2] + tets[tet].gradients[3];
    var waves: Local;
    for (var corner = 0u; corner < 4u; corner++) {
        let vertex = tets[tet].corners[corner];
        let weight = (tets[tet].gradients[corner] & place) / (unit & place);
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
'''

# Exact lowest quadrupole gravitational wave of the round three-sphere, oscillating at `sqrt(8)` in the frames.
quadrupole_frequency = np.sqrt(8.0)

quadrupole_source = tets_source + '''
var<uniform> swing: f32;

const quadrupole = (mv.x * (mv.x & Step) - mv.y * (mv.y & Step)).cast(Strain);

fn local_waves(tet: u32, place: Point, time: f32) -> Local {
    let phase = frequency * time;
    var waves: Local;
    waves.strain = quadrupole * (2.0 * swing * cos(phase));
    waves.strain_rate = quadrupole * (-2.0 * swing * frequency * sin(phase));
    return waves;
}
'''

wave_legs_source = '''
var<uniform> origin: Point;
var<uniform> translating: Translating;
var<uniform> moving: Moving;
var<uniform> turning: Rolling;
var<uniform> step: f32;
var<uniform> n_rolls: u32;

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

fn connection(steps: Step, waves: Local) -> Rate {
    let stretch = translating(waves.strain(steps));
    return moving(steps + stretch * 0.5) + turning(steps - stretch * 2.0 + steps * (waves.trace * 0.5))
           + waves.turn(steps);
}

fn round_step(heading: Step, waves: Local) -> Step {
    return heading - translating(waves.strain(heading)) * 0.5;
}

fn heading_change(heading: Step, waves: Local) -> Step {
    return -heading.commutator(connection(round_step(heading, waves), waves)).cast(Step)
           + translating(waves.strain_rate(heading)) * 0.5;
}

fn frame(place: Point) -> Motor {
    return ((1.0 + mv.xyzw) * (place / origin) + (1.0 - mv.xyzw)) * 0.5;
}

fn next_leg(place: Point, heading: Step, tet: u32, entry: u32, time: f32) -> Leg {
    let unit = tets[tet].gradients[0] + tets[tet].gradients[1] + tets[tet].gradients[2] + tets[tet].gradients[3];
    let chart = place / (unit & place);
    let start_waves = local_waves(tet, place, time);
    let step_end = (frame(place) * (moving(round_step(heading, start_waves)) * (-0.5 * step)).exp()) >> origin;
    let step_segment = step_end / (unit & step_end) - chart;
    var share = 1.0;
    var exit = 4u;
    for (var corner = 0u; corner < 4u; corner++) {
        let change = tets[tet].gradients[corner] & step_segment;
        if (corner != entry && change < 0.0) {
            let run_out = max(-(tets[tet].gradients[corner] & chart) / change, 0.0);
            if (run_out < share) {
                share = run_out;
                exit = corner;
            }
        }
    }
    let length = acos(clamp(place.dot((chart + step_segment * share).normalized()), -1.0, 1.0));
    let n_leg_rolls = max(u32(ceil(f32(n_rolls) * share)), 1u);
    let roll = length / f32(n_leg_rolls);
    var leg_end = place;
    var end_heading = heading;
    for (var k = 0u; k < n_leg_rolls; k++) {
        let roll_time = time - roll * f32(k);
        let roll_waves = local_waves(tet, leg_end, roll_time);
        let half_heading = (end_heading + heading_change(end_heading, roll_waves) * (0.5 * roll)).normalized();
        let half_place = (frame(leg_end) * (moving(round_step(end_heading, roll_waves)) * (-0.25 * roll)).exp())
                         >> origin;
        let half_waves = local_waves(tet, half_place, roll_time - 0.5 * roll);
        leg_end = (frame(leg_end) * (moving(round_step(half_heading, half_waves)) * (-0.5 * roll)).exp()) >> origin;
        end_heading = (end_heading + heading_change(half_heading, half_waves) * roll).normalized();
    }
    var this_leg = Leg(unit, chart, leg_end / (unit & leg_end) - chart, length, end_heading, tet, 4u);
    if (exit < 4u) {
        this_leg.tet = tets[tet].neighbours[exit];
        this_leg.entry = tets[tet].entries[exit];
    }
    return this_leg;
}
'''

wave_walk_source = wave_tables_source + wave_legs_source
quadrupole_walk_source = quadrupole_source + wave_legs_source
