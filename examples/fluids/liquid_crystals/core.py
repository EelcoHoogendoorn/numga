"""Orientational torque and active substrate stress drive half-charge defects in a nematic liquid-crystal film.

A nematic records a molecular axis without choosing either end. Its alignment is modelled as the mean projector
onto those axes, less half the identity in the film. That symmetric traceless map measures the degree of order by
its norm, which vanishes smoothly at a defect core. Bulk free energy favours unit order while elastic energy
penalizes spatial variation. Differentiating with respect to a local bivector turn yields a linear form on turns,
paired with a restoring couple through the regressive product. Pure rotation relaxes only orientation, whereas the
full molecular field also alters the degree of order.

In the active film, the divergence of the active stress balances against local substrate drag to drive a velocity
that advects and corotates the alignment without flow-alignment strain coupling. Perimeter vertices hold their
initial alignment at zero velocity. Activity injects mechanical work, while the passive film descends its
Landau–de Gennes free energy.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import jax
import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

ga = Algebra("x+y+")
mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                      # Scalar+
Point = ga.gatype.antivector()                                   # Antivector-
Plane = ga.gatype.vector()                                       # Vector+
Turn = ga.gatype.bivector()                                      # Bivector+
Couple = ga.gatype.antibivector()                                # Antibivector-
Order = ga.gatype((Point, Point))                                # Point- <- Point-
Slope = ga.gatype((Scalar, Turn))                                # Scalar+ <- Turn+
work = Couple & Turn                                             # [] Scalar+ <- (Couple-, Turn+)


# --- math -----------------------------------------------------------------------------
def alignment(directors: Point, strength: Scalar) -> Order:
    """Unit molecular axes `directors` and scalar `strength` define the symmetric traceless alignment
    map (`Point- <- Point-`)."""
    return strength * (directors * (directors.dual_inverse() & Point) - Point / 2)


def defect_pair(vertices: Point, centres: Point, core_radius: float, bearing: Turn) -> Order:
    """Two opposite half-charge defects at `centres` (`[2] Point-`) set an alignment field
    (`(Point- <- Point-)[V]`) whose order vanishes smoothly within `core_radius` of each core."""
    positive_offset, negative_offset = vertices - centres[0], vertices - centres[1]  # Point-[V], Point-[V]
    # The relative radial turn winds once at one core and backwards at the other. Its quarter power turns a
    # molecular axis through half that angle; crossing the logarithm's cut only reverses the axis.
    relative_rotor = positive_offset.normalized().dual_inverse() * negative_offset.normalized().dual_inverse()
    rotor = (relative_rotor.log() / 4 - bearing / 2).exp()
    directors = rotor >> mv.x.dual()                             # Point-[V]
    radii = positive_offset.norm() * negative_offset.norm()      # Scalar+[V]
    strength = radii / ((positive_offset.scalar_norm_squared() + core_radius ** 2) *
                        (negative_offset.scalar_norm_squared() + core_radius ** 2)).square_root()
    return alignment(directors, strength)


def disordered_alignment(vertices: Point, wavevectors: Plane, phases: np.ndarray,
                         amplitudes: Order, softness: float) -> Order:
    """Superposed plane waves (`[...] Plane+`) saturate into a smooth disordered alignment field
    (`(Point- <- Point-)[V]`) with half-charge defects where the wave sum vanishes."""
    waves = (wavevectors & vertices[..., None]) + phases         # [...] Scalar+[V]
    order = (amplitudes * waves.cos()).sum(axis=-1)              # (Point- <- Point-)[V]
    # Saturation gives strongly aligned domains while leaving each zero a continuous, disordered core.
    strength_squared = 2 * order(order).batch().trace().field()  # Scalar+[V]
    return order / (strength_squared + softness ** 2).square_root()


def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Triangle geometry on `vertices` (`Point-[V]`) gives the triangle areas (`Scalar+[F]`),
    simplex gradients (`[3] Plane+[F]`), and lumped dual-cell areas (`Scalar+[V]`)."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)  # [3] Point-[F]
    double_areas = sides[0] & sides[1]                            # Scalar-[F]
    gradients = sides / double_areas                              # [3] Plane+[F]
    areas = double_areas.norm() / 2                               # Scalar+[F]
    dual_areas = (topology.incidence[2, 0].adjugate() * (areas / 3)).sum(axis=0)  # Scalar+[V]
    return areas, gradients, dual_areas


def elasticity(topology: Topology, areas: Scalar, gradients: Plane) -> SparseExtensor:
    """Triangle areas and simplex gradients assemble into the symmetric Dirichlet stiffness
    (`Scalar+[V <- V]`) on vertex fields."""
    corners = topology.incidence[2, 0]                            # [3] Scalar+[F <- V]
    local_stiffness = areas * (gradients[:, None] & gradients[None, :].dual())  # [3, 3] Scalar+[F]
    return (corners[:, None].adjugate() * spdiag(local_stiffness) * corners[None, :]).sum(axis=0).sum(axis=0)


def free_energy(stiffness: SparseExtensor, dual_areas: Scalar, elastic: float, bulk: float, order: Order) -> Scalar:
    """Bulk order preference and elastic distortion integrate over `dual_areas` (`Scalar+[V]`) into the
    Landau–de Gennes free energy (`[...] Scalar+`)."""
    # Trace each local alignment map at its vertex without contracting across spatial sites.
    squared_order = order(order).batch().trace().field()          # [...] Scalar+[V]
    well = bulk * (squared_order - 0.5) ** 2 / 2                  # [...] Scalar+[V]
    distortion = order(stiffness * order).batch().trace().field()  # [...] Scalar+[V]
    return (dual_areas * well + elastic * distortion / 2).sites.sum()


def molecular_field(stiffness: SparseExtensor, dual_areas: Scalar, elastic: float, bulk: float,
                    order: Order) -> Order:
    """Minus the free-energy slope per unit area defines the symmetric traceless molecular field
    (`[...] (Point- <- Point-)[V]`)."""
    squared_order = order(order).batch().trace().field()          # [...] Scalar+[V]
    return -2 * bulk * (squared_order - 0.5) * order - elastic * (stiffness * order) / dual_areas


def turn_slope(order: Order, molecular: Order) -> Slope:
    """Commuting `order` with `Turn` and pairing against `molecular` gives the orientational energy
    slope (`[...] (Scalar+ <- Turn+)[V]`)."""
    # Both the direction carried by the map and the direction it reads turn with the molecules.
    tangent = order.commutator(Turn) - order(Point.commutator(Turn))
    # The trace pairs the molecular field with the map's change; the molecular field is minus the slope.
    return -molecular(tangent).batch().trace(0, 1).field()


def windings(topology: Topology, order: Order, reference: Point) -> Turn:
    """Edge rotor turns of `order(reference)` sum around each triangle into its director winding
    (`[...] Turn+[F]`) in whole turns."""
    # A traceless alignment acting on a fixed direction turns through twice the molecular angle. Its edge
    # logarithms therefore count half-charge defects without choosing a sign for any molecular axis.
    ends = topology.incidence[1, 0] * order(reference)[..., None]  # [..., 2] Point-[E]
    tails, heads = ends[..., 0], ends[..., 1]                      # [...] Point-[E], Point-[E]
    turns = (heads.dual_inverse() * tails.dual_inverse()).normalized().log()
    return (topology.d[1] * turns) / (4 * np.pi)


def defects(topology: Topology, vertices: Point, order: Order, reference: Point) -> tuple[Point, Scalar]:
    """Linear interpolation across each winding triangle locates the alignment zeros (`[...] Point-`)
    and reads their charges (`[...] Scalar+`) in whole turns."""
    charges = windings(topology, order, reference) & mv.xy.dual()  # Scalar+[F]
    winding_faces = np.flatnonzero(charges.norm() > 0.25)
    tails, heads = topology.incidence[1, 0] * order(reference)     # Point-[E], Point-[E]
    # The triangle of alignment values holds the origin at the weights its opposite sides span with it;
    # the same weights on the triangle's corners place the zero.
    zero_spans = (topology.boundary[2, 1] * (tails & heads)).batch()[:, winding_faces]  # [3, ...] Scalar-
    corner_weights = zero_spans / zero_spans.sum(axis=0)           # [3, ...] Scalar+
    corners = (topology.incidence[2, 0] * vertices).batch()[:, winding_faces]  # [3, ...] Point-
    return (corners * corner_weights).sum(axis=0), charges.batch()[winding_faces]


def with_defects(topology: Topology, vertices: Point, states: Iterable[Order],
                 reference: Point) -> Iterator[tuple[Order, list[tuple[Point, Scalar]]]]:
    """Yields each batched alignment state (`[...] (Point- <- Point-)[V]`) paired with the defect
    positions and charges of every film in the batch."""
    for order in states:
        yield order, [defects(topology, vertices, film, reference) for film in order]


def turn_relaxation(topology: Topology, stiffness: SparseExtensor, dual_areas: Scalar, elastic: float, bulk: float,
                    initial: Order, time_step: float, n_frames: int, n_substeps: int) -> Iterator[Order]:
    """Yields `n_frames` alignment states (`[...] (Point- <- Point-)[V]`) as rotational friction relaxes
    the molecular orientation at fixed order strength."""
    # Unit rotational friction is a positive form on bivectors. Solving it against the negative derivative
    # gives the angular velocity whose dissipated power is the couple's work.
    friction = -Turn.scalar_product(Turn)                         # [] Scalar+ <- (Turn+, Turn+)
    boundary = topology.reading(topology.exterior[0], 0)          # Scalar+[V_exterior <- V]

    def step(order: Order, _) -> tuple[Order, None]:
        molecular = molecular_field(stiffness, dual_areas, elastic, bulk, order)
        angular_velocity = friction.solve(-turn_slope(order, molecular))
        angular_velocity = angular_velocity - boundary.adjugate() * (boundary * angular_velocity)
        rotor = (-angular_velocity * time_step / 2).exp()
        return rotor >> order(rotor << Point), None

    @jax.jit
    def advance(order: Order) -> Order:
        return jax.lax.scan(step, order, None, length=n_substeps)[0]

    order = initial
    yield order
    for _ in range(n_frames - 1):
        order = advance(order)
        yield order


def active_velocity(topology: Topology, areas: Scalar, gradients: Plane, dual_areas: Scalar,
                    activity: np.ndarray, order: Order) -> Point:
    """Active stress divergence balanced against unit substrate drag drives the interior film velocity
    (`[...] Point-[V]`)."""
    corners = topology.incidence[2, 0]                            # [3] Scalar+[F <- V]
    face_order = corners * order[..., None]                       # [..., 3] (Point- <- Point-)[F]
    # Isotropic substrate drag turns the stress divergence into a velocity along its force.
    divergence = face_order(gradients.dual()).sum(axis=-1)        # [...] Point-[F]
    velocity = -activity * (corners.adjugate() * (areas * divergence / 3)[..., None]).sum(axis=-1) / dual_areas
    boundary = topology.reading(topology.exterior[0], 0)          # Scalar+[V_exterior <- V]
    return velocity - boundary.adjugate() * (boundary * velocity)


def rate(topology: Topology, areas: Scalar, gradients: Plane, dual_areas: Scalar, stiffness: SparseExtensor,
         elastic: float, bulk: float, activity: np.ndarray, order: Order) -> Order:
    """Molecular relaxation, active advection, and vorticity corotation sum into the interior alignment
    rate (`[...] (Point- <- Point-)[V]`)."""
    molecular = molecular_field(stiffness, dual_areas, elastic, bulk, order)
    velocity = active_velocity(topology, areas, gradients, dual_areas, activity, order)
    corners = topology.incidence[2, 0]                            # [3] Scalar+[F <- V]
    local_order = corners * order[..., None]                      # [..., 3] (Point- <- Point-)[F]
    local_velocity = corners * velocity[..., None]                # [..., 3] Point-[F]
    face_velocity = local_velocity.mean(axis=-1)                  # [...] Point-[F]
    advection = (local_order * (gradients & face_velocity[..., None])).sum(axis=-1)
    velocity_gradient = (local_velocity * (gradients & Point)).sum(axis=-1)
    # Half the curl is the plane and rate of the fluid's rotation.
    rotation_rate = Point.reverse().commutator(velocity_gradient(Point)).batch().contract(1, 2).field() / 2
    face_order = local_order.mean(axis=-1)
    corotation = face_order.commutator(rotation_rate) - face_order(Point.commutator(rotation_rate))
    transport = (corners.adjugate() * (areas * (corotation - advection) / 3)[..., None]).sum(axis=-1) / dual_areas
    total_rate = molecular + transport
    boundary = topology.reading(topology.exterior[0], 0)          # Scalar+[V_exterior <- V]
    return total_rate - boundary.adjugate() * (boundary * total_rate)


def evolution(topology: Topology, areas: Scalar, gradients: Plane, dual_areas: Scalar, stiffness: SparseExtensor,
              elastic: float, bulk: float, activity: np.ndarray, initial: Order, time_step: float,
              n_frames: int, n_substeps: int) -> Iterator[Order]:
    """Yields `n_frames` alignment states (`[...] (Point- <- Point-)[V]`) advanced by explicit midpoint
    substeps."""
    def step(order: Order, _) -> tuple[Order, None]:
        midpoint_order = order + time_step / 2 * rate(topology, areas, gradients, dual_areas, stiffness,
                                                      elastic, bulk, activity, order)
        next_order = order + time_step * rate(topology, areas, gradients, dual_areas, stiffness,
                                              elastic, bulk, activity, midpoint_order)
        return next_order, None

    @jax.jit
    def advance(order: Order) -> Order:
        return jax.lax.scan(step, order, None, length=n_substeps)[0]

    order = initial
    yield order
    for _ in range(n_frames - 1):
        order = advance(order)
        yield order
