"""Nematic defects in a film: orientational torque and active transport of an alignment map.

A nematic records a molecular axis without choosing either end. Its alignment is modelled as the mean projector
onto those axes, less half the identity in the film. The map is symmetric and traceless; its size measures order,
which can vanish at a defect core. A bulk free energy favours order and an elastic energy penalizes its variation.
The derivative with respect to a local bivector turn is a form on turns, represented by a couple through the
regressive pairing. It measures only the orientational part of relaxation; the full map also changes its size.

The active film uses a local substrate drag: the divergence of the active stress drives a velocity, which carries
and turns the alignment. This is a compressible, friction-dominated model with corotation and no flow alignment.
The boundary holds the initial alignment and the velocity vanishes there. Activity supplies energy; the passive
film descends its Landau–de Gennes free energy.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

from numga import Algebra
from numga.backend.jax import JaxContext
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

ga = Algebra("x+y+")
mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                      # Scalar+
Point = ga.gatype.antivector()                                   # Antivector-
Plane = ga.gatype.vector()                                       # Vector+
Turn = ga.gatype.bivector()                                      # Bivector+
Couple = ga.gatype.antibivector()                                # Antibivector-
Order = ga.gatype((Point, Point))                                # Point- <- Point-
Slope = ga.gatype((Scalar, Turn))                                # Scalar+ <- Turn+
work = Couple & Turn                                            # [] Scalar+ <- (Couple-, Turn+)


# --- math -----------------------------------------------------------------------------
def alignment(directors: Point, strength: Scalar) -> Order:
    """The alignment map of these unit molecular axes, with this degree of order."""
    return strength * (directors * (directors.dual_inverse() & Point) - Point / 2)


def defect_pair(vertices: Point, centres: Point, core_radius: float, bearing: Turn) -> Order:
    """Opposite half-charge defects, with a smooth loss of order at their cores."""
    positive, negative = vertices - centres[0], vertices - centres[1]
    # The relative radial turn winds once at one core and backwards at the other. Its quarter power turns a
    # molecular axis through half that angle; crossing the logarithm's cut only reverses the axis.
    relative = positive.normalized().dual_inverse() * negative.normalized().dual_inverse()
    turn = (relative.log() / 4 + bearing * -0.5).exp()
    directors = turn >> mv.x.dual()                              # Point-[vertices]
    radii = positive.norm() * negative.norm()
    strength = radii / ((positive.scalar_norm_squared() + core_radius ** 2) *
                        (negative.scalar_norm_squared() + core_radius ** 2)).square_root()
    return alignment(directors, strength)


def disordered_alignment(vertices: Point, wavevectors: Plane, phases: np.ndarray,
                         amplitudes: Order, softness: float) -> Order:
    """A smooth, disordered alignment, with half-charge defects where the wave sum vanishes."""
    waves = (wavevectors & vertices[..., None]) + phases
    order = (amplitudes * waves.cos()).sum(axis=-1)
    # Saturation gives strongly aligned domains while leaving each zero a continuous, disordered core.
    strength_squared = 2 * order(order).batch().trace().field()
    return order / (strength_squared + softness ** 2).square_root()


def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """The triangles' areas, their corner gradients and the vertices' shares of area."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)  # [corners] Point-[faces]
    volume = sides[0] & sides[1]                                # Scalar-[faces]
    gradients = sides / volume                                 # [corners] Plane+[faces]
    areas = volume.norm() / 2                                   # Scalar+[faces]
    cells = (~topology.incidence[2, 0] * (areas / 3)).sum(axis=0) # Scalar+[vertices]
    return areas, gradients, cells


def elasticity(topology: Topology, areas: Scalar, gradients: Plane) -> SparseExtensor:
    """The Dirichlet form on vertex fields, assembled from the triangles' corner gradients."""
    corners = topology.incidence[2, 0]                           # [corners] Scalar+[faces <- vertices]
    local = areas * (gradients[:, None] & gradients[None, :].dual())
    return (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)


def free_energy(stiffness: SparseExtensor, cells: Scalar, elastic: float, bulk: float, order: Order) -> Scalar:
    """The film's free energy: a bulk preference for unit order and the elastic cost of spatial variation."""
    # This is the trace of each local map, leaving the spatial sites uncontracted.
    squared_order = order(order).batch().trace().field()          # Scalar+[vertices]
    well = bulk * (squared_order - 0.5) ** 2 / 2                # Scalar+[vertices]
    distortion = order(stiffness * order).batch().trace().field() # Scalar+[vertices]
    return (cells * well + elastic * distortion / 2).sites.sum()


def molecular_field(stiffness: SparseExtensor, cells: Scalar, elastic: float, bulk: float, order: Order) -> Order:
    """The negative free-energy derivative per unit area, within symmetric traceless maps."""
    squared_order = order(order).batch().trace().field()          # Scalar+[vertices]
    return -2 * bulk * (squared_order - 0.5) * order - elastic * (stiffness * order) / cells


def turn_slope(order: Order, molecular: Order) -> Slope:
    """The free-energy density's first derivative with respect to a bivector turn of the alignment."""
    # Both the direction carried by the map and the direction it reads turn with the molecules.
    tangent = order.commutator(Turn) - order(Point.commutator(Turn))
    # The trace pairs the molecular field with the map's change; the molecular field is minus the slope.
    return -molecular(tangent).batch().trace(0, 1).field()


def windings(topology: Topology, order: Order, reference: Point) -> Turn:
    """Each triangle's director winding, in whole turns, represented in the film's turning plane."""
    # A traceless alignment acting on a fixed direction turns through twice the molecular angle. Its edge
    # logarithms therefore count half-charge defects without choosing a sign for any molecular axis.
    ends = topology.incidence[1, 0] * order(reference)[..., None]
    tails, heads = ends[..., 0], ends[..., 1]
    turns = (heads.dual_inverse() * tails.dual_inverse()).normalized().log()
    return (topology.d[1] * turns) / (4 * np.pi)


def turn_relaxation(topology: Topology, stiffness: SparseExtensor, cells: Scalar, elastic: float, bulk: float,
                    initial: Order, time_step: float, n_frames: int, n_substeps: int) -> Iterator[Order]:
    """The alignment rotating down its energy slope, with its degree of order fixed at every site."""
    # Unit rotational friction is a positive form on bivectors. Solving it against the negative derivative
    # gives the angular velocity whose dissipated power is the couple's work.
    friction = -Turn.scalar_product(Turn)                        # [] Scalar+ <- (Turn+, Turn+)
    boundary = topology.reading(topology.exterior[0], 0)

    def step(order: Order, _) -> tuple[Order, None]:
        molecular = molecular_field(stiffness, cells, elastic, bulk, order)
        angular_velocity = friction.solve(-turn_slope(order, molecular))
        angular_velocity = angular_velocity - ~boundary * (boundary * angular_velocity)
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


def active_velocity(topology: Topology, areas: Scalar, gradients: Plane, cells: Scalar,
                    activity: np.ndarray, order: Order) -> Point:
    """The velocity driven by minus the alignment's divergence, at unit substrate drag."""
    corners = topology.incidence[2, 0]
    face_order = corners * order[..., None]                     # [..., corners] (Point- <- Point-)[faces]
    # Isotropic substrate drag turns the stress divergence into a velocity along its force.
    divergence = face_order(gradients.dual()).sum(axis=-1)       # [...] Point-[faces]
    velocity = -activity * ((~corners * (areas * divergence / 3)[..., None]).sum(axis=-1)) / cells
    boundary = topology.reading(topology.exterior[0], 0)
    return velocity - ~boundary * (boundary * velocity)


def rate(topology: Topology, areas: Scalar, gradients: Plane, cells: Scalar, stiffness: SparseExtensor,
         elastic: float, bulk: float, activity: np.ndarray, order: Order) -> Order:
    """Relaxation, advection and corotation of the film's alignment."""
    molecular = molecular_field(stiffness, cells, elastic, bulk, order)
    velocity = active_velocity(topology, areas, gradients, cells, activity, order)
    corners = topology.incidence[2, 0]
    local_order = corners * order[..., None]
    local_velocity = corners * velocity[..., None]
    face_velocity = local_velocity.mean(axis=-1)                # [...] Point-[faces]
    advection = (local_order * (gradients & face_velocity[..., None])).sum(axis=-1)
    velocity_gradient = (local_velocity * (gradients & Point)).sum(axis=-1)
    # Half the curl is the plane and rate of the fluid's rotation.
    spin = Point.reverse().commutator(velocity_gradient(Point)).batch().contract(1, 2).field() / 2
    face_order = local_order.mean(axis=-1)
    corotation = face_order.commutator(spin) - face_order(Point.commutator(spin))
    transported = (~corners * (areas * (corotation - advection) / 3)[..., None]).sum(axis=-1) / cells
    changing = molecular + transported
    boundary = topology.reading(topology.exterior[0], 0)
    return changing - ~boundary * (boundary * changing)


def evolution(topology: Topology, areas: Scalar, gradients: Plane, cells: Scalar, stiffness: SparseExtensor,
              elastic: float, bulk: float, activity: np.ndarray, initial: Order, time_step: float,
              n_frames: int, n_substeps: int) -> Iterator[Order]:
    """The film's alignment at each frame, advanced with a midpoint step."""
    def step(order: Order, _) -> tuple[Order, None]:
        halfway = order + time_step / 2 * rate(topology, areas, gradients, cells, stiffness,
                                               elastic, bulk, activity, order)
        following = order + time_step * rate(topology, areas, gradients, cells, stiffness,
                                              elastic, bulk, activity, halfway)
        return following, None

    @jax.jit
    def advance(order: Order) -> Order:
        return jax.lax.scan(step, order, None, length=n_substeps)[0]

    order = initial
    yield order
    for _ in range(n_frames - 1):
        order = advance(order)
        yield order
