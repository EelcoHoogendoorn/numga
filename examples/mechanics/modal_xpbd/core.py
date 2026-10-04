"""Small elastic deformations carried by freely moving rigid frames, in the projective geometric
algebra of the plane.

Each body is a girder of bars. Its large motion is a motor, and its small bending is a few of its
vibration modes, each a field of displacements over its points with an amplitude. A point of the
body is its rest point moved by the modes, carried into the world by the motor. The modes come
from the girder's stiffness against its masses: a sparse extensor from the points to the bars,
taking each bar's tail from its head, gives the bars' stretches, and its reverse carries their
pulls back to the points.

The bodies are stepped by extended position-based dynamics: each step predicts the motion, then
moves the bodies until their pins close, and every spring is a compliance solved with them. A pin
joins an anchor point on one body to one on another; a hinge is one pin, a splice two. How an
anchor moves for a twist of its body is a map, the commutator of the anchor with the open twist,
`Direction <- Twist`, and over all pins and bodies these maps are the cells of a sparse extensor;
its adjugate carries the forces at the pins back to the forques they exert on the bodies. Through
the inverse inertia the two compose to how every pin opens for a force at any other, and the
vibration modes add how far they yield. One sparse solve gives the forces that close every pin
together: solved one joint at a time, a stiff structure converges too slowly to carry a load.

Inertia is evaluated at rest and the coupling of rotation and vibration is left out, so the bodies
should turn slowly compared with their retained vibrations.

In the notation of extended position-based dynamics the system reads as
$S = J_b M_b^{-1} J_b^\\top + J_q W J_q^\\top + \\tilde\\alpha_p$, with
$W = \\tilde\\alpha_m / (1 + \\tilde\\alpha_m)$ the modes' mobility.
"""

from dataclasses import dataclass, replace

import numpy as np

from numga import NumpyContext
from numga.algebras import PGA2D
from numga.sparse import SparseExtensor, spdiag
from examples.mechanics import lie_integrators as lie


ga = PGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Twist = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
# The lines x and y through the origin, the polars of the directions: a displacement or a force in
# the plane is one of them, and joined with a direction, `force & direction`, it gives the work done
# moving that way.
Force = ga.gatype(ga.subspace("x y"))
Direction = ga.gatype(Force.output_subspace.complement())
Inertia = ga.gatype((Forque, Twist))                                       # Forque <- Twist
InverseInertia = ga.gatype((Twist, Forque))                                # Twist <- Forque


# --- plumbing -------------------------------------------------------------------------
@dataclass(frozen=True)
class Shape:
    """A girder at rest and its retained vibrations."""
    rest: Point                    # [vertices] Point, centred on mass
    modes: Direction               # [modes, vertices] Direction
    frequencies: Scalar            # [modes] Scalar, angular frequencies
    compliance: Scalar             # [modes] Scalar, inverse squared frequencies
    masses: np.ndarray             # [vertices]
    edges: np.ndarray              # [edges, ends]
    inertia: Inertia               # [] Forque <- Twist


@dataclass(frozen=True)
class Bodies:
    """Bodies in motion, batched over cases: their motors and twists, and their modes' amplitudes and
    rates."""
    motor: Motor                   # [..., bodies] Motor
    rate: Twist                    # [..., bodies] Twist, in each body's frame
    amplitudes: Scalar             # [..., bodies, modes] Scalar
    rates: Scalar                  # [..., bodies, modes] Scalar
    compliance: Scalar             # [..., bodies, modes] Scalar
    frequencies: Scalar            # [..., bodies, modes] Scalar
    damping: np.ndarray            # [..., bodies, modes], damping ratio
    masses: np.ndarray             # [..., bodies]
    inertia: Inertia               # [..., bodies] Forque <- Twist
    inverse_inertia: InverseInertia  # [..., bodies] Twist <- Forque, zero for a fixed body


@dataclass(frozen=True)
class Pins:
    """Anchor points of two bodies pinned together: the bodies, the anchors in each body's frame,
    and the body's modes sampled there."""
    bodies: np.ndarray             # [pins, ends]
    anchors: Point                 # [pins, ends] Point
    modes: Direction               # [pins, ends, modes] Direction
    compliance: Scalar             # [pins] Scalar


def truss(cells: int, length: float, height: float) -> tuple[Force, np.ndarray]:
    """A girder's points at rest, two levels at every station, as lines through the origin `[vertices]
    Force`, and its bars `[edges, ends]`: longitudinal bars, both diagonals, and one upright at every
    station."""
    stations = np.arange(cells + 1)
    levels = np.array([-0.5, 0.5])
    positions = (mv.x * (stations[:, None] * length / cells) + mv.y * (levels[None, :] * height)).reshape(-1)  # [vertices] Force
    lower = 2 * stations[:-1]
    # Longitudinal bars, both diagonals, and one upright at every station.
    edges = np.concatenate([
        np.stack([lower, lower + 2], axis=-1),
        np.stack([lower + 1, lower + 3], axis=-1),
        np.stack([lower, lower + 3], axis=-1),
        np.stack([lower + 1, lower + 2], axis=-1),
        np.stack([2 * stations, 2 * stations + 1], axis=-1),
    ])
    return positions, edges


def incidence(cells, rows: np.ndarray, columns: np.ndarray, shape: tuple[int, int]) -> SparseExtensor:
    """Cells `[..., *rows.shape]` as a sparse extensor, each coupling the row and the column named at
    its place."""
    return SparseExtensor(cells.reshape(cells.shape[:cells.ndim - rows.ndim] + (rows.size,)), rows.reshape(-1), columns.reshape(-1), shape)


# --- math -----------------------------------------------------------------------------
# Leading ... axes hold independent cases.
def girder(cells: int, length: float, height: float, stiffness: float, density: float, modes: int) -> Shape:
    """A girder of cross-braced bars, reduced to its lowest free vibration modes."""
    positions, edges = truss(cells, length, height)                       # [vertices] Force, [edges, ends]
    vertices = positions.shape[0]
    # Each bar's tail taken from its head.
    ends = SparseExtensor.from_columns(edges, mv.scalar((np.ones_like(edges) * [-1, 1])[..., None]), vertices)  # [edges, vertices] Scalar
    difference = ends * positions                                         # [edges] Force
    lengths = difference.norm()                                           # [edges] Scalar
    directions = difference / lengths                                     # [edges] Force
    # Each bar measures extension along itself and pulls back along the same direction.
    bars = spdiag(directions * (directions | Force) * stiffness / lengths)  # [edges, edges] Force <- Force
    masses = np.full(vertices, density * length * height / vertices)
    weights = spdiag(mv.scalar(masses[:, None])) * Force                    # [vertices, vertices] Force <- Force
    # The stiffness against the masses; its three motions at zero are rigid and belong to the motor.
    values, fields = (~ends * bars(ends * Force)).eigh(weights, 3 + modes)  # [3 + modes] Scalar, [3 + modes, vertices] Force
    values, fields = values[3:], fields[3:]                               # [modes] Scalar, [modes, vertices] Force
    centre = (positions * masses).sum(axis=0) / masses.sum()              # [] Force
    rest = (positions - centre + mv.w).dual()                             # [vertices] Point
    inertia = ((rest & rest.commutator(Twist)) * masses).sum(axis=0)      # [] Forque <- Twist
    return Shape(rest, fields.dual(), values.square_root(), 1 / values, masses, edges, inertia)


def points(bodies: Bodies, shape: Shape) -> Point:
    """Every body's points in the world: its rest points moved by its modes, carried by its motor."""
    local = shape.rest + (shape.modes * bodies.amplitudes[..., None]).sum(axis=-2)  # [..., bodies, vertices] Point
    return bodies.motor[..., None] >> local                               # [..., bodies, vertices] Point


def modal_terms(bodies: Bodies, previous: Scalar, dt: float) -> tuple[Scalar, Scalar, Scalar]:
    """Each mode's compliance and residual over the step, with its implicit dashpot folded in, and
    how far it yields to a unit load: `[..., bodies, modes] Scalar` each."""
    damping = 2 * bodies.damping * bodies.frequencies * bodies.compliance / dt  # [..., bodies, modes] Scalar
    compliance = bodies.compliance / dt**2 / (1 + damping)                # [..., bodies, modes] Scalar
    residual = (bodies.amplitudes + damping * (bodies.amplitudes - previous)) / (1 + damping)  # [..., bodies, modes] Scalar
    return compliance, residual, 1 / (1 + compliance)


def coupling(bodies: Bodies, pins: Pins) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor, Direction]:
    """How every pin opens: for the bodies' twists `[..., pins, bodies] Direction <- Twist` and for
    their modes' amplitudes `[..., pins, bodies * modes] Direction`; the work a force at each pin does
    through each mode `[..., bodies * modes, pins] Scalar <- Force`; and the gap `[..., pins]
    Direction`."""
    count, modes = bodies.motor.shape[-1], bodies.amplitudes.shape[-1]
    motor = bodies.motor[..., pins.bodies]                                # [..., pins, ends] Motor
    local = pins.anchors + (pins.modes * bodies.amplitudes[..., pins.bodies, :]).sum(axis=-1)  # [..., pins, ends] Point
    signs = np.array([-1, 1])
    # An open twist measures how an anchor moves, the two ends taken from each other.
    motion = (motor >> local.commutator(Twist)) * signs                   # [..., pins, ends] Direction <- Twist
    shapes = (motor[..., None] >> pins.modes) * signs[:, None]            # [..., pins, ends, modes] Direction
    gap = ((motor >> local) * signs).sum(axis=-1)                         # [..., pins] Direction
    pin = np.broadcast_to(np.arange(len(pins.bodies))[:, None], pins.bodies.shape)  # [pins, ends]
    mode = pins.bodies[..., None] * modes + np.arange(modes)              # [pins, ends, modes]
    pin_mode = np.broadcast_to(pin[..., None], mode.shape)                # [pins, ends, modes]
    return (
        incidence(motion, pin, pins.bodies, (len(pins.bodies), count)),
        incidence(shapes, pin_mode, mode, (len(pins.bodies), count * modes)),
        incidence(Force & shapes, mode, pin_mode, (count * modes, len(pins.bodies))),
        gap,
    )


def project(bodies: Bodies, pins: Pins, previous: Scalar, lambdas: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """The bodies moved so that every pin closes, all solved together, and the modes' accumulated
    reactions `[..., bodies, modes] Scalar`."""
    compliance, residual, yielding = modal_terms(bodies, previous, dt)     # [..., bodies, modes] Scalar each
    motion, shapes, loads, gap = coupling(bodies, pins)
    batch = bodies.amplitudes.shape[:-2]
    mode_load = -(residual + compliance * lambdas)                        # [..., bodies, modes] Scalar
    free_step = (yielding * mode_load).reshape(batch + (-1,))             # [..., bodies * modes] Scalar
    # Each mode, eliminated, yields by its mobility; each pin's own compliance moves its anchors apart
    # along the force, its polar.
    mobility = spdiag((yielding * compliance).reshape(batch + (-1,)))        # [..., bodies * modes, bodies * modes] Scalar
    weld = spdiag(pins.compliance / dt**2 * Force.dual())                 # [pins, pins] Direction <- Force
    inertias = spdiag(bodies.inverse_inertia)                               # [..., bodies, bodies] Twist <- Forque
    system = motion(inertias(motion.adjugate())) + shapes * (mobility * loads) + weld  # [..., pins, pins] Direction <- Force
    forces = system.solve((-gap - shapes * free_step).cast(Direction))    # [..., pins] Force
    # Equal and opposite forces act through both the rigid and elastic responses.
    displacement = bodies.inverse_inertia(motion.adjugate()(forces))      # [..., bodies] Twist
    mode_forces = loads(forces).reshape(bodies.amplitudes.shape)          # [..., bodies, modes] Scalar
    modal_reaction = yielding * (mode_load - mode_forces)                  # [..., bodies, modes] Scalar
    moved = replace(
        bodies,
        motor=bodies.motor * (displacement * -0.5).exp(),
        amplitudes=bodies.amplitudes + mode_forces + modal_reaction,
    )
    return moved, lambdas + modal_reaction


def step(bodies: Bodies, pins: Pins, dt: float, gravity: Direction) -> Bodies:
    """The bodies one step on: predicted, their springs relaxed, their pins closed, and their rates
    read from how far they moved."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity pulls through each body's centre of mass, the origin of its frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses          # [..., bodies] Forque

    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    _, residual, yielding = modal_terms(bodies, previous.amplitudes, dt)  # [..., bodies, modes] Scalar each
    lambdas = -residual * yielding                                         # [..., bodies, modes] Scalar
    bodies, _ = project(replace(bodies, amplitudes=bodies.amplitudes + lambdas), pins, previous.amplitudes, lambdas, dt)
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )
