r"""Thin vortex filaments moving through an ideal fluid by their own bending and by each other's pull.

A filament is modelled as a closed polygon of vorticity in a fluid without viscosity that cannot be compressed. By the Biot–Savart law every part of a filament moves the fluid everywhere, and the fluid carries every other part along. The part from close to each point dominates and moves the point across the curve, perpendicular to the curve and to the way it bends, at a speed set by how sharply it bends. Keeping that part alone is called the smoke ring flow, or the local induction approximation; under it a circle flies along its axis without changing shape, and the surface a curve sweeps over time is called its Hashimoto surface, after Hoffmann's discrete flow for polygons of unit edges (2000). Keeping the whole law lets two rings, one behind the other, take turns passing through each other.

The filament's points and edges are intrinsic antivectors, and its circulation is an intrinsic scalar. The turn between the two edges at a vertex is the rotor `outgoing * ~incoming`, and the edges' commutator over the turn's scalar part plus the product of their lengths is the tangent of half the turning angle in the turn's plane; times the circulation it is a point, the vertex's step along the binormal, and on unit edges exactly Hoffmann's velocity. The polygon's vector area, the sum of `point & next_point` over its edges, is the filament's impulse, and the flow keeps it. Hoffmann's Bäcklund transformation steps the flow in discrete time by turning each offset about a quad's diagonal with a quaternion, `((edge - offset) + twist).normalized() >> offset`, and the whole Biot–Savart law sums `separations.commutator(segments)` over every segment of every filament.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                      # Scalar+
Vector = ga.gatype.vector()                                                      # Vector+
Rotor = ga.gatype.rotor()                                                        # Rotor+
# Points, and their differences, the edges and offsets, are antivectors, intrinsic.
Point = ga.gatype.antivector()                                                   # Antivector-
# The filament's circulation changes sign under a mirror, an intrinsic scalar.
Circulation = ga.gatype.antipseudoscalar()                                       # Antipseudoscalar-
# The Bäcklund transformation's parameter, added to the quad's diagonal, a point, is an intrinsic scalar too.
Parameter = ga.gatype.antipseudoscalar()                                         # Antipseudoscalar-
# A polygon's vector area, the join of its points, is an intrinsic vector, a patch.
Patch = ga.gatype.antibivector()                                                 # Antibivector-


# --- math -----------------------------------------------------------------------------
def edges(points: Point) -> Point:
    """The edges of closed polygons, edge k running from point k to point k + 1."""
    closed_polyline = concatenate([points, points[..., :1]], axis=-1)            # [...] Point-
    return closed_polyline[..., 1:] - closed_polyline[..., :-1]                  # [...] Point-


def velocity(points: Point, circulation: Circulation) -> Point:
    """The smoke ring flow's velocity at each vertex of closed polygonal filaments of the given circulation, twice the
    Cayley parameter of the turn at the vertex times the circulation, over the mean length of its two edges."""
    # The polygon closed at both ends, so that every vertex has an incoming and an outgoing side.
    wrapped = concatenate([points[..., -1:], points, points[..., :1]], axis=-1)  # [...] Point-
    sides = wrapped[..., 1:] - wrapped[..., :-1]                                 # [...] Point-
    incoming, outgoing = sides[..., :-1], sides[..., 1:]                         # [...] Point-, Point-
    lengths = outgoing.norm() * incoming.norm()                                  # [...] Scalar+
    # The commutator of the two edges is the bivector of the turn; over the lengths plus the edges paired,
    # it is the tangent of half the turning angle.
    binormal_step = 2 * incoming.commutator(outgoing) * circulation / (lengths + outgoing.dot(incoming))   # [...] Point-
    return binormal_step / ((outgoing.norm() + incoming.norm()) / 2)             # [...] Point-


def impulse(points: Point) -> Patch:
    """The vector area of closed polygons, twice over, the sum of each point joined with the next."""
    return (points & (points + edges(points))).sum(axis=-1)                      # [...] Patch-


def induced_velocity(points: Point, circulations: Circulation, core_radius: float) -> Point:
    """The velocity at every vertex of closed polygonal vortex filaments of the given circulations, induced by every
    segment of every filament, the commutator of the vertex's separation from the segment's middle with the segment
    over the softened cube of the separation."""
    n_rings, n_vertices = points.shape
    segments = edges(points)                                                     # [n_rings, n_vertices] Point-
    separations = points.reshape(-1)[:, None] - (points + segments / 2).reshape(-1)[None, :]   # [n_rings * n_vertices, n_rings * n_vertices] Point-
    soft_squares = separations.norm_squared() + core_radius**2                   # [n_rings * n_vertices, n_rings * n_vertices] Scalar+
    strengths = circulations[..., None].broadcast_to(points.shape).reshape(-1) / (4 * np.pi)   # [n_rings * n_vertices] Circulation-
    pulls = separations.commutator(segments.reshape(-1)) * strengths / (soft_squares * soft_squares.square_root())
    return pulls.sum(axis=1).reshape(points.shape)                               # [n_rings, n_vertices] Point-


def flow(points: Point, rate: Callable[[Point], Point], step: float) -> Iterator[Point]:
    """Yield the polygons of a flow with the given velocity at every classical Runge–Kutta step, without end."""
    while True:
        yield points
        first = rate(points)
        second = rate(points + first * (step / 2))
        third = rate(points + second * (step / 2))
        fourth = rate(points + third * step)
        points = points + (first + 2 * second + 2 * third + fourth) * (step / 6)


def next_offset(edge: Point, offset: Point, twist: Parameter) -> Point:
    """The next offset of a Bäcklund transformation across an edge, the current one turned about the diagonal
    `edge - offset` of the quad between them by the quaternion the diagonal and the parameter make together."""
    return ((edge - offset) + twist).normalized() >> offset                      # [...] Point-


def backlund(points: Point, start: Point, twist: Parameter) -> Point:
    """The Bäcklund transformation of open polygons with the given parameter and first offset, every later offset
    the one before folded across the edge between them."""
    sides = points[..., 1:] - points[..., :-1]                                   # [...] Point-

    # The offsets are folded one after another; a product of the folds' Lorentz rotors grows from edge to edge until
    # round-off swamps the offsets.
    def offsets() -> Iterator[Point]:
        offset = start
        yield offset
        for index in range(sides.shape[-1]):
            offset = next_offset(sides[..., index], offset, twist)
            yield offset

    return points + stack(list(offsets()), axis=-1)                              # [...] Point-


def periodic_offset(points: Point, twist: Parameter) -> Point:
    """The unit offset at the first vertex of closed polygons of unit edges that comes round to itself after a lap,
    minus the last edge carried once around by the lap's Lorentz transformation onto its attracting fixed point."""
    sides = edges(points)                                                        # [...] Point-
    # Every fold is a boost along its edge and a turn about it, the boost's generator the direction the edge's point
    # stands for, its rapidity and the turn's angle the real and imaginary parts of the logarithm of 1 - 2 i / twist.
    rapidity = (1 + 4 / twist.squared()).log() / 2                               # [] Scalar+
    angle = -(2 * twist.inverse()).arctan()                                      # [] Parameter-
    folds = (-(rapidity * sides.dual_inverse() + angle * sides) / 2).exp()       # [...] Multivector+
    lap = folds.cumprod(axis=-1)[..., -1]                                        # [...] Multivector+
    # A Lorentz transformation acts on paravectors by the sandwich with the reverse; the offset's null paravector is
    # one plus the direction it stands for.
    image = lap * (1 - sides[..., -1].dual_inverse()) * lap.reverse()            # [...] Multivector+
    # The paravector's bivector and volume parts are zero.
    return (image.cast(Vector) / image.cast(Scalar)).dual()                      # [...] Point-


def time_step(points: Point, twist: Parameter) -> Point:
    """The closed polygons of unit edges one step of the discrete-time smoke ring flow later, by the Bäcklund
    transformation with periodic offsets relabelled by one vertex, a time of about half the parameter."""
    closed_polyline = concatenate([points, points[..., :1]], axis=-1)            # [...] Point-
    return backlund(closed_polyline, periodic_offset(points, twist), twist)[..., 1:]   # [...] Point-


def turns(points: Point) -> Rotor:
    """The turn at every inner vertex of polygons of unit edges, the rotor from the incoming edge to the outgoing
    one, the square of the rotor that carries one onto the other."""
    sides = points[..., 1:] - points[..., :-1]                                   # [...] Point-
    return sides[..., 1:] * ~sides[..., :-1]                                     # [...] Rotor+
