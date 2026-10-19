"""The smoke ring flow of a closed polygon and the Hashimoto surface it sweeps, and vortex rings moving
each other by the Biot–Savart law, in the geometric algebra of three-dimensional space.

A thin vortex filament, the core of a smoke ring, moves in the direction its own curvature points
across, by an amount set by how sharply it bends: each point of the curve moves along the binormal, the
direction perpendicular to the curve and to the way it turns, at a speed equal to the curvature. This
is the local induction approximation of a vortex filament, the Hashimoto or smoke ring flow, and a
circle under it flies along its axis without changing shape. Hoffmann, "Discrete Hashimoto surfaces and
a doubly discrete smoke ring flow" (2000), gives it for polygons with edges of unit length, as a flow
that keeps every edge's length and moves discrete elastic curves rigidly; the surface the curve sweeps
over time is its Hashimoto surface.

At a vertex between the incoming edge `S_prev` and the outgoing edge `S`, the product `S * S_prev` is
the turn from one edge to the next, `|S| |S_prev| + S | S_prev` its scalar part and `S ^ S_prev` its
plane. Their ratio is the tangent of half the turning angle times the unit plane, the Cayley parameter
of the turn. Its dual is the binormal, an intrinsic vector. Multiplying by the filament's circulation,
an intrinsic scalar, gives an extrinsic vector. Twice that, over the mean length of the two edges, is
the vertex's velocity: on unit edges exactly Hoffmann's
`2 * (S ^ S_prev).dual() * circulation / (1 + S | S_prev)`, and in the limit of short edges the curvature
times the binormal. The velocity is perpendicular to both edges
at its vertex, so every edge keeps its length, whatever the lengths are; the vector area of the polygon,
the sum of `point ^ next_point` over its edges, is the filament's impulse, which the flow keeps too.

The smoke ring flow is an approximation to Euler flow, the flow of an ideal fluid, without viscosity and
incompressible. In Euler flow, vorticity gathered in a thin tube along a curve moves the fluid
everywhere by the Biot–Savart law, so every part of the filament moves every other part, and every
filament moves every other filament. The part of that integral from close to each point dominates, by
a factor growing with the log of the curve's radius of curvature over the tube's radius; it is the
curvature times the binormal, and keeping it alone gives the smoke ring flow, with time rescaled by that
factor. Kept whole, the integral over straight segments of a polygonal filament gives, at a point, the
dual of the plane of each segment and the point's separation from the segment's middle, over the cube of
the separation, softened by a core radius so that it stays finite on the filament itself: the vortex
filament of Rosenhead and Moore. Two coaxial rings moved this way leapfrog: the rear ring shrinks,
speeds up and passes through the front one, which widens and slows, and then they change places.

Time can be made discrete too, by Hoffmann's Bäcklund transformation. A new polygon `points + offsets`,
every offset of one length and every new edge again of unit length, follows from the old one by folding:
across an edge `S`, the next offset is the current one `v` turned about the diagonal `S - v` of the quad
between them, `((S - v - twist) * unit_volume).normalized() >> v`. A vector plus a volume, times the
pseudoscalar, is a scalar plus a bivector: a rotor about the vector, whose sandwich is the turn; the
pseudoscalar that no motion changes keeps the sandwich free of any handedness; the transformation's
parameter enters as the volume `twist`, a multiple of the pseudoscalar, and sets how far the fold turns,
by twice the angle whose tangent is the diagonal's length over the parameter. The turn keeps each new
edge's part along the diagonal, so the new edge is as long as the old.

On the sphere of unit offsets, folding across a unit edge keeps the edge's two directions fixed, and
turns and scales the sphere around them alike for every edge: it is a Lorentz transformation of the
sphere, a boost along the edge and a turn about it, the sphere taken as the sphere of light rays of
spacetime. In the algebra of space such a transformation acts on paravectors, a scalar plus a vector,
by the sandwich with the reverse, and the null paravector `1 + v` stands for the offset `v`; across an
edge `S` it is `exp(-log(1 + 2 / twist) * S / 2)`, its rapidity and its angle the real and the volume
part of one logarithm. Folding once around a closed polygon is the product of these; the boosts in a
long product grow from edge to edge until round-off swamps the offsets, so the offsets are folded one
after another, and the product serves where a whole lap is meant. For a closed polygon the offsets must come round to where they started: carried
once around the lap, any start lands on the lap's attracting fixed point. With offsets near minus the
edge before, of unit length, the new polygon lies one vertex back along the old one, moved by the smoke
ring flow over a time of about half the parameter: relabelled by one vertex, the transformation is a
time step that keeps every edge exactly of unit length, and moves a discrete elastic curve, a ring among
them, exactly rigidly.

One transformation of a straight line makes a soliton, Hasimoto's: a loop that runs along the line and
turns about it without changing shape. Every fold across a straight line keeps its two directions fixed,
the forward one repelling offsets and the backward one attracting them; an offset leaning just off the
forward direction crosses over to the backward one, and the new polygon loops where it crosses, within
one offset of the line. Time steps from minus the first edge carry the loop along, and through it the
tangent turns once around: the angles of the turns at the vertices, the outgoing edge times the incoming
one, add up to a full turn.

In the notation of the paper the flow reads $\\dot\\gamma_k = 2\\, S_k \\times S_{k-1} / (1 +
\\langle S_k, S_{k-1}\\rangle)$, and in the smooth limit $\\dot\\gamma = \\kappa B$, the Frenet curvature
times the binormal; the Biot–Savart law reads $u(x) = \\frac{\\Gamma}{4\\pi} \\oint \\frac{d\\ell \\times
(x - y)}{(|x - y|^2 + \\delta^2)^{3/2}}$ for a filament of circulation $\\Gamma$ and core radius $\\delta$.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
Pseudoscalar = ga.gatype.pseudoscalar()
# The filament's circulation changes sign under a mirror: an intrinsic scalar.
Circulation = Scalar.derive.flipped_orientation
# The unit volume that no motion changes, intrinsic: a product with it is a complement.
unit_volume = mv.antiscalar([1.0])


# --- math -----------------------------------------------------------------------------
def edges(points: Vector) -> Vector:
    """The edges of closed polygons `[..., n]`, edge k from point k to point k + 1."""
    return concatenate([points[..., 1:], points[..., :1]], axis=-1) - points    # [..., n] Vector


def velocity(points: Vector, circulation: Circulation) -> Vector:
    """The smoke ring flow's velocity at each vertex `[..., n]` of a filament of the given circulation:
    twice the dual of the turn's Cayley parameter, times the circulation, over the mean edge length."""
    outgoing = edges(points)                                                     # [..., n] Vector
    incoming = concatenate([outgoing[..., -1:], outgoing[..., :-1]], axis=-1)    # [..., n] Vector
    lengths = outgoing.norm() * incoming.norm()                                  # [..., n] Scalar
    halving = 2 * (outgoing ^ incoming).dual() * circulation / (lengths + (outgoing | incoming))   # [..., n] Vector
    return halving / ((outgoing.norm() + incoming.norm()) / 2)                   # [..., n] Vector


def impulse(points: Vector) -> Bivector:
    """The vector area of closed polygons, twice over: the sum of each point wedged with the next."""
    return (points ^ (points + edges(points))).sum(axis=-1)                     # [...] Bivector


def induced(points: Vector, circulations: Circulation, core_radius: float) -> Vector:
    """The velocity at every vertex of closed polygonal vortex filaments `[rings, n]`, of the given
    circulations `[rings]`, induced by every segment of every filament: the dual of the plane of the
    segment and the vertex's separation from the segment's middle, over the softened cube of the
    separation."""
    segments = edges(points)                                                     # [rings, n] Vector
    separations = points.reshape(-1)[:, None] - (points + segments / 2).reshape(-1)[None, :]   # [rings n, rings n] Vector
    softened = separations.norm_squared() + core_radius**2                       # [rings n, rings n] Scalar
    strengths = circulations[..., None].broadcast_to(points.shape).reshape(-1) / (4 * np.pi)   # [rings n] Circulation
    pulls = (segments.reshape(-1) ^ separations).dual() * strengths / (softened * softened.square_root())
    return pulls.sum(axis=1).reshape(points.shape)                               # [rings, n] Vector


def flow(points: Vector, rate: Callable[[Vector], Vector], step: float) -> Iterator[Vector]:
    """A flow of polygons with the given velocity, by classical Runge–Kutta steps of duration `step`: the
    polygons at every step, without end."""
    while True:
        yield points
        first = rate(points)
        second = rate(points + first * (step / 2))
        third = rate(points + second * (step / 2))
        fourth = rate(points + third * step)
        points = points + (first + 2 * second + 2 * third + fourth) * (step / 6)


def fold(edge: Vector, offset: Vector, twist: Pseudoscalar) -> Vector:
    """The next offset of a Bäcklund transformation across an edge: the current one turned about the
    diagonal `edge - offset` of the quad between them, by the rotor that a vector plus a volume is times
    the pseudoscalar."""
    return ((edge - offset - twist) * unit_volume).normalized() >> offset        # [...] Vector


def backlund(points: Vector, start: Vector, twist: Pseudoscalar) -> Vector:
    """One Bäcklund transformation of a polygon `[..., n]` with the parameter as the oriented volume
    `twist`, from the offset at its first vertex: every later offset is the one before, folded across the
    edge between them."""
    sides = points[..., 1:] - points[..., :-1]                                   # [..., n - 1] Vector

    def offsets() -> Iterator[Vector]:
        offset = start
        yield offset
        for index in range(sides.shape[-1]):
            offset = fold(sides[..., index], offset, twist)
            yield offset

    return points + stack(list(offsets()), axis=-1)                              # [..., n] Vector


def periodic(points: Vector, twist: Pseudoscalar) -> Vector:
    """The unit offset at the first vertex of closed polygons of unit edges `[..., n]` that comes round to
    itself after a lap: minus the last edge, carried once around by the lap's Lorentz transformation,
    which lands it on the lap's attracting fixed point."""
    sides = edges(points)                                                        # [..., n] Vector
    # The boost and turn of every fold as one logarithm.
    folds = (-(1 + 2 * twist.inverse()).log() * sides / 2).exp()                 # [..., n] Lorentz rotor
    lap = folds.cumprod(axis=-1)[..., -1]                                        # [...] Lorentz rotor
    image = lap * (1 - sides[..., -1]) * lap.reverse()                           # [...] null paravector
    # The paravector's bivector and volume parts are zero.
    return image.cast(Vector) / image.cast(Scalar)                               # [...] Vector


def advance(points: Vector, twist: Pseudoscalar) -> Vector:
    """One step of the discrete-time smoke ring flow of closed polygons of unit edges `[..., n]`: the
    Bäcklund transformation with periodic offsets, relabelled by one vertex, a time of about half the
    parameter along the flow."""
    return backlund(closed(points), periodic(points, twist), twist)[..., 1:]     # [..., n] Vector


def turns(points: Vector) -> Rotor:
    """The turn at every inner vertex of polygons of unit edges `[..., n]`: the rotor from the incoming
    edge to the outgoing one, the square of the rotor that carries one onto the other."""
    sides = points[..., 1:] - points[..., :-1]                                   # [..., n - 1] Vector
    return sides[..., 1:] * sides[..., :-1]                                      # [..., n - 2] Rotor


# --- plumbing -------------------------------------------------------------------------
def closed(points: Vector) -> Vector:
    """Closed polygons `[..., n]` as polylines `[..., n + 1]`: the first vertex again at the end."""
    return concatenate([points, points[..., :1]], axis=-1)                     # [..., n + 1] Vector


def ring(count: int) -> Vector:
    """A regular polygon of `count` unit edges around the z axis."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return ((mv.xy * (-angles / 2)).exp() >> mv.x) / (2 * np.sin(np.pi / count))   # [count] Vector


def coaxial_rings(radius: float, count: int, heights: np.ndarray) -> Vector:
    """Regular polygons of `count` vertices and the given radius around the z axis, one at each height
    `[rings]` along it."""
    return (ring(count) * (radius * 2 * np.sin(np.pi / count)))[None, :] + mv.z * heights[:, None]   # [rings, count] Vector


def wavy_ring(count: int, mode: int, tilt: float) -> Vector:
    """A closed polygon of `count` unit edges around the z axis, each edge tilted out of the plane by
    `tilt` times the sine of `mode` times its angle around the axis: a ring with a standing wave on it.
    The tilts' sines and their cosines have no component going once around the axis, so the edges close."""
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    around = (mv.xy * (-angles / 2)).exp()                                      # [count] Rotor
    lifted = (mv.yz * (-tilt * np.sin(mode * angles) / 2)).exp() >> mv.y        # [count] Vector: before turning around
    steps = around >> lifted                                                    # [count] Vector
    return concatenate([mv.vector([[0, 0, 0]]), steps[:-1]]).cumsum(axis=0)    # [count] Vector


def torus_knot(around_axis: int, around_tube: int, major_radius: float, minor_radius: float, count: int) -> Vector:
    """A closed polygon of `count` points on a torus knot, winding `around_axis` times around the z axis
    and `around_tube` times around the tube of the given radii."""
    turns = np.linspace(0, 2 * np.pi, count, endpoint=False)
    tube = (mv.zx * (-around_tube * turns / 2)).exp() >> (mv.x * minor_radius)   # [count] Vector
    return (mv.xy * (-around_axis * turns / 2)).exp() >> (mv.x * major_radius + tube)   # [count] Vector
