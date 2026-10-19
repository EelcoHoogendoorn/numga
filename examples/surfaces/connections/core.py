"""Connections on curved surfaces, in the geometric algebra of three-dimensional space: parallel transport
and its holonomy, the connection Laplacian, trivial connections that design vector fields with chosen
singularities, and stripe patterns along those fields.

Parallel transport carries a tangent vector along a surface while keeping it tangent and turning it
no more than the surface itself bends. Between two neighbouring points with unit normals `n_tail`
and `n_head`, the shortest turn that carries the first normal onto the second is the square root of
their geometric product, `turn = (n_head * n_tail).square_root()`. That rotor turns in the plane of
the two normals and leaves the hinge line where their tangent planes meet untouched. Its action on
vectors, `turn >> Vector`, is the parallel transport map from the tail's tangent plane to the head's.
Chaining those turns around a closed loop of normals carries a tangent vector around the loop, and
when the loop closes the vector has turned in the starting tangent plane by the solid angle the
normals sweep out on the unit sphere. On the equator of a sphere the normal sweeps a flat great
circle of solid angle `2 * np.pi`, so the vector returns unchanged; on the latitude circle sixty
degrees from the pole the cap has solid angle `np.pi`, so the vector returns pointing the opposite way.

A connection on a triangle mesh is a rotor from each end of each edge to its head. Those rotors give
the covariant derivative, a sparse map of ambient vector fields `covariant: Vector[E] <- Vector[V]`
that turns each edge's tail vector into its head's tangent plane and subtracts it from the head's
vector. Weighting each edge by its cotangent ratio `H1` and composing with the adjoint gives the
connection Laplacian `L = covariant.adjoint()(H1 * covariant)`. Because each edge's rotor carries
the tail's normal onto the head's normal, `covariant` and `L` never mix tangent vectors with normal
vectors: any tangent field stays perpendicular to the vertex normals everywhere, with no local
two-dimensional coordinate charts or reference edges chosen at the vertices. With the surface's own
transport, one backward-Euler step `(M0 * Vector + time * L).solve(M0 * source)` diffuses a single
tangent vector across the surface along all paths at once, the vector heat method, and normalizing
the diffused vectors recovers parallel transport from the source to every vertex. Adding a stiff
weight along the normals lifts the normal bundle out of the low spectrum, so the lowest eigenfields of
`L` against the vertex areas `M0 * Vector` are the smoothest global tangent vector fields on the
surface. Turning every vector a quarter turn in its own tangent plane, `u | normals.dual()`, commutes
with every edge transport, so every tangent eigenvalue comes in an equal pair.

In the complex line-bundle notation of discrete differential geometry, each vertex chooses a local
orthonormal basis of its tangent plane, each edge rotor reads as a unit complex transition factor
$e^{i\\theta_{ij}}$ between the two frames' reference directions, and the connection Laplacian reads
as a Hermitian matrix of complex numbers coupled by those phases.

A tangent vector field on a closed surface must vanish or spin somewhere, unless the surface is a
torus: the indices of its singularities, how often the field turns as one goes once around each,
add up to the surface's Euler characteristic. Choosing where the singularities sit and how they turn,
and finding the smoothest field that has exactly those, is the design problem of Crane, Desbrun and
Schröder, "Trivial connections on discrete surfaces" (2010). Carried around the three edges of a
face, a tangent vector comes back turned in its tangent plane: the product of the three rotors turns
about the normal by the face's holonomy, the curvature inside the face, and the holonomies of all
faces add up to `2 pi` times the Euler characteristic. A whole turn of a rotor is minus one, so a
loop's angle is read from the rotor's square, the turn itself, by the plane of its logarithm:
`(loop * loop).log() | normal.dual()`, an angle in (-pi, pi].

A connection is trivial when transport around every loop returns every vector unchanged; then
carrying one vector out from anywhere gives a field with no turning at all along the surface. The
surface's own transport is not trivial, because the surface is curved. Each edge gets an extra turn
`angle` in the head's tangent plane, `(n_head.dual() * (-angle / 2)).exp()`, and since every turn in a
tangent plane commutes with the transport into that plane, a loop's holonomy grows by exactly the sum
of the extra turns along it, signed by direction: around the faces, the circulation `d1(angles)`.
Asking every face to turn by `2 pi` times its chosen index and every non-contractible loop, a
generator of the surface's handles, by nothing is a linear system for the edge turns, solved with the
smallest turns, which also turn the field the least. The system can be met only when the indices
add up to the Euler characteristic: around every face the circulation of the extra turns adds up to
nothing, so Gauss–Bonnet and Poincaré–Hopf become the condition for a solution. The field itself is
the null space of the connection Laplacian of the adjusted transport: two fields, one a quarter turn
of the other in every tangent plane, with energy zero, unit length everywhere, and every edge's
adjusted transport carrying one end onto the other.

In the notation of discrete exterior calculus, the extra turns are a dual 1-form or a primal 1-form
$\\theta$ on edges, the constraints read $d\\theta = 2\\pi k - K$ on faces and $\\oint_\\gamma \\theta =
-\\Omega_\\gamma$ around the generators $\\gamma$, with $K$ the Gaussian curvature and $\\Omega_\\gamma$ the
holonomy of the Levi-Civita connection, and the smallest solution is $\\theta = d^{+}(2\\pi k - K)$.

A stripe pattern is a field of evenly spaced stripes running across a surface at right angles to a
chosen direction field: the stripes' phase advances along the field at a chosen frequency. Where the
field has a singularity, or where the surface's curvature leaves no room for evenly spaced stripes,
the stripes fork. Knöppel, Crane, Pinkall and Schröder, "Stripe patterns on surfaces" (2015), find
the phase as the smoothest section of a connection that advances by the frequency along the field.
Across each edge, the phase should advance by the frequency times the field's component along the
edge: the edge's target. The phase is carried by a tangent vector at every vertex, measured as its
angle from the direction field. A direction field designed by a trivial connection is parallel for that
connection: carrying it across an edge with the connection's transport lands it exactly on the field
at the head. So a tangent vector carried by the same transport, turned further by the edge's target in
the head's tangent plane, keeps its angle from the field plus the target: its angle from the field
advances by exactly the target. The stripe field is the smoothest tangent field of that connection,
the trivial connection's turns with the targets added: the lowest eigenfield of its connection
Laplacian, with one more turn on every edge. It comes as a pair, a field and its quarter turn, two
stripe patterns shifted by a quarter of a stripe; the stripes are the sine of the angle from the stripe
field to the direction field. Where the targets around a face add up to more than the face's
singularity allows, no phase can follow them all, and the lowest eigenfield gives up there: the
stripes fork.

In the complex notation of the paper, the phase is a section $\\psi$ of the trivial line bundle, the
connection is $e^{i \\omega_{ij}}$ with $\\omega_{ij} = \\nu \\langle X, e_{ij} \\rangle$ the target of
the edge $e_{ij}$ for the field $X$ and frequency $\\nu$, and the stripes are the smallest eigenvector of
$\\sum_{ij} w_{ij} |\\psi_j - e^{i \\omega_{ij}} \\psi_i|^2$ against the vertex areas.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import breadth_first_order

from numga import concatenate, stack
from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.mesh", ga)
Mesh, as_ga_sparse, as_scalar, context = mesh_module.Mesh, mesh_module.as_ga_sparse, mesh_module.as_scalar, mesh_module.ctx
mv = context.multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Point = mesh_module.Point
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()


# --- math -----------------------------------------------------------------------------
def transport_step(normal_tail: Vector, normal_head: Vector) -> Rotor:
    """The shortest rotor carrying `normal_tail` onto `normal_head`: parallel transport between their
    tangent planes."""
    return (normal_head * normal_tail).square_root()                            # [...] Rotor


def transported(normals: Vector, start: Vector) -> Iterator[Vector]:
    """Parallel transport of a tangent vector `start` along a sequence of unit normals
    `[steps + 1, ...] Vector`: the vector at each of them."""
    yield start
    for tail, head in zip(normals[:-1], normals[1:]):
        start = transport_step(tail, head) >> start                             # [...] Vector
        yield start


def turn(rotors: Rotor, normals: Vector) -> Scalar:
    """The angle by which each rotor turns tangent vectors about its unit normal, in (-pi, pi]: read
    from the rotor's square, the turn itself, so that whole turns read as nothing."""
    return (rotors * rotors).log() | normals.dual()                              # [...] Scalar


def loop_turn(normals: Vector) -> Scalar:
    """The holonomy of transport around closed loops of unit normals `[..., steps]`: the turn of the
    running product of the transport rotors, each later step multiplying from the left."""
    following = concatenate([normals[..., 1:], normals[..., :1]], axis=-1)      # [..., steps] Vector
    steps = transport_step(normals, following)                                   # [..., steps] Rotor
    around = steps.cumprod(axis=-1)[..., -1]                                     # [...] Rotor
    return turn(around, normals[..., 0])                                         # [...] Scalar


def adjusted_transport(mesh: Mesh, angles: Scalar) -> Rotor:
    """The rotors from each end of each edge to its head `[E, 2]`: the surface's own transport from
    the tail followed by the edge's extra turn `Scalar[E]` in the head's tangent plane, and none at the
    head. With no extra turns, this is the surface's own parallel transport."""
    normals = mesh.vertex_normals.batch()                                         # [V] Vector
    tails, heads = normals[mesh.edges[:, 0]], normals[mesh.edges[:, 1]]           # [E] Vector each
    extra = (heads.dual() * (-angles.batch() / 2)).exp()                         # [E] Rotor
    return stack([extra * transport_step(tails, heads), mv.rotor().broadcast_to(extra.shape)], axis=-1)


def connection(mesh: Mesh, transports: Rotor, weights: Scalar) -> tuple[SparseExtensor, SparseExtensor]:
    """The sparse covariant derivative of the rotors `[E, 2]` from each end of each edge to its head,
    and its connection Laplacian on vertex vector fields with each edge weighted by `Scalar[E]`."""
    covariant = as_ga_sparse(mesh.edges, (transports >> Vector) * np.array([-1, 1]))   # [E, V] Vector <- Vector
    laplacian = covariant.adjoint()(spdiag(weights) * covariant)                # [V, V] Vector <- Vector
    return covariant, laplacian


def low_modes(laplacian: SparseExtensor, normals: Vector, areas: Scalar, normal_stiffness: float, count: int
              ) -> tuple[Scalar, Vector]:
    """The `count` smoothest vector fields of a connection Laplacian against the vertex areas
    `Scalar[V]`, and their Dirichlet energies per unit area, with the normal bundle along the unit
    normals `Vector[V]` lifted out of the low spectrum by `normal_stiffness`."""
    unit_normals = normals.batch()                                               # [V] Vector
    normal_penalty = spdiag((unit_normals * (unit_normals | Vector)).field()) * normal_stiffness   # [V, V] Vector <- Vector
    return (laplacian + normal_penalty).eigh(spdiag(areas) * Vector, count)     # [count] Scalar, [count] Vector[V]


def heat_transport(laplacian: SparseExtensor, areas: Scalar, source: Vector, step_time: float, steps: int
                   ) -> Iterator[tuple[Vector, Vector]]:
    """Parallel transport of a tangent field `source` across the mesh by repeated backward-Euler
    vector heat steps of duration `step_time` with the connection Laplacian of the surface's own
    transport and the vertex areas `Scalar[V]`: yields the diffused field and the unit-normalized
    field at each step."""
    area_weights = spdiag(areas)                                                # [V, V] Scalar
    step_operator = area_weights * Vector + step_time * laplacian               # [V, V] Vector <- Vector
    field = source                                                              # Vector[V]
    for _ in range(steps):
        field = step_operator.solve(area_weights * field)                       # Vector[V]
        yield field, field / field.norm()                                       # Vector[V], Vector[V]


def trivial_connection(mesh: Mesh, indices: np.ndarray, generators: list[np.ndarray]) -> tuple[Scalar, SparseExtensor, Scalar]:
    """The smallest extra turns on the edges `Scalar[E]` that make every face turn by `2 pi` times its
    index `[F]` and every generator, a closed loop of vertices, by nothing; with the circulation of
    edge turns around the faces and generators and its target, for checking."""
    normals = mesh.vertex_normals.batch()                                         # [V] Vector
    curvature = loop_turn(normals[mesh.faces])                                    # [F] Scalar
    # Around each generator, the extra turns undo the surface's own holonomy.
    undone = [-loop_turn(normals[loop[None]]) for loop in generators]              # [1] Scalar each
    circulation = loops(mesh, generators)                                         # [F + G, E] Scalar <- Scalar
    target = concatenate([2 * np.pi * as_scalar(indices) - curvature, *undone]).field()   # Scalar[F + G]
    return circulation.lstsq(target), circulation, target


def targets(mesh: Mesh, field: Vector, frequency: float) -> Scalar:
    """The phase advance across every edge `Scalar[E]`: the frequency times the field paired with the
    edge's displacement, the field averaged over the edge's two ends."""
    ends = field.batch()[mesh.edges]                                             # [E, 2] Vector
    return ((ends.sum(axis=-1) / 2 & mesh.edge_displacements.batch()) * frequency).field()


def quarter_turn(mesh: Mesh, field: Vector) -> Vector:
    """The field turned a quarter turn about every vertex normal `Vector[V]`: parallel wherever the field is,
    since a turn in the tangent plane commutes with the transport into it."""
    return (field.batch() | mesh.vertex_normals.batch().dual()).field()          # Vector[V]


def carried_out(mesh: Mesh, angles: Scalar) -> Vector:
    """A unit tangent vector at every vertex `Vector[V]`, carried out from the first vertex by the transport
    of the given turns along a breadth-first tree: one sparse solve, each tree edge asking its head to be
    its tail carried across it, and the first vertex asking to point across its first edge. Where the turns
    make a trivial connection whose faces turn by quarter turns, this is a cross field: carried across
    any edge, the tail's vector lands on the head's turned by a whole number of quarter turns."""
    children, tree = spanning_tree(mesh)
    count = len(children) + 1
    turns = adjusted_transport(mesh, angles)[tree]                               # [V - 1, 2] Rotor
    along_tree = SparseExtensor.from_indices((turns >> Vector) * np.array([-1, 1]), children[:, None], mesh.edges[tree], (count, count))
    first = SparseExtensor.from_indices(as_scalar([1.0]), np.array([0]), np.array([0]), (count, count)) * Vector
    normal = mesh.vertex_normals.batch()[0]                                      # Vector
    # the first edge met with the normal: a tangent direction across it
    seed = (mesh.edge_displacements.batch()[0] | normal).normalized()          # Vector
    return (along_tree + first).solve((seed * as_scalar(np.arange(count) == 0)).field())


def cross_stripes(mesh: Mesh, angles: Scalar, cross: Vector, regular: Scalar, frequency: float, normal_stiffness: float
                  ) -> tuple[Vector, Vector, Scalar]:
    """Two families of stripes across the two lines of a cross field `Vector[V]` parallel up to quarter
    turns for the connection of the given turns: the lines `[V, 2] Vector`, the stripe field of each
    `[V, 2] Vector`, and the targets across every edge for each line at its head `[E, 2] Scalar`.

    Carried across an edge, each of the tail's lines lands on one of the head's lines, pointing along it
    or against it. A stripe phase along a line reads backwards along the line reversed, so the stripe
    field carried onto a line it lands against is mirrored: its component along that line reversed, the
    stripes' angle from the line negated. The two families are one field over the cross's two lines at
    every vertex, coupled across the edges where the lines trade places, around the cross's singularities.
    Each face adds its half-cotangents to its edges' weights only where it is regular `Scalar[F]`."""
    count = mesh.vertex_areas.batch().shape[0]
    forward = adjusted_transport(mesh, angles)[:, 0]                             # [E] Rotor
    lines = stack([cross.batch(), quarter_turn(mesh, cross).batch()], axis=-1)   # [V, 2] Vector
    tails, heads = mesh.edges[:, 0], mesh.edges[:, 1]
    ahead = lines[heads]                                                         # [E, 2] Vector
    # One where the tail's line b lands along the head's line a, minus one where it lands against it, and
    # zero where it lands across it.
    match = ahead[:, :, None] | (forward[:, None] >> lines[tails])[:, None, :]  # [E, 2, 2] Scalar
    behind = (match * lines[tails][:, None, :]).sum(axis=-1)                     # [E, 2] Vector
    advance = (((ahead + behind) / 2) & mesh.edge_displacements.batch()[:, None]) * frequency   # [E, 2] Scalar
    turn = (mesh.vertex_normals.batch()[heads][:, None].dual() * (-advance / 2)).exp()     # [E, 2] Rotor
    # Carried, then mirrored onto the head's line where it lands against it.
    moved = forward[:, None, None] >> Vector                                     # [E, 1, 1] Vector <- Vector
    along = ahead[:, :, None]                                                    # [E, 2, 1] Vector
    landed = match * match * moved + (match - match * match) * (along * (along | moved))   # [E, 2, 2] Vector <- Vector
    # The head's stripe field turned back by the target, against the tail's carried and mirrored.
    cells = stack([turn.reverse() >> Vector, -landed[..., 0], -landed[..., 1]], axis=-1)    # [E, 2, 3] Vector <- Vector
    rows = np.arange(len(heads))[:, None, None] * 2 + np.arange(2)[None, :, None]           # [E, 2, 1]
    columns = np.stack(np.broadcast_arrays(heads[:, None] * 2 + np.arange(2), tails[:, None] * 2, tails[:, None] * 2 + 1), axis=-1)
    covariant = SparseExtensor.from_indices(cells, rows, columns, (2 * len(heads), 2 * count))   # [2E, 2V] Vector <- Vector
    weights = mesh.weighted_edge_ratio(regular).batch()                          # [E] Scalar
    vertex = np.arange(2 * count) // 2                                           # each line's vertex
    laplacian = covariant.adjoint()(spdiag(weights[np.arange(2 * len(heads)) // 2].field()) * covariant)   # [2V, 2V]
    _, stripe_fields = low_modes(laplacian, mesh.vertex_normals.batch()[vertex].field(),
                                 mesh.vertex_areas.batch()[vertex].field(), normal_stiffness, 1)
    stripe = stripe_fields.batch()[0].reshape(count, 2)                          # [V, 2] Vector
    return lines, stripe / stripe.norm(), advance


# --- plumbing -------------------------------------------------------------------------
def latitude_rings(colatitudes: np.ndarray, steps: int) -> tuple[Vector, Vector]:
    """Closed latitude circles on the unit sphere at the given colatitudes `[rings]` over `steps`
    segments, and the unit meridian tangent vector at the start of each ring."""
    longitudes = np.linspace(0.0, 2 * np.pi, steps + 1)
    tilt = (mv.zx * (-colatitudes / 2)).exp()                                   # [rings] Rotor
    spin = (mv.xy * (-longitudes[:, None] / 2)).exp()                           # [steps + 1, 1] Rotor
    normals = spin >> (tilt >> mv.z)                                            # [steps + 1, rings] Vector
    starts = tilt >> mv.x                                                       # [rings] Vector
    return normals, starts


def loops(mesh: Mesh, generators: list[np.ndarray]) -> SparseExtensor:
    """The circulation of edge values around each face, counter-clockwise, then around each generator,
    a closed loop of vertices `[steps]`, in its own order: `[F + G, E]` Scalar <- Scalar."""
    rows, columns, signs = [np.repeat(np.arange(len(mesh.faces)), 3)], [mesh.face_edges.reshape(-1)], [mesh.face_edge_orientation.reshape(-1)]
    keys = mesh.edges[:, 0] * len(mesh.edges) + mesh.edges[:, 1]                 # [E], sorted
    for row, loop in enumerate(generators, start=len(mesh.faces)):
        steps = np.stack([loop, np.roll(loop, -1)], axis=-1)                     # [steps, 2]
        low, high = steps.min(axis=-1), steps.max(axis=-1)
        rows.append(np.full(len(loop), row))
        columns.append(np.searchsorted(keys, low * len(mesh.edges) + high))
        signs.append(np.where(steps[:, 0] < steps[:, 1], 1.0, -1.0))
    shape = (len(mesh.faces) + len(generators), len(mesh.edges))
    return SparseExtensor.from_indices(as_scalar(np.concatenate(signs)), np.concatenate(rows), np.concatenate(columns), shape) * Scalar


def faces_around(mesh: Mesh, vertex: int) -> int:
    """A face with the given vertex as a corner."""
    return int(np.flatnonzero((mesh.faces == vertex).any(axis=-1))[0])


def spanning_tree(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Every vertex but the first `[V - 1]`, and the edge joining each to its parent `[V - 1]`, in a
    breadth-first tree from the first vertex."""
    count = mesh.edges.max() + 1
    graph = coo_matrix((np.ones(len(mesh.edges)), (mesh.edges[:, 0], mesh.edges[:, 1])), shape=(count, count))
    order, parents = breadth_first_order(graph.tocsr(), 0, directed=False)
    children = order[1:]
    low, high = np.minimum(children, parents[children]), np.maximum(children, parents[children])
    return children, np.searchsorted(mesh.edges[:, 0] * count + mesh.edges[:, 1], low * count + high)
