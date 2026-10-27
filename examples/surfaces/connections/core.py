"""Connections on curved surfaces, in the geometric algebra of three-dimensional space: parallel transport
and its holonomy, the connection Laplacian, trivial connections that design vector fields with chosen
singularities, and stripe patterns along those fields.

Parallel transport carries a tangent vector along a surface while keeping it tangent and turning it
no more than the surface itself bends. Between two neighbouring points with unit normals `n_tail`
and `n_head`, the shortest turn that carries the first normal onto the second is the square root of
their geometric product, `turn = (n_head * n_tail).square_root()`. That rotor turns in the plane of
the two normals and leaves the hinge line where their tangent planes meet untouched. Tangent vectors are
directions, antivectors, intrinsic like all geometry, and the rotor's action on them, `turn >> Point`, is the
parallel transport map from the tail's tangent plane to the head's. The normals are the unit patches of the
tangent planes, joins of edges, and a normal's dual is the plane a tangent vector turns in.
Chaining those turns around a closed loop of normals carries a tangent vector around the loop, and
when the loop closes the vector has turned in the starting tangent plane by the solid angle the
normals sweep out on the unit sphere. On the equator of a sphere the normal sweeps a flat great
circle of solid angle `2 * np.pi`, so the vector returns unchanged; on the latitude circle sixty
degrees from the pole the cap has solid angle `np.pi`, so the vector returns pointing the opposite way.

A connection on a triangle mesh is a rotor from each end of each edge to its head. Those rotors give
the covariant derivative, a sparse map of ambient direction fields `(Point- <- Point-)[edges <- vertices]`
that turns each edge's tail vector into its head's tangent plane and subtracts it from the head's
vector. Weighting each edge by its cotangent ratio `H1` and composing with the adjoint gives the
connection Laplacian `L = covariant.adjoint()(H1 * covariant)`. Because each edge's rotor carries
the tail's normal onto the head's normal, `covariant` and `L` never mix tangent vectors with normal
vectors: any tangent field stays perpendicular to the vertex normals everywhere, with no local
two-dimensional coordinate charts or reference edges chosen at the vertices. With the surface's own
transport, one backward-Euler step `(M0 * Point + time * L).solve(M0 * source)` diffuses a single
tangent vector across the surface along all paths at once, the vector heat method, and normalizing
the diffused vectors recovers parallel transport from the source to every vertex. Adding a stiff
weight along the normals lifts the normal bundle out of the low spectrum, so the lowest eigenfields of
`L` against the vertex areas `M0 * Point` are the smoothest global tangent vector fields on the
surface. Turning every vector a quarter turn in its own tangent plane, its commutator with the plane,
`vector.commutator(normals.dual())`, commutes with every edge transport, so every tangent eigenvalue comes in an equal pair.

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

from numga import concatenate, stack
from numga.algebras import VGA3D as ga
from numga.sparse import SparseExtensor, spdiag
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_ga_sparse, as_scalar, context = mesh_module.Mesh, mesh_module.as_ga_sparse, mesh_module.as_scalar, mesh_module.ctx
mv = context.multivector
Scalar = ga.gatype.scalar()                         # Scalar+
# Positions and tangent vectors are points and directions, antivectors.
Point = mesh_module.Point                           # Antivector-
# The unit normals: the patches of the tangent planes, joins of two edges.
Vector = ga.gatype.vector()                         # Vector-
Rotor = ga.gatype.rotor()                           # Rotor+


# --- math -----------------------------------------------------------------------------
def vertex_normals(mesh: Mesh, vertices: Point) -> Vector:
    """The unit normal at each vertex, `Vector-[vertices]`, is the sum of its faces' normals, each the join of two
    of the face's edges, twice the face's area large."""
    edges = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point-[faces]
    spread = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces, dtype=float)))   # Scalar+[faces <- vertices]
    return (~spread * (edges[0] & edges[1])).normalized()


def vertex_areas(mesh: Mesh, vertices: Point) -> Scalar:
    """Each vertex's area, `Scalar+[vertices]`, is a third of the area of each face at it, a face's area half the
    size of its plane."""
    edges = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point-[faces]
    thirds = as_ga_sparse(mesh.faces, as_scalar(np.ones_like(mesh.faces) / 3))   # Scalar+[faces <- vertices]
    return ~thirds * (edges[1].commutator(edges[0]).norm() / 2)


def cotangent_weights(mesh: Mesh, vertices: Point, face_weights: Scalar) -> Scalar:
    """Each edge's weight, `Scalar+[edges]`, is half the cotangents of the angles facing it, each face's share
    weighted by `Scalar+[faces]`: the cotangent at a corner is one edge's slice read along the other, over the size
    of the plane they span; head to tail, the corner's own edges point the other way."""
    edges = ((mesh.d0 * vertices).batch()[mesh.face_edges.T] * mesh.face_edge_orientation.T).field()   # [3] Point-[faces]
    after, before = edges[[1, 2, 0]], edges[[2, 0, 1]]                          # [3] Point-[faces], Point-[faces]
    cotangents = -(after.dual_inverse() & before) / after.commutator(before).norm()   # [3] Scalar+[faces]
    return ~mesh.by_corner(cotangents / 2) * face_weights


def transported(normals: Vector, start: Point) -> Iterator[Point]:
    """Parallel transport carries a tangent vector `start` along a sequence of unit normals `[steps + 1, ...]`,
    and yields it at each of them."""
    yield start
    for tail, head in zip(normals[:-1], normals[1:]):
        start = (head * tail).square_root() >> start                            # [...] Point-
        yield start


def loop_turn(normals: Vector) -> Scalar:
    """The holonomy of transport around closed loops of unit normals `[..., steps]` is the turn of the running
    product of the transport rotors, each later step multiplying from the left, read from the rotor's square so
    that whole turns read as nothing."""
    following = concatenate([normals[..., 1:], normals[..., :1]], axis=-1)      # [..., steps] Vector-
    steps = (following * normals).square_root()                                  # [..., steps] Rotor+
    around = steps.cumprod(axis=-1)[..., -1]                                     # [...] Rotor+
    return (around * around).log() | normals[..., 0].dual()                      # [...] Scalar+


def adjusted_transport(mesh: Mesh, normals: Vector, angles: Scalar) -> Rotor:
    """The rotors from each end of each edge to its head, `[edges, 2]`, are the surface's own transport from the
    tail followed by the edge's extra turn, `Scalar+[edges]`, in the head's tangent plane, and none at the head.
    With no extra turns, this is the surface's own parallel transport."""
    unit_normals = normals.batch()                                               # [vertices] Vector-
    tails, heads = unit_normals[mesh.edges[:, 0]], unit_normals[mesh.edges[:, 1]]   # [edges] Vector-, [edges] Vector-
    extra = (heads.dual() * (-angles.batch() / 2)).exp()                         # [edges] Rotor+
    return stack([extra * (heads * tails).square_root(), mv.rotor().broadcast_to(extra.shape)], axis=-1)


def connection(mesh: Mesh, transports: Rotor, weights: Scalar) -> tuple[SparseExtensor, SparseExtensor]:
    """The sparse covariant derivative of the rotors `[edges, 2]` from each end of each edge to its head comes with
    its connection Laplacian on vertex fields of tangent vectors, each edge weighted by `Scalar+[edges]`."""
    covariant = as_ga_sparse(mesh.edges, (transports >> Point) * np.array([-1, 1]))   # (Point- <- Point-)[edges <- vertices]
    laplacian = covariant.adjoint()(spdiag(weights) * covariant)                # (Point- <- Point-)[vertices <- vertices]
    return covariant, laplacian


def low_modes(laplacian: SparseExtensor, normals: Vector, areas: Scalar, normal_stiffness: float, count: int
              ) -> tuple[Scalar, Point]:
    """The `count` smoothest tangent fields of a connection Laplacian against the vertex areas, `Scalar+[vertices]`,
    come with their Dirichlet energies per unit area, the normal bundle along the unit normals,
    `Vector-[vertices]`, lifted out of the low spectrum by `normal_stiffness`."""
    unit_normals = normals.batch()                                               # [vertices] Vector-
    # A direction's part along the normal: the normal's direction, its dual, times the direction's pairing with
    # the normal's patch.
    normal_penalty = spdiag((unit_normals.dual() * (unit_normals & Point)).field()) * normal_stiffness   # (Point- <- Point-)[vertices <- vertices]
    return (laplacian + normal_penalty).eigh(spdiag(areas) * Point, count)      # [count] Scalar+, [count] Point-[vertices]


def heat_transport(laplacian: SparseExtensor, areas: Scalar, source: Point, step_time: float, steps: int
                   ) -> Iterator[tuple[Point, Point]]:
    """Repeated backward-Euler vector heat steps of duration `step_time`, with the connection Laplacian of the
    surface's own transport and the vertex areas `Scalar+[vertices]`, carry a tangent field `source` across the
    mesh; they yield the diffused field and the unit field at each step."""
    area_weights = spdiag(areas)                                                # Scalar+[vertices <- vertices]
    step_operator = area_weights * Point + step_time * laplacian                # (Point- <- Point-)[vertices <- vertices]
    field = source                                                              # Point-[vertices]
    for _ in range(steps):
        field = step_operator.solve(area_weights * field)                       # Point-[vertices]
        yield field, field / field.norm()                                       # Point-[vertices], Point-[vertices]


def trivial_connection(mesh: Mesh, normals: Vector, indices: np.ndarray, generators: list[np.ndarray]) -> tuple[Scalar, SparseExtensor, Scalar]:
    """The smallest extra turns on the edges, `Scalar+[edges]`, make every face turn by `2 pi` times its index
    `[faces]` and every generator, a closed loop of vertices, by nothing; they come with the circulation of edge
    turns around the faces and generators and its target."""
    unit_normals = normals.batch()                                                # [vertices] Vector-
    curvature = loop_turn(unit_normals[mesh.faces])                               # [faces] Scalar+
    # Around each generator, the extra turns undo the surface's own holonomy.
    undone = [-loop_turn(unit_normals[loop[None]]) for loop in generators]        # [generators] list of [1] Scalar+
    circulation = mesh.circulation(generators)                                       # (Scalar+ <- Scalar+)[faces + generators <- edges]
    target = concatenate([2 * np.pi * as_scalar(indices) - curvature, *undone]).field()   # Scalar+[faces + generators]
    return circulation.lstsq(target), circulation, target


def carried_out(mesh: Mesh, vertices: Point, normals: Vector, angles: Scalar) -> Point:
    """A unit tangent vector at every vertex, `Point-[vertices]`, is carried out from the first vertex by the
    transport of the given turns along a breadth-first tree: one sparse solve, each tree edge asking its head to
    be its tail carried across it, and the first vertex asking to point across its first edge. Where the turns
    make a trivial connection whose faces turn by quarter turns, this is a cross field: carried across any edge,
    the tail's vector lands on the head's turned by a whole number of quarter turns."""
    children, tree = mesh.spanning_tree
    count = len(children) + 1
    turns = adjusted_transport(mesh, normals, angles)[tree]                      # [vertices - 1, 2] Rotor+
    along_tree = SparseExtensor.from_indices((turns >> Point) * np.array([-1, 1]), children[:, None], mesh.edges[tree], (count, count))
    first = SparseExtensor.from_indices(as_scalar([1.0]), np.array([0]), np.array([0]), (count, count)) * Point
    # The direction across the first edge, in the tangent plane: the plane through the normal and the edge's
    # direction, which is that direction as an antivector.
    seed = (normals.batch()[0] ^ (mesh.d0 * vertices).batch()[0].dual_inverse()).normalized()   # [] Point-
    return (along_tree + first).solve((seed * as_scalar(np.arange(count) == 0)).field())


def cross_stripes(mesh: Mesh, vertices: Point, normals: Vector, areas: Scalar, angles: Scalar, cross: Point,
                  regular: Scalar, frequency: float, normal_stiffness: float) -> tuple[Point, Point, Scalar]:
    """Two families of stripes run across the two lines of a cross field, `Point-[vertices]`, parallel up to
    quarter turns for the connection of the given turns: the lines `[vertices, 2] Point-`, the stripe field of
    each `[vertices, 2] Point-`, and the targets across every edge for each line at its head `[edges, 2] Scalar+`.

    Carried across an edge, each of the tail's lines lands on one of the head's lines, pointing along it
    or against it. A stripe phase along a line reads backwards along the line reversed, so the stripe
    field carried onto a line it lands against is mirrored: its component along that line reversed, the
    stripes' angle from the line negated. The two families are one field over the cross's two lines at
    every vertex, coupled across the edges where the lines trade places, around the cross's singularities.
    Each face adds its half-cotangents to its edges' weights only where it is regular, `Scalar+[faces]`."""
    count = mesh.n_vertices
    unit_normals = normals.batch()                                               # [vertices] Vector-
    forward = adjusted_transport(mesh, normals, angles)[:, 0]                    # [edges] Rotor+
    # The cross's second line: the first turned a quarter turn by its commutator with the tangent plane.
    lines = stack([cross.batch(), cross.batch().commutator(unit_normals.dual())], axis=-1)   # [vertices, 2] Point-
    tails, heads = mesh.edges[:, 0], mesh.edges[:, 1]
    ahead = lines[heads]                                                         # [edges, 2] Point-
    # One where the tail's line b lands along the head's line a, minus one where it lands against it, and
    # zero where it lands across it.
    match = ahead[:, :, None].dot((forward[:, None] >> lines[tails])[:, None, :])   # [edges, 2, 2] Scalar+
    behind = (match * lines[tails][:, None, :]).sum(axis=-1)                     # [edges, 2] Point-
    # The field read along each edge, by the metric: the paper's inner product of the field with the edge.
    advance = ((ahead + behind) / 2).dot((mesh.d0 * vertices).batch()[:, None]) * frequency   # [edges, 2] Scalar+
    turn = (unit_normals[heads][:, None].dual() * (-advance / 2)).exp()          # [edges, 2] Rotor+
    # Carried, then mirrored onto the head's line where it lands against it.
    moved = forward[:, None, None] >> Point                                      # [edges, 1, 1] Point- <- Point-
    along = ahead[:, :, None]                                                    # [edges, 2, 1] Point-
    landed = match * match * moved + (match - match * match) * (along * along.dot(moved))   # [edges, 2, 2] Point- <- Point-
    # The head's stripe field turned back by the target, against the tail's carried and mirrored.
    cells = stack([turn.reverse() >> Point, -landed[..., 0], -landed[..., 1]], axis=-1)     # [edges, 2, 3] Point- <- Point-
    rows = np.arange(len(heads))[:, None, None] * 2 + np.arange(2)[None, :, None]           # [edges, 2, 1]
    columns = np.stack(np.broadcast_arrays(heads[:, None] * 2 + np.arange(2), tails[:, None] * 2, tails[:, None] * 2 + 1), axis=-1)
    covariant = SparseExtensor.from_indices(cells, rows, columns, (2 * len(heads), 2 * count))   # (Point- <- Point-)[edges * 2 <- vertices * 2]
    weights = cotangent_weights(mesh, vertices, regular).batch()                 # [edges] Scalar+
    vertex = np.arange(2 * count) // 2                                           # each line's vertex
    laplacian = covariant.adjoint()(spdiag(weights[np.arange(2 * len(heads)) // 2].field()) * covariant)   # (Point- <- Point-)[vertices * 2 <- vertices * 2]
    _, stripe_fields = low_modes(laplacian, unit_normals[vertex].field(),
                                 areas.batch()[vertex].field(), normal_stiffness, 1)
    stripe = stripe_fields.batch()[0].reshape(count, 2)                          # [vertices, 2] Point-
    return lines, stripe / stripe.norm(), advance


# --- plumbing -------------------------------------------------------------------------
def latitude_rings(colatitudes: np.ndarray, steps: int) -> tuple[Vector, Point]:
    """Closed latitude circles on the unit sphere at the given colatitudes `[rings]`, over `steps` segments, come
    as their unit normals, the patches of the sphere's tangent planes, with the unit meridian tangent vector at the
    start of each ring."""
    longitudes = np.linspace(0.0, 2 * np.pi, steps + 1)
    tilt = (mv.zx * (-colatitudes / 2)).exp()                                   # [rings] Rotor+
    spin = (mv.xy * (-longitudes[:, None] / 2)).exp()                           # [steps + 1, 1] Rotor+
    normals = spin >> (tilt >> mv.xy.dual())                                    # [steps + 1, rings] Vector-
    starts = tilt >> mv.x.dual()                                                # [rings] Point-
    return normals, starts
