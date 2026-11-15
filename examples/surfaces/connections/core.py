"""Connections on curved surfaces in Euclidean three-dimensional geometric algebra.

Parallel transport carries a tangent vector along a surface while keeping it tangent and turning it
no more than the surface itself bends. Between two neighbouring vertices with unit surface patches
`tail_patch` and `head_patch`, the shortest turn carrying the tail patch onto the head patch is the
square root of their geometric product, `turn = (head_patch * tail_patch).square_root()`, which
rotates in the plane of the two normals and leaves the hinge line between their tangent planes fixed.
Tangent vectors are intrinsic directions (`Point-`), surface normals are unit intrinsic area patches
(`Patch-`) formed by joining triangle edges, and `Patch.dual()` gives the extrinsic turning bivector
in each tangent plane. Chained around a closed curve of normals, those rotors turn a tangent vector
by the solid angle swept out on the unit sphere.

On a triangulated surface, assigning a rotor from each end of every edge to its head defines a
sparse covariant derivative `(Point- <- Point-)[E <- V]` directly on ambient three-dimensional
direction fields, without local two-dimensional coordinate charts or per-vertex reference edges.
Pairing each edge difference against its dual plane `Point.dual()` with cotangent weights and pulling
back via `.adjugate()` builds the connection Laplacian `(Plane+ <- Point-)[V <- V]`. Because every
edge rotor maps the tail patch onto the head patch, neither the covariant derivative nor the
connection Laplacian mixes tangent and normal components. Backward-Euler vector heat steps diffuse a
localized tangent impulse across the surface along all paths at once, while penalizing the normal
direction lifts the normal bundle out of the low spectrum so the generalized eigenproblem against
the vertex area form yields the smoothest global tangent fields in equal-energy quarter-turn pairs.

Following Crane, Desbrun, and Schröder (2010), adding an extra tangent-plane turn on each edge shifts
every triangle's holonomy and every handle loop's holonomy linearly by the edge circulation, solving
for trivial connections whose parallel fields have prescribed singularity indices summing to the
Euler characteristic. Following Knöppel, Crane, Pinkall, and Schröder (2015), adding a directional
phase advance on every edge of a trivial connection yields globally smooth stripe and cross-stripe
fields that fork automatically where curvature or singularities prevent uniform spacing.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, concatenate, stack
from numga.algebras import VGA3D as ga
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology

context = NumpyContext(ga)
mv = context.multivector
Scalar = ga.gatype.scalar()                                  # Scalar+
# Surface positions and tangent directions are intrinsic antivectors.
Point = ga.gatype.antivector()                               # Antivector-
Plane = ga.gatype.vector()                                   # Vector+
# Unit surface normals are intrinsic tangent-plane patches formed by joining two edge directions.
Patch = Plane.derive.flipped_orientation                     # Vector-
Rotor = ga.gatype.rotor()                                    # Rotor+


# --- math -----------------------------------------------------------------------------
def latitude_rings(colatitudes: np.ndarray, n_ring_steps: int) -> tuple[Patch, Point]:
    """Rotating a north-pole patch and meridian tangent to `colatitudes` (`[n_rings]`) and sweeping
    around the vertical axis over `n_ring_steps` segments forms closed latitude loops of unit
    surface patches (`[n_ring_steps + 1, n_rings] Patch-`) and initial meridian directions
    (`[n_rings] Point-`)."""
    longitudes = np.linspace(0, 2 * np.pi, n_ring_steps + 1)
    tilt_rotors = (mv.zx * (-colatitudes / 2)).exp()                                      # [n_rings] Rotor+
    spin_rotors = (mv.xy * (-longitudes[:, None] / 2)).exp()                              # [n_ring_steps + 1, 1] Rotor+
    tilted_patches = tilt_rotors >> mv.xy.dual()                                          # [n_rings] Patch-
    patches = spin_rotors >> tilted_patches                                               # [n_ring_steps + 1, n_rings] Patch-
    starts = tilt_rotors >> mv.x.dual()                                                   # [n_rings] Point-
    return patches, starts


def parallel_transport(patches: Patch, start: Point) -> Iterator[Point]:
    """Yields `start` (`[...] Point-`) carried step by step along `patches` (`[n_steps + 1, ...] Patch-`)
    by the shortest dihedral rotor between consecutive unit surface patches."""
    yield start
    for tail, head in zip(patches[:-1], patches[1:]):
        turn = (head * tail).square_root()                                                # [...] Rotor+
        start = turn >> start                                                             # [...] Point-
        yield start


def vertex_patches(topology: Topology, vertices: Point) -> Patch:
    """Joining two boundary edges on each triangle and pulling their area-weighted patches back to
    incident vertices gives the normalized vertex surface patches (`Patch-[V]`)."""
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)                          # [3] Point-[F]
    return (topology.incidence[2, 0].sum(axis=0).adjugate() * (edges[0] & edges[1])).normalized()


def vertex_areas(topology: Topology, vertices: Point) -> Scalar:
    """Distributing one third of each triangle's area (`(edges[0] & edges[1]).norm() / 2`) to its
    three corners gives the lumped vertex areas (`Scalar+[V]`)."""
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)                          # [3] Point-[F]
    corner_mean = topology.incidence[2, 0].mean(axis=0)                                   # Scalar+[F <- V]
    return corner_mean.adjugate() * ((edges[0] & edges[1]).norm() / 2)


def cotangent_weights(topology: Topology, vertices: Point, face_weights: Scalar) -> Scalar:
    """Half-cotangents of opposite corner angles weighted by `face_weights` (`Scalar+[F]`) sum into
    the symmetric edge weights (`Scalar+[E]`)."""
    edges = topology.boundary[2, 1] * (topology.d[0] * vertices)                          # [3] Point-[F]
    tails, heads = topology.corners[2, 1]
    tail_edges, head_edges = edges[tails], edges[heads]                                   # [3] Point-[F], [3] Point-[F]
    cotangents = (tail_edges | head_edges) / (tail_edges & head_edges).norm()             # [3] Scalar+[F]
    return (topology.incidence[2, 1].adjugate() * (face_weights * cotangents / 2)).sum(axis=0)


def connection_operators(topology: Topology, transports: Rotor,
                         weights: Scalar) -> tuple[SparseExtensor, SparseExtensor]:
    """Edge endpoint rotors `transports` (`[2] Rotor+[E]`) and edge `weights` (`Scalar+[E]`) assemble
    the sparse covariant derivative (`(Point- <- Point-)[E <- V]`) and connection Laplacian
    (`(Plane+ <- Point-)[V <- V]`)."""
    # Carry both endpoints' vectors into the head tangent plane and take their oriented difference.
    covariant = (spdiag(transports >> Point) * topology.boundary[1, 0]).sum(axis=0)       # (Point- <- Point-)[E <- V]
    # Pair each edge difference with its dual plane, weight by `weights`, and pull back via `.adjugate()`.
    laplacian = covariant.adjugate()((weights * Point.dual())(covariant))                 # (Plane+ <- Point-)[V <- V]
    return covariant, laplacian


def heat_diffusion(laplacian: SparseExtensor, areas: Scalar, source: Point,
                   step_time: float, n_steps: int) -> Iterator[Point]:
    """Yields the diffused tangent vector field (`Point-[V]`) after each of `n_steps` backward-Euler
    vector heat steps of duration `step_time` from `source`."""
    mass = areas * Point.dual()                                                           # (Plane+ <- Point-)[V]
    step_operator = mass + step_time * laplacian                                          # (Plane+ <- Point-)[V <- V]
    field = source                                                                        # Point-[V]
    for _ in range(n_steps):
        field = step_operator.solve(mass(field))                                          # Point-[V]
        yield field


def low_modes(laplacian: SparseExtensor, patches: Patch, areas: Scalar,
              normal_stiffness: float, n_modes: int) -> tuple[Scalar, Point]:
    """Penalizing normal components by `normal_stiffness` lifts the normal bundle out of the low
    spectrum and solves for the `n_modes` smoothest tangent eigenfields (`[n_modes] Point-[V]`) and
    their Dirichlet energies (`[n_modes] Scalar+`) against vertex `areas`."""
    unit_patches = patches.batch()                                                        # [V] Patch-
    # Project each direction onto the surface normal direction and pair with its dual plane.
    normal_part = (unit_patches.dual() * (unit_patches & Point)).field()                  # (Point- <- Point-)[V]
    normal_penalty = normal_stiffness * Point.dual()(normal_part)                         # (Plane+ <- Point-)[V]
    return (laplacian + normal_penalty).eigh(spdiag(areas * Point.dual()), n_modes)


def loop_holonomy(patches: Patch) -> Scalar:
    """Chaining dihedral transport rotors around closed loops of unit `patches` (`[n_steps, ...] Patch-`)
    reads each loop's signed holonomy angle (`[...] Scalar+`) in `(-pi, pi]` from the squared loop rotor."""
    next_patches = concatenate([patches[1:], patches[:1]])                                # [n_steps, ...] Patch-
    step_rotors = (next_patches * patches).square_root()                                  # [n_steps, ...] Rotor+
    loop_rotor = step_rotors.cumprod(axis=0)[-1]                                          # [...] Rotor+
    return (loop_rotor * loop_rotor).log() | patches[0].dual()                            # [...] Scalar+


def adjusted_transport(topology: Topology, patches: Patch, angles: Scalar) -> Rotor:
    """Composing each edge's dihedral rotor from tail to head with an extra tangent-plane rotation by
    `angles` (`Scalar+[E]`) forms the adjusted edge endpoint rotors (`[2] Rotor+[E]`)."""
    tails, heads = topology.incidence[1, 0] * patches                                     # Patch-[E], Patch-[E]
    extra_turns = (heads.dual() * (-angles / 2)).exp()                                    # Rotor+[E]
    edge_rotors = extra_turns * (heads * tails).square_root()                             # Rotor+[E]
    return stack([edge_rotors, mv.rotor().broadcast_to(edge_rotors.batch().shape).field()])  # [2] Rotor+[E]


def trivial_connection(topology: Topology, patches: Patch, indices: np.ndarray,
                       generators: list[np.ndarray]) -> tuple[Scalar, SparseExtensor, Scalar]:
    """Least-squares edge turns (`Scalar+[E]`) cancel face curvature up to `2 * pi * indices` on
    triangles and cancel handle holonomy around `generators`, returning the turns alongside the
    sparse circulation operator and target holonomy field."""
    corner_patches = (topology.incidence[2, 0] * patches).batch()                         # [3, F] Patch-
    loop_patches = [(topology.reading(loop[:, None], 0) * patches).batch() for loop in generators]
    curvature = loop_holonomy(corner_patches)                                             # [F] Scalar+
    handle_targets = [-loop_holonomy(handle_patches) for handle_patches in loop_patches]
    circulation = topology.circulation(generators) * Scalar                               # (Scalar+ <- Scalar+)[F_loops <- E]
    target = concatenate([2 * np.pi * indices - curvature, *handle_targets]).field()      # Scalar+[F_loops]
    return circulation.lstsq(target), circulation, target


def parallel_field(topology: Topology, patches: Patch, areas: Scalar, weights: Scalar,
                   angles: Scalar, normal_stiffness: float) -> tuple[Scalar, Point]:
    """Extracting the two lowest eigenmodes of the adjusted connection with edge turns `angles`
    returns their energies (`[2] Scalar+`) and the unit parallel tangent field (`Point-[V]`)."""
    _, laplacian = connection_operators(topology, adjusted_transport(topology, patches, angles), weights)
    energies, fields = low_modes(laplacian, patches, areas, normal_stiffness, 2)
    field = fields.batch()[0]                                                             # [V] Point-
    return energies, (field / field.norm()).field()


def stripe_family(topology: Topology, vertices: Point, patches: Patch, areas: Scalar,
                  angles: Scalar, field: Point, regular_faces: Scalar, frequency: float,
                  normal_stiffness: float) -> tuple[Point, Point, Scalar, Scalar]:
    """Adding directional phase advances of `frequency` along `field` (`Point-[V]`) on regular faces
    (`regular_faces`, `Scalar+[F]`) solves for the smoothest unit stripe field (`Point-[V]`), its edge
    advances (`Scalar+[E]`), and the lowest two stripe energies (`[2] Scalar+`)."""
    edge_means = topology.incidence[1, 0].mean(axis=0)                                    # Scalar+[E <- V]
    advance = (-((edge_means * field) | (topology.d[0] * vertices))) * frequency          # Scalar+[E]
    _, laplacian = connection_operators(
        topology, adjusted_transport(topology, patches, angles + advance),
        cotangent_weights(topology, vertices, regular_faces),
    )
    energies, stripe_fields = low_modes(laplacian, patches, areas, normal_stiffness, 2)
    stripe = stripe_fields.batch()[0]                                                     # [V] Point-
    return field, (stripe / stripe.norm()).field(), advance, energies


def spanning_tree_field(topology: Topology, vertices: Point, patches: Patch, angles: Scalar) -> Point:
    """Propagating a unit tangent seed from the first vertex across `topology.spanning_tree` under
    `angles` (`Scalar+[E]`) in one sparse linear solve produces a unit cross-field representative
    (`Point-[V]`) parallel up to quarter turns across non-tree edges."""
    children, tree = topology.spanning_tree
    turns = adjusted_transport(topology, patches, angles)                                 # [2] Rotor+[E]
    covariant = (spdiag(turns >> Point) * topology.boundary[1, 0]).sum(axis=0)            # (Point- <- Point-)[E <- V]
    tree_edges = topology.reading(tree, 1)                                                # Scalar+[V_tree <- E]
    tree_covariant = topology.reading(children, 0).adjugate() * (tree_edges * covariant)  # (Point- <- Point-)[V <- V]
    root_vertex = topology.reading(np.array([0]), 0)                                      # Scalar+[1 <- V]
    first_edge = topology.reading(np.array([0]), 1)                                       # Scalar+[1 <- E]
    # Turn the first edge vector in the root vertex's tangent plane to obtain an initial unit tangent seed.
    seed = (-(first_edge * (topology.d[0] * vertices)).commutator((root_vertex * patches).dual())).normalized()  # Point-[1]
    return (tree_covariant + root_vertex.adjugate() * root_vertex * Point).solve(root_vertex.adjugate() * seed)


def cross_stripes(topology: Topology, vertices: Point, patches: Patch, areas: Scalar,
                  angles: Scalar, cross: Point, regular_faces: Scalar, frequency: float,
                  normal_stiffness: float) -> tuple[Point, Point, Scalar]:
    """Coupling the two orthogonal cross-field lines (`[2] Point-[V]`) across edges where quarter or
    half turns swap or reverse them solves for both coupled unit stripe families (`[2] Point-[V]`)
    and their edge phase advances (`[2] Scalar+[E]`)."""
    V, E, F = topology.n_elements
    edge_rotors = adjusted_transport(topology, patches, angles)[0]                        # Rotor+[E]
    # Turn each cross line by a quarter turn in its tangent plane via commutator with the dual patch.
    lines = stack([cross, cross.commutator(patches.dual())])                              # [2] Point-[V]
    tails, heads = topology.incidence[1, 0]                                               # Scalar+[E <- V], Scalar+[E <- V]
    tail_lines, head_lines = tails * lines, heads * lines                                 # [2] Point-[E], [2] Point-[E]
    transported_lines = edge_rotors >> tail_lines                                         # [2] Point-[E]
    # Signed alignment (+1, -1, or 0) matching each transported tail line onto each head line.
    alignment = -(head_lines[:, None] | transported_lines[None, :])                       # [2, 2] Scalar+[E]
    matched_tails = (alignment * tail_lines[None, :]).sum(axis=1)                         # [2] Point-[E]
    advance = (-((head_lines + matched_tails) / 2 | (topology.d[0] * vertices))) * frequency  # [2] Scalar+[E]
    target_turn = ((heads * patches).dual() * (-advance / 2)).exp()                       # [2] Rotor+[E]
    # Mirror the carried stripe vector along the head line whenever transport lands against that line.
    transport_map = edge_rotors >> Point                                                  # (Point- <- Point-)[E]
    head_axis = head_lines[:, None]                                                       # [2, 1] Point-[E]
    coupled_transport = (alignment * alignment * transport_map
                         + (alignment - alignment * alignment) * (head_axis * -(head_axis | transport_map)))  # [2, 2] (Point- <- Point-)[E]
    vertex_lines = Sparse.selection(np.arange(2 * V).reshape(V, 2).T, 2 * V)              # [2] Scalar+[V <- 2 * V]
    edge_lines = Sparse.selection(np.arange(2 * E).reshape(E, 2).T, 2 * E)                # [2] Scalar+[E <- 2 * E]
    line_differences = (spdiag(target_turn.reverse() >> Point) * heads * vertex_lines
                        - (spdiag(coupled_transport) * tails * vertex_lines[None]).sum(axis=1))  # [2] (Point- <- Point-)[E <- 2 * V]
    covariant = (edge_lines.adjugate() * line_differences).sum(axis=0)                    # (Point- <- Point-)[2 * E <- 2 * V]
    weights = (edge_lines.adjugate() * cotangent_weights(topology, vertices, regular_faces)).sum(axis=0)  # Scalar+[2 * E]
    laplacian = covariant.adjugate()((weights * Point.dual())(covariant))                 # (Plane+ <- Point-)[2 * V <- 2 * V]
    _, stripe_fields = low_modes(
        laplacian, (vertex_lines.adjugate() * patches).sum(axis=0),
        (vertex_lines.adjugate() * areas).sum(axis=0), normal_stiffness, 1,
    )
    stripe = vertex_lines * stripe_fields[0]                                              # [2] Point-[V]
    return lines, stripe / stripe.norm(), advance


def turned_fields(patches: Patch, fields: Point, n_frames: int) -> Iterator[Point]:
    """Yields `fields` (`[...] Point-[V]`) rotated about every vertex normal by `n_frames` evenly
    spaced angles over one full turn."""
    planes = patches.dual()                                                               # Bivector+[V]
    for angle in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        turn = (planes * (-angle / 2)).exp()                                              # Rotor+[V]
        yield turn >> fields                                                              # [...] Point-[V]
