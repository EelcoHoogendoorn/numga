"""Thin elastic shells deform as triangulated surfaces that stretch along their edges and bend at
their interior hinges in three-dimensional projective geometric algebra, following Grinspun,
Hirani, Desbrun and Schröder, "Discrete shells" (2003).

Every edge acts as a spring along the line joining its endpoints, storing energy in changes of its
length, while every interior hinge between two triangles folds one triangle's patch onto the other
about their shared edge line and stores energy in the squared deviation from its rest dihedral
angle. Differentiating this energy pulls edge endpoints along their edge lines and twists each
hinge's two incident triangles equally and oppositely, pushing the opposite corners perpendicular
to their triangle patches and dividing the counter-forces between the hinge endpoints by the lever
rule; because every contribution is internal, the vertex force lines sum to zero net force and zero
net moment.

Incompatible rest lengths or natural curvatures prevent a flat rest state in three-dimensional
space: rest lengths that grow toward the rim of a disc buckle it into a saddle, and a natural
curvature form with opposite signs along two crossed directions twists a strip into a ribbon.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.util import shapes
from numga import NumpyContext, stack
from numga.algebras import PGA3D as ga
from numga.sparse import Sparse
from numga.topology import Topology

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                      # Scalar+
Chiral = Scalar.derive.flipped_orientation                                       # Chiral-
Plane = ga.gatype.vector()                                                       # Plane+
Patch = ga.gatype.antitrivector()                                                # Patch-
Point = ga.gatype.antivector()                                                   # Point-
# Lines joining points: edges, hinges, and the action lines of vertex forces.
Line = ga.gatype.antibivector()                                                  # Line-
Motor = ga.gatype.rotor()                                                        # Motor+
# Symmetric form evaluating normal curvature across unit lines in the surface.
Curvature = ga.gatype((Scalar, Line, Line))                                      # Scalar+ <- (Line-, Line-)


# --- math -----------------------------------------------------------------------------
def rest_geometry(
    topology: Topology, points: Point, curvature: Curvature, top: Plane, thickness: float,
) -> tuple[Sparse, Sparse, Chiral, Scalar, Scalar]:
    """From a rest configuration (`Point-[V]`) of uniform `thickness`, unit modulus, and unit
    density, `curvature` toward `top` determines the interior-hinge endpoint and opposite-corner
    selectors (`[2] Scalar+[E_interior <- V]`), chiral rest dihedral angles
    (`Chiral-[E_interior]`), hinge bending stiffnesses (`Scalar+[E_interior]`), and lumped vertex
    masses (`Scalar+[V]`)."""
    # Select each interior hinge's two endpoints in the +1 face's winding order and the opposite corners in both faces.
    hinge_ends = topology.reading(topology.interior[1], 1) * topology.incidence[1, 0]   # [2] Scalar+[E_interior <- V]
    hinge_corners = topology.reading(topology.facet_opposites.T, 0)              # [2] Scalar+[E_interior <- V]
    ends, tips = hinge_ends * points                                             # Point-[E_interior], Point-[E_interior]
    across, opposite = hinge_corners * points                                    # Point-[E_interior], Point-[E_interior]
    line = ends & tips                                                           # Line-[E_interior]
    first, second = line & across, -(line & opposite)                            # Patch-[E_interior], Patch-[E_interior]
    # Normal curvature across the hinge times the centroid-to-centroid span (one third of the two heights) sets the rest bend.
    unit = line.normalized()                                                     # Line-[E_interior]
    spans = (first.norm() + second.norm()) / (3 * line.norm())                   # Scalar+[E_interior]
    face_corners = topology.incidence[2, 0] * points                             # [3] Point-[F]
    patches = face_corners[0] & face_corners[1] & face_corners[2]                # Patch-[F]
    # Pairing the face patches with `top` supplies the chiral orientation sign toward the chosen top side.
    facing = (patches | top).sites.sum()                                         # [] Chiral-
    angles = curvature(unit, unit) * spans * (facing / facing.norm())            # Chiral-[E_interior]
    bending = thickness ** 3 / 2 * -(line | line) / (first.norm() + second.norm())   # Scalar+[E_interior]
    # Lumping one third of each triangle's mass onto each of its three corners gives the vertex masses.
    masses = topology.incidence[2, 0].sum(axis=0).adjugate() * patches.norm() * (thickness / 6)   # Scalar+[V]
    return hinge_ends, hinge_corners, angles, bending, masses


def shell_energy(
    topology: Topology,
    hinge_ends: Sparse,
    hinge_corners: Sparse,
    lengths: Scalar,
    angles: Chiral,
    stretching: float,
    bending: Scalar,
    points: Point,
) -> Scalar:
    """Edge stretching against `lengths` (`Scalar+[E]`) and interior-hinge bending against `angles`
    (`Chiral-[E_interior]`) sum to the total elastic energy (`[] Scalar+`) at `points`
    (`Point-[V]`)."""
    tails, heads = topology.incidence[1, 0] * points                             # Point-[E], Point-[E]
    stretched = (tails & heads).norm() - lengths                                 # Scalar+[E]
    ends, tips = hinge_ends * points                                             # Point-[E_interior], Point-[E_interior]
    across, opposite = hinge_corners * points                                    # Point-[E_interior], Point-[E_interior]
    line = ends & tips                                                           # Line-[E_interior]
    # Wind both face patches consistently so a coplanar hinge has unit motor and zero folding angle.
    first, second = line & across, -(line & opposite)                            # Patch-[E_interior], Patch-[E_interior]
    # Multiplying the two unit patches turns twice about their shared hinge line; twice the square root's log reads the angle.
    motor = (second.normalized() * first.normalized()).square_root()             # Motor+[E_interior]
    bent = -2 * (motor.log() | line.normalized()) - angles                       # Chiral-[E_interior]
    stretch_energy = (stretching * stretched * stretched / lengths).sites.sum()  # [] Scalar+
    bend_energy = (bending * bent.squared()).sites.sum()                         # [] Scalar+
    return (stretch_energy + bend_energy) / 2


def shell_forces(
    topology: Topology,
    hinge_ends: Sparse,
    hinge_corners: Sparse,
    lengths: Scalar,
    angles: Chiral,
    stretching: float,
    bending: Scalar,
    points: Point,
) -> Point:
    """Negative energy gradients on edges and interior hinges scatter back through the topological
    adjugates to give the net internal force (`Point-[V]`) at `points` (`Point-[V]`)."""
    tails, heads = topology.incidence[1, 0] * points                             # Point-[E], Point-[E]
    current_lengths = (tails & heads).norm()                                     # Scalar+[E]
    pull = (heads - tails) * (stretching * (current_lengths - lengths) / (lengths * current_lengths))   # Point-[E]

    ends, tips = hinge_ends * points                                             # Point-[E_interior], Point-[E_interior]
    across, opposite = hinge_corners * points                                    # Point-[E_interior], Point-[E_interior]
    line = ends & tips                                                           # Line-[E_interior]
    first, second = line & across, -(line & opposite)                            # Patch-[E_interior], Patch-[E_interior]
    motor = (second.normalized() * first.normalized()).square_root()             # Motor+[E_interior]
    moment = line.norm() * bending * (-2 * (motor.log() | line.normalized()) - angles)   # Chiral-[E_interior]
    # Each face pushes its opposite corner normal to its patch along `(patch | corner) ^ w`, scaled by inverse height.
    pushes = [((patch | corner) ^ mv.w) * (moment / patch.norm_squared())
              for patch, corner in ((first, across), (second, opposite))]        # [2] Point-[E_interior]
    # Divide the equal-and-opposite reaction between the hinge endpoints by where each opposite corner projects along the hinge.
    shares = [((ends & corner) | line) / (line | line) for corner in (across, opposite)]   # [2] Scalar+[E_interior]
    at_tips = -(pushes[0] * shares[0] + pushes[1] * shares[1])                   # Point-[E_interior]
    at_ends = -(pushes[0] + pushes[1]) - at_tips                                 # Point-[E_interior]

    # Pull back edge tensions and hinge corner reactions onto vertices via the coboundary and reading adjugates.
    return (-(topology.d[0].adjugate() * pull)
            + (hinge_ends.adjugate() * stack([at_ends, at_tips])).sum(axis=0)
            + (hinge_corners.adjugate() * stack(pushes)).sum(axis=0))


def relax(
    topology: Topology,
    hinge_ends: Sparse,
    hinge_corners: Sparse,
    lengths: Scalar,
    angles: Chiral,
    stretching: float,
    bending: Scalar,
    masses: Scalar,
    points: Point,
    step: float,
    damping: float,
    n_frames: int,
    n_steps_per_frame: int,
) -> Iterator[Point]:
    """Yields `n_frames + 1` vertex configurations (`Point-[V]`) spaced `n_steps_per_frame` damped
    symplectic-Euler steps apart as the shell relaxes from rest at `points`."""
    import jax
    from numga.backend.jax import JaxContext

    def substep(state: tuple[Point, Point], _) -> tuple[tuple[Point, Point], None]:
        current_points, velocity = state
        acceleration = shell_forces(
            topology, hinge_ends, hinge_corners, lengths, angles, stretching, bending, current_points,
        ) / masses                                                               # Point-[V]
        next_velocity = (velocity + acceleration * step) * (1 - damping)         # Point-[V]
        return (current_points + next_velocity * step, next_velocity), None

    # Compile one frame's worth of damped time steps into a single scanned loop.
    advance = jax.jit(lambda state: jax.lax.scan(substep, state, length=n_steps_per_frame)[0])
    context = JaxContext(ga)
    state = (points.materialize(context), (points - points).materialize(context))   # Point-[V], Point-[V]
    for _ in range(n_frames):
        yield state[0].materialize(mv.context)
        state = advance(state)
    yield state[0].materialize(mv.context)


# --- plumbing -------------------------------------------------------------------------
def flat_triangulation(coordinates: np.ndarray, spacing: float) -> tuple[Topology, Point]:
    """Delaunay triangulation of planar `coordinates` (`[V, 2]`) drops degenerate boundary slivers
    smaller than `spacing` squared and embeds the vertices in the `z == 0` plane (`Point-[V]`)."""
    points = mv.point(np.concatenate([coordinates, np.zeros((len(coordinates), 1))], axis=-1)).field()   # Point-[V]
    delaunay, _ = shapes.delaunay(coordinates)
    corners = delaunay.incidence[2, 0] * points                                  # [3] Point-[F]
    kept = (corners[0] & corners[1] & corners[2]).norm().batch() > 1e-6 * spacing ** 2
    return delaunay.subset(kept, -1), points


def triangular_lattice(spacing: float, radius: float) -> np.ndarray:
    """Equilateral triangular lattice points of step `spacing` span a region covering `radius`
    around the origin (`[..., 2]`)."""
    n_rings = int(radius / spacing) + 2
    rows, columns = np.meshgrid(
        np.arange(-2 * n_rings, 2 * n_rings + 1), np.arange(-n_rings - 1, n_rings + 2), indexing="ij",
    )
    return np.stack([rows + columns / 2, columns * np.sqrt(3) / 2], axis=-1).reshape(-1, 2) * spacing
