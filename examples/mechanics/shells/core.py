"""Thin elastic shells as triangle meshes that stretch along their edges and bend at their hinges, in the
projective geometric algebra of three-dimensional space, after Grinspun, Hirani, Desbrun and Schröder,
"Discrete shells" (2003).

A shell resists two things: stretching, which changes lengths along the surface, and bending, which
changes how the surface curves. Stretching is held by every edge, a spring along the line joining its
two ends, whose length is the line's norm. Bending is held by every hinge, an edge with a face on either
side. Each face's plane joins the hinge's line with the corner across it, and the motor that folds one
face onto the other is a turn about the hinge: its logarithm is half the bending angle times the hinge's
unit line. A hinge stores energy as the square of how far its bending angle is from its rest angle.

A hinge bent away from its rest angle twists its two faces about its line, equally and oppositely. Each
face pushes the corner across the hinge square to the face, by the twist over the corner's distance from
the hinge, and the hinge's two ends take the opposite push, shared by the lever rule. The forces are the
exact downhill direction of the energy, and as internal forces they balance: the lines along which they
act, the joins of each vertex with its force, add up to no force and no moment at all.

A sheet's rest state need not fit in space. Rest lengths that grow towards the rim of a disc leave too
much rim for a flat disc, and the disc buckles out of its plane into a saddle; rest angles from a natural
curvature that no unstretched surface can take, a saddle, as in the two layers of a seed pod curving
across each other, leave a strip to bend partly both ways, and it twists into a ribbon. The rest angles
come from a natural curvature, a form on lines that gives the curvature across each hinge.

In the notation of the discrete shells paper the bending energy is
$\\sum_e (\\theta_e - \\bar\\theta_e)^2 \\, 3 \\bar\\ell_e^2 / \\bar A_e$ with $\\bar A_e$ the summed area of
the two faces, and the gradient of the angle at the corner across the hinge is $n / h$, the unit normal
over the corner's height above the hinge.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from numga import stack
from numga.algebras import PGA3D as ga
from numga.sparse import SparseExtensor
from examples import instantiate

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, as_scalar = mesh_module.Mesh, mesh_module.as_scalar
mv = mesh_module.ctx.multivector
Scalar = ga.gatype.scalar()
Plane = ga.gatype.vector()
Point = ga.gatype.antivector()
Direction = Point.derive.degenerate
# Lines joining points: edges, hinges and the lines forces act along.
Line = ga.gatype.antibivector()
Motor = ga.gatype.rotor()
# The form of a natural curvature, on unit lines in the sheet: the curvature across them.
Curvature = ga.gatype((Scalar, Line, Line))                  # Scalar <- (Line, Line)


@dataclass(frozen=True)
class Sheet:
    """A shell's rest state on a mesh: the maps reading every hinge's two ends and the corners across it
    `[H, V]`, every edge's rest length `Scalar[E]` and every hinge's rest angle `Scalar[H]`, the stiffness
    of both, and the mass at every vertex `Scalar[V]`."""

    mesh: Mesh
    corners: tuple[SparseExtensor, SparseExtensor, SparseExtensor, SparseExtensor]
    lengths: Scalar
    angles: Scalar
    stretching: float
    bending: Scalar
    masses: Scalar


# --- math -----------------------------------------------------------------------------
def hinge_motors(ends: Point, tips: Point, across: Point, opposite: Point) -> tuple[Line, Plane, Plane, Motor]:
    """Every hinge's line `Line[H]`, the planes of its two faces `Plane[H]` each, and the motor that folds
    the first face onto the second `Motor[H]`, from its two ends and the corners across it."""
    line = ends & tips                                                           # Line[H]
    # Both planes wound as the first face, so that a flat hinge folds by nothing.
    first, second = line & across, -(line & opposite)                            # Plane[H] each
    # The product of two unit planes is twice the turn about their common line: its square root folds one onto the other.
    return line, first, second, (second.normalized() * first.normalized()).square_root()


def bending_angles(line: Line, motor: Motor) -> Scalar:
    """Every hinge's bending angle `Scalar[H]`: twice the logarithm of its motor, along its unit line."""
    return -2 * (motor.log() | line.normalized())


def energy(sheet: Sheet, points: Point) -> Scalar:
    """The elastic energy of the sheet at the given points `Point[V]`."""
    tails, heads = (end * points for end in sheet.mesh.edge_ends)
    stretched = (tails & heads).norm() - sheet.lengths                           # Scalar[E]
    line, _, _, motor = hinge_motors(*(corner * points for corner in sheet.corners))
    bent = bending_angles(line, motor) - sheet.angles                             # Scalar[H]
    return ((sheet.stretching * stretched * stretched / sheet.lengths).sites.sum() + (sheet.bending * bent * bent).sites.sum()) / 2


def forces(sheet: Sheet, points: Point) -> Direction:
    """The force at every vertex `Direction[V]`, the energy's downhill direction: the edges' springs pulling
    their ends together or apart, and the hinges twisting their faces back towards their rest angles."""
    tails, heads = (end * points for end in sheet.mesh.edge_ends)
    lengths = (tails & heads).norm()                                             # Scalar[E]
    pull = (heads - tails) * (sheet.stretching * (lengths - sheet.lengths) / (sheet.lengths * lengths))   # Direction[E]

    corners = [corner * points for corner in sheet.corners]                     # Point[H] each
    ends, _, across, opposite = corners
    line, first, second, motor = hinge_motors(*corners)
    moment = line.norm() * sheet.bending * (bending_angles(line, motor) - sheet.angles)   # Scalar[H]
    # Each face pushes the corner across the hinge square to its plane, along `(plane | corner) ^ w`, by the
    # moment over the corner's distance from the hinge.
    pushes = [((plane | corner) ^ mv.w) * (moment / plane.norm_squared()) for plane, corner in ((first, across), (second, opposite))]
    # The hinge's ends take the opposite pushes, shared by the lever rule: by where each corner lies along the hinge.
    shares = [((ends & corner) | line) / (line | line) for corner in (across, opposite)]
    at_tips = -(pushes[0] * shares[0] + pushes[1] * shares[1])                   # Direction[H]
    at_ends = -(pushes[0] + pushes[1]) - at_tips                                 # Direction[H]

    # The incidence's reverse sums every edge's pull at its two ends, and each reading map's reverse sums
    # the hinges' pushes at their corners.
    at_ends_map, at_tips_map, across_map, opposite_map = sheet.corners
    return (-(~sheet.mesh.d0 * pull) + ~at_ends_map * at_ends + ~at_tips_map * at_tips
            + ~across_map * pushes[0] + ~opposite_map * pushes[1])


def forque(points: Point, pushes: Direction) -> Line:
    """The sum of the lines along which forces act, each through its vertex: their total force and moment."""
    return (points & pushes).sites.sum()


def relax(sheet: Sheet, points: Point, step: float, damping: float, n_frames: int, n_steps_per_frame: int
          ) -> Iterator[Point]:
    """The sheet moving under its own forces from the given points `Point[V]`, from rest, losing a share
    `damping` of its velocity every step of the given size until it settles: the points at every frame,
    so many steps apart, and at the end."""
    import jax
    from numga.backend.jax import JaxContext

    def substep(state: tuple[Point, Direction], _) -> tuple[tuple[Point, Direction], None]:
        points, velocity = state
        velocity = (velocity + forces(sheet, points) * (step / sheet.masses)) * (1 - damping)
        return (points + velocity * step, velocity), None

    # The steps between two frames, compiled once into one loop.
    frame = jax.jit(lambda state: jax.lax.scan(substep, state, length=n_steps_per_frame)[0])
    context = JaxContext(ga)
    state = (points.materialize(context), (points - points).materialize(context))   # Point[V], Direction[V]
    for _ in range(n_frames):
        yield state[0].materialize(mv.context)
        state = frame(state)
    yield state[0].materialize(mv.context)


def sheet(mesh: Mesh, lengths: Scalar, curvature: Curvature, top: Plane, thickness: float) -> Sheet:
    """The rest state of a sheet of the given thickness, of unit stiffness and density, at the mesh's
    vertices, with the given rest lengths `Scalar[E]` and the rest angles of the given natural curvature,
    which bends towards the given top side of the sheet."""
    points = mesh.vertices
    corners = tuple(mesh.reading(index) for index in hinges(mesh).T)             # [H, V] each
    line, first, second, _ = hinge_motors(*(corner * points for corner in corners))
    # The curvature across each hinge, times the distance between its faces' centres, a third of the sum of
    # the corners' distances from the hinge, is the angle the hinge bends by on a surface of that curvature.
    unit = line.normalized()
    angles = curvature(unit, unit) * (first.norm() + second.norm()) / (3 * line.norm())   # Scalar[H]
    # The plane through each face's corners, wound as the face, has twice the face's area as its norm; read
    # against the top side, it says which way the bending angles count.
    corner_points = [corner * points for corner in mesh.face_corners]           # Point[F] each
    planes = corner_points[0] & corner_points[1] & corner_points[2]              # Plane[F]
    facing = (planes | top).sites.sum()                                          # [] Scalar
    # Each vertex carries a third of the mass of every face at it.
    double_areas = planes.norm()                                                 # Scalar[F]
    masses = ~mesh.at_corners(stack([double_areas] * 3)) * as_scalar(np.ones(len(mesh.faces))).field() * (thickness / 6)
    return Sheet(
        mesh=mesh,
        corners=corners,
        lengths=lengths,
        angles=angles * (facing / facing.norm()),
        stretching=thickness,
        bending=thickness ** 3 / 2 * -(line | line) / (first.norm() + second.norm()),
        masses=masses,
    )


# --- plumbing -------------------------------------------------------------------------
def hinges(mesh: Mesh) -> np.ndarray:
    """Every edge between two faces `[H, 4]`: its two ends as the first face winds them, then the corner
    across it in that face and in the other. A face's edge at each place faces the face's corner at that
    place, so each corner across an edge sits at one of the edge's own two places among the faces."""
    places = mesh.face_edges.reshape(-1)                                         # [3F] the edge at every place
    order = np.argsort(places, kind="stable")
    counts = np.bincount(places)
    starts = (np.cumsum(counts) - counts)[counts == 2]
    (first, at_first), (second, at_second) = np.divmod(order[starts], 3), np.divmod(order[starts + 1], 3)
    faces = mesh.faces
    return np.stack([faces[first, (at_first + 1) % 3], faces[first, (at_first + 2) % 3],
                     faces[first, at_first], faces[second, at_second]], axis=-1)


def flat_mesh(coords: np.ndarray, spacing: float) -> Mesh:
    """The Delaunay triangulation of points in the xy plane `[V, 2]`, every face wound as the largest, without
    the slivers along straight rows of the boundary. Which way that is the sheet reads off its top side."""
    faces = Delaunay(coords).simplices
    first, second = coords[faces[:, 1]] - coords[faces[:, 0]], coords[faces[:, 2]] - coords[faces[:, 0]]
    turning = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    faces = np.where((turning * turning[np.argmax(np.abs(turning))] < 0)[:, None], faces[:, [0, 2, 1]], faces)
    points = mv.point(np.concatenate([coords, np.zeros((len(coords), 1))], axis=-1)).field()   # Point[V]
    return Mesh(points, faces[np.abs(turning) > 1e-6 * spacing ** 2])


def lattice(spacing: float, radius: float) -> np.ndarray:
    """The points of a triangular lattice of the given spacing, covering the given radius around the origin `[n, 2]`."""
    count = int(radius / spacing) + 2
    rows, columns = np.meshgrid(np.arange(-2 * count, 2 * count + 1), np.arange(-count - 1, count + 2), indexing="ij")
    return np.stack([rows + columns / 2, columns * np.sqrt(3) / 2], axis=-1).reshape(-1, 2) * spacing
