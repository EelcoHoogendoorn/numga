"""Penrose tilings by cut and project: a plane through the integer lattice of five-dimensional space,
cut at the slope of a five-fold turn, in the geometric algebra of five dimensions.

A Penrose tiling covers the plane with two kinds of rhombus, a thick and a thin one, in a pattern
that never repeats yet has five-fold symmetry everywhere. It is the shadow of an ordinary lattice in
more dimensions: the cubes of the integer lattice of five-dimensional space, cut by a plane tilted so
that it never passes through a second lattice point. Each square face of the lattice that the plane
passes close enough to becomes a tile, laid flat onto the plane. Shifting the plane without tilting it
changes which faces are close enough, and the tiling rearranges itself by local flips, never sliding
as a whole.

The lattice's points and axes are antivectors, intrinsic like all geometry. Turning the five axes into
one another in a cycle is a rotation, the product of four reflections in the mirrors between
neighbouring axes. It turns one plane by a fifth of a turn, another by two fifths, and leaves the
diagonal fixed. Averaged over its five powers, each weighted by the cosine of a fifth of a turn times
the power, the turn projects onto the first plane, `along(Antivector)`, and the cut is the join of two
axes laid flat, `along(first) & along(second)`. A square face of the lattice is a tile where the cut
passes through the cube it is dual to, the cube through its centre along the three other axes. Whether
it does is read by the planes that pair to one with one of those axes and to nothing with the other
two and the cut: Cramer's rule with joins alone, `(second & third & cut) / (first & second & third &
cut)`, with no metric. How often a face is a tile is the size of its dual cube's shadow square to the
cut, the join of its three axes under `Antivector - along`, which equals the size of its tile, the join
of its two sides laid flat: thick tiles outnumber thin ones by the golden ratio.

In the notation of de Bruijn's pentagrid, a tiling vertex is the integer point
$k_j = \\lceil \\mathbf x \\cdot \\mathbf e_j + \\gamma_j \\rceil$ of the five grids of lines
$\\mathbf x \\cdot \\mathbf e_j + \\gamma_j \\in \\mathbb Z$, with
$\\mathbf e_j = (\\cos\\frac{2\\pi j}{5}, \\sin\\frac{2\\pi j}{5})$ and $\\sum_j \\gamma_j = 0$.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack

ga = Algebra("a+b+c+d+e+")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
# Planes, read by pairing with points: the mirrors, and the planes that read a point's offsets.
Plane = ga.gatype.vector()
# Points and directions are antivectors, intrinsic; the join of two is a plane through them, of three a
# space.
Antivector = ga.gatype.antivector()
Join = ga.gatype.antibivector()
Rotor = ga.gatype.rotor()
Projection = ga.gatype((Antivector, Antivector))          # Antivector <- Antivector: onto one of the turn's planes

mirrors = mv.basis()                                      # [5] Plane: across each axis
axes = mirrors.dual()                                     # [5] Antivector
# The faces of the lattice: each pair of axes, and the three other axes of each.
FACES = list(itertools.combinations(range(5), 2))
SIDES = np.array(FACES)                                   # [faces, 2]
OTHERS = np.array([[k for k in range(5) if k not in face] for face in FACES])   # [faces, 3]


# --- math -----------------------------------------------------------------------------
def turn() -> Rotor:
    """The rotation turning each axis into the next and the last into the first: four reflections."""
    # The mirror between two neighbouring axes swaps them.
    normals = (mirrors[:-1] - mirrors[1:]) / np.sqrt(2)                       # [4] Plane
    return normals[0] * normals[1] * normals[2] * normals[3]                  # [] Rotor


def projection(harmonic: int) -> Projection:
    """The projection onto the plane the turn turns by `harmonic` fifths of a turn: its five powers as
    maps on directions, the fifth being the identity, each weighted by the cosine of that turn times the power."""
    maps = stack(list(powers(turn() >> Antivector, 5)))                       # [5] Antivector <- Antivector
    weights = 2 / 5 * np.cos(2 * np.pi * harmonic * np.arange(1, 6) / 5)
    return (maps * weights).sum(axis=0)                                       # [] Antivector <- Antivector


def powers(shift: Projection, n_powers: int) -> Iterator[Projection]:
    """The map applied once, twice, and so on."""
    power = shift                                                             # [] Antivector <- Antivector
    for _ in range(n_powers):
        yield power
        power = shift(power)


def reciprocal(cut: Join) -> Plane:
    """Per face, the reciprocal frame of its three other axes and the cut `[faces, 3]`: the planes whose
    pairings with a point read how far along each of the three axes it is, blind to the cut."""
    others = axes[OTHERS]                                                      # [faces, 3] Antivector
    # Cramer's rule with joins: per axis, the other two joined with the cut, a plane that pairs to nothing
    # with them and with the cut, over the volume all three span with the cut.
    cofactors = others[:, [1, 2, 0]] & others[:, [2, 0, 1]] & cut             # [faces, 3] Plane
    volume = others[:, 0] & others[:, 1] & others[:, 2] & cut                 # [faces] Scalar
    return cofactors / volume[:, None]                                        # [faces, 3] Plane


def margins(shift: Antivector, centres: Antivector, frames: Plane) -> Scalar:
    """How far inside the dual cube of each face the cut passes, along each of its three other axes
    `[..., faces, 3]`, for the cut through `shift`: the face is a tile where all three are positive."""
    offsets = frames & (shift - centres)[..., None]                           # [..., faces, 3] Scalar
    return 1 / 4 - offsets.squared()                                          # [..., faces, 3] Scalar


def shadows(along: Projection) -> tuple[Scalar, Scalar]:
    """Per face `[faces]`: its area laid flat onto the cut, and the volume of the shadow its dual cube
    casts square to the cut, how often the face is a tile among faces of every kind."""
    sides, others = axes[SIDES], axes[OTHERS]                                  # [faces, 2], [faces, 3] Antivector
    # The tile is the join of the face's two sides laid flat; the shadow the join of its three other axes
    # with their parts in the cut taken away.
    square = Antivector - along                                               # [] Antivector <- Antivector
    tiles = along(sides[:, 0]) & along(sides[:, 1])                           # [faces] Join
    shadows = square(others[:, 0]) & square(others[:, 1]) & square(others[:, 2])   # [faces] Antitrivector
    return tiles.norm(), shadows.norm()


# --- plumbing -------------------------------------------------------------------------
def lattice(radius: int) -> Antivector:
    """The points of the integer lattice within `radius` of the origin along every axis `[points]`."""
    coordinates = np.array(list(itertools.product(range(-radius, radius + 1), repeat=5)))   # [points, 5]
    return (axes * coordinates).sum(axis=-1)                                  # [points] Antivector
