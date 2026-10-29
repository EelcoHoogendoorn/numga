"""Penrose tilings by cut and project: a two-dimensional cut through the integer lattice of
five-dimensional space, at the slope of a five-fold turn, in the geometric algebra of five dimensions.

A Penrose tiling covers two-dimensional space with two kinds of rhombus, a thick and a thin one, in a
pattern that never repeats yet has five-fold symmetry everywhere. It is the shadow of an ordinary lattice
in more dimensions: the cubes of the integer lattice of five-dimensional space, cut by a two-dimensional
slice, the cut, tilted so that it never passes through a second lattice point. Each square face of the
lattice that the cut passes close enough to becomes a tile, laid flat onto the cut. Shifting the cut
without tilting it changes which faces are close enough, and the tiling rearranges itself by local flips,
never sliding as a whole.

The lattice's points and axes are antivectors, intrinsic like all geometry. Turning the five axes into
one another in a cycle is a rotation, the product of four reflections in the mirrors between
neighbouring axes. It turns the directions of its first bivector by a fifth of a turn, those of its
second by two fifths, and leaves the diagonal fixed. Averaged over its five powers, each weighted by the
cosine of a fifth of a turn times the power, the turn projects onto the directions it turns by a fifth,
`along(Antivector)`, and the cut is the join of two axes laid flat, `along(first) & along(second)`. A
square face of the lattice is a tile where the cut passes through the cube it is dual to, the cube
through its centre along the three other axes. Whether it does is read by the planes that pair to one
with one of those axes and to nothing with the other two and the cut: Cramer's rule with joins alone,
`(second & third & cut) / (first & second & third & cut)`, with no metric. How often a face is a tile is
the size of its dual cube's shadow square to the cut, the join of its three axes under
`Antivector - along`, which equals the size of its tile, the join of its two sides laid flat: thick
tiles outnumber thin ones by the golden ratio.

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
Scalar = ga.gatype.scalar()                               # Scalar+
# Planes, read by pairing with points: the mirrors, and the planes that read a point's offsets.
Plane = ga.gatype.vector()                                # Vector+
# Points and directions are antivectors, intrinsic; the join of two directions is an oriented area, such as
# the cut or a tile, of three a volume.
Antivector = ga.gatype.antivector()                       # Antivector-
Join = ga.gatype.antibivector()                           # Antibivector-
# A map on directions: the turn, its powers, and the projections onto the directions it turns by one angle.
Projection = ga.gatype((Antivector, Antivector))          # Antivector- <- Antivector-

# The mirror across each axis, and the axis, its dual.
mirrors = mv.basis()                                      # [axes] Plane+
axes = mirrors.dual()                                     # [axes] Antivector-
# The faces of the lattice: each pair of axes, and the three other axes of each.
faces_axes = list(itertools.combinations(range(5), 2))
sides_index = np.array(faces_axes)                        # [faces, 2]
others_index = np.array([[k for k in range(5) if k not in face] for face in faces_axes])   # [faces, 3]


# --- math -----------------------------------------------------------------------------
# The mirror between two neighbouring axes swaps them; the four swaps in a row turn each axis into the next
# and the last into the first.
neighbour_mirrors = (mirrors[:-1] - mirrors[1:]).normalized()                 # [axes - 1] Plane+
turn = neighbour_mirrors[0] * neighbour_mirrors[1] * neighbour_mirrors[2] * neighbour_mirrors[3]   # [] Rotor+
# The turn as a map on directions.
turning = turn >> Antivector                                                  # [] Antivector- <- Antivector-


def powers(cycle: Projection, n_powers: int) -> Iterator[Projection]:
    """The map applied once, twice, and so on."""
    power = cycle                                                             # [] Antivector- <- Antivector-
    for _ in range(n_powers):
        yield power
        power = cycle(power)


def projection(harmonic: int) -> Projection:
    """The projection onto the directions the turn turns by `harmonic` fifths of a turn: its five powers as
    maps on directions, the fifth being the identity, each weighted by the cosine of that turn times the power."""
    maps = stack(list(powers(turning, 5)))                                    # [powers] Antivector- <- Antivector-
    weights = 2 / 5 * np.cos(2 * np.pi * harmonic * np.arange(1, 6) / 5)
    return (maps * weights).sum(axis=0)                                       # [] Antivector- <- Antivector-


def reciprocal(cut: Join) -> Plane:
    """Per face, the reciprocal frame of its three other axes and the cut `[faces, 3]`: the planes whose
    pairings with a point read how far along each of the three axes it is, blind to the cut."""
    others = axes[others_index]                                               # [faces, 3] Antivector-
    # Cramer's rule with joins: per axis, the other two joined with the cut, a plane that pairs to nothing
    # with them and with the cut, over the volume all three span with the cut.
    cofactors = others[:, [1, 2, 0]] & others[:, [2, 0, 1]] & cut             # [faces, 3] Plane+
    volume = others[:, 0] & others[:, 1] & others[:, 2] & cut                 # [faces] Scalar+
    return cofactors / volume[:, None]                                        # [faces, 3] Plane+


def margins(shift: Antivector, centres: Antivector, frames: Plane) -> Scalar:
    """How far inside the dual cube of each face the cut passes, along each of its three other axes
    `[..., faces, 3]`, for the cut through `shift`: the face is a tile where all three are positive."""
    offsets = frames & (shift - centres)[..., None]                           # [..., faces, 3] Scalar+
    return 1 / 4 - offsets.squared()                                          # [..., faces, 3] Scalar+


def shadows(along: Projection) -> tuple[Scalar, Scalar]:
    """Per face `[faces]`: its area laid flat onto the cut, and the volume of the shadow its dual cube
    casts square to the cut, how often the face is a tile among faces of every kind."""
    sides = axes[sides_index]                                                 # [faces, 2] Antivector-
    others = axes[others_index]                                               # [faces, 3] Antivector-
    # The tile is the join of the face's two sides laid flat; the shadow the join of its three other axes
    # with their parts in the cut taken away.
    square = Antivector - along                                               # [] Antivector- <- Antivector-
    tiles = along(sides[:, 0]) & along(sides[:, 1])                           # [faces] Join-
    cube_shadows = square(others[:, 0]) & square(others[:, 1]) & square(others[:, 2])   # [faces] Antitrivector-
    return tiles.norm(), cube_shadows.norm()


# --- plumbing -------------------------------------------------------------------------
def lattice(radius: int) -> Antivector:
    """The points of the integer lattice within `radius` of the origin along every axis `[points]`."""
    coordinates = np.array(list(itertools.product(range(-radius, radius + 1), repeat=5)))   # [points, 5]
    return (axes * coordinates).sum(axis=-1)                                  # [points] Antivector-
