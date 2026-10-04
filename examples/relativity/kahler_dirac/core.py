"""Waves of a field over the whole geometric algebra of the plane and time, `x+y+t-`, obeying
`d(phi) == mass * phi`: one equation for every grade at once, stepped on a lattice.

The geometric derivative is `d = e_x * d/dx + e_y * d/dy - e_t * d/dt`. Solved for the time
derivative, the equation reads `d/dt phi = e_t * (mass * phi - d_space(phi))`: the spatial
derivative and the mass, turned by `e_t`. On the lattice the spatial derivative splits each axis's
product into its two parts, `e_a * phi == e_a.left_contraction(phi) + (e_a ^ phi)`: the wedge
takes its difference with the cell ahead and the contraction with the cell behind. Swapping inputs
and outputs turns one into minus the other, so the derivative is antisymmetric, and applied twice it
is the lattice Laplacian on every component.

The field's blades without t, `Space = 1 x y xy`, and those with t, `Time = t xt yt xyt`, each keep
to themselves under the spatial derivative, and `e_t` swaps them. So the time step is a leapfrog: the
time part from the space part, then the space part from the time part, each through its own sparse
extensor, and the energy is kept exactly. The space blades have reverse norm plus one and the time
blades minus one, so the density is `space.scalar_norm_squared() - time.scalar_norm_squared()`.
Twice over, the step is the Klein-Gordon equation: `-(to_space(to_time(space)))` is the mass squared
less the Laplacian.

In the notation of Kähler, with the Clifford product of differential forms, the equation reads as
$d\\phi - \\delta\\phi = m\\,\\phi$, its derivative split into the exterior derivative and the
codifferential.

Lengths are in lattice spacings, and the speed of light is one.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack
from numga.gatype import GAType
from numga.sparse import SparseExtensor

ga = Algebra("x+y+t-")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
# The blades without t, and those with t.
Space = ga.gatype(ga.subspace("1 x y xy"))
Time = ga.gatype(ga.subspace("t xt yt xyt"))
axes = stack([mv.x, mv.y])                                              # [2] Vector: x and y


# --- math -----------------------------------------------------------------------------
def derivative(grid: Grid, part: GAType) -> SparseExtensor:
    """The spatial geometric derivative of a field of one part: along each axis, the wedge with it
    against the cell ahead and the contraction with it against the cell behind."""
    raised, lowered = axes ^ part, axes.left_contraction(part)                 # [2] part <- part
    couplings = stack([-raised, raised, lowered, -lowered], axis=-1)            # [2, 4] part <- part
    cells = grid.here.shape[0]
    return SparseExtensor.from_columns(grid.stencil, couplings.broadcast_to((cells, 2, 4)).reshape(cells, 8), cells)


def step(grid: Grid, mass: Scalar, part: GAType) -> SparseExtensor:
    """The time derivative of the other part, from this part: the mass less the spatial derivative,
    turned by e_t."""
    turn = SparseExtensor.from_diagonal(mv.t * np.ones(grid.here.shape[0])) * part   # [cells, cells] other <- part
    return turn(SparseExtensor.from_diagonal(mass) * part - derivative(grid, part))  # [cells, cells] other <- part


def leapfrog(to_time: SparseExtensor, to_space: SparseExtensor, space: Space, time: Time, interval: float,
             count: int) -> Iterator[tuple[Space, Time]]:
    """The space part at each whole step and the time part half a step after it: the time part moved
    on by the space part, then the space part by the time part."""
    for _ in range(count):
        time = time + to_time(space) * interval                                # [cells] Time
        yield space, time
        space = space + to_space(time) * interval                              # [cells] Space


def density(space: Space, time: Time) -> Scalar:
    """The field's density: the space blades' reverse norm, positive, less the time blades', negative."""
    return space.scalar_norm_squared() - time.scalar_norm_squared()


def oscillation(to_time: SparseExtensor, waves: Space, frequencies: Scalar, phases: np.ndarray) -> tuple[Space, Time]:
    """Standing waves at the given phases of their periods `[waves, phases, cells]`: the space part
    a cosine, and the time part, what the step makes of it over the frequency, a sine."""
    space = waves[:, None] * np.cos(phases)[:, None]                           # [waves, phases, cells] Space
    time = (to_time(waves) / frequencies[:, None])[:, None] * np.sin(phases)[:, None]   # [waves, phases, cells] Time
    return space, time


def modes(grid: Grid, mass: Scalar, count: int) -> tuple[Scalar, Space]:
    """The count standing waves of least frequency: the space parts whose time derivative, twice over,
    is minus their frequency squared times themselves."""
    to_time, to_space = step(grid, mass, Space), step(grid, mass, Time)
    unit = SparseExtensor.from_diagonal(mv.scalar(np.ones((grid.here.shape[0], 1)))) * Space
    return (-to_space(to_time)).eigh(unit, count)                             # [count] Scalar, [count, cells] Space


# --- plumbing -------------------------------------------------------------------------
class Grid:
    """A periodic square lattice of `side` cells along x and y, the cell centres `[cells] Vector` about
    the origin, each cell's index, and for each the cells it couples to `[cells, 8]`: along x, itself
    and the cell ahead, itself and the cell behind; then the same along y."""

    def __init__(self, side: int) -> None:
        i, j = (index.ravel() for index in np.meshgrid(np.arange(side), np.arange(side), indexing="ij"))
        self.side = side
        self.here = i * side + j
        self.positions = axes[0] * (i - (side - 1) / 2) + axes[1] * (j - (side - 1) / 2)   # [cells] Vector
        ahead = np.stack([((i + 1) % side) * side + j, i * side + (j + 1) % side])   # [2, cells]
        behind = np.stack([((i - 1) % side) * side + j, i * side + (j - 1) % side])  # [2, cells]
        here = np.broadcast_to(self.here, ahead.shape)
        self.stencil = np.stack([here, ahead, here, behind], axis=-1).transpose(1, 0, 2).reshape(-1, 8)


def disk(grid: Grid, centre: Vector, radius: float, edge: float) -> Scalar:
    """One inside a disk and zero outside, falling off over the edge's width."""
    distance = (grid.positions - centre).norm()                                # [cells] Scalar
    return (1 - ((distance - radius) / edge).tanh()) * 0.5
