"""A Rubik's cube as points: turned, drawn and solved with the same quarter-turn maps, in the geometric
algebra of three-dimensional space.

The cube is 27 cubies, their centres at -1, 0 and 1 along each axis, and a sticker is a cubie's face
on the cube's surface: a face is a side and an axis, and a cubie shows the faces whose side it lies
on. Each sticker is a point, its cubie's centre plus half its face, in units of half a cubie,
`2 * cubie + face`, so every coordinate is an integer, and so are its corners. A face's layer is the
stickers whose point reaches past 1 along the face, and a quarter turn is the rotor
`1 + face * mv.xyz`, the plane of the face, as a map on vectors halved: its sandwich doubles lengths,
so `((1 + plane) >> Vector) * 0.5` turns by exactly a quarter, clockwise seen from outside, with
coefficients 0 and 1, and `1 - plane` turns it back. A point turned any number of times keeps integer
coordinates, and the cube is solved when every sticker is back at its own point.

The same maps turn the whole frontier of a search at once: every state by both directions of every
face in one batched application. The solver meets in the middle, widening the states reachable from
the scramble and from the solved cube a quarter turn at a time until they share one. Drawing turns a
layer by the rotor of a fraction of the quarter, `(plane * (-angle / 2)).exp()`.

In Singmaster's notation the faces along +x, -x, +y, -y, +z and -z read as R, L, B, F, U and D.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
Vector = ga.gatype.vector()

# The six faces, by side and axis, and each face's plane, clockwise seen from outside.
sides = np.array([1.0, -1.0])
faces = sides[:, None] * mv.basis()                                        # [2, 3] Vector
planes = faces * mv.xyz                                                    # [2, 3] Bivector
# The quarter turns, each face clockwise and back.
directions = np.array([1.0, -1.0])
turns = ((1 + directions[:, None, None] * planes) >> Vector) * 0.5         # [2, 2, 3] Vector <- Vector


# --- math -----------------------------------------------------------------------------
def turned(stickers: Vector, turn: Vector, face: Vector) -> Vector:
    """The stickers after a quarter turn of a face: those in its layer turned, the rest left."""
    layer = (stickers | face) > 1                                          # [...] boolean
    return stickers + layer * (turn(stickers) - stickers)                  # [...] Vector


def frontier(stickers: Vector) -> Vector:
    """Every quarter turn, each way round each face, applied to each of the given states:
    `[2, 2, 3, states, 54]`."""
    return turned(stickers, turns[..., None, None], faces[..., None, None])


def solve(scramble: Vector, solved: Vector, depth: int) -> np.ndarray:
    """The quarter turns `[turns, 3]`, each a direction, a side and an axis, that take the scrambled
    cube to the solved one, found by widening the states reachable from each, a turn at a time and each
    side in turn, up to depth turns each, until they meet."""
    searches = [Search(scramble[None]), Search(solved[None])]
    for level in range(2 * depth):
        search, other = searches[level % 2], searches[1 - level % 2]
        search.widen()
        here, there = meeting(search.states[-1], other.states[-1])
        if len(here):
            forward, backward = (search, other) if level % 2 == 0 else (other, search)
            ends = (here[0], there[0]) if level % 2 == 0 else (there[0], here[0])
            back = backward.path(ends[1])[::-1]                                # [turns, 3]
            return np.concatenate([forward.path(ends[0]), np.concatenate([1 - back[:, :1], back[:, 1:]], axis=-1)])
    raise ValueError(f"no solution within {depth} quarter turns from each side")


def animate(stickers: Vector, corners: Vector, moves: np.ndarray, steps: int) -> Iterator[tuple[Vector, Vector]]:
    """The stickers' corners `[54, 4]` and the cut under the turning layer, on both sides `[2, 9, 4]`,
    through the given quarter turns `[turns, 3]`, each turn's layer turned smoothly over the given
    steps."""
    for direction, side, axis in moves:
        layer = (stickers | faces[side, axis]) > 1                             # [54] boolean
        # The cut: the faces of the layer's cubies towards the middle.
        cut = 2 * cubies[outward[:, side, axis]][:, None] + squares[1 - side, axis]   # [9, 4] Vector
        for fraction in np.arange(steps) / steps:
            rotor = (planes[side, axis] * (np.pi / 4 * fraction * directions[direction])).exp()   # [] Rotor
            yield corners + layer[:, None] * ((rotor >> corners) - corners), stack([cut, rotor >> cut])
        turn, face = turns[direction, side, axis], faces[side, axis]
        stickers, corners = turned(stickers, turn, face), corners + layer[:, None] * (turn(corners) - corners)
    yield corners, stack([cut, cut])


# --- plumbing -------------------------------------------------------------------------
# The cubies' centres, and the faces each shows on the cube's surface, those whose side it lies on.
cubies = mv.vector(np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), axis=-1).reshape(-1, 3))   # [27] Vector
outward = (cubies[:, None, None] | faces) > 0.5                            # [27, 2, 3] boolean


def face_corners() -> Vector:
    """The corners of each face of a cubie, about its centre, in order round the face: one corner on
    the face's side, then its quarter turns about it."""
    corner = mv.vector([1.0, 1.0, 1.0]) - mv.basis() + faces               # [2, 3] Vector
    around = [corner]
    for _ in range(3):
        around.append(turns[0](around[-1]))
    return stack(around, axis=-1)                                          # [2, 3, 4] Vector


squares = face_corners()                                                   # [2, 3, 4] Vector


def solved_cube() -> tuple[Vector, Vector, np.ndarray]:
    """The solved cube: its 54 sticker points, their corners `[54, 4]`, and the face each belongs to,
    `side * 3 + axis`."""
    points = (2 * cubies[:, None, None] + faces)[outward]                    # [54] Vector
    corners = (2 * cubies[:, None, None, None] + squares)[outward]           # [54, 4] Vector
    _, side, axis = np.nonzero(outward)
    return points, corners, side * 3 + axis


class Search:
    """The states reached from one cube, a quarter turn at a time: each level's new states, with the
    state one turn back and the turn that led there, a direction, a side and an axis."""

    def __init__(self, start: Vector) -> None:
        self.states = [start]                                                  # [level] [states, 54] Vector
        self.parents = [np.zeros(1, dtype=int)]
        self.moves = [np.zeros((1, 3), dtype=int)]
        self.seen = keys(start)

    def widen(self) -> None:
        """The states one turn on from the last level, not reached before."""
        flat = frontier(self.states[-1]).reshape(-1, 54)                       # [12 * states, 54] Vector
        index = fresh(flat, self.seen)
        self.states.append(flat[index])
        *move, parent = np.unravel_index(index, (2, 2, 3, len(self.states[-2])))
        self.parents.append(parent)
        self.moves.append(np.stack(move, axis=-1))
        self.seen |= keys(flat[index])

    def path(self, index: int) -> np.ndarray:
        """The turns `[turns, 3]` from the start to the given state of the last level."""
        moves = []
        for level in range(len(self.states) - 1, 0, -1):
            moves.append(self.moves[level][index])
            index = self.parents[level][index]
        return np.array(moves[::-1], dtype=int).reshape(-1, 3)


def keys(states: Vector) -> set[bytes]:
    """Each state's sticker coordinates as bytes: exact, since they are integers."""
    return {row.tobytes() for row in np.rint(states.kernel).astype(np.int8).reshape(len(states), -1)}


def fresh(states: Vector, seen: set[bytes]) -> np.ndarray:
    """The indices of the first copy of each state not already seen."""
    rows = np.rint(states.kernel).astype(np.int8).reshape(len(states), -1)
    _, first = np.unique(rows, axis=0, return_index=True)
    first = np.sort(first)
    return first[[rows[i].tobytes() not in seen for i in first]]


def meeting(these: Vector, those: Vector) -> tuple[np.ndarray, np.ndarray]:
    """The indices of the states the two sets share, in each."""
    index = {row.tobytes(): i for i, row in enumerate(np.rint(those.kernel).astype(np.int8).reshape(len(those), -1))}
    rows = np.rint(these.kernel).astype(np.int8).reshape(len(these), -1)
    here = [i for i, row in enumerate(rows) if row.tobytes() in index]
    return np.array(here, dtype=int), np.array([index[rows[i].tobytes()] for i in here], dtype=int)
