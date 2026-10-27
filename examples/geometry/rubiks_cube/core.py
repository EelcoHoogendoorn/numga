"""A Rubik's cube models its 54 stickers as integer-coordinate points in 3D space that are turned, drawn, and solved with the same quarter-turn rotor maps.

Each sticker is a point at integer coordinates in half-cubie units, and each face is a plane whose pairing `face & stickers > 1` selects the stickers in its turning layer. A quarter turn of a face is the rotor `1 + plane` for the bivector plane the face turns in; its sandwich doubles lengths, so as a halved map on points, `((1 + plane) >> Point) * 0.5`, it turns by a quarter turn with coefficients 0 and ±1 only. Stickers keep integer coordinates through any number of turns, so cube states compare without round-off. Applying the batch of twelve quarter-turn maps to a batch of states turns every state every way at once, and bidirectional breadth-first search widens the states reachable from the scrambled and solved cubes one quarter turn at a time until the two frontiers share a state.

In Singmaster's notation the faces along $+x, -x, +y, -y, +z,$ and $-z$ read as $\\mathrm{R}, \\mathrm{L}, \\mathrm{B}, \\mathrm{F}, \\mathrm{U},$ and $\\mathrm{D}$ for clockwise quarter turns seen from outside and $\\mathrm{R}', \\mathrm{L}', \\mathrm{B}', \\mathrm{F}', \\mathrm{U}',$ and $\\mathrm{D}'$ for their inverses, generating the Rubik's cube group as a subgroup of the symmetric group $S_{48}$ on the 48 moving facelets.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# Stickers, cubies, and corners are points (antivectors), and faces are planes through the origin (vectors).
Point = ga.gatype.antivector()                      # Antivector-
Plane = ga.gatype.vector()                          # Vector+
Turn = ga.gatype((Point, Point))

# Each of the six faces is a plane along an axis on either side; `face & point` reads how far a point reaches along it.
sides = np.array([1, -1])
faces = sides[:, None] * mv.basis()                                         # [sides, axes] Plane+
# Each face turns in the bivector plane perpendicular to its axis, oriented clockwise seen from outside as in Singmaster's notation.
planes = sides[:, None] * stack([mv.yz, mv.zx, mv.xy])                      # [sides, axes] Bivector+
# Each quarter turn is a map on points, clockwise and anticlockwise around each face.
directions = np.array([1, -1])
turns = ((1 + directions[:, None, None] * planes) >> Point) * 0.5           # [directions, sides, axes] Point- <- Point-


# --- math -----------------------------------------------------------------------------
def turn_layer(stickers: Point, turn: Turn, face: Plane) -> Point:
    """The stickers after a quarter turn of `face` turns the stickers in its outer layer and leaves the rest in place."""
    layer = (face & stickers) > 1                                           # [...]
    return stickers + layer * (turn(stickers) - stickers)                   # [...] Point-


def scramble(stickers: Point, moves: np.ndarray) -> Point:
    """The stickers after applying the sequence of quarter turns `moves` one after another."""
    for direction, side, axis in moves:
        stickers = turn_layer(stickers, turns[direction, side, axis], faces[side, axis])  # [stickers] Point-
    return stickers


def solve(scrambled_stickers: Point, solved_stickers: Point, depth: int) -> np.ndarray:
    """The sequence of quarter turns `(direction, side, axis)` that carries `scrambled_stickers` to `solved_stickers`, found by bidirectional search up to `depth` turns from each end."""
    searches = [Search(scrambled_stickers[None]), Search(solved_stickers[None])]
    for level in range(2 * depth):
        active, opposite = searches[level % 2], searches[1 - level % 2]
        active.widen()
        active_matches, opposite_matches = meeting(active.states[-1], opposite.states[-1])
        if len(active_matches):
            forward, backward = (active, opposite) if level % 2 == 0 else (opposite, active)
            forward_end, backward_end = (active_matches[0], opposite_matches[0]) if level % 2 == 0 else (opposite_matches[0], active_matches[0])
            backward_moves = backward.path(backward_end)[::-1]              # [turns, 3]
            return np.concatenate([forward.path(forward_end), np.concatenate([1 - backward_moves[:, :1], backward_moves[:, 1:]], axis=-1)])
    raise ValueError(f"no solution within {depth} quarter turns from each side")


def animate(stickers: Point, corners: Point, moves: np.ndarray, n_steps: int) -> Iterator[tuple[Point, Point]]:
    """The sticker corners and the two sides of the inner cut across `moves`, with each quarter turn interpolated smoothly over `n_steps` frames."""
    for direction, side, axis in moves:
        layer = (faces[side, axis] & stickers) > 1                          # [stickers]
        # The cut consists of the inner square faces of the turning layer's cubies.
        cut = 2 * cubies[outward[:, side, axis]][:, None] + squares[1 - side, axis]  # [layer_cubies, corners] Point-
        for fraction in np.arange(n_steps) / n_steps:
            rotor = (planes[side, axis] * (np.pi / 4 * fraction * directions[direction])).exp()  # [] Rotor+
            yield corners + layer[:, None] * ((rotor >> corners) - corners), stack([cut, rotor >> cut])
        turn, face = turns[direction, side, axis], faces[side, axis]
        stickers, corners = turn_layer(stickers, turn, face), corners + layer[:, None] * (turn(corners) - corners)
    yield corners, stack([cut, cut])


# --- plumbing -------------------------------------------------------------------------
# The 27 cubie centres lie at -1, 0, and 1 along each axis, and a cubie shows the faces whose side it lies on.
cubies = mv.vector(np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)).dual()  # [cubies] Point-
outward = (faces & cubies[:, None, None]) > 0.5                             # [cubies, sides, axes]


def face_corners() -> Point:
    """The four corners of each face of a cubie about its centre, ordered around the face by quarter turns."""
    first_corner = (mv.vector([1, 1, 1]) - mv.basis() + faces).dual()       # [sides, axes] Point-
    corners = [first_corner]
    for _ in range(3):
        corners.append(turns[0](corners[-1]))
    return stack(corners, axis=-1)                                          # [sides, axes, corners] Point-


squares = face_corners()                                                    # [sides, axes, corners] Point-


def solved_cube() -> tuple[Point, Point, np.ndarray]:
    """The solved cube's 54 sticker points, their four corners each, and the face index `side * 3 + axis` of each sticker."""
    stickers = (2 * cubies[:, None, None] + faces.dual())[outward]          # [stickers] Point-
    corners = (2 * cubies[:, None, None, None] + squares)[outward]          # [stickers, corners] Point-
    _, side, axis = np.nonzero(outward)
    return stickers, corners, side * 3 + axis


class Search:
    """A breadth-first search tree records the states reached from a starting cube one quarter turn at a time, together with each state's parent index and move."""

    def __init__(self, start: Point) -> None:
        self.states = [start]                                               # [levels][states, stickers] Point-
        self.parents = [np.zeros(1, dtype=int)]
        self.moves = [np.zeros((1, 3), dtype=int)]
        self.visited_keys: set[bytes] = set()
        fresh(start, self.visited_keys)

    def widen(self) -> None:
        """Advance one quarter turn from the last level, keeping only states not visited before."""
        candidates = turn_layer(self.states[-1], turns[..., None, None], faces[..., None, None]).reshape(-1, 54)  # [candidates, stickers] Point-
        indices = fresh(candidates, self.visited_keys)
        self.states.append(candidates[indices])
        *move, parent = np.unravel_index(indices, (2, 2, 3, len(self.states[-2])))
        self.parents.append(parent)
        self.moves.append(np.stack(move, axis=-1))

    def path(self, index: int) -> np.ndarray:
        """The sequence of moves `(direction, side, axis)` from the starting cube to state `index` of the last level."""
        moves = []
        for level in range(len(self.states) - 1, 0, -1):
            moves.append(self.moves[level][index])
            index = self.parents[level][index]
        return np.array(moves[::-1], dtype=int).reshape(-1, 3)


def codes(states: Point) -> np.ndarray:
    """Each state's sticker coordinates packed as a row of exact small integers for state comparison."""
    return np.rint(states.kernel).astype(np.int8).reshape(len(states), -1)


def fresh(states: Point, visited_keys: set[bytes]) -> np.ndarray:
    """The indices of the first occurrence of each state not in `visited_keys`, recording the newly visited keys."""
    rows = codes(states)
    _, first_indices = np.unique(rows, axis=0, return_index=True)
    first_indices = np.sort(first_indices)
    kept = first_indices[[rows[i].tobytes() not in visited_keys for i in first_indices]]
    visited_keys.update(rows[i].tobytes() for i in kept)
    return kept


def meeting(first_states: Point, second_states: Point) -> tuple[np.ndarray, np.ndarray]:
    """The matching state indices in `first_states` and `second_states` where the two frontiers share a state."""
    lookup = {row.tobytes(): i for i, row in enumerate(codes(second_states))}
    rows = codes(first_states)
    first_matches = [i for i, row in enumerate(rows) if row.tobytes() in lookup]
    return np.array(first_matches, dtype=int), np.array([lookup[rows[i].tobytes()] for i in first_matches], dtype=int)
