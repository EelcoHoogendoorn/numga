"""A Rubik's cube models its 54 stickers as integer-coordinate points in 3D space that are turned, drawn, and solved with the same quarter-turn rotor maps.

Each sticker is a point at integer coordinates in half-cubie units, and each face is a plane whose pairing `face & stickers > 1` selects the stickers in its turning layer. A quarter turn of a face is the rotor `1 + plane` for the bivector plane the face turns in; its sandwich doubles lengths, so as a halved map on points, `((1 + plane) >> Point) * 0.5`, it turns by a quarter turn with coefficients 0 and ±1 only. Stickers keep integer coordinates through any number of turns, so cube states compare without round-off. Applying the batch of twelve quarter-turn maps to a batch of states turns every state every way at once, and bidirectional breadth-first search widens the states reachable from the scrambled and solved cubes one quarter turn at a time until the two frontiers share a state.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import VGA3D as ga

mv = NumpyContext(ga).multivector
# Stickers, cubies, and corners are points (antivectors), and faces are planes through the origin (vectors).
Point = ga.gatype.antivector()                                              # Antivector-
Plane = ga.gatype.vector()                                                  # Vector+
Turn = ga.gatype((Point, Point))                                            # Point- <- Point-

# Each of the six faces is a plane along an axis on either side; `face & point` reads how far a point reaches along it.
sides = np.array([1, -1])                                                   # [2]
faces = sides[:, None] * mv.basis()                                         # [2, 3] Plane+
# Each face turns in the bivector plane perpendicular to its axis, oriented clockwise seen from outside as in Singmaster's notation.
planes = sides[:, None] * stack([mv.yz, mv.zx, mv.xy])                      # [2, 3] Bivector+
# Each quarter turn is a map on points, clockwise and anticlockwise around each face.
directions = np.array([1, -1])                                              # [2]
turns = ((1 + directions[:, None, None] * planes) >> Point) * 0.5           # [2, 2, 3] Point- <- Point-


# --- math -----------------------------------------------------------------------------
# The 27 cubie centres lie at -1, 0, and 1 along each axis, and a cubie shows the faces whose side it lies on.
cubies = mv.vector(np.stack(np.meshgrid(*[[-1, 0, 1]] * 3, indexing="ij"), axis=-1).reshape(-1, 3)).dual()  # [27] Point-
outward_faces = (faces & cubies[:, None, None]) > 0.5                       # [27, 2, 3]


def face_corners() -> Point:
    """The four corners of each face of a cubie lie at integer half-cubie offsets ordered around the face by quarter turns (`[2, 3, 4] Point-`)."""
    first_corner = (mv.vector([1, 1, 1]) - mv.basis() + faces).dual()       # [2, 3] Point-
    corners = [first_corner]
    for _ in range(3):
        corners.append(turns[0](corners[-1]))
    return stack(corners, axis=-1)                                          # [2, 3, 4] Point-


squares = face_corners()                                                    # [2, 3, 4] Point-


def solved_cube() -> tuple[Point, Point, np.ndarray]:
    """The solved cube places 54 sticker centres (`[54] Point-`), their four corners each (`[54, 4] Point-`), and each sticker's face index (`[54]`)."""
    stickers = (2 * cubies[:, None, None] + faces.dual())[outward_faces]    # [54] Point-
    corners = (2 * cubies[:, None, None, None] + squares)[outward_faces]    # [54, 4] Point-
    _, side, axis = np.nonzero(outward_faces)
    return stickers, corners, side * 3 + axis


def turned_layer(stickers: Point, turn: Turn, face: Plane) -> Point:
    """A quarter turn of `face` turns the stickers in its outer layer (`face & stickers > 1`) while leaving the remaining stickers fixed (`[..., 54] Point-`)."""
    layer = (face & stickers) > 1                                           # [..., 54]
    return stickers + layer * (turn(stickers) - stickers)                   # [..., 54] Point-


def scrambled_cube(stickers: Point, moves: np.ndarray) -> Point:
    """Applying the quarter turns in `moves` in order carries `stickers` to a scrambled cube (`[54] Point-`)."""
    for direction, side, axis in moves:
        stickers = turned_layer(stickers, turns[direction, side, axis], faces[side, axis])  # [54] Point-
    return stickers


def state_codes(states: Point) -> np.ndarray:
    """Each state's 54 sticker face incidences `faces & states > 2` pack into exact bytes (`[..., 41]`) that uniquely identify the cube state."""
    incidences = (faces.reshape(-1) & states[..., None]) > 2                # [..., 54, 6]
    return np.packbits(incidences.reshape(len(states), -1), axis=-1)        # [..., 41]


def next_frontier(states: Point, parents: list[np.ndarray], moves: list[np.ndarray],
                  visited_keys: set[bytes]) -> Point:
    """One batched step of all twelve quarter turns widens `states` to newly visited child states (`[..., 54] Point-`) and records their parents and moves."""
    candidates = turned_layer(states, turns[..., None, None], faces[..., None, None]).reshape(-1, 54)  # [..., 54] Point-
    rows = state_codes(candidates)                                          # [..., 41]
    _, first_indices = np.unique(rows, axis=0, return_index=True)
    first_indices = np.sort(first_indices)
    kept_indices = first_indices[[rows[i].tobytes() not in visited_keys for i in first_indices]]
    visited_keys.update(rows[i].tobytes() for i in kept_indices)
    *move, parent = np.unravel_index(kept_indices, (2, 2, 3, len(states)))
    parents.append(parent)
    moves.append(np.stack(move, axis=-1))
    return candidates[kept_indices]                                         # [..., 54] Point-


def move_path(parents: list[np.ndarray], moves: list[np.ndarray], index: int) -> np.ndarray:
    """Walking parent links back from state `index` recovers the sequence of quarter turns `(direction, side, axis)` from the root (`[..., 3]`)."""
    path_moves = []
    for level in range(len(moves) - 1, -1, -1):
        path_moves.append(moves[level][index])
        index = parents[level][index]
    return np.array(path_moves[::-1], dtype=int).reshape(-1, 3)             # [..., 3]


def solution_moves(scrambled_stickers: Point, solved_stickers: Point, n_levels: int) -> np.ndarray:
    """Bidirectional breadth-first search up to `n_levels` quarter turns from each end finds a sequence of turns `(direction, side, axis)` from `scrambled_stickers` to `solved_stickers` (`[..., 3]`)."""
    frontiers = [scrambled_stickers[None], solved_stickers[None]]           # [2][..., 54] Point-
    parents: list[list[np.ndarray]] = [[], []]
    moves: list[list[np.ndarray]] = [[], []]
    visited = [{state_codes(frontiers[0])[0].tobytes()}, {state_codes(frontiers[1])[0].tobytes()}]
    for step in range(2 * n_levels):
        end = step % 2
        frontiers[end] = next_frontier(frontiers[end], parents[end], moves[end], visited[end])
        lookup = {row.tobytes(): i for i, row in enumerate(state_codes(frontiers[1]))}
        for forward_index, row in enumerate(state_codes(frontiers[0])):
            backward_index = lookup.get(row.tobytes())
            if backward_index is not None:
                backward_moves = move_path(parents[1], moves[1], backward_index)[::-1]  # [..., 3]
                return np.concatenate([
                    move_path(parents[0], moves[0], forward_index),
                    np.concatenate([1 - backward_moves[:, :1], backward_moves[:, 1:]], axis=-1),
                ])
    return np.zeros((0, 3), dtype=int)


def turn_frames(stickers: Point, corners: Point, moves: np.ndarray, n_steps: int) -> Iterator[tuple[Point, Point]]:
    """Yields the sticker corners (`[54, 4] Point-`) and the two sides of the inner cut (`[2, 9, 4] Point-`) across `moves`, interpolating each quarter turn smoothly over `n_steps` frames."""
    for direction, side, axis in moves:
        layer = (faces[side, axis] & stickers) > 1                          # [54]
        # The cut consists of the inner square faces of the nine cubies in the turning layer.
        cut = 2 * cubies[outward_faces[:, side, axis]][:, None] + squares[1 - side, axis]  # [9, 4] Point-
        for fraction in np.arange(n_steps) / n_steps:
            rotor = (planes[side, axis] * (np.pi / 4 * fraction * directions[direction])).exp()  # [] Rotor+
            turned_corners = rotor >> corners                               # [54, 4] Point-
            turned_cut = rotor >> cut                                       # [9, 4] Point-
            yield corners + layer[:, None] * (turned_corners - corners), stack([cut, turned_cut])
        turn, face = turns[direction, side, axis], faces[side, axis]
        stickers = turned_layer(stickers, turn, face)                       # [54] Point-
        corners = corners + layer[:, None] * (turn(corners) - corners)      # [54, 4] Point-
    yield corners, stack([cut, cut])
