"""Scenes for the Rubik's cube generate a random scramble of quarter turns, solve it by bidirectional breadth-first search, and animate the scramble and solution."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from examples.geometry.rubiks_cube import core

# The scramble's quarter turns and its seed, and how many turns the search goes from each side.
N_SCRAMBLE = 10
SEED = 4
DEPTH = 5
# The animation's frames per quarter turn, and a frame's duration in milliseconds.
N_STEPS = 6
FRAME_DURATION = 60


# --- math -----------------------------------------------------------------------------
def random_turns(rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Random quarter turns without end, each a direction, a side and an axis, none undoing the one
    before."""
    last = rng.integers((2, 2, 3))
    yield last
    while True:
        move = rng.integers((2, 2, 3))
        if (1 - move[0], *move[1:]) != tuple(last):
            last = move
            yield move


def scramble(stickers: core.Point, moves: np.ndarray) -> core.Point:
    """The stickers after the quarter turns `[turns, 3]`, one after another."""
    for direction, side, axis in moves:
        stickers = core.turned(stickers, core.turns[direction, side, axis], core.faces[side, axis])   # [54] Point
    return stickers


def scrambled_and_solved():
    """A scramble of random quarter turns and the turns that solve it, each a direction, a side and an
    axis `[turns, 3]`, with the scrambled cube."""
    solved, _, _ = core.solved_cube()
    moves = np.array(list(islice(random_turns(np.random.default_rng(SEED)), N_SCRAMBLE)))
    scrambled = scramble(solved, moves)
    return moves, core.solve(scrambled, solved, DEPTH), scrambled


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.geometry.rubiks_cube import render

    moves, solution, scrambled = scrambled_and_solved()
    solved, corners, colours = core.solved_cube()
    states = core.animate(solved, corners, np.concatenate([moves, solution]), N_STEPS)
    save_animation(render.animate(states, colours), "rubiks_cube", FRAME_DURATION)

    # checks
    # Every turn moves stickers onto sticker points: the scrambled cube's points are the solved cube's. The
    # solution takes the scrambled cube back exactly, in no more turns than the scramble.
    points = lambda cube: sorted(map(bytes, core.codes(cube[None]).reshape(-1, 3)))
    assert points(scrambled) == points(solved)
    assert (core.codes(scramble(scrambled, solution)[None]) == core.codes(solved[None])).all()
    assert len(solution) <= N_SCRAMBLE


if __name__ == "__main__":
    main()
