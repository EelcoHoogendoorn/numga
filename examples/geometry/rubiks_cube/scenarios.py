"""Scenes for the Rubik's cube generate a random scramble of quarter turns, solve it by bidirectional breadth-first search, and animate the scramble and solution."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from examples.geometry.rubiks_cube import core

# The scramble's quarter-turn count and random seed, and the search depth from each end.
N_SCRAMBLE = 10
SEED = 4
DEPTH = 5
# The animation's frames per quarter turn, and each frame's duration in milliseconds.
N_STEPS = 9
FRAME_DURATION = 40


# --- math -----------------------------------------------------------------------------
def random_turns(rng: np.random.Generator) -> Iterator[np.ndarray]:
    """An endless stream of random quarter turns `(direction, side, axis)` with no move immediately undoing the one before."""
    previous_move = rng.integers((2, 2, 3))
    yield previous_move
    while True:
        move = rng.integers((2, 2, 3))
        if (1 - move[0], *move[1:]) != tuple(previous_move):
            previous_move = move
            yield move


def scramble_and_solution() -> tuple[np.ndarray, np.ndarray, core.Point]:
    """A random scramble of quarter turns, the sequence of turns that solves it, and the scrambled sticker points."""
    solved_stickers, _, _ = core.solved_cube()                              # [stickers] Point-
    moves = np.array(list(islice(random_turns(np.random.default_rng(SEED)), N_SCRAMBLE)))
    scrambled_stickers = core.scramble(solved_stickers, moves)              # [stickers] Point-
    return moves, core.solve(scrambled_stickers, solved_stickers, DEPTH), scrambled_stickers


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.geometry.rubiks_cube import render

    moves, solution, scrambled_stickers = scramble_and_solution()
    solved_stickers, solved_corners, colours = core.solved_cube()
    states = core.animate(solved_stickers, solved_corners, np.concatenate([moves, solution]), N_STEPS)
    save_animation(render.animate(states, colours), "rubiks_cube", FRAME_DURATION)

    # checks
    # Every turn permutes the 54 sticker positions, and the solution restores the solved cube in no more turns than the scramble.
    np.testing.assert_allclose(
        np.unique(np.rint(scrambled_stickers.kernel), axis=0),
        np.unique(np.rint(solved_stickers.kernel), axis=0),
    )
    np.testing.assert_allclose((core.scramble(scrambled_stickers, solution) - solved_stickers).kernel, 0.0, atol=1e-12)
    assert len(solution) <= N_SCRAMBLE


if __name__ == "__main__":
    main()
