"""Scenes for the Rubik's cube generate a random scramble of quarter turns, solve it by bidirectional breadth-first search, and animate the scramble and solution."""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from examples.geometry.rubiks_cube import core

# The scramble's quarter-turn count and random seed, and the search depth from each end.
n_scramble = 10
seed = 4
n_levels = 5
# The animation's frames per quarter turn, and each frame's duration in milliseconds.
n_steps = 9
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def random_turns(rng: np.random.Generator) -> Iterator[np.ndarray]:
    """Yields random quarter turns `(direction, side, axis)` (`[3]`) with no move immediately undoing the one before."""
    previous_move = rng.integers((2, 2, 3))
    yield previous_move
    while True:
        move = rng.integers((2, 2, 3))
        if (1 - move[0], *move[1:]) != tuple(previous_move):
            previous_move = move
            yield move


def scramble_moves(n_moves: int, rng_seed: int) -> np.ndarray:
    """Draws `n_moves` non-cancelling quarter turns `(direction, side, axis)` from `rng_seed` (`[n_moves, 3]`)."""
    return np.array(list(islice(random_turns(np.random.default_rng(rng_seed)), n_moves)))  # [n_moves, 3]


def scramble_and_solution(n_moves: int, search_levels: int) -> tuple[np.ndarray, np.ndarray, core.Point]:
    """Scrambles the solved cube by `n_moves` random quarter turns and solves it within `search_levels` turns from each end (`[n_moves, 3]`, `[..., 3]`, `[54] Point-`)."""
    solved_stickers, _, _ = core.solved_cube()                              # [54] Point-
    moves = scramble_moves(n_moves, seed)                                   # [n_moves, 3]
    scrambled_stickers = core.scrambled_cube(solved_stickers, moves)        # [54] Point-
    return moves, core.solution_moves(scrambled_stickers, solved_stickers, search_levels), scrambled_stickers


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.geometry.rubiks_cube import render

    moves, solution, scrambled_stickers = scramble_and_solution(n_scramble, n_levels)
    solved_stickers, solved_corners, colours = core.solved_cube()
    recorded_states: list[tuple[core.Point, core.Point]] = []

    def streamed_states() -> Iterator[tuple[core.Point, core.Point]]:
        for state in core.turn_frames(solved_stickers, solved_corners, np.concatenate([moves, solution]), n_steps):
            recorded_states.append(state)
            yield state

    save_animation(render.cube_animation(colours, streamed_states()), "rubiks_cube", frame_duration_ms)

    # checks
    # Every turn permutes the 54 sticker positions, and the streamed scramble-then-solution restores all sticker corners.
    matches = (scrambled_stickers[:, None] - solved_stickers[None, :]).scalar_norm_squared() < 1e-12
    assert (matches.sum(axis=1) == 1).all() and (matches.sum(axis=0) == 1).all()
    final_corners, _ = recorded_states[-1]
    assert ((final_corners - solved_corners).scalar_norm_squared() < 1e-12).all()
    assert len(solution) <= n_scramble


if __name__ == "__main__":
    main()
