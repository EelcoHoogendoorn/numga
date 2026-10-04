"""Scenes for the cube: a scramble of random quarter turns, solved by meeting in the middle, and the
cube turned through the scramble and back through the solution."""

from __future__ import annotations

import numpy as np

from examples.geometry.rubiks_cube import core

# The scramble's quarter turns and its seed, and how many turns the search goes from each side.
SCRAMBLE = 10
SEED = 4
DEPTH = 5
# The animation's frames per quarter turn.
STEPS = 6


# --- math -----------------------------------------------------------------------------
def scrambled_and_solved():
    """A scramble of random quarter turns, none undoing the one before, and the turns that solve it,
    each a direction, a side and an axis `[turns, 3]`."""
    solved, _, _ = core.solved_cube()
    rng = np.random.default_rng(SEED)
    scramble = [rng.integers((2, 2, 3))]
    while len(scramble) < SCRAMBLE:
        move = rng.integers((2, 2, 3))
        scramble += [move] if (1 - move[0], *move[1:]) != tuple(scramble[-1]) else []
    scrambled = solved
    for direction, side, axis in scramble:
        scrambled = core.turned(scrambled, core.turns[direction, side, axis], core.faces[side, axis])   # [54] Vector
    solution = core.solve(scrambled, solved, DEPTH)

    # --- checks
    # Every turn moves stickers onto sticker points: the scrambled cube's points are the solved cube's.
    order = lambda cube: np.sort(np.rint(cube.kernel).astype(int).view([("", int)] * 3).ravel())
    np.testing.assert_array_equal(order(scrambled), order(solved))
    # The solution takes the scrambled cube back exactly, in no more turns than the scramble.
    cube = scrambled
    for direction, side, axis in solution:
        cube = core.turned(cube, core.turns[direction, side, axis], core.faces[side, axis])
    np.testing.assert_array_equal(cube.kernel, solved.kernel)
    assert len(solution) <= SCRAMBLE
    return np.array(scramble), solution


if __name__ == "__main__":
    from examples.animation import save_animation
    from examples.geometry.rubiks_cube import render

    scramble, solution = scrambled_and_solved()
    solved, corners, colours = core.solved_cube()
    states = core.animate(solved, corners, np.concatenate([scramble, solution]), STEPS)
    save_animation(render.animate(states, colours), "rubiks_cube", 60)
