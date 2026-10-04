"""The Rubik's cube: quarter turns are exact and cycle in fours, the batched frontier is every turn of
every state, and the scene solves its scramble and draws."""

from __future__ import annotations

import numpy as np

from examples.geometry.rubiks_cube import core, render, scenarios


def test_quarter_turns_are_exact_cycle_in_fours_and_undo():
    solved, _, _ = core.solved_cube()
    for direction, side, axis in np.ndindex(2, 2, 3):
        cube = solved
        for _ in range(4):
            cube = core.turned(cube, core.turns[direction, side, axis], core.faces[side, axis])
        np.testing.assert_array_equal(cube.kernel, solved.kernel)
        once = core.turned(solved, core.turns[direction, side, axis], core.faces[side, axis])
        back = core.turned(once, core.turns[1 - direction, side, axis], core.faces[side, axis])
        np.testing.assert_array_equal(back.kernel, solved.kernel)


def test_the_frontier_is_every_turn_of_every_state():
    solved, _, _ = core.solved_cube()
    states = core.frontier(solved[None]).reshape(-1, 54)                       # [12, 54] Vector
    children = core.frontier(states)                                           # [2, 2, 3, 12, 54] Vector
    for direction, side, axis, parent in np.ndindex(2, 2, 3, 12):
        expected = core.turned(states[parent], core.turns[direction, side, axis], core.faces[side, axis])
        np.testing.assert_array_equal(children[direction, side, axis, parent].kernel, expected.kernel)


def test_the_scene_solves_and_draws():
    scramble, solution = scenarios.scrambled_and_solved()
    solved, corners, colours = core.solved_cube()
    images = render.animate(core.animate(solved, corners, solution[:1], 2), colours)
    assert len(images) == 3
