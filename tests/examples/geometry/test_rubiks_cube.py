"""Tests for the Rubik's cube verify that quarter turns are exact and cycle in fours, the batched layer turn reaches every child state, and the scene solves its scramble and renders."""

from __future__ import annotations

import numpy as np

from examples.geometry.rubiks_cube import core, render, scenarios


def test_quarter_turns_are_exact_cycle_in_fours_and_undo():
    solved_stickers, _, _ = core.solved_cube()
    for direction, side, axis in np.ndindex(2, 2, 3):
        cube = solved_stickers
        for _ in range(4):
            cube = core.turn_layer(cube, core.turns[direction, side, axis], core.faces[side, axis])
        np.testing.assert_array_equal(cube.kernel, solved_stickers.kernel)
        after_turn = core.turn_layer(solved_stickers, core.turns[direction, side, axis], core.faces[side, axis])
        restored_cube = core.turn_layer(after_turn, core.turns[1 - direction, side, axis], core.faces[side, axis])
        np.testing.assert_array_equal(restored_cube.kernel, solved_stickers.kernel)


def test_the_frontier_is_every_turn_of_every_state():
    solved_stickers, _, _ = core.solved_cube()
    states = core.turn_layer(solved_stickers[None], core.turns[..., None, None], core.faces[..., None, None]).reshape(-1, 54)  # [states, stickers] Point-
    children = core.turn_layer(states, core.turns[..., None, None], core.faces[..., None, None])  # [directions, sides, axes, states, stickers] Point-
    for direction, side, axis, parent in np.ndindex(2, 2, 3, 12):
        expected = core.turn_layer(states[parent], core.turns[direction, side, axis], core.faces[side, axis])
        np.testing.assert_array_equal(children[direction, side, axis, parent].kernel, expected.kernel)


def test_the_scene_solves_and_draws():
    _, solution, _ = scenarios.scramble_and_solution()
    solved_stickers, solved_corners, colours = core.solved_cube()
    images = list(render.animate(core.animate(solved_stickers, solved_corners, solution[:1], 2), colours))
    assert len(images) == 3
