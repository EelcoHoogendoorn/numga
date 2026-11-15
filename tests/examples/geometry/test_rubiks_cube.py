"""Tests for the Rubik's cube verify that quarter turns are exact and cycle in fours, the batched layer turn reaches every child state, and a one-turn scramble solves and draws."""

from __future__ import annotations

import numpy as np

from examples.geometry.rubiks_cube import core, render, scenarios


def test_quarter_turns_are_exact_cycle_in_fours_and_undo():
    solved_stickers, _, _ = core.solved_cube()
    for direction, side, axis in np.ndindex(2, 2, 3):
        cube = solved_stickers
        for _ in range(4):
            cube = core.turned_layer(cube, core.turns[direction, side, axis], core.faces[side, axis])
        assert ((cube - solved_stickers).scalar_norm_squared() < 1e-12).all()
        turned_stickers = core.turned_layer(solved_stickers, core.turns[direction, side, axis], core.faces[side, axis])
        restored_cube = core.turned_layer(turned_stickers, core.turns[1 - direction, side, axis], core.faces[side, axis])
        assert ((restored_cube - solved_stickers).scalar_norm_squared() < 1e-12).all()


def test_the_frontier_is_every_turn_of_every_state():
    solved_stickers, _, _ = core.solved_cube()
    states = core.turned_layer(solved_stickers[None], core.turns[..., None, None], core.faces[..., None, None]).reshape(-1, 54)  # [12, 54] Point-
    children = core.turned_layer(states, core.turns[..., None, None], core.faces[..., None, None])  # [2, 2, 3, 12, 54] Point-
    for direction, side, axis, parent in np.ndindex(2, 2, 3, 12):
        expected = core.turned_layer(states[parent], core.turns[direction, side, axis], core.faces[side, axis])
        assert ((children[direction, side, axis, parent] - expected).scalar_norm_squared() < 1e-12).all()


def test_a_one_turn_scramble_solves_and_draws():
    _, solution, _ = scenarios.scramble_and_solution(1, 1)
    solved_stickers, solved_corners, colours = core.solved_cube()
    next(render.cube_animation(colours, core.turn_frames(solved_stickers, solved_corners, solution, 1)))
