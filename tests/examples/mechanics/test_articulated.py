"""The humanoid dropped for one frame of one step: its energy and its forward dynamics computed, and the frame drawn."""

from __future__ import annotations

from examples.mechanics.articulated import core, render, scenarios


def test_a_dropped_humanoid_steps_and_a_frame_draws():
    tree, robot = scenarios.robot()
    state = scenarios.dropped()
    core.energy(tree, robot, scenarios.physics(), scenarios.ground(), scenarios.stops(), state)
    motors = core.forward_kinematics(tree, robot, state.motor, state.angles)
    weights = scenarios.physics().gravity & (motors >> robot.mass_points)      # Forque-[links]
    _, forces = core.inverse_dynamics(tree, robot, motors, state.rate, state.speeds, state.rate * 0, state.speeds * 0,
                                      weights)
    core.forward_dynamics(tree, robot, motors, state.rate, state.speeds, forces, weights)
    _, _, _, _, radii = scenarios.home_pose()
    centres = scenarios.fall(state, n_frames=1, n_steps=1)
    list(render.animate(centres, radii, len(scenarios.capsule_bodies)))
