"""Finite-horizon optimality, physical thruster work, and docking trajectories."""

import numpy as np

from numga import NumpyContext, stack
from examples.mechanics.riccati import core, scenarios


def condensed_control(dynamics: np.ndarray, actuation: np.ndarray,
                      state_cost: np.ndarray, effort_cost: np.ndarray,
                      terminal: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Minimize over every control at once, independently of the Riccati recursion."""
    dimension, controls = actuation.shape
    initial_map = np.eye(dimension)
    control_map = np.zeros((dimension, steps * controls))
    hessian = np.kron(np.eye(steps), effort_cost)
    cross = np.zeros((steps * controls, dimension))
    initial_cost = np.zeros((dimension, dimension))
    for step in range(steps):
        hessian += control_map.T @ state_cost @ control_map
        cross += control_map.T @ state_cost @ initial_map
        initial_cost += initial_map.T @ state_cost @ initial_map
        initial_map = dynamics @ initial_map
        control_map = dynamics @ control_map
        control_map[:, step * controls:(step + 1) * controls] += actuation
    hessian += control_map.T @ terminal @ control_map
    cross += control_map.T @ terminal @ initial_map
    initial_cost += initial_map.T @ terminal @ initial_map
    optimum = -np.linalg.solve(hessian, cross)
    return optimum.reshape(steps, controls, dimension), initial_cost + cross.T @ optimum


def test_riccati_matches_a_single_dense_optimization_with_general_dynamics_and_actuation():
    context = NumpyContext(core.ga)
    steps = 9
    dynamics = np.array([
        [[1.03, 0.14, -0.04], [0, 0.96, 0.12], [0.03, 0, 1.01]],
        [[0.93, -0.18, 0.05], [0.09, 1.04, -0.02], [0, 0.07, 0.91]],
    ])
    actuation = np.array([
        [[0.22, 0.05, 0], [-0.03, 0.19, 0.04], [0.01, 0.02, 0.16]],
        [[0.17, -0.03, 0.02], [0.04, 0.23, 0], [0.01, -0.04, 0.18]],
    ])
    state_cost = np.array([
        [[2, 0.3, -0.2], [0.3, 1.4, 0.1], [-0.2, 0.1, 0.9]],
        [[1.1, -0.1, 0.2], [-0.1, 2.3, -0.3], [0.2, -0.3, 1.7]],
    ])
    effort_cost = np.array([
        [[0.8, 0.1, 0], [0.1, 0.6, -0.1], [0, -0.1, 0.5]],
        [[0.5, -0.05, 0.1], [-0.05, 0.7, 0], [0.1, 0, 1.0]],
    ])
    terminal = 3 * state_cost
    # A form's kernel is its matrix on coefficients, and a map's kernel acts on them.
    dynamics_map = context.extensor(core.Dynamics, dynamics)
    actuation_map = context.extensor(core.Actuation, actuation)
    values, backward_gains = zip(*core.riccati(
        context.extensor(core.StateCost, terminal[:, None]), dynamics_map, actuation_map,
        context.extensor(core.StateCost, state_cost[:, None]),
        context.extensor(core.EffortCost, effort_cost[:, None]), steps,
    ))
    expected_actions, expected_values = zip(*(
        condensed_control(drift, response, tracking, effort, final, steps)
        for drift, response, tracking, effort, final in
        zip(dynamics, actuation, state_cost, effort_cost, terminal)
    ))
    expected_actions, expected_values = np.stack(expected_actions), np.stack(expected_values)
    np.testing.assert_allclose(values[-1].kernel, expected_values[:, None], rtol=0, atol=1e-10)

    # Three independent initial errors test the policy's action on the whole state space.
    initial = core.mv.bivector(np.eye(3)).broadcast_to((len(actuation), 3))
    feedbacks = stack(backward_gains[::-1])
    errors = stack(tuple(core.rollout(
        initial, dynamics_map[:, None], actuation_map[:, None], feedbacks[:, :, None],
    )))
    pushes = feedbacks[:, :, None](errors[:-1])
    np.testing.assert_allclose(pushes.kernel, expected_actions.transpose(1, 0, 3, 2),
                               rtol=0, atol=1e-10)


def test_docking_realizes_its_bellman_cost_in_tracking_displacements_and_commands():
    values, feedbacks, errors, commands = scenarios.docking()
    lines = scenarios.TRACKING_POINTS[:, None] & scenarios.AXES[None, :]
    displacements = lines & errors[..., None, None]
    tracking = (displacements.squared() * scenarios.TRACKING_WEIGHTS).sum(axis=(-2, -1)) * scenarios.DT
    command_effort = (commands.squared() * scenarios.THRUSTER_WEIGHTS).sum(axis=-1)
    running = tracking[:-1] + command_effort * scenarios.EFFORT_SCALES * scenarios.DT
    landing = tracking[-1] * scenarios.LANDING
    predicted = values[-1](errors[0], errors[0])
    np.testing.assert_allclose((running.sum(axis=0) + landing - predicted).kernel, 0, atol=1e-9)

    # What is left to pay falls by exactly each step's running cost, down to the landing's.
    remaining = values[::-1](errors[:-1], errors[:-1])
    decrease = remaining[:-1] - remaining[1:]
    np.testing.assert_allclose((decrease - running[:-1]).kernel, 0, atol=1e-10)
    np.testing.assert_allclose((remaining[-1] - running[-1] - landing).kernel, 0, atol=1e-10)
    # Both approaches reach the target by the deadline, to within a millimetre.
    np.testing.assert_allclose(errors[-1].kernel, 0, atol=1e-3)


def test_thruster_commands_reproduce_the_feedback_forque_and_geometric_work():
    _, feedbacks, errors, commands = scenarios.docking()
    thrusters = scenarios.MOUNTS & scenarios.DIRECTIONS
    supplied = (commands * thrusters).sum(axis=-1)
    requested = feedbacks(errors[:-1])
    np.testing.assert_allclose((supplied - requested).kernel, 0, atol=1e-10)

    # A thruster's work on a twist is its force dotted into the velocity the twist gives its mount;
    # this pins the sign of the motor convention against the physical thrust arrows.
    trials = core.mv.bivector([[0.2, -0.3, 0.15], [-0.1, 0.25, -0.12]])[:, None]
    velocities = scenarios.MOUNTS.commutator(trials)
    power = scenarios.DIRECTIONS.dual() | velocities.dual()
    np.testing.assert_allclose((power - (thrusters & trials)).kernel, 0, atol=1e-10)


def test_docking_figures_and_animation_draw_without_saving():
    import matplotlib.pyplot as plt

    from examples.mechanics.riccati import render

    values, _, errors, commands = scenarios.docking()
    poses = (errors * -0.5).exp()
    figures = (
        render.draw_setup(scenarios.HULL, scenarios.MOUNTS, scenarios.DIRECTIONS,
                          scenarios.TRACKING_POINTS),
        render.draw_approaches(scenarios.HULL, poses, scenarios.LABELS),
        render.draw_costs(values, scenarios.LABELS),
    )
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)
    frames = render.animate(scenarios.HULL, scenarios.MOUNTS, scenarios.DIRECTIONS, poses[:2],
                            commands[:2], scenarios.LABELS, scenarios.ARROW_SCALE)
    assert frames[0].ndim == 3 and frames[0].shape == frames[1].shape
    assert not np.array_equal(frames[0], frames[1])
