"""Tests finite-horizon Riccati optimality, Bellman cost accounting, thruster work, and single-step rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext, concatenate, stack
from examples.mechanics.riccati import core, scenarios


def condensed_control(dynamics: np.ndarray, actuation: np.ndarray,
                      state_cost: np.ndarray, effort_cost: np.ndarray,
                      terminal_cost: np.ndarray, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Solves the condensed multi-step quadratic program directly for comparison with the Riccati recursion."""
    n_states, n_controls = actuation.shape
    initial_map = np.eye(n_states)
    control_map = np.zeros((n_states, n_steps * n_controls))
    hessian = np.kron(np.eye(n_steps), effort_cost)
    cross = np.zeros((n_steps * n_controls, n_states))
    initial_cost = np.zeros((n_states, n_states))
    for step in range(n_steps):
        hessian += control_map.T @ state_cost @ control_map
        cross += control_map.T @ state_cost @ initial_map
        initial_cost += initial_map.T @ state_cost @ initial_map
        initial_map = dynamics @ initial_map
        control_map = dynamics @ control_map
        control_map[:, step * n_controls:(step + 1) * n_controls] += actuation
    hessian += control_map.T @ terminal_cost @ control_map
    cross += control_map.T @ terminal_cost @ initial_map
    initial_cost += initial_map.T @ terminal_cost @ initial_map
    optimum = -np.linalg.solve(hessian, cross)
    return optimum.reshape(n_steps, n_controls, n_states), initial_cost + cross.T @ optimum


def test_riccati_matches_dense_optimization_with_general_dynamics_and_actuation():
    """Verifies that backward Riccati gains and value forms match a single condensed quadratic solve."""
    context = NumpyContext(core.ga)
    n_steps = 4
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
    terminal_cost = 3 * state_cost
    dynamics_map = context.extensor(core.Dynamics, dynamics)
    actuation_map = context.extensor(core.Actuation, actuation)
    effort_forms = context.extensor(core.EffortCost, effort_cost[:, None])
    values, backward_gains = zip(*core.riccati(
        context.extensor(core.StateCost, terminal_cost[:, None]), dynamics_map, actuation_map,
        context.extensor(core.StateCost, state_cost[:, None]),
        effort_forms, n_steps,
    ))
    expected_actions, expected_values = zip(*(
        condensed_control(drift, response, tracking, effort, terminal, n_steps)
        for drift, response, tracking, effort, terminal in
        zip(dynamics, actuation, state_cost, effort_cost, terminal_cost)
    ))
    expected_actions, expected_values = np.stack(expected_actions), np.stack(expected_values)
    expected_value_forms = context.extensor(core.StateCost, expected_values[:, None])
    trial_twists = core.mv.bivector(np.eye(3))
    value_difference = (
        values[-1][:, None, None](trial_twists, trial_twists[:, None])
        - expected_value_forms[:, None, None](trial_twists, trial_twists[:, None])
    )
    assert (value_difference.abs() < 1e-10).all()

    initial_errors = trial_twists.broadcast_to((len(actuation), 3))
    feedbacks = stack(backward_gains[::-1])
    errors = stack(tuple(core.rollout(
        initial_errors, dynamics_map[:, None], actuation_map[:, None], feedbacks[:, :, None],
    )))
    pushes = feedbacks[:, :, None](errors[:-1])
    expected_pushes = context.extensor(core.Forque, expected_actions.transpose(1, 0, 3, 2))
    push_residual = pushes - expected_pushes
    assert (effort_forms[None, :, None](push_residual, push_residual).abs() < 1e-12).all()


def test_docking_realizes_bellman_cost_and_thruster_work():
    """Checks Bellman step-by-step cost decrease, thruster forque synthesis, and virtual work sign."""
    n_steps = 4
    costs, gains = zip(*core.riccati(
        scenarios.state_cost * scenarios.landing_weight * np.ones(scenarios.n_cases),
        scenarios.dynamics, scenarios.actuation, scenarios.state_cost, scenarios.effort_cost, n_steps,
    ))
    values = stack(costs)                                                      # [n_steps, n_cases] Scalar+ <- (Twist+, Twist+)
    feedbacks = stack(gains)[::-1]                                             # [n_steps, n_cases] Forque- <- Twist+
    errors = stack(tuple(core.rollout(
        scenarios.initial_error, scenarios.dynamics, scenarios.actuation, feedbacks,
    )))                                                                        # [n_steps + 1, n_cases] Twist+
    requested_pushes = feedbacks(errors[:-1])                                  # [n_steps, n_cases] Forque-
    commands = (scenarios.resistance(requested_pushes)[..., None] & scenarios.thrusters) / scenarios.thruster_weights

    tracking_lines = scenarios.tracking_points[:, None] & scenarios.axes[None, :]
    displacements = errors[..., None, None] & tracking_lines
    tracking_costs = (displacements.squared() * scenarios.tracking_weights).sum(axis=(-2, -1)) * scenarios.dt
    command_efforts = (commands.squared() * scenarios.thruster_weights).sum(axis=-1)
    running_costs = tracking_costs[:-1] + command_efforts * scenarios.effort_scales * scenarios.dt
    landing_cost = tracking_costs[-1] * scenarios.landing_weight
    predicted_cost = values[-1](errors[0], errors[0])
    assert ((running_costs.sum(axis=0) + landing_cost - predicted_cost).abs() < 1e-8).all()

    remaining_costs = values[::-1](errors[:-1], errors[:-1])
    step_decreases = remaining_costs[:-1] - remaining_costs[1:]
    assert ((step_decreases - running_costs[:-1]).abs() < 1e-8).all()
    assert ((remaining_costs[-1] - running_costs[-1] - landing_cost).abs() < 1e-8).all()

    supplied_pushes = (commands * scenarios.thrusters).sum(axis=-1)
    forque_residual = supplied_pushes - requested_pushes
    assert ((scenarios.resistance(forque_residual) & forque_residual).abs() < 1e-9).all()

    trials = core.mv.bivector([[0.2, -0.3, 0.15], [-0.1, 0.25, -0.12]])[:, None]
    velocities = scenarios.mounts.commutator(trials)
    power = scenarios.directions.dual() | velocities.dual()
    assert ((power - (trials & scenarios.thrusters)).abs() < 1e-10).all()


def test_single_step_renders(capsys):
    """Exercises all figure builders, cost table printout, and one animation frame on a single step."""
    from examples.mechanics.riccati import render

    costs, gains = zip(*core.riccati(
        scenarios.state_cost * scenarios.landing_weight * np.ones(scenarios.n_cases),
        scenarios.dynamics, scenarios.actuation, scenarios.state_cost, scenarios.effort_cost, 1,
    ))
    values = stack(costs)                                                      # [1, n_cases] Scalar+ <- (Twist+, Twist+)
    feedbacks = stack(gains)[::-1]                                             # [1, n_cases] Forque- <- Twist+
    errors = stack(tuple(core.rollout(
        scenarios.initial_error, scenarios.dynamics, scenarios.actuation, feedbacks,
    )))                                                                        # [2, n_cases] Twist+
    pushes = feedbacks(errors[:-1])                                            # [1, n_cases] Forque-
    commands = (scenarios.resistance(pushes)[..., None] & scenarios.thrusters) / scenarios.thruster_weights
    display_commands = concatenate((commands, commands[-1:] * 0))              # [2, n_cases, 3] Scalar+
    poses = (errors * -0.5).exp()                                              # [2, n_cases] Motor+

    render.setup_figure(scenarios.hull, scenarios.mounts, scenarios.directions, scenarios.tracking_points)
    render.approaches_figure(scenarios.hull, poses)
    render.costs_figure(values)
    tracking_paid = scenarios.state_cost(errors[:-1], errors[:-1]).sum(axis=0)
    effort_paid = scenarios.effort_cost(pushes, pushes).sum(axis=0)
    landing_paid = scenarios.state_cost(errors[-1], errors[-1]) * scenarios.landing_weight
    predicted_cost = values[-1](scenarios.initial_error, scenarios.initial_error)
    render.cost_summary(tracking_paid, effort_paid, landing_paid, predicted_cost, scenarios.labels)
    assert "Lower effort penalty" in capsys.readouterr().out
    frame = next(render.docking_frames(scenarios.hull, scenarios.mounts, scenarios.directions, poses, display_commands,
                                       scenarios.arrow_scale))
    assert frame.ndim == 3
    plt.close("all")
