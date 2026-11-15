"""Defines the three-thruster docking scenario and saves its figures and animation."""

import numpy as np

from numga import concatenate, stack
from examples.mechanics.riccati import core

mv, Twist, Forque = core.mv, core.Twist, core.Forque

labels = ("Lower effort penalty", "Higher effort penalty")
n_cases = len(labels)
effort_scales = np.array([0.05, 5.0])                                      # [n_cases]
thruster_weights = np.array([1.0, 2.0, 3.0])                               # [3]
# Drag resists surge and sway through the centre along with yaw turning.
drag_weights = np.array([2.0, 3.0, 1.0])                                   # [3]
# Rows weight bow and stern sites; columns weight forward and sideways alignment.
tracking_weights = np.array([[4.0, 8.0], [1.0, 2.0]])                      # [2, 2]
dt = 0.1
n_steps = 72
# Pricing deadline error steeply enforces arrival at the target pose.
landing_weight = 1e6
frame_duration_ms = 40
arrow_scale = 0.04

hull = mv.point([[-0.45, -0.23], [0.28, -0.23], [0.55, 0], [0.28, 0.23], [-0.45, 0.23]])  # [5] Point-
mounts = mv.point([[-0.3, 0.2], [-0.3, -0.2], [0.35, 0]])                  # [3] Point-
directions = mv.direction([[1, 0], [1, 0], [0, 1]])                        # [3] Point-
tracking_points = mv.point([[0.45, 0], [-0.35, 0]])                        # [2] Point-
axes = mv.direction([[1, 0], [0, 1]])                                      # [2] Point-
centre = mv.point([0, 0])                                                  # [] Point-
initial_error = (mv.xw * 0.4 + mv.yw * 0.5 + mv.xy * 0.35) * np.ones(n_cases)  # [n_cases] Twist+

thrusters = mounts & directions                                            # [3] Forque-
authority = (thrusters * (Twist & thrusters) / thruster_weights).sum(axis=0)  # [] Forque- <- Twist+
resistance = authority.inverse()                                           # [] Twist+ <- Forque-
effort_cost = (resistance(Forque) & Forque) * effort_scales * dt           # [n_cases] Scalar+ <- (Forque-, Forque-)
state_cost = core.tracking_cost(tracking_points, axes, tracking_weights, dt)  # [] Scalar+ <- (Twist+, Twist+)
mobility = core.drag_mobility(centre, axes, drag_weights)                  # [] Twist+ <- Forque-
dynamics = Twist                                                           # [] Twist+ <- Twist+
actuation = mobility * dt                                                  # [] Twist+ <- Forque-


def main() -> None:
    """Runs the docking recursion and rollout once, writes figures and animation, and verifies invariants."""
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.riccati import render

    costs, gains = zip(*core.riccati(
        state_cost * landing_weight * np.ones(n_cases), dynamics, actuation, state_cost, effort_cost, n_steps,
    ))
    values = stack(costs)                                                  # [n_steps, n_cases] Scalar+ <- (Twist+, Twist+)
    feedbacks = stack(gains)[::-1]                                         # [n_steps, n_cases] Forque- <- Twist+
    errors = stack(tuple(core.rollout(initial_error, dynamics, actuation, feedbacks)))  # [n_steps + 1, n_cases] Twist+
    pushes = feedbacks(errors[:-1])                                        # [n_steps, n_cases] Forque-
    commands = (resistance(pushes)[..., None] & thrusters) / thruster_weights  # [n_steps, n_cases, 3] Scalar+
    poses = (errors * -0.5).exp()                                          # [n_steps + 1, n_cases] Motor+

    save_figure(render.setup_figure(hull, mounts, directions, tracking_points), "riccati_setup")
    save_figure(render.approaches_figure(hull, poses), "riccati_approaches")
    save_figure(render.costs_figure(values), "riccati_costs")
    # Append a zero-command frame so the landed pose settles with thrusters off.
    display_commands = concatenate((commands, commands[-1:] * 0))          # [n_steps + 1, n_cases, 3] Scalar+
    save_animation(render.docking_frames(hull, mounts, directions, poses, display_commands, arrow_scale),
                   "riccati", frame_duration_ms)

    # checks
    tracking_paid = state_cost(errors[:-1], errors[:-1]).sum(axis=0)       # [n_cases] Scalar+
    effort_paid = effort_cost(pushes, pushes).sum(axis=0)                  # [n_cases] Scalar+
    landing_paid = state_cost(errors[-1], errors[-1]) * landing_weight     # [n_cases] Scalar+
    predicted_cost = values[-1](initial_error, initial_error)              # [n_cases] Scalar+
    assert ((tracking_paid + effort_paid + landing_paid - predicted_cost).abs() < 1e-6).all()
    forque_residual = (commands * thrusters).sum(axis=-1) - pushes         # [n_steps, n_cases] Forque-
    assert ((resistance(forque_residual) & forque_residual).abs() < 1e-9).all()
    landing_offsets = (poses[-1, :, None] >> tracking_points) - tracking_points  # [n_cases, 2] Point-
    assert (landing_offsets.antinorm() < 1e-3).all()


if __name__ == "__main__":
    main()
