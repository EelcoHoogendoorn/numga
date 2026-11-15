"""Coverage checks verify forward kinematics, inverse and forward dynamics, linearly implicit stepping, mechanical energy, and GPU frame rendering on the dropped humanoid."""

from __future__ import annotations

from functools import partial

from examples.mechanics.articulated import core, render, scenarios


def test_dropped_humanoid_steps_and_draws_frame():
    tree, home_axes, inertias, mass_points = scenarios.humanoid_model()
    _, _, _, centres, radii = scenarios.home_pose()
    gravity = -scenarios.mv.z.dual() * scenarios.gravity_strength              # [] Point-
    stiffness, damping = scenarios.hinge_parameters()
    hinge_stiffness = stiffness * core.Scalar.dual()                           # (Force- <- Scalar+)[E]
    hinge_damping = damping * core.Scalar.dual()                               # (Force- <- Scalar+)[E]
    low_stops, high_stops, stop_raw = scenarios.stop_parameters()
    stop_stiffness = stop_raw * core.Scalar.dual()                             # (Force- <- Scalar+)[E]
    ground, surfaces, contact_stiffness, ground_damping, ground_friction = scenarios.ground_parameters()

    state = scenarios.initial_drop_state()
    initial_energy = core.mechanical_energy(
        tree,
        home_axes,
        inertias,
        mass_points,
        gravity,
        hinge_stiffness,
        low_stops,
        high_stops,
        stop_stiffness,
        ground,
        surfaces,
        contact_stiffness,
        state,
    )
    assert initial_energy.isfinite().all()

    motor, rate, angles, speeds = state
    motors = core.forward_kinematics(tree, home_axes, motor, angles)
    weights = gravity & (motors >> mass_points)                                # [1] Forque-[V]
    forque, forces = core.inverse_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, rate, speeds, weights
    )
    loaded = weights + tree[0].adjugate() * forque[..., None].field()          # [1] Forque-[V]
    recovered_accelerations = core.forward_dynamics(
        tree, home_axes, inertias, motors, rate, speeds, forces, loaded
    )[1]
    assert ((recovered_accelerations - speeds).norm() < 1e-8).all()

    step_fn = partial(
        core.drop_step,
        tree,
        home_axes,
        inertias,
        mass_points,
        gravity,
        hinge_stiffness,
        hinge_damping,
        low_stops,
        high_stops,
        stop_stiffness,
        ground,
        surfaces,
        contact_stiffness,
        ground_damping,
        ground_friction,
    )
    forward_fn = partial(core.forward_kinematics, tree, home_axes)
    figure = render.pose(centres, radii, len(scenarios.capsule_bodies))
    plt_figure_axes = figure.axes
    assert len(plt_figure_axes) == 1

    states = core.trajectory(step_fn, state, scenarios.dt, 1, 2)
    frames = list(render.animate_states(forward_fn, tree[4], centres, radii, len(scenarios.capsule_bodies), states))
    assert len(frames) == 2

    chart = render.throughput(scenarios.robot_counts[:2], scenarios.robot_counts[:2] * 100.0, 1 / scenarios.dt)
    assert len(chart.axes) == 1

    import matplotlib.pyplot as plt
    plt.close(figure)
    plt.close(chart)
