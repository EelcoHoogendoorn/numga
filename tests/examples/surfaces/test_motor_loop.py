"""Relative local turns bend the motor-subdivided surface while shared turns cancel, and one streamed frame draws."""

import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core, render, scenarios


def test_only_relative_turns_between_frames_bend_the_motor_surface():
    control_topology = core.Topology.grid(3, 2)
    V, E, F = control_topology.n_elements
    pinned = np.isin(np.arange(V), [0, 1, 4, 5])                            # [V]
    refined_topology = control_topology.loop_subdivided(pinned, 1)
    z_coords, x_coords = scenarios.ribbon_grid(3, 2, 2.0, 1.0)              # [V], [V]
    rest_motors = core.ribbon_rest_motors(z_coords, x_coords)               # [V] Motor+
    markers = core.frame_markers(scenarios.axis_length)                     # [4] Point-
    axes, phases = scenarios.turning_axes(control_topology)

    states = list(core.turning_states(refined_topology, markers, rest_motors, axes, phases, scenarios.turn_amplitude, 4))
    turning_pair = stack([state[1] for state in states]).batch()            # [4, 2, V_refined] Point-
    point_surfaces, motor_surfaces = turning_pair[:, 0], turning_pair[:, 1]  # [4, V_refined] Point-, [4, V_refined] Point-
    assert ((point_surfaces & point_surfaces[:1]).norm() < 1e-12).all()
    assert ((motor_surfaces & point_surfaces).norm() > 0.02).any()

    shared_turn = (core.mv.yz * (-0.7 / 2)).exp()                           # [] Motor+
    shared_pair = (core.subdivided_motors(refined_topology, stack([rest_motors, rest_motors * shared_turn]).field()) >> core.origin).batch()  # [2, V_refined] Point-
    assert ((shared_pair[0] & shared_pair[1]).norm() < 1e-12).all()


def test_wound_ribbon_frame_draws():
    control_topology, refined_topology, z_coords, x_coords = scenarios.ribbon_setup()
    markers = core.frame_markers(scenarios.axis_length)                     # [4] Point-
    rest_motors = core.ribbon_rest_motors(z_coords, x_coords)               # [V] Motor+
    view = render.ribbon_view(scenarios.winding_bounds(rest_motors, markers, z_coords))
    states = core.winding_states(refined_topology, markers, rest_motors, z_coords, scenarios.length, scenarios.twist_angle, 2)
    frame = next(render.surface_animation(control_topology, refined_topology, view, states))
    assert frame.ndim == 3 and frame.shape[2] == 3
