"""Only relative turns between frames bend the motor surface, and a frame of the wound ribbon draws."""

import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core, render, scenarios


def test_only_the_turns_between_frames_bend_the_motor_surface():
    _, surfaces, _, _ = scenarios.turning()
    point_surfaces, motor_surfaces = surfaces[:, 0].batch(), surfaces[:, 1].batch()  # [frames, refined] Point-
    # Turning the frames about their origins leaves the control points fixed while bending the motor surface.
    assert np.all((point_surfaces & point_surfaces[:1]).norm() < 1e-12)
    assert np.any((motor_surfaces & point_surfaces).norm() > 0.05)
    # A rigid turn shared by every control frame leaves the subdivided surface unchanged.
    z_coords, x_coords = scenarios.grid()
    rest_motors = (core.mv.zw * (z_coords / 2)).exp() * (core.mv.xw * (x_coords / 2)).exp()  # [vertices] Motor+
    shared_turn = (core.mv.yz * (-0.7 / 2)).exp()                              # [] Motor+
    cases = scenarios.subdivide_ribbon(stack([rest_motors, rest_motors * shared_turn]).field())[1][:, 1].batch()  # [cases, refined] Point-
    assert np.all((cases[0] & cases[1]).norm() < 1e-12)


def test_the_wound_ribbon_draws():
    control_frames, surfaces, control_topology, refined_topology = scenarios.winding(n_frames=1)
    assert len(list(render.animate(control_frames, surfaces, control_topology, refined_topology))) == 1
