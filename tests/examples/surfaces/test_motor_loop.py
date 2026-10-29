"""A twisted ribbon keeps its width when subdivided as motors and wrings thin as points, and only relative turns between frames bend the motor surface."""

import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core, render, scenarios


def test_motors_keep_the_twisted_ribbon_wide_where_points_wring_it_thin():
    control_frames, surfaces, control_mesh, refined_mesh = scenarios.winding()
    # Measure how far each surface reaches from the z axis on average, at rest and at peak twist.
    centre_line = core.mv.w.dual() & core.mv.point([0.0, 0.0, 1.0])            # [] Line-
    distance = (centre_line & surfaces[[0, scenarios.N_FRAMES // 2]].batch()).norm().mean(axis=-1)  # [cases, schemes] Scalar+
    (points_rest, motors_rest), (points_peak, motors_peak) = distance
    # Untwisted, the two schemes agree; wound up, point subdivision wrings the ribbon toward its axis while motor subdivision does not.
    assert (points_rest - motors_rest).abs() < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest
    assert len(list(render.animate(control_frames[:1], surfaces[:1], control_mesh, refined_mesh))) == 1


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
