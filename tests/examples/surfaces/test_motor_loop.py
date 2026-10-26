"""Loop subdivision of motors: a twisted ribbon keeps its width as motors and wrings thin as points, and only
the turns between frames bend it."""

from numga import stack
from examples.surfaces.motor_loop import core, render, scenarios


def test_motors_keep_the_twisted_ribbon_wide_where_points_wring_it_thin():
    control, surfaces, control_faces, faces = scenarios.winding()
    # How far each surface reaches from the z axis on average, at rest and wound up.
    centre_line = core.mv.w.dual() & core.mv.point([0.0, 0.0, 1.0])           # [] Line
    reach = (centre_line & surfaces[[0, scenarios.N_FRAMES // 2]].batch()).norm().mean(axis=-1)   # [cases, schemes] Scalar
    (points_rest, motors_rest), (points_peak, motors_peak) = reach.to_array()
    # Untwisted the two agree; wound up, the points wring the ribbon towards its axis and the motors do not.
    assert abs(points_rest - motors_rest) < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest
    assert len(list(render.animate(control[:1], surfaces[:1], control_faces, faces))) == 1


def test_only_the_turns_between_frames_bend_the_motor_surface():
    control, surfaces, control_faces, faces = scenarios.turning()
    points, motors = surfaces[:, 0].batch(), surfaces[:, 1].batch()            # [frames, refined] Point
    # Turning the frames about their origins leaves the points alone and bends the motors.
    assert (points & points[:1]).norm().to_array().max() < 1e-12
    assert (motors & points).norm().to_array().max() > 0.05
    # A turn common to every frame changes nothing.
    along, across = scenarios.grid()
    flat = (core.mv.z * core.mv.w * (along / 2)).exp() * (core.mv.x * core.mv.w * (across / 2)).exp()   # [vertices] Motor
    cases = scenarios.refined(stack([flat, flat * (core.mv.yz * (-0.7 / 2)).exp()]).field())[1][:, 1].batch()   # [cases, refined] Point
    assert (cases[0] & cases[1]).norm().to_array().max() < 1e-12
