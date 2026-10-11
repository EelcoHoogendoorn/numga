"""Loop subdivision of motors: a twisted ribbon keeps its width as motors and wrings thin as points, and only
the turns between frames bend it."""

from numga import stack
from examples.surfaces.motor_loop import core, render, scenarios


def test_motors_keep_the_twisted_ribbon_wide_where_points_wring_it_thin():
    control, surfaces, control_faces, faces = scenarios.winding()
    (points_rest, motors_rest), (points_peak, motors_peak) = render.reach(surfaces[[0, scenarios.FRAMES // 2]])
    # Untwisted the two agree; wound up, the points wring the ribbon towards its axis and the motors do not.
    assert abs(points_rest - motors_rest) < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest
    assert len(render.animate(control[:1], surfaces[:1], control_faces, faces)) == 1


def test_only_the_turns_between_frames_bend_the_motor_surface():
    control, surfaces, control_faces, faces = scenarios.turning()
    points, motors = surfaces[:, 0], surfaces[:, 1]                            # [frames] Point[refined]
    # Turning the frames about their origins leaves the points alone and bends the motors.
    assert render.separation(points, points[:1]) < 1e-12
    assert render.separation(motors, points) > 0.05
    # A turn common to every frame changes nothing.
    flat = core.ribbon(*scenarios.grid(), 0.0)                                 # [vertices] Motor
    cases = scenarios.refined(stack([flat, flat * core.turn(core.mv.yz, 0.7)]).field())[1][:, 1]   # [cases] Point[refined]
    assert render.separation(cases[0], cases[1]) < 1e-12
