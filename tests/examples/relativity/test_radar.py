"""Null propagation, reconstruction from clock readings, and covariance of the radar experiment."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from examples.relativity.radar import core, render, scenarios
from examples.util.panels import image_figure

tolerance = 1e-9


@pytest.mark.parametrize("velocity", [-0.6, 0, 0.25, 0.85])
def test_echo_maps_and_reconstruction(velocity):
    distance = 2.3
    times = np.array([0.2, 0.6, 1.1])
    placement = (core.mv.xw * (distance / 2)).exp() * (core.mv.xt * (np.arctanh(velocity) / 2)).exp()
    target = placement >> core.mv.x
    emitted = core.origin + core.time * times
    reflection, echo = core.echo_maps(core.mv.x, target, core.outgoing, core.incoming, core.origin)
    reflected, received = reflection(emitted), echo(emitted)

    # Analytic round-trip times, independently of the homogeneous intersection construction.
    expected_stretch = (1 + velocity) / (1 - velocity)
    expected_return = 2 * distance / (1 - velocity) + expected_stretch * times
    assert (((core.mv.t & received) - expected_return).abs() < tolerance).all()
    assert (((core.mv.x & reflected) - (distance + velocity * (core.mv.t & reflected))).abs() < tolerance).all()
    assert ((target & reflected).abs() < tolerance).all()
    assert ((core.mv.x & received).abs() < tolerance).all()
    assert ((emitted & reflected).squared().abs() < tolerance).all()
    assert ((reflected & received).squared().abs() < tolerance).all()
    assert ((core.mv.t & (reflected - emitted)) > 0).all()
    assert ((core.mv.t & (received - reflected)) > 0).all()

    located, simultaneous, radar_time, radar_range = core.radar_assignment(
        emitted, received, core.mv.t, core.origin, core.time, core.across)
    assert ((core.mv.x & (located - reflected)).abs() < tolerance).all()
    assert ((core.mv.t & (located - reflected)).abs() < tolerance).all()
    assert ((core.mv.t & (located - simultaneous)).abs() < tolerance).all()
    assert ((radar_time - (core.mv.t & reflected)).abs() < tolerance).all()
    assert ((radar_range - (core.mv.x & reflected)).abs() < tolerance).all()
    stretch, inferred_velocity, initial_range = core.reflector_motion(emitted, received, core.mv.t)
    assert (stretch - expected_stretch).abs() < tolerance
    assert (inferred_velocity - velocity).abs() < tolerance
    assert (initial_range - distance).abs() < tolerance


@pytest.mark.parametrize("mirrored", [False, True])
def test_common_frame_change_preserves_events_and_clock_readings(mirrored):
    worldlines, emitted, reflected, received = scenarios.pulse_exchange()
    transform = (core.mv.xw * 0.4 - core.mv.tw * 0.3).exp() * (core.mv.xt * -0.4).exp()
    if mirrored:
        transform = transform * core.mv.x
    target_placement = core.mv.xw.exp() * (core.mv.xt * (np.arctanh(0.25) / 2)).exp()
    target = transform >> (target_placement >> core.mv.x)
    clock, reference = transform >> core.mv.t, transform >> core.origin
    sent = transform >> emitted
    reflection, echo = core.echo_maps(transform >> core.mv.x, target,
                                      transform >> core.outgoing, transform >> core.incoming, reference)
    hit, returned = reflection(sent), echo(sent)
    expected_hit, expected_returned = transform >> reflected, transform >> received
    assert ((core.mv.x & (hit - expected_hit)).abs() < tolerance).all()
    assert ((core.mv.t & (hit - expected_hit)).abs() < tolerance).all()
    assert ((core.mv.x & (returned - expected_returned)).abs() < tolerance).all()
    assert ((core.mv.t & (returned - expected_returned)).abs() < tolerance).all()
    assert (((clock & returned) - (core.mv.t & received)).abs() < tolerance).all()
    located, simultaneous, radar_time, ranges = core.radar_assignment(
        sent, returned, clock, reference, transform >> core.time, transform >> core.across)
    assert ((core.mv.x & (located - hit)).abs() < tolerance).all()
    assert ((core.mv.t & (located - hit)).abs() < tolerance).all()
    assert ((ranges - (core.mv.x & reflected)).abs() < tolerance).all()
    stretch, velocity, distance = core.reflector_motion(sent, returned, clock)
    assert (stretch - 5 / 3).abs() < tolerance
    assert (velocity - 0.25).abs() < tolerance
    assert (distance - 2).abs() < tolerance

    if not mirrored:
        next(render.exchange_animation(worldlines, emitted, reflected, received, n_frames=1))
        fig = image_figure(render.reconstruction_image(
            worldlines, emitted, reflected, received, located, simultaneous, radar_time, ranges, clock, render.exchange_window))
        plt.close(fig)
        states = core.boosted_frames(worldlines, emitted, reflected, received, -0.65, 0.3, 0.2, n_frames=1)
        next(render.frames_animation(states))
