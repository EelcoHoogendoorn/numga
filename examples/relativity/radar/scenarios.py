"""Scene parameters and pulse exchanges trace radar echoes off a receding target across boosted and translated frames."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.relativity.radar.core import (
    Event, across, boosted_frames, echo_maps, incoming, mv, origin, outgoing, radar_assignment, reflector_motion, time,
)

# The reflector's distance at the clock's zero and its velocity; the pulses, the time between them, and how long the
# worldlines run on past the last echo.
distance, velocity = 2, 0.25
n_pulses, pulse_spacing = 3, 0.75
end_margin = 0.5
# The moving frames: the peak rapidity of the boost and its shifts in space and time.
final_rapidity, spatial_shift, time_shift = -0.65, 0.3, 0.2
# The frames of the animations, and the duration of a frame.
n_echo_frames = 99
n_boost_frames = 196
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def pulse_exchange() -> tuple[Event, Event, Event, Event]:
    """Transmitter and reflector worldline segments (`[2, 2] Event-`) pair with the emitted, reflected, and received events (`[n_pulses] Event-`)."""
    # xt squares to +1 and generates boosts; arctanh(velocity) is the reflector's rapidity.
    # xw squares to zero and generates the spatial translation, applied after the boost.
    placement = (mv.xw * (distance / 2)).exp() * (mv.xt * (np.arctanh(velocity) / 2)).exp()  # [] Motor+
    target = placement >> mv.x                                                               # [] Plane+
    emitted = origin + time * (np.arange(n_pulses) * pulse_spacing)                          # [n_pulses] Event-
    reflection, echo = echo_maps(mv.x, target, outgoing, incoming, origin)
    reflected, received = reflection(emitted), echo(emitted)                                 # [n_pulses] Event-, Event-

    finish = (mv.t & received[-1]) + end_margin                                              # [] Scalar+
    target_end = target ^ (mv.t - mv.w * finish)                                             # [] Event-
    target_end = target_end / (mv.w & target_end)                                            # [] Event-
    target_start = placement >> origin                                                       # [] Event-
    worldlines = stack((stack((origin, origin + time * finish)),
                        stack((target_start, target_end))))                                  # [2, 2] Event-
    return worldlines, emitted, reflected, received


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.relativity.radar import render
    from examples.util.animation import save_animation, save_figure
    from examples.util.panels import image_figure

    worldlines, emitted, reflected, received = pulse_exchange()
    geometry = worldlines, emitted, reflected, received
    save_animation(render.exchange_animation(*geometry, n_echo_frames), "radar_echoes", frame_duration_ms)

    located, simultaneous, radar_time, radar_range = radar_assignment(
        emitted, received, mv.t, origin, time, across)
    stretch, inferred_velocity, initial_range = reflector_motion(emitted, received, mv.t)
    save_figure(image_figure(render.reconstruction_image(
        *geometry, located, simultaneous, radar_time, radar_range, mv.t, render.exchange_window)), "radar_reconstruction")

    streamed = []

    def recorded_states():
        for state in boosted_frames(*geometry, final_rapidity, spatial_shift, time_shift, n_boost_frames):
            streamed.append(state)
            yield state

    save_animation(render.frames_animation(recorded_states()), "radar_frames", frame_duration_ms)

    # checks
    # Outbound and return light lines are null, located events match actual reflections, and radar invariants hold across frames.
    assert (((emitted & reflected).squared()).abs() < 1e-10).all()
    assert (((reflected & received).squared()).abs() < 1e-10).all()
    assert ((mv.x & (located - reflected)).abs() < 1e-10).all()
    assert ((mv.t & (located - reflected)).abs() < 1e-10).all()
    assert (inferred_velocity - velocity).abs() < 1e-10
    assert (initial_range - distance).abs() < 1e-10
    peak_state = streamed[n_boost_frames // 2]
    assert ((peak_state[6] - radar_time).abs() < 1e-10).all()
    assert ((peak_state[7] - radar_range).abs() < 1e-10).all()


if __name__ == "__main__":
    main()
