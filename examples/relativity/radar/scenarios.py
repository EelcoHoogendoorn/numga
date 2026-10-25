"""Light pulses reflected by a receding target, and the same experiment in moving frames."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.relativity.radar.core import (
    Event, across, echo_maps, incoming, mv, origin, outgoing, reconstruct, time,
)


def exchange() -> tuple[Event, Event, Event, Event]:
    """The two worldline segments and the emitted, reflected, and received events."""
    distance, velocity = 2, 0.25
    n_pulses, pulse_spacing = 3, 0.75
    end_margin = 0.5

    # xt squares to +1 and generates boosts; arctanh(velocity) is the reflector's rapidity.
    # xw squares to zero and generates the spatial translation, applied after the boost.
    placement = (mv.xw * (distance / 2)).exp() * (mv.xt * (np.arctanh(velocity) / 2)).exp()
    target = placement >> mv.x
    emitted = origin + time * (np.arange(n_pulses) * pulse_spacing)
    reflection, echo = echo_maps(mv.x, target, outgoing, incoming, origin)
    reflected, received = reflection(emitted), echo(emitted)

    finish = (mv.t & received[-1]) + end_margin
    target_end = target ^ (mv.t - mv.w * finish)
    target_end = target_end / (mv.w & target_end)
    worldlines = stack((stack((origin, origin + time * finish)),
                        stack((placement >> origin, target_end))))
    return worldlines, emitted, reflected, received


def frames(worldlines: Event, emitted: Event, reflected: Event, received: Event,
           n_frames: int):
    """A common boost about the middle of the experiment, followed by a translation."""
    final_rapidity, spatial_shift, time_shift = -0.65, 0.3, 0.2
    midpoint_time = (mv.t & (emitted[0] + received[-1])) / 2
    centre = (-mv.tw * (midpoint_time / 2)).exp()
    for fraction in np.linspace(0, 1, n_frames):
        boost = (mv.xt * (final_rapidity * fraction / 2)).exp()
        # The translation generator squares to zero, so its exponential is one plus the generator.
        translation = ((mv.xw * spatial_shift - mv.tw * time_shift) * (fraction / 2)).exp()
        # Conjugating the boost by centre anchors it at the midpoint event.
        motor = translation * centre * boost * centre.reverse()
        # The clock's unit and zero reading move with the experiment.
        clock = motor >> mv.t
        sent, returned = motor >> emitted, motor >> received
        located, simultaneous, _, _ = reconstruct(
            sent, returned, clock, motor >> origin, motor >> time, motor >> across)
        yield motor >> worldlines, sent, motor >> reflected, returned, located, simultaneous, clock


if __name__ == "__main__":
    from examples.relativity.radar import render
    from examples.util.animation import save_animation, save_figure
    from examples.util.wgsl import image_figure

    n_frames, duration_ms = 72, 55
    worldlines, emitted, reflected, received = exchange()
    geometry = worldlines, emitted, reflected, received
    save_animation(render.exchange_frames(*geometry, n_frames), "radar_echoes", duration_ms)
    located, simultaneous, _, _ = reconstruct(emitted, received, mv.t, origin, time, across)
    save_figure(image_figure(render.reconstruction(*geometry, located, simultaneous, mv.t,
                                                render.EXCHANGE_WINDOW)), "radar_reconstruction")
    save_animation((render.reconstruction(*state, render.FRAME_WINDOW)
                    for state in frames(*geometry, n_frames)), "radar_frames", duration_ms,
                   forward_then_back=True)
