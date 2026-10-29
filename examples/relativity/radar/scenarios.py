"""Light pulses reflected by a receding target, and the same experiment in moving frames."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.relativity.radar.core import (
    Event, Scalar, Slice, across, echo_maps, incoming, mv, origin, outgoing, reconstruct, time,
)

# The reflector's distance at the clock's zero and its velocity; the pulses, the time between them, and how long the
# worldlines run on past the last echo.
DISTANCE, VELOCITY = 2, 0.25
N_PULSES, PULSE_SPACING = 3, 0.75
END_MARGIN = 0.5
# The moving frames: the rapidity of the last frame's boost and its shifts in space and time.
FINAL_RAPIDITY, SPATIAL_SHIFT, TIME_SHIFT = -0.65, 0.3, 0.2
# The frames of the animations, and the duration of a frame.
N_FRAMES, DURATION_MS = 99, 40


# --- math -----------------------------------------------------------------------------
def exchange() -> tuple[Event, Event, Event, Event]:
    """The two worldline segments and the emitted, reflected, and received events."""
    # xt squares to +1 and generates boosts; arctanh(velocity) is the reflector's rapidity.
    # xw squares to zero and generates the spatial translation, applied after the boost.
    placement = (mv.xw * (DISTANCE / 2)).exp() * (mv.xt * (np.arctanh(VELOCITY) / 2)).exp()
    target = placement >> mv.x
    emitted = origin + time * (np.arange(N_PULSES) * PULSE_SPACING)
    reflection, echo = echo_maps(mv.x, target, outgoing, incoming, origin)
    reflected, received = reflection(emitted), echo(emitted)

    finish = (mv.t & received[-1]) + END_MARGIN
    target_end = target ^ (mv.t - mv.w * finish)
    target_end = target_end / (mv.w & target_end)
    worldlines = stack((stack((origin, origin + time * finish)),
                        stack((placement >> origin, target_end))))
    return worldlines, emitted, reflected, received


def frames(worldlines: Event, emitted: Event, reflected: Event, received: Event,
           n_frames: int) -> Iterator[tuple[Event, Event, Event, Event, Event, Event, Scalar, Scalar, Slice]]:
    """A common boost about the middle of the experiment, followed by a translation."""
    midpoint_time = (mv.t & (emitted[0] + received[-1])) / 2
    centre = (-mv.tw * (midpoint_time / 2)).exp()
    for fraction in np.linspace(0, 1, n_frames):
        boost = (mv.xt * (FINAL_RAPIDITY * fraction / 2)).exp()
        # The translation generator squares to zero, so its exponential is one plus the generator.
        translation = ((mv.xw * SPATIAL_SHIFT - mv.tw * TIME_SHIFT) * (fraction / 2)).exp()
        # Conjugating the boost by centre anchors it at the midpoint event.
        motor = translation * centre * boost * centre.reverse()
        # The clock's unit and zero reading move with the experiment.
        clock = motor >> mv.t
        sent, returned = motor >> emitted, motor >> received
        located, simultaneous, radar_time, radar_range = reconstruct(
            sent, returned, clock, motor >> origin, motor >> time, motor >> across)
        yield motor >> worldlines, sent, motor >> reflected, returned, located, simultaneous, radar_time, radar_range, clock


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.relativity.radar import render
    from examples.util.animation import save_animation, save_figure
    from examples.util.wgsl import image_figure

    worldlines, emitted, reflected, received = exchange()
    geometry = worldlines, emitted, reflected, received
    save_animation(render.exchange_frames(*geometry, N_FRAMES), "radar_echoes", DURATION_MS)
    located, simultaneous, radar_time, radar_range = reconstruct(emitted, received, mv.t, origin, time, across)
    save_figure(image_figure(render.reconstruction(*geometry, located, simultaneous, radar_time, radar_range, mv.t,
                                                render.EXCHANGE_WINDOW)), "radar_reconstruction")
    save_animation((render.reconstruction(*state, render.FRAME_WINDOW)
                    for state in frames(*geometry, N_FRAMES)), "radar_frames", DURATION_MS,
                   forward_then_back=True)


if __name__ == "__main__":
    main()
