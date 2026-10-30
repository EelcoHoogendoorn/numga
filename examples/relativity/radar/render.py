"""Spacetime diagrams at equal spatial and temporal scales, with clock readings beside events."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from numga import concatenate, stack, where
from numga.multivector.factory import MultivectorFactory
from examples.relativity.radar.core import Event, Scalar, Slice
from examples.util.wgsl import planar
from examples.util.wgsl.planar import Cap, Marker, Marks, Polygons, SOLID, Strokes

BLUE, ORANGE, GREY, INK, GREEN = (
    np.array(to_rgb(colour)) for colour in ("#3196b4", "#de9439", "#aab9c1", "#485961", "#458875"))
FAINT, WHITE = np.array(to_rgb("#e2e8eb")), np.ones(3)
FONT = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 13)
EXCHANGE_WINDOW = planar.window((-1.55, 4.8), (-0.65, 8.75), 620)
FRAME_WINDOW = planar.window((-4, 6), (-2.65, 10.15), 620)


def coordinates(events: Event) -> np.ndarray:
    """Position and coordinate time of events of unit weight `[..., 2]`, read only at the drawing boundary."""
    mv = events.context.multivector
    return (stack((mv.x, mv.t)) & events[..., None]).to_array()


def gpu_points(events: Event) -> planar.Point:
    """Draw position across and coordinate time up, keeping the readings typed through the diagram mapping."""
    mv = events.context.multivector
    readings = stack((mv.x, mv.t)) & events[..., None]
    frame = stack((planar.mv.x, planar.mv.y)).dual()
    return planar.origin + (frame * planar.context.include_from_algebra(readings)).sum(axis=-1)



def strokes(events: Event, width: float, colour: np.ndarray, opacity: float,
            dashes: tuple[float, float]) -> Strokes:
    return Strokes(gpu_points(events), width, colour, opacity, dashes, Cap.ROUND)


def dots(events: Event, colour: np.ndarray, radius: float) -> list:
    placed = gpu_points(events)
    return [Marks(placed, Marker.DISC, 0, radius + 1.5, WHITE, 1),
            Marks(placed, Marker.DISC, 0, radius, colour, 1)]


def axis_ends(window: planar.Window, mv: MultivectorFactory) -> tuple[Event, Event]:
    """The ends of the space and time axes through the origin, a little inside the window: x first, then t."""
    left, right, bottom, top = window.extent
    margin = 0.2
    # The origin, of unit weight, and the ideal events along space and along time.
    origin, across, time = mv.w.dual(), mv.x.dual(), -mv.t.dual()             # [] Event-, Event-, Event-
    starts = stack((origin + across * (left + margin), origin + time * (bottom + margin)))   # [axes] Event-
    ends = stack((origin + across * (right - margin), origin + time * (top - margin)))       # [axes] Event-
    return starts, ends


def written(window: planar.Window, layers: list, labels: list, mv: MultivectorFactory) -> np.ndarray:
    """Composite the geometry and attach labels, each beside its event, in the same native pixel grid."""
    starts, ends = axis_ends(window, mv)
    axes = strokes(stack((starts, ends), axis=1), 1, FAINT, 1, SOLID)
    canvas = Image.fromarray(planar.draw(window, [axes] + layers, WHITE))
    pen = ImageDraw.Draw(canvas)
    labels = [(ends[0], "x", INK, (0, 12), "mm"), (ends[1], "t", INK, (10, 0), "lm")] + labels
    places = coordinates(stack([event for event, *_ in labels]))               # [labels, 2]
    for (x, t), (_, text, colour, offset, anchor) in zip(places, labels):
        place = ((x - window.left) / window.pixel + offset[0],
                 (window.top - t) / window.pixel + offset[1])
        pen.text(place, text, font=FONT, fill=tuple(np.rint(colour * 255).astype(int)), anchor=anchor)
    return np.asarray(canvas)


def exchange_frames(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                    n_frames: int):
    """Light pulses advancing through the diagram; their completed paths stay visible."""
    mv = emitted.context.multivector
    starts, ends = concatenate((emitted, reflected)), concatenate((reflected, received))   # [paths] Event-, Event-
    n_pulses = len(emitted)
    colours = np.repeat(np.stack((ORANGE, BLUE)), n_pulses, axis=0)
    paths = stack((starts, ends), axis=1)                                      # [paths, 2] Event-
    # The light's lines through spacetime, each pulse's way out and back.
    light_lines = starts & ends                                                # [paths] Line-
    first, finish = (mv.t & emitted[0]).to_array(), (mv.t & received[-1]).to_array()
    for now in np.linspace(first - 0.3, finish + 0.7, n_frames):
        # The events at coordinate time `now` pair to zero with this slice, the earlier ones negatively.
        present = mv.t - mv.w * now                                            # [] Slice+
        begun = (present & starts) < 0
        arrived = (present & ends) < 0
        travelling = begun & ~arrived
        # Each pulse in flight is where its line meets the present; one that has arrived stays at its end.
        crossings = light_lines ^ present                                      # [paths] Event-
        fronts = where(arrived, ends, crossings / (mv.w & crossings))          # [paths] Event-
        layers = [strokes(worldlines, 2, np.stack((INK, GREY)), 1, SOLID),
                  strokes(paths, 1, colours, 0.16, SOLID),
                  strokes(stack((starts, fronts), axis=1)[begun], 2, colours[begun], 1, SOLID)]
        layers += dots(fronts[travelling], colours[travelling], 4)
        layers += dots(reflected[(present & reflected) <= 0], ORANGE, 3)
        clock_events = concatenate((emitted[(present & emitted) <= 0], received[(present & received) <= 0]))
        layers += dots(clock_events, INK, 3)
        labels = [(event, f"{reading:.2f}", INK, (-10, 0), "rm")
                  for event, reading in zip(clock_events, (mv.t & clock_events).to_array())]
        yield written(EXCHANGE_WINDOW, layers, labels, mv)


def reconstruction(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                   located: Event, simultaneous: Event, radar_time: Scalar, radar_range: Scalar, clock: Slice,
                   window: planar.Window) -> np.ndarray:
    """The radar assignment for the middle pulse, with every recovered reflection marked and the middle one's clock
    readings, radar time and range written beside it."""
    index = len(emitted) // 2
    triangle = stack((emitted[index], reflected[index], received[index]))[None]   # [1, corners] Event-
    outgoing = stack((emitted, reflected), axis=1)                            # [pulses, 2] Event-
    incoming = stack((reflected, received), axis=1)                           # [pulses, 2] Event-
    layers = [Polygons(gpu_points(triangle), BLUE, 0.045),
              strokes(worldlines, 2, np.stack((INK, GREY)), 1, SOLID),
              strokes(outgoing, 1, ORANGE, 0.2, SOLID),
              strokes(incoming, 1, BLUE, 0.2, SOLID),
              strokes(outgoing[index:index + 1], 2, ORANGE, 1, SOLID),
              strokes(incoming[index:index + 1], 2, BLUE, 1, SOLID),
              strokes(stack((simultaneous[index], located[index]))[None], 1.7, GREEN, 1, (5, 4)),
              strokes(located[None], 2, GREEN, 1, SOLID)]
    layers += dots(located, GREEN, 4)
    layers += dots(stack((emitted[index], received[index])), INK, 3)
    layers += dots(simultaneous[index:index + 1], GREEN, 3)
    readings = (clock & stack((emitted[index], received[index]))).to_array()
    middle, distance = radar_time[index].to_array(), radar_range[index].to_array()
    # The range written halfway along the line of simultaneity to the located reflection.
    range_label = (simultaneous[index] + located[index]) / 2                   # [] Event-
    labels = [(emitted[index], f"τ− = {readings[0]:.2f}", INK, (-10, 0), "rm"),
              (received[index], f"τ+ = {readings[1]:.2f}", INK, (-10, 0), "rm"),
              (simultaneous[index], f"T = {middle:.2f}", GREEN, (-10, 0), "rm"),
              (range_label, f"R = {distance:.2f}", GREEN, (0, -12), "mm")]
    return written(window, layers, labels, emitted.context.multivector)
