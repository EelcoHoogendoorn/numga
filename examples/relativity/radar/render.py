"""Spacetime diagrams at equal spatial and temporal scales, with clock readings beside events."""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from numga import stack
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
    """Position and coordinate time of events `[..., 2]`, read only at the drawing boundary."""
    mv = events.context.multivector
    return (stack((mv.x, mv.t)) & events[..., None]).to_array()


def strokes(points: np.ndarray, width: float, colour: np.ndarray, opacity: float,
            dashes: tuple[float, float]) -> Strokes:
    return Strokes(planar.mv.point(points), width, colour, opacity, dashes, Cap.ROUND)


def dots(points: np.ndarray, colour: np.ndarray, radius: float) -> list:
    placed = planar.mv.point(points)
    return [Marks(placed, Marker.DISC, 0, radius + 1.5, WHITE, 1),
            Marks(placed, Marker.DISC, 0, radius, colour, 1)]


def axes(window: planar.Window) -> list:
    left, right, bottom, top = window.extent
    ends = np.array([[right - 0.2, 0], [0, top - 0.2]])
    starts = np.array([[left + 0.2, 0], [0, bottom + 0.2]])
    return [strokes(np.stack((starts, ends), axis=1), 1, FAINT, 1, SOLID)]


def written(window: planar.Window, layers: list, labels: list) -> np.ndarray:
    """Composite the geometry and attach labels in the same native pixel grid."""
    canvas = Image.fromarray(planar.draw(window, axes(window) + layers, WHITE))
    pen = ImageDraw.Draw(canvas)
    _, right, _, top = window.extent
    labels = [(right - 0.2, 0, "x", INK, (0, 12), "mm"),
              (0, top - 0.2, "t", INK, (10, 0), "lm")] + labels
    for x, t, text, colour, offset, anchor in labels:
        place = ((x - window.left) / window.pixel + offset[0],
                 (window.top - t) / window.pixel + offset[1])
        pen.text(place, text, font=FONT, fill=tuple(np.rint(colour * 255).astype(int)), anchor=anchor)
    return np.asarray(canvas)


def exchange_frames(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                    n_frames: int):
    """Light pulses advancing through the diagram; their completed paths stay visible."""
    tracks, sent, hits, returned = map(coordinates, (worldlines, emitted, reflected, received))
    starts, ends = np.concatenate((sent, hits)), np.concatenate((hits, returned))
    n_pulses = len(sent)
    colours = np.repeat(np.stack((ORANGE, BLUE)), n_pulses, axis=0)
    paths = np.stack((starts, ends), axis=1)
    finish = returned[-1, 1]
    for now in np.linspace(sent[0, 1] - 0.3, finish + 0.7, n_frames):
        # Clip only the visible part of each already computed light segment to drawing time.
        fraction = np.clip((now - starts[:, 1]) / (ends[:, 1] - starts[:, 1]), 0, 1)
        fronts = starts + (ends - starts) * fraction[:, None]
        begun = fraction > 0
        travelling = begun & (fraction < 1)
        layers = [strokes(tracks, 2, np.stack((INK, GREY)), 1, SOLID),
                  strokes(paths, 1, colours, 0.16, SOLID),
                  strokes(np.stack((starts[begun], fronts[begun]), axis=1), 2, colours[begun], 1, SOLID)]
        layers += dots(fronts[travelling], colours[travelling], 4)
        layers += dots(hits[hits[:, 1] <= now], ORANGE, 3)
        clock_events = np.concatenate((sent[sent[:, 1] <= now], returned[returned[:, 1] <= now]))
        layers += dots(clock_events, INK, 3)
        labels = [(x, t, f"{t:.2f}", INK, (-10, 0), "rm") for x, t in clock_events]
        yield written(EXCHANGE_WINDOW, layers, labels)


def reconstruction(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                   located: Event, simultaneous: Event, radar_time: Scalar, radar_range: Scalar, clock: Slice,
                   window: planar.Window) -> np.ndarray:
    """The radar assignment for the middle pulse, with every recovered reflection marked and the middle one's clock
    readings, radar time and range written beside it."""
    tracks, sent, hits, returned, measured, midpoints = map(
        coordinates, (worldlines, emitted, reflected, received, located, simultaneous))
    index = len(sent) // 2
    triangle = np.stack((sent[index], hits[index], returned[index]))[None]
    outgoing = np.stack((sent, hits), axis=1)
    incoming = np.stack((hits, returned), axis=1)
    layers = [Polygons(planar.mv.point(triangle), BLUE, 0.045),
              strokes(tracks, 2, np.stack((INK, GREY)), 1, SOLID),
              strokes(outgoing, 1, ORANGE, 0.2, SOLID),
              strokes(incoming, 1, BLUE, 0.2, SOLID),
              strokes(outgoing[index:index + 1], 2, ORANGE, 1, SOLID),
              strokes(incoming[index:index + 1], 2, BLUE, 1, SOLID),
              strokes(np.stack((midpoints[index], measured[index]))[None], 1.7, GREEN, 1, (5, 4)),
              strokes(measured[None], 2, GREEN, 1, SOLID)]
    layers += dots(measured, GREEN, 4)
    layers += dots(np.stack((sent[index], returned[index])), INK, 3)
    layers += dots(midpoints[index:index + 1], GREEN, 3)
    readings = (clock & stack((emitted[index], received[index]))).to_array()
    middle, distance = radar_time[index].to_array(), radar_range[index].to_array()
    range_label = (midpoints[index] + measured[index]) / 2
    labels = [(sent[index, 0], sent[index, 1], f"τ− = {readings[0]:.2f}", INK, (-10, 0), "rm"),
              (returned[index, 0], returned[index, 1], f"τ+ = {readings[1]:.2f}", INK, (-10, 0), "rm"),
              (*midpoints[index], f"T = {middle:.2f}", GREEN, (-10, 0), "rm"),
              (*range_label, f"R = {distance:.2f}", GREEN, (0, -12), "mm")]
    return written(window, layers, labels)
