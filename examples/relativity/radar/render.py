"""Spacetime diagrams at equal spatial and temporal scales, with clock readings beside events."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from numga import concatenate, stack, where
from numga.multivector.factory import MultivectorFactory
from examples.relativity.radar.core import Event, Scalar, Slice
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera

BLUE, ORANGE, GREY, INK, GREEN = (
    np.array(to_rgb(colour)) for colour in ("#3196b4", "#de9439", "#aab9c1", "#485961", "#458875"))
FAINT, WHITE = np.array(to_rgb("#e2e8eb")), np.ones(3)
FONT = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 13)
# A solid stroke: one dash, no gap.
SOLID = (1.0, 0.0)


class Window(NamedTuple):
    """The positions and coordinate times a diagram shows, and its rows of pixels down the times."""
    positions: tuple[float, float]
    times: tuple[float, float]
    rows: int


EXCHANGE_WINDOW = Window((-1.55, 4.8), (-0.65, 8.75), 620)
FRAME_WINDOW = Window((-4, 6), (-2.65, 10.15), 620)


def coordinates(events: Event) -> np.ndarray:
    """Position and coordinate time of events of unit weight `[..., 2]`, read only at the drawing boundary."""
    mv = events.context.multivector
    return (stack((mv.x, mv.t)) & events[..., None]).to_array()


def diagram_camera(window: Window, mv: MultivectorFactory) -> Camera:
    """The camera of a diagram, as many square pixels across its positions as fit, about their middle: the events as
    they are, position across, read by the plane x, and coordinate time up, read by the plane t, over their weights,
    their pairings with the plane w."""
    pixel = (window.times[1] - window.times[0]) / window.rows
    columns = round((window.positions[1] - window.positions[0]) / pixel)
    return Camera(mv.x, mv.t, 0 * mv.x, mv.w, 0.0, np.array([sum(window.positions), sum(window.times)]) / 2,
                  np.array([columns, window.rows]) * pixel / 2, columns)


def strokes(events: Event, width: float, colour: np.ndarray, opacity: float, dashes: tuple[float, float],
            camera: Camera) -> np.ndarray:
    """Polylines through events `[n, k]`, `width` pixels across, one colour `[3]` or one each `[n, 3]`: their
    picture."""
    size, n_events = events.shape
    colours = np.repeat(np.broadcast_to(colour, (size, 3)), n_events - 1, axis=0)
    return simplices.strokes(events.reshape(-1), simplices.Mesh.chains([n_events] * size), colours, width, camera,
                             dashes=dashes, opacity=opacity)


def dots(events: Event, colour: np.ndarray, radius: float, camera: Camera) -> list[np.ndarray]:
    """Dots at events `[n]`, `radius` pixels, ringed in white: their pictures."""
    return [simplices.markers(events, WHITE, radius + 1.5, camera),
            simplices.markers(events, colour, radius, camera)]


def axis_ends(camera: Camera, mv: MultivectorFactory) -> tuple[Event, Event]:
    """The ends of the space and time axes through the origin, a little inside the picture: x first, then t."""
    left, right, bottom, top = camera.extent
    margin = 0.2
    # The origin, of unit weight, and the ideal events along space and along time.
    origin, across, time = mv.w.dual(), mv.x.dual(), -mv.t.dual()             # [] Event-, Event-, Event-
    starts = stack((origin + across * (left + margin), origin + time * (bottom + margin)))   # [axes] Event-
    ends = stack((origin + across * (right - margin), origin + time * (top - margin)))       # [axes] Event-
    return starts, ends


def written(camera: Camera, layers: list[np.ndarray], labels: list, mv: MultivectorFactory) -> np.ndarray:
    """Composite the geometry over white and attach labels, each beside its event, in the same native pixel grid."""
    starts, ends = axis_ends(camera, mv)
    axes = strokes(stack((starts, ends), axis=1), 1, FAINT, 1, SOLID, camera)
    picture = simplices.composited([axes] + layers)
    seen = WHITE * (1 - picture[..., 3:]) + picture[..., :3] * picture[..., 3:]
    canvas = Image.fromarray(np.round(np.clip(seen, 0, 1) * 255).astype(np.uint8))
    pen = ImageDraw.Draw(canvas)
    labels = [(ends[0], "x", INK, (0, 12), "mm"), (ends[1], "t", INK, (10, 0), "lm")] + labels
    places = coordinates(stack([event for event, *_ in labels]))               # [labels, 2]
    left, _, _, top = camera.extent
    for (x, t), (_, text, colour, offset, anchor) in zip(places, labels):
        place = ((x - left) / camera.pixel + offset[0], (top - t) / camera.pixel + offset[1])
        pen.text(place, text, font=FONT, fill=tuple(np.rint(colour * 255).astype(int)), anchor=anchor)
    return np.asarray(canvas)


def exchange_frames(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                    n_frames: int):
    """Light pulses advancing through the diagram; their completed paths stay visible."""
    mv = emitted.context.multivector
    camera = diagram_camera(EXCHANGE_WINDOW, mv)
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
        layers = [strokes(worldlines, 2, np.stack((INK, GREY)), 1, SOLID, camera),
                  strokes(paths, 1, colours, 0.16, SOLID, camera),
                  strokes(stack((starts, fronts), axis=1)[begun], 2, colours[begun], 1, SOLID, camera)]
        layers += dots(fronts[travelling], colours[travelling], 4, camera)
        layers += dots(reflected[(present & reflected) <= 0], ORANGE, 3, camera)
        clock_events = concatenate((emitted[(present & emitted) <= 0], received[(present & received) <= 0]))
        layers += dots(clock_events, INK, 3, camera)
        labels = [(event, f"{reading:.2f}", INK, (-10, 0), "rm")
                  for event, reading in zip(clock_events, (mv.t & clock_events).to_array())]
        yield written(camera, layers, labels, mv)


def reconstruction(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                   located: Event, simultaneous: Event, radar_time: Scalar, radar_range: Scalar, clock: Slice,
                   window: Window) -> np.ndarray:
    """The radar assignment for the middle pulse, with every recovered reflection marked and the middle one's clock
    readings, radar time and range written beside it."""
    mv = emitted.context.multivector
    camera = diagram_camera(window, mv)
    index = len(emitted) // 2
    triangle = stack((emitted[index], reflected[index], received[index]))   # [corners] Event-
    outgoing = stack((emitted, reflected), axis=1)                            # [pulses, 2] Event-
    incoming = stack((reflected, received), axis=1)                           # [pulses, 2] Event-
    layers = [contours.picture(triangle, simplices.Mesh(3, np.array([[0, 1, 2]])), np.zeros(3), BLUE, camera,
                               opacity=0.045),
              strokes(worldlines, 2, np.stack((INK, GREY)), 1, SOLID, camera),
              strokes(outgoing, 1, ORANGE, 0.2, SOLID, camera),
              strokes(incoming, 1, BLUE, 0.2, SOLID, camera),
              strokes(outgoing[index:index + 1], 2, ORANGE, 1, SOLID, camera),
              strokes(incoming[index:index + 1], 2, BLUE, 1, SOLID, camera),
              strokes(stack((simultaneous[index], located[index]))[None], 1.7, GREEN, 1, (5, 4), camera),
              strokes(located[None], 2, GREEN, 1, SOLID, camera)]
    layers += dots(located, GREEN, 4, camera)
    layers += dots(stack((emitted[index], received[index])), INK, 3, camera)
    layers += dots(simultaneous[index:index + 1], GREEN, 3, camera)
    readings = (clock & stack((emitted[index], received[index]))).to_array()
    middle, distance = radar_time[index].to_array(), radar_range[index].to_array()
    # The range written halfway along the line of simultaneity to the located reflection.
    range_label = (simultaneous[index] + located[index]) / 2                   # [] Event-
    labels = [(emitted[index], f"τ− = {readings[0]:.2f}", INK, (-10, 0), "rm"),
              (received[index], f"τ+ = {readings[1]:.2f}", INK, (-10, 0), "rm"),
              (simultaneous[index], f"T = {middle:.2f}", GREEN, (-10, 0), "rm"),
              (range_label, f"R = {distance:.2f}", GREEN, (0, -12), "mm")]
    return written(camera, layers, labels, mv)
