"""Spacetime diagrams draw worldlines, light rays, and clock readings at equal spatial and temporal scales."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import NamedTuple

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from numga import concatenate, stack, where
from numga.multivector.factory import MultivectorFactory
from examples.relativity.radar.core import Event, Plane, Scalar, mv
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices

blue_colour, orange_colour, grey_colour, ink_colour, green_colour = (
    np.array(to_rgb(colour)) for colour in ("#3196b4", "#de9439", "#aab9c1", "#485961", "#458875"))
faint_colour, white_colour = np.array(to_rgb("#e2e8eb")), np.ones(3)
label_font = ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans")), 13)
# A solid stroke: one dash, no gap.
solid_dash = (1.0, 0.0)


class Window(NamedTuple):
    """Spatial and coordinate-time bounds of a spacetime diagram cover `n_rows` vertical pixels."""
    positions: tuple[float, float]
    times: tuple[float, float]
    n_rows: int


exchange_window = Window((-1.55, 4.8), (-0.65, 8.75), 620)
frame_window = Window((-4, 6), (-2.65, 10.15), 620)


def event_coordinates(events: Event) -> np.ndarray:
    """Position and coordinate time of unit-weight events form a `[..., 2]` coordinate array at the drawing boundary."""
    return (stack((mv.x, mv.t)) & events[..., None]).to_array()               # [..., 2]


def diagram_camera(window: Window, mv: MultivectorFactory) -> Camera:
    """A square-pixel camera reads spatial position with the `x` plane, coordinate time with the `t` plane, and homogeneous weight with `w`."""
    pixel = (window.times[1] - window.times[0]) / window.n_rows
    n_columns = round((window.positions[1] - window.positions[0]) / pixel)
    return Camera(mv.x, mv.t, 0 * mv.x, mv.w, 0.0, np.array([sum(window.positions), sum(window.times)]) / 2,
                  np.array([n_columns, window.n_rows]) * pixel / 2, n_columns)


def polyline_strokes(events: Event, width: float, colour: np.ndarray, opacity: float, dashes: tuple[float, float],
                     camera: Camera) -> np.ndarray:
    """Renders polylines through event chains (`[n_chains, n_events] Event-`) at `width` pixels across."""
    n_chains, n_events = events.shape
    colours = np.repeat(np.broadcast_to(colour, (n_chains, 3)), n_events - 1, axis=0)
    return simplices.strokes(events.reshape(-1), Topology.chains([n_events] * n_chains), colours, width, camera,
                             dashes=dashes, opacity=opacity)


def event_markers(events: Event, colour: np.ndarray, radius: float, camera: Camera) -> list[np.ndarray]:
    """Renders white-ringed circular markers of `radius` pixels at `events`."""
    return [simplices.markers(events, white_colour, radius + 1.5, camera),
            simplices.markers(events, colour, radius, camera)]


def axis_endpoints(camera: Camera, mv: MultivectorFactory) -> tuple[Event, Event]:
    """Start and end events (`[2] Event-`) span the spatial and temporal coordinate axes inside `camera`."""
    left, right, bottom, top = camera.extent
    margin = 0.2
    # The origin, of unit weight, and the ideal events along space and along time.
    origin, across, time = mv.w.dual(), mv.x.dual(), -mv.t.dual()             # Event-, Event-, Event-
    starts = stack((origin + across * (left + margin), origin + time * (bottom + margin)))   # [2] Event-
    ends = stack((origin + across * (right - margin), origin + time * (top - margin)))       # [2] Event-
    return starts, ends


def labelled_diagram(camera: Camera, layers: list[np.ndarray], labels: list, mv: MultivectorFactory) -> np.ndarray:
    """Composites the spacetime geometry over white and draws text labels beside their events."""
    starts, ends = axis_endpoints(camera, mv)
    axes = polyline_strokes(stack((starts, ends), axis=1), 1, faint_colour, 1, solid_dash, camera)
    picture = drawing.composited([axes] + layers)
    seen = white_colour * (1 - picture[..., 3:]) + picture[..., :3] * picture[..., 3:]
    canvas = Image.fromarray(np.round(np.clip(seen, 0, 1) * 255).astype(np.uint8))
    pen = ImageDraw.Draw(canvas)
    all_labels = [(ends[0], "x", ink_colour, (0, 12), "mm"), (ends[1], "t", ink_colour, (10, 0), "lm")] + labels
    n_labels = len(all_labels)
    places = event_coordinates(stack([event for event, *_ in all_labels]))    # [n_labels, 2]
    left, _, _, top = camera.extent
    for (x, t), (_, text, colour, offset, anchor) in zip(places, all_labels):
        place = ((x - left) / camera.pixel + offset[0], (top - t) / camera.pixel + offset[1])
        pen.text(place, text, font=label_font, fill=tuple(np.rint(colour * 255).astype(int)), anchor=anchor)
    return np.asarray(canvas)


def exchange_animation(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                       n_frames: int) -> Iterator[np.ndarray]:
    """Yields rendered frames as light pulses propagate out to the reflector and back to the clock."""
    camera = diagram_camera(exchange_window, mv)
    n_pulses = len(emitted)
    starts, ends = concatenate((emitted, reflected)), concatenate((reflected, received))   # [2 * n_pulses] Event-, Event-
    colours = np.repeat(np.stack((orange_colour, blue_colour)), n_pulses, axis=0)
    paths = stack((starts, ends), axis=1)                                      # [2 * n_pulses, 2] Event-
    # The light's lines through spacetime, each pulse's way out and back.
    light_lines = starts & ends                                                # [2 * n_pulses] Line-
    first_time, finish_time = (mv.t & emitted[0]).to_array(), (mv.t & received[-1]).to_array()
    for slice_time in np.linspace(first_time - 0.3, finish_time + 0.7, n_frames):
        # The events at coordinate time `slice_time` pair to zero with this plane, the earlier ones negatively.
        present = mv.t - mv.w * slice_time                                     # [] Plane+
        has_begun = (present & starts) < 0
        has_arrived = (present & ends) < 0
        in_flight = has_begun & ~has_arrived
        # Each pulse in flight is where its line meets the present; one that has arrived stays at its end.
        crossings = light_lines ^ present                                      # [2 * n_pulses] Event-
        fronts = where(has_arrived, ends, crossings / (mv.w & crossings))      # [2 * n_pulses] Event-
        layers = [polyline_strokes(worldlines, 2, np.stack((ink_colour, grey_colour)), 1, solid_dash, camera),
                  polyline_strokes(paths, 1, colours, 0.16, solid_dash, camera),
                  polyline_strokes(stack((starts, fronts), axis=1)[has_begun], 2, colours[has_begun], 1, solid_dash, camera)]
        layers += event_markers(fronts[in_flight], colours[in_flight], 4, camera)
        layers += event_markers(reflected[(present & reflected) <= 0], orange_colour, 3, camera)
        clock_events = concatenate((emitted[(present & emitted) <= 0], received[(present & received) <= 0]))
        layers += event_markers(clock_events, ink_colour, 3, camera)
        labels = [(event, f"{reading:.2f}", ink_colour, (-10, 0), "rm")
                  for event, reading in zip(clock_events, (mv.t & clock_events).to_array())]
        yield labelled_diagram(camera, layers, labels, mv)


def reconstruction_image(worldlines: Event, emitted: Event, reflected: Event, received: Event,
                         located: Event, simultaneous: Event, radar_time: Scalar, radar_range: Scalar, clock: Plane,
                         window: Window) -> np.ndarray:
    """Renders the radar assignment for the middle pulse, marking all recovered reflections and labelling clock times, radar time, and range."""
    camera = diagram_camera(window, mv)
    n_pulses = len(emitted)
    index = n_pulses // 2
    triangle = stack((emitted[index], reflected[index], received[index]))     # [3] Event-
    outgoing_legs = stack((emitted, reflected), axis=1)                        # [n_pulses, 2] Event-
    incoming_legs = stack((reflected, received), axis=1)                       # [n_pulses, 2] Event-
    layers = [contours.picture(triangle, Topology(3, np.array([[0, 1, 2]])), np.zeros(3), blue_colour, camera,
                               opacity=0.045),
              polyline_strokes(worldlines, 2, np.stack((ink_colour, grey_colour)), 1, solid_dash, camera),
              polyline_strokes(outgoing_legs, 1, orange_colour, 0.2, solid_dash, camera),
              polyline_strokes(incoming_legs, 1, blue_colour, 0.2, solid_dash, camera),
              polyline_strokes(outgoing_legs[index:index + 1], 2, orange_colour, 1, solid_dash, camera),
              polyline_strokes(incoming_legs[index:index + 1], 2, blue_colour, 1, solid_dash, camera),
              polyline_strokes(stack((simultaneous[index], located[index]))[None], 1.7, green_colour, 1, (5, 4), camera),
              polyline_strokes(located[None], 2, green_colour, 1, solid_dash, camera)]
    layers += event_markers(located, green_colour, 4, camera)
    layers += event_markers(stack((emitted[index], received[index])), ink_colour, 3, camera)
    layers += event_markers(simultaneous[index:index + 1], green_colour, 3, camera)
    readings = (clock & stack((emitted[index], received[index]))).to_array()
    middle_time, distance = radar_time[index].to_array(), radar_range[index].to_array()
    # The range written halfway along the line of simultaneity to the located reflection.
    range_label = (simultaneous[index] + located[index]) / 2                   # [] Event-
    labels = [(emitted[index], f"τ− = {readings[0]:.2f}", ink_colour, (-10, 0), "rm"),
              (received[index], f"τ+ = {readings[1]:.2f}", ink_colour, (-10, 0), "rm"),
              (simultaneous[index], f"T = {middle_time:.2f}", green_colour, (-10, 0), "rm"),
              (range_label, f"R = {distance:.2f}", green_colour, (0, -12), "mm")]
    return labelled_diagram(camera, layers, labels, mv)


def frames_animation(states: Iterable[tuple[Event, Event, Event, Event, Event, Event, Scalar, Scalar, Plane]]) -> Iterator[np.ndarray]:
    """Yields the spacetime reconstruction diagram for each boosted and translated state."""
    for state in states:
        yield reconstruction_image(*state, frame_window)
