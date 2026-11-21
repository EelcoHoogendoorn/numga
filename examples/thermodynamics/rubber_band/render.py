"""A temperature-coloured rubber band hangs from a support bar with a weight at its lower end and a glowing lamp beside
it, shown alone or beside a hand-pulled band under a shared support.

Spatial positions are read off the vertices by pairing with the planes x and y. Because the plane joined with entropy has
no projective weight direction, every vertex carries unit spatial weight in `Camera`. Vertex temperature colours blend
across each triangle by evaluating its corner simplex gradients, on a shared palette from blue below room temperature
through grey at room temperature to red above it.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from numga import concatenate, dim, stack
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices
from examples.util.panels import image_figure
from examples.thermodynamics.rubber_band import core

mv = core.mv
# Temperature palette from cooler than the room through grey at room temperature to warmer, plus stroke and weight inks.
temperature_colours = LinearSegmentedColormap.from_list(
    "temperature", ((0.0, "#2f6eb5"), (0.5, "#7d8794"), (0.75, "#e2913f"), (1.0, "#c0391d")))
outline_colour = np.array(to_rgb("#4b4f55"))
support_colour = np.array(to_rgb("#2b2f33"))
weight_colour = np.array(to_rgb("#5a6068"))
# Lamp bulb colours (off and lit) and concentric translucent markers for its halo, measured in band widths.
bulb_colours = np.array([to_rgb("#a3a8ae"), to_rgb("#ffd36b")])
glow_colour = np.array(to_rgb("#ffaa33"))
glow_reach, n_glow, glow_opacity = mv.scalar(0.75, unit=dim.length), 12, 0.09
# Resolution (in pixels per band width) and margins (in band widths) for single and side-by-side views, plus stroke widths and bulb radius in pixels.
pixels_per_width = mv.scalar(64.0, unit=dim.pixel / dim.length)
margin = mv.scalar(0.12, unit=dim.length)
pair_pixels_per_width = mv.scalar(48.0, unit=dim.pixel / dim.length)
pair_margin = mv.scalar(0.45, unit=dim.length)
pulled_across = mv.scalar(2.6, unit=dim.length)
outline_width, edge_width, support_width, bulb_radius = 1.5, 1.5, 3.5, 4.5
support_reach = mv.scalar(0.8, unit=dim.length)


# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Premultiplied RGBA image composited over white as RGB bytes."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def tinted(temperatures: core.Scalar, room_temperature: core.Scalar, reach: core.Scalar) -> np.ndarray:
    """Vertex RGB colours for the vertex temperatures: grey at room temperature, red above it and blue below."""
    change = ((temperatures - room_temperature) / reach).to_array()          # [V]
    return temperature_colours(np.clip(0.5 + change / 2, 0, 1))[:, :3]        # [V, 3]


def scenery(beside: core.Scalar) -> core.Spatial:
    """Endpoints of the horizontal support bar spanning the band at the origin and a second band shifted horizontally."""
    across = mv.x.dual()
    return stack((-support_reach * across, (beside + support_reach) * across))


def framing(scenes: list[core.Point], resolution: core.Scalar, room: core.Scalar) -> Camera:
    """Camera framing the given point sets with margin `room` at `resolution` (`pixel/length`), reading unweighted
    spatial offsets into `dim.pixel`."""
    across, up = resolution * mv.x, resolution * mv.y                         # Vector+, pixel/length
    xs = concatenate([(across & points.batch()).cast(core.Scalar).reshape(-1) for points in scenes])   # Scalar+, pixel
    ys = concatenate([(up & points.batch()).cast(core.Scalar).reshape(-1) for points in scenes])       # Scalar+, pixel
    pad = resolution * room                                                   # [] Scalar+, pixel
    low = stack((xs.min(), ys.min())) - pad                                   # [2] Scalar+, pixel
    high = stack((xs.max(), ys.max())) + pad                                  # [2] Scalar+, pixel
    return Camera(across, up, 0 * across, mv.zeros(core.Upright), 1.0, (low + high) / 2, (high - low) / 2)


def hanging(places: core.Point, topology: Topology, band: Topology, weight: Topology, colours: np.ndarray,
            camera: Camera) -> list:
    """GPU layers for the hanging band: temperature-shaded band triangles, dark weight triangles, and outer and weight
    border strokes."""
    V, E, F = topology.n_elements
    # Pass a constant scalar field so triangles show vertex colours without contour lines.
    unmarked = np.zeros(V)                                                    # [V]
    return [contours.picture(places, band, unmarked, colours, camera),
            contours.picture(places, weight, unmarked, weight_colour, camera),
            simplices.strokes(places, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(places, weight.border, outline_colour, outline_width, camera)]


# --- drawing --------------------------------------------------------------------------
def draw_band(topology: Topology, in_weight: np.ndarray, places: core.Spatial):
    """Draw the unloaded band hanging from its support bar, with triangle edges overlaid and the weight shaded dark."""
    V, E, F = topology.n_elements
    unmarked = np.zeros(V)                                                    # [V]
    support = scenery(mv.scalar(0.0, unit=dim.length))
    camera = framing([places, support], pixels_per_width, margin)
    return image_figure(on_white(drawing.composited([
        contours.picture(places, topology.subset(~in_weight, -1), unmarked, temperature_colours(0.5)[:3], camera),
        contours.picture(places, topology.subset(in_weight, -1), unmarked, weight_colour, camera),
        simplices.strokes(places, topology, outline_colour, edge_width, camera),
        simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)])))


def animate(topology: Topology, in_weight: np.ndarray, lamp: core.Spatial, room_temperature: core.Scalar,
            hottest: core.Scalar, states: Iterable[tuple[core.Point, core.Scalar, float]]) -> Iterator[np.ndarray]:
    """Yield RGB frames of the lamp-driven band, colouring vertex temperatures up to the peak lamp temperature and
    drawing the glowing lamp beside it, framed on the initial state."""
    states = iter(states)
    first = next(states)
    support = scenery(mv.scalar(0.0, unit=dim.length))
    band, weight = topology.subset(~in_weight, -1), topology.subset(in_weight, -1)
    glow = lamp + glow_reach * stack((mv.x.dual(), -mv.x.dual(), mv.y.dual(), -mv.y.dual()))   # [4] Spatial-, length
    camera = framing([first[0], support, glow], pixels_per_width, margin)
    radii = pixels_per_width * glow_reach * np.linspace(1 / n_glow, 1, n_glow)[::-1]   # [n_glow] Scalar+, pixel
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for places, temperatures, lit in chain([first], states):
        bulb = bulb_colours[0] + lit * (bulb_colours[1] - bulb_colours[0])
        yield on_white(drawing.composited([
            simplices.markers(stack([lamp] * n_glow), glow_colour, radii, camera, opacity=lit * glow_opacity),
            *hanging(places, topology, band, weight, tinted(temperatures, room_temperature, hottest - room_temperature),
                     camera),
            fixed,
            simplices.markers(stack([lamp]), bulb, bulb_radius, camera)]))


def animate_pulled(topology: Topology, in_weight: np.ndarray, room_temperature: core.Scalar, swing: core.Scalar,
                   drop: core.Scalar, states: Iterable[tuple[core.Point, core.Scalar, float]]) -> Iterator[np.ndarray]:
    """Yield RGB frames of the hand-pulled band, colouring temperature excursions above and below room temperature and
    reserving vertical room below the initial state for the downward pull."""
    states = iter(states)
    first = next(states)
    support = scenery(mv.scalar(0.0, unit=dim.length))
    band, weight = topology.subset(~in_weight, -1), topology.subset(in_weight, -1)
    camera = framing([first[0], first[0] - drop * mv.y.dual(), support], pixels_per_width, margin)
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for places, temperatures, _ in chain([first], states):
        yield on_white(drawing.composited([
            *hanging(places, topology, band, weight, tinted(temperatures, room_temperature, swing), camera),
            fixed]))


def animate_together(topology: Topology, in_weight: np.ndarray, lamp: core.Spatial, room_temperature: core.Scalar,
                     hottest: core.Scalar, swing: core.Scalar, drop: core.Scalar,
                     states: Iterable[tuple[tuple[core.Point, core.Scalar, float],
                                            tuple[core.Point, core.Scalar, float]]]) -> Iterator[np.ndarray]:
    """Yield side-by-side RGB frames of the lamp-driven band on the left and the hand-pulled band on the right under a
    shared support bar, each tinted on its own temperature range."""
    states = iter(states)
    first = next(states)
    # Shift the pulled band horizontally for side-by-side display.
    beside = pulled_across * mv.x.dual()                                      # [] Spatial-, length
    support = scenery(pulled_across)
    band, weight = topology.subset(~in_weight, -1), topology.subset(in_weight, -1)
    glow = lamp + glow_reach * stack((mv.x.dual(), -mv.x.dual(), mv.y.dual(), -mv.y.dual()))   # [4] Spatial-, length
    pulled_start = first[1][0] + beside                                       # Point-[V]
    camera = framing([first[0][0], pulled_start, pulled_start - drop * mv.y.dual(), support, glow],
                     pair_pixels_per_width, pair_margin)
    radii = pair_pixels_per_width * glow_reach * np.linspace(1 / n_glow, 1, n_glow)[::-1]   # [n_glow] Scalar+, pixel
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for (places, temperatures, lit), (pulled_places, pulled_temperatures, _) in chain([first], states):
        bulb = bulb_colours[0] + lit * (bulb_colours[1] - bulb_colours[0])
        yield on_white(drawing.composited([
            simplices.markers(stack([lamp] * n_glow), glow_colour, radii, camera, opacity=lit * glow_opacity),
            *hanging(places, topology, band, weight, tinted(temperatures, room_temperature, hottest - room_temperature),
                     camera),
            *hanging(pulled_places + beside, topology, band, weight,
                     tinted(pulled_temperatures, room_temperature, swing), camera),
            fixed,
            simplices.markers(stack([lamp]), bulb, bulb_radius, camera)]))
