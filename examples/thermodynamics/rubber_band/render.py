"""Drawing for the rubber band engine: the band coloured by its temperature, hanging from its support with the weight,
its lowest rows, at its lower end, and the lamp beside it, glowing while it is on; and the band a hand pulls, alone or
beside the lamp's under one support.

The band's points are read by the planes x and y. The plane joined with the entropy has no projective weight, so a point
is an offset from the origin, its weight one. The colours of the temperatures at the vertices are interpolated across
the triangles by their corners' hats, on one palette for every band: grey at the room's temperature, towards red where
the band is warmer and towards blue where it is cooler, each band reaching red and blue at a change of its own.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb

from numga import stack
from examples.util.wgsl import contours, image_figure, simplices
from examples.util.wgsl.camera import Camera
from examples.thermodynamics.rubber_band import core

mv = core.mv
Topology = simplices.Topology
# The bands' colours, from cooler than the room through the room's grey to warmer, by orange to red; the outline's,
# the support's and the weight's.
temperature_colours = LinearSegmentedColormap.from_list(
    "temperature", ((0.0, "#2f6eb5"), (0.5, "#7d8794"), (0.75, "#e2913f"), (1.0, "#c0391d")))
outline_colour = np.array(to_rgb("#4b4f55"))
support_colour = np.array(to_rgb("#2b2f33"))
weight_colour = np.array(to_rgb("#5a6068"))
# The lamp's bulb, dark and lit, and its glow: dots of the glow's colour, each covering a little, wider and wider out to
# the glow's reach, in band widths.
bulb_colours = np.array([to_rgb("#a3a8ae"), to_rgb("#ffd36b")])
glow_colour = np.array(to_rgb("#ffaa33"))
glow_reach, n_glow, glow_opacity = 0.75, 12, 0.09
# The pictures' pixels to a band width and their room around the scene, in band widths, for one band and for two side
# by side; the pulled band's place across from the lamp's, in band widths; the widths of the outline, the triangles'
# edges and the support and the bulb's radius, in pixels; the support's reach beyond the bands' middles, in band widths.
pixels_per_width, margin = 64, 0.12
pair_pixels_per_width, pair_margin = 48, 0.45
pulled_across = 2.6
outline_width, edge_width, support_width, bulb_radius = 1.5, 1.5, 3.5, 4.5
support_reach = 0.8
# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def tinted(temperatures: core.Scalar, room_temperature: float, reach: float) -> np.ndarray:
    """The colours `[vertices, 3]` of the temperatures `Scalar+[vertices]` on the one palette: grey at the room's
    temperature, red this reach above it and blue this reach below."""
    change = (temperatures.to_array() - room_temperature) / reach            # [vertices]
    return temperature_colours(np.clip(0.5 + change / 2, 0, 1))[:, :3]


def scenery(beside: float) -> core.Point:
    """The support, a bar across the top of the band at the origin and of one this many band widths across from it,
    `[2] Point-`."""
    across = mv.x.dual()
    return stack((-support_reach * across, (beside + support_reach) * across))


def framing(scenes: list[core.Point], resolution: float, room: float) -> Camera:
    """The camera looking at these points with this room around them, in band widths, at this resolution, in pixels to
    a band width: points read by the planes x and y of their algebra, each of weight one."""
    xs = np.concatenate([(mv.x & points).to_array().ravel() for points in scenes])
    ys = np.concatenate([(mv.y & points).to_array().ravel() for points in scenes])
    low = np.array([xs.min(), ys.min()]) - room
    high = np.array([xs.max(), ys.max()]) + room
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, (low + high) / 2, (high - low) / 2,
                  round((high - low)[0] * resolution))


def hanging(places: core.Point, topology: Topology, band: Topology, weight: Topology, colours: np.ndarray,
            camera: Camera) -> list:
    """The layers of the band at its places `Point-[vertices]`: the band's own triangles in their colours at the
    vertices `[vertices, 3]`, the weight's dark, and the outlines of the whole and of the weight."""
    # No level lines: the colours alone fill the triangles.
    unmarked = np.zeros(topology.n_vertices)
    return [contours.picture(places, band, unmarked, colours, camera),
            contours.picture(places, weight, unmarked, weight_colour, camera),
            simplices.strokes(places, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(places, weight.border, outline_colour, outline_width, camera)]


# --- drawing --------------------------------------------------------------------------
def draw_band(topology: Topology, in_weight: np.ndarray, places: core.Point):
    """The band at rest at its places `Point-[vertices]`, the weight's triangles `[faces]` dark, its triangles' edges
    over it, hanging from the support."""
    support = scenery(0.0)
    camera = framing([places, support], pixels_per_width, margin)
    return image_figure(on_white(simplices.composited([
        contours.picture(places, topology.subset(~in_weight, -1), np.zeros(topology.n_vertices),
                         temperature_colours(0.5)[:3], camera),
        contours.picture(places, topology.subset(in_weight, -1), np.zeros(topology.n_vertices), weight_colour, camera),
        simplices.strokes(places, topology, outline_colour, edge_width, camera),
        simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)])))


def animate(topology: Topology, in_weight: np.ndarray, lamp: core.Point, room_temperature: float, hottest: float,
            states: Iterable[tuple[core.Point, core.Scalar, float]]) -> Iterator[np.ndarray]:
    """The band at each state as it comes, at its points `Point-[vertices]`, read for their places, coloured by its
    temperatures `Scalar+[vertices]`, red at the hottest, the weight's triangles `[faces]` dark; and the lamp at its
    place, glowing while it is on. The picture is framed on the first state, the band at
    its longest."""
    states = iter(states)
    first = next(states)
    support = scenery(0.0)
    band, weight = topology.subset(~in_weight, -1), topology.subset(in_weight, -1)
    glow = lamp + glow_reach * stack((mv.x.dual(), -mv.x.dual(), mv.y.dual(), -mv.y.dual()))   # [4] Point-
    camera = framing([first[0], support, glow], pixels_per_width, margin)
    radii = glow_reach / camera.pixel * np.linspace(1 / n_glow, 1, n_glow)[::-1]   # [glow]
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for places, temperatures, lit in chain([first], states):
        bulb = bulb_colours[0] + lit * (bulb_colours[1] - bulb_colours[0])
        yield on_white(simplices.composited([
            simplices.markers(stack([lamp] * n_glow), glow_colour, radii, camera, opacity=lit * glow_opacity),
            *hanging(places, topology, band, weight, tinted(temperatures, room_temperature, hottest - room_temperature),
                     camera),
            fixed,
            simplices.markers(stack([lamp]), bulb, bulb_radius, camera)]))


def animate_pulled(topology: Topology, in_weight: np.ndarray, room_temperature: float, swing: float, drop: float,
                   states: Iterable[tuple[core.Point, core.Scalar, float]]) -> Iterator[np.ndarray]:
    """The band pulled by hand at each state as it comes, at its points `Point-[vertices]`, read for their places,
    coloured by how much warmer or cooler than the room its temperatures `Scalar+[vertices]` are, red at this swing
    above and blue at it below, the weight's triangles `[faces]` dark. The picture is framed on the first state and
    that state lowered by this drop, about as far as the hand pulls the weight down."""
    states = iter(states)
    first = next(states)
    support = scenery(0.0)
    band, weight = topology.subset(~in_weight, -1), topology.subset(in_weight, -1)
    camera = framing([first[0], first[0] - drop * mv.y.dual(), support], pixels_per_width, margin)
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for places, temperatures, _ in chain([first], states):
        yield on_white(simplices.composited([
            *hanging(places, topology, band, weight, tinted(temperatures, room_temperature, swing), camera),
            fixed]))


def animate_together(topology: Topology, in_weight: np.ndarray, lamp: core.Point, room_temperature: float,
                     hottest: float, swing: float, drop: float,
                     states: Iterable[tuple[tuple[core.Point, core.Scalar, float],
                                            tuple[core.Point, core.Scalar, float]]]) -> Iterator[np.ndarray]:
    """The two bands side by side under one support, at each pair of states as it comes, each at its points
    `Point-[vertices]`, read for their places, the weight's triangles `[faces]` dark: on the left the band the lamp
    warms, red at the hottest, and the lamp at its place, glowing while it is on; on the right the band the hand pulls,
    red at this swing above the room's temperature and blue at it below. The picture is framed on the first pair, the
    pulled band also lowered by this drop, about as far as the hand pulls the weight down."""
    states = iter(states)
    first = next(states)
    # The pulled band's places moved across beside the lamp's band, for the picture alone.
    beside = pulled_across * mv.x.dual()                                      # [] Point-
    support = scenery(pulled_across)
    band, weight = topology.subset(~in_weight, -1), topology.subset(in_weight, -1)
    glow = lamp + glow_reach * stack((mv.x.dual(), -mv.x.dual(), mv.y.dual(), -mv.y.dual()))   # [4] Point-
    pulled_start = first[1][0] + beside                                       # Point-[vertices]
    camera = framing([first[0][0], pulled_start, pulled_start - drop * mv.y.dual(), support, glow],
                     pair_pixels_per_width, pair_margin)
    radii = glow_reach / camera.pixel * np.linspace(1 / n_glow, 1, n_glow)[::-1]   # [glow]
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for (places, temperatures, lit), (pulled_places, pulled_temperatures, _) in chain([first], states):
        bulb = bulb_colours[0] + lit * (bulb_colours[1] - bulb_colours[0])
        yield on_white(simplices.composited([
            simplices.markers(stack([lamp] * n_glow), glow_colour, radii, camera, opacity=lit * glow_opacity),
            *hanging(places, topology, band, weight, tinted(temperatures, room_temperature, hottest - room_temperature),
                     camera),
            *hanging(pulled_places + beside, topology, band, weight,
                     tinted(pulled_temperatures, room_temperature, swing), camera),
            fixed,
            simplices.markers(stack([lamp]), bulb, bulb_radius, camera)]))
