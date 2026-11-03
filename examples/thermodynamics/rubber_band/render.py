"""Drawing for the rubber band engine: the band coloured by its temperature, hanging from its support with the weight,
its lowest rows, at its lower end, and the lamp beside it, glowing while it is on.

The band's points are read by the planes x and y. The plane joined with the entropy has no projective weight, so a point
is an offset from the origin, its weight one. The colours of the temperatures at the vertices are interpolated across
the triangles by their corners' hats.
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

Topology = simplices.Topology
# The band's colours from the room's temperature to the lamp's, its outline's, the support's and the weight's.
band_colours = LinearSegmentedColormap.from_list("rubber", ("#7d8794", "#e2913f", "#c0391d"))
# The pulled band's colours, from cooler than the room through the room's grey to warmer.
change_colours = LinearSegmentedColormap.from_list("change", ("#2f6eb5", "#7d8794", "#c0391d"))
outline_colour = np.array(to_rgb("#4b4f55"))
support_colour = np.array(to_rgb("#2b2f33"))
weight_colour = np.array(to_rgb("#5a6068"))
# The lamp's bulb, dark and lit, and its glow: dots of the glow's colour, each covering a little, wider and wider out to
# the glow's reach, in band widths.
bulb_colours = np.array([to_rgb("#a3a8ae"), to_rgb("#ffd36b")])
glow_colour = np.array(to_rgb("#ffaa33"))
glow_reach, n_glow, glow_opacity = 0.75, 12, 0.09
# The pictures' pixels to a band width and their room around the scene, in band widths; the widths of the outline, the
# triangles' edges and the support and the bulb's radius, in pixels; the support's reach either side of the band's
# middle, in band widths.
pixels_per_width, margin = 64, 0.12
outline_width, edge_width, support_width, bulb_radius = 1.5, 1.5, 3.5, 4.5
support_reach = 0.8
# --- plumbing -------------------------------------------------------------------------
def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def scenery(points: core.Point) -> core.Point:
    """The support, a bar across the band's top at the origin `[2] Point-`, in the algebra of these points."""
    across = points.context.multivector.x.dual()
    return stack((-support_reach * across, support_reach * across))


def framing(scenes: list[core.Point]) -> Camera:
    """The camera looking at these points with the margin around them, `pixels_per_width` to a band width: points
    read by the planes x and y of their algebra, each of weight one."""
    mv = scenes[0].context.multivector
    xs = np.concatenate([(mv.x & points).to_array().ravel() for points in scenes])
    ys = np.concatenate([(mv.y & points).to_array().ravel() for points in scenes])
    low = np.array([xs.min(), ys.min()]) - margin
    high = np.array([xs.max(), ys.max()]) + margin
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, (low + high) / 2, (high - low) / 2,
                  round((high - low)[0] * pixels_per_width))


# --- drawing --------------------------------------------------------------------------
def draw_band(topology: Topology, in_weight: np.ndarray, places: core.Point):
    """The band at rest at its places `Point-[vertices]`, the weight's triangles `[faces]` dark, its triangles' edges
    over it, hanging from the support."""
    support = scenery(places)
    camera = framing([places, support])
    return image_figure(on_white(simplices.composited([
        contours.picture(places, topology.subset(~in_weight), np.zeros(topology.n_vertices), band_colours(0.0)[:3],
                         camera),
        contours.picture(places, topology.subset(in_weight), np.zeros(topology.n_vertices), weight_colour, camera),
        simplices.strokes(places, topology, outline_colour, edge_width, camera),
        simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)])))


def animate(topology: Topology, in_weight: np.ndarray, lamp: core.Point, coolest: float, hottest: float,
            states: Iterable[tuple[core.Point, core.Scalar, core.Scalar, float, object]]) -> Iterator[np.ndarray]:
    """The band at each state as it comes, at its places `Point-[vertices]`, the entropies they hold left undrawn, coloured by its temperatures
    `Scalar+[vertices]` from the coolest to the hottest, the weight's triangles `[faces]` dark; and the lamp at its
    place, glowing while it is on; the budget is left undrawn. The picture is framed on the first state, the band at
    its longest."""
    states = iter(states)
    first = next(states)
    mv = lamp.context.multivector
    support = scenery(lamp)
    band, weight = topology.subset(~in_weight), topology.subset(in_weight)
    glow = lamp + glow_reach * stack((mv.x.dual(), -mv.x.dual(), mv.y.dual(), -mv.y.dual()))   # [4] Point-
    camera = framing([first[0], support, glow])
    radii = glow_reach / camera.pixel * np.linspace(1 / n_glow, 1, n_glow)[::-1]   # [glow]
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for places, _, temperatures, lit, _ in chain([first], states):
        warmth = temperatures.to_array()                                      # [vertices]
        colours = band_colours((warmth - coolest) / (hottest - coolest))[:, :3]
        bulb = bulb_colours[0] + lit * (bulb_colours[1] - bulb_colours[0])
        yield on_white(simplices.composited([
            simplices.markers(stack([lamp] * n_glow), glow_colour, radii, camera, opacity=lit * glow_opacity),
            contours.picture(places, band, warmth, colours, camera),
            contours.picture(places, weight, np.zeros(topology.n_vertices), weight_colour, camera),
            simplices.strokes(places, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(places, weight.border, outline_colour, outline_width, camera),
            fixed,
            simplices.markers(stack([lamp]), bulb, bulb_radius, camera)]))


def animate_pulled(topology: Topology, in_weight: np.ndarray, room_temperature: float, swing: float, drop: float,
                   states: Iterable[tuple[core.Point, core.Scalar, core.Scalar, float, object]]) -> Iterator[np.ndarray]:
    """The band pulled by hand at each state as it comes, at its places `Point-[vertices]`, the entropies they hold left undrawn, coloured by how much
    warmer or cooler than the room its temperatures `Scalar+[vertices]` are, red to this swing above and blue to it
    below, the weight's triangles `[faces]` dark. The picture is framed on the first state and that state lowered by
    this drop, about as far as the hand pulls the weight down."""
    states = iter(states)
    first = next(states)
    mv = first[0].context.multivector
    support = scenery(first[0])
    band, weight = topology.subset(~in_weight), topology.subset(in_weight)
    camera = framing([first[0], first[0] - drop * mv.y.dual(), support])
    fixed = simplices.strokes(support, Topology.chains([2]), support_colour, support_width, camera)
    for places, _, temperatures, _, _ in chain([first], states):
        change = (temperatures.to_array() - room_temperature) / swing         # [vertices]
        colours = change_colours(np.clip(0.5 + change / 2, 0, 1))[:, :3]
        yield on_white(simplices.composited([
            contours.picture(places, band, change, colours, camera),
            contours.picture(places, weight, np.zeros(topology.n_vertices), weight_colour, camera),
            simplices.strokes(places, topology.border, outline_colour, outline_width, camera),
            simplices.strokes(places, weight.border, outline_colour, outline_width, camera),
            fixed]))
