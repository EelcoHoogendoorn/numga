"""Drawing shared by the thermodynamics examples: diagrams of one quantity against another, drawn on the GPU by
`planar` with their names written over them, and the apparatus, a cylinder under a piston traced by `solids`.

A diagram's chart is a plane whose points carry its two quantities, read back by pairing, `mv.x & point` across and
`mv.y & point` up; the diagram is an affine map from that plane onto the plane of its picture, in pixels, each quantity
at its own scale. Curves are polylines through chart points, areas are polygons, tangents are joins of a point with a
direction and their crossings meets, and the conics of a chart are carried into the picture through the map; the
picture's points are read as pixels only where Pillow writes text.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from contourpy import contour_generator

from numga import Extensor, concatenate, stack
from examples.util import charts
from examples.util.wgsl import image_figure, planar, solids
from examples.util.wgsl.planar import SOLID, Cap, Marker, Marks, Polygons, Strokes

mv = planar.mv
Point = planar.Point                                                         # Antivector-
Plane = planar.Plane                                                         # Vector+
Quadric = planar.Quadric                                                     # Plane+ <- Point-
Placement = planar.ga.gatype((Point, Point))                                 # Point- <- Point-


def rgb(colour: str) -> np.ndarray:
    return np.array(to_rgb(colour))


blue, orange, grey, ink, green = rgb("#3196b4"), rgb("#de9439"), rgb("#aab9c1"), rgb("#485961"), rgb("#458875")
# The faint colour of the lines a diagram is read against where no ramp of its own suits, and the white behind
# everything.
faint = rgb("#cfd9dc")
white = np.ones(3)
# The colour of each family of lines a diagram is read against, one plain colour a family: isotherms red, isobars
# purple, isochores green, adiabats teal, levels of humidity blue, of enthalpy brown, other levels grey-blue. Phase
# boundaries, the lines the regions of a diagram part along, are black.
families = dict(temperature=rgb("#d9604c"), pressure=rgb("#8a68b8"), volume=rgb("#5aa260"), entropy=rgb("#3f9e96"),
                humidity=rgb("#4a82bd"), enthalpy=rgb("#a8763f"), level=rgb("#8a9ea6"))
boundary = rgb("#1c2326")

# Pixels around a diagram's plot: room for the name of the quantity up on the left, of the one across below, and a
# little on the other two sides.
left_margin, bottom_margin, right_margin, top_margin = 30, 24, 12, 16
# The axes' width and arrowheads, in pixels: the axes as wide as a chart's frame; and how far a tick mark reaches out
# from its axis.
axis_width, head_length, head_width, tick_length = charts.FRAME_PX, 7.0, 5.0, 4.0
# The widths of the strokes, in pixels, those of the charts: the lines a diagram is read against as thin as a chart's
# frame, a curve as wide as a chart's, and the path the diagram follows half as wide again.
thin, medium, bold = charts.FRAME_PX, charts.LINE_PX, 1.5 * charts.LINE_PX
# The size of the text, the quantities' names and the labels on curves and regions, and of the numbers at the ticks
# and the subscripts, in pixels: the charts' text and tick labels.
text_size, tick_size = charts.TEXT_PX, charts.TICK_PX
fonts = {size: ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", style="italic")), size)
         for size in (text_size, tick_size)}
# The width of the white halo around written text, in pixels.
halo = 2
# The radius of a dot marking a state, and of the white ring around it, in pixels.
dot, ring = 4.0, 5.5


# --- diagrams --------------------------------------------------------------------------
def chart_points(across: np.ndarray, up: np.ndarray) -> Point:
    """The points of a chart's plane at values of its two quantities `[...]`: the origin moved by each value along its
    quantity's direction. Every value a chart shows enters its plane here."""
    return planar.origin + mv.x.dual() * np.asarray(across) + mv.y.dual() * np.asarray(up)


def chart_directions(across: np.ndarray, up: np.ndarray) -> Point:
    """The directions of a chart's plane, points at infinity, along changes of its two quantities `[...]`."""
    return mv.x.dual() * np.asarray(across) + mv.y.dual() * np.asarray(up)


@dataclass(frozen=True)
class Diagram:
    """A diagram of one quantity across against another up: the range of each shown, the values at which the axes
    cross, the picture's rows and columns of pixels, and the names of the two quantities."""
    across: tuple[float, float]
    up: tuple[float, float]
    crossing: tuple[float, float]
    rows: int
    columns: int
    names: tuple[str, str]

    @cached_property
    def placement(self) -> Placement:
        """The affine map from the chart's plane onto the picture's, in pixels right and up from its bottom left
        corner: the starts of the two ranges to the plot's corner inside the margins, each range across the plot's
        width or up its height."""
        corner = planar.origin + mv.x.dual() * left_margin + mv.y.dual() * bottom_margin   # [] Point-
        # A unit of each quantity in the picture: so many pixels right, and so many up.
        width, height = self.columns - left_margin - right_margin, self.rows - bottom_margin - top_margin
        rightward = mv.x.dual() * (width / (self.across[1] - self.across[0]))       # [] Point-
        upward = mv.y.dual() * (height / (self.up[1] - self.up[0]))                 # [] Point-
        # The chart's planes where the two ranges start: a chart point pairs with each to how far past the start it
        # lies.
        left_edge = mv.x - mv.w * self.across[0]                                # [] Plane+
        bottom_edge = mv.y - mv.w * self.up[0]                                  # [] Plane+
        return corner * (mv.w & Point) + rightward * (left_edge & Point) + upward * (bottom_edge & Point)

    @cached_property
    def bounds(self) -> Plane:
        """The four planes at the ends of the chart's ranges, each positive on the side of the ranges `[4]`."""
        return stack((mv.x - mv.w * self.across[0], mv.w * self.across[1] - mv.x,
                      mv.y - mv.w * self.up[0], mv.w * self.up[1] - mv.y))

    @cached_property
    def window(self) -> planar.Window:
        return planar.window((0, self.columns), (0, self.rows), self.rows)


def pixel_places(points: Point) -> np.ndarray:
    """The pixels right and up from a picture's bottom left corner at which its points of unit weight `[...]` lie,
    `[..., 2]`, read by pairing: where Pillow writes text."""
    return (stack((mv.x, mv.y)) & points[..., None]).to_array()


def arrowheads(starts: Point, ends: Point, colour: np.ndarray) -> Polygons:
    """Heads on segments of the picture's plane from starts to ends `[n]`, each a triangle beyond its end, pointing
    on along the segment."""
    along = ends - starts                                                    # [n] Point-
    # A direction has no weight to normalise by: its length is the norm of its dual, the plane square to it.
    unit = along / along.dual().norm()                                       # [n] Point-
    # The direction square to it, a quarter turn in the plane of the picture.
    square = unit.commutator(mv.xy)                                          # [n] Point-
    corners = stack((ends + unit * head_length, ends + square * (head_width / 2), ends - square * (head_width / 2)),
                    axis=-1)                                                 # [n, 3] Point-
    return Polygons(corners, colour, 1.0)


def axes(diagram: Diagram) -> list:
    """The two axes, crossing where the diagram says, each ending in an arrowhead."""
    x0, y0 = diagram.crossing
    corner = diagram.placement(chart_points(x0, y0))                          # [] Point-
    ends = diagram.placement(chart_points(np.array([diagram.across[1], x0]),
                                          np.array([y0, diagram.up[1]])))           # [2] Point-
    starts = corner.broadcast_to((2,))                                       # [2] Point-
    return [Strokes(stack((starts, ends), axis=-1), axis_width, ink, 1.0, SOLID, Cap.BUTT),
            arrowheads(starts, ends, ink)]


def within(diagram: Diagram, lines: Point, width: float, colour: np.ndarray, opacity: float) -> list[Strokes]:
    """Polylines through chart points `[curves, samples]`, each in its colour `[curves, 3]`, cut to the runs of samples
    inside the diagram's ranges."""
    return [stroke for curve in range(lines.shape[0])
            for stroke in polylines(diagram, [lines[curve]], width, colour[curve], opacity, SOLID)]


def polylines(diagram: Diagram, lines: list[Point], width: float, colour: np.ndarray, opacity: float,
              dashes: tuple[float, float]) -> list[Strokes]:
    """Polylines through chart points of unit weight, each its own number of samples, cut to the runs of samples
    inside the diagram's ranges; dashed `dashes` pixels on and off, `SOLID` for solid."""
    strokes = []
    for points in lines:
        inside = ((diagram.bounds & points[:, None]) >= 0).all(axis=-1)
        # The runs of samples inside, between the steps into and out of the ranges.
        steps = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
        for start, end in zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)):
            if end - start > 1:
                cap = Cap.ROUND if dashes == SOLID else Cap.BUTT
                strokes.append(Strokes(diagram.placement(points[start:end])[None], width, colour, opacity, dashes, cap))
    return strokes


def level_lines(across: np.ndarray, up: np.ndarray, values: np.ndarray,
                levels: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Where a function sampled on a grid, `values[up, across]` at the grid's coordinates `across` and `up`, takes
    each of the levels: each level with a polyline `[samples, 2]` of the two coordinates, as many as it has."""
    lines = contour_generator(across, up, values)
    return [(level, line) for level in levels for line in lines.lines(level) if len(line) > 1]


def family(diagram: Diagram, lines: list[Point], name: str, width: float) -> list[Strokes]:
    """A family of lines, polylines of chart points, in the family's colour."""
    return polylines(diagram, lines, width, families[name], 1.0, SOLID)


def tangent_lines(points: Point, directions: Point, across: np.ndarray) -> Point:
    """Lines of a chart's plane through points along directions `[...]`, sampled where they cross the planes of the
    across quantity's values `[samples]`, `[..., samples]`: each line, the join of its point and its direction, met
    with each of those."""
    lines = points & directions                                              # [...] Line-
    crossings = mv.x - mv.w * np.asarray(across)                             # [samples] Plane+
    return planar.unit(lines[..., None] ^ crossings)


def arrows(diagram: Diagram, starts: Point, toward: Point, length: float, width: float,
           colour: np.ndarray) -> list:
    """Arrows `length` pixels long from chart points `[n]` toward others, each a stroke ending in a head."""
    tails, aims = diagram.placement(starts), diagram.placement(toward)      # [n] Point-, Point-
    along = aims - tails                                                     # [n] Point-
    tips = tails + along * (length / along.dual().norm())                    # [n] Point-
    return [Strokes(stack((tails, tips), axis=-1), width, colour, 1.0, SOLID, Cap.BUTT),
            arrowheads(tails, tips, colour)]


def ellipses(diagram: Diagram, quadrics: Quadric) -> Quadric:
    """Quadrics of a chart's plane `[n]` as quadrics of the picture's: each pixel's point carried back onto the chart,
    measured there, and its plane carried forward by the adjugate, which keeps every pairing of a plane with a
    point."""
    reading = diagram.placement.solve(Point)                                 # [] Point- <- Point-
    return reading.adjugate()(quadrics(reading))


def dashed(diagram: Diagram, points: Point, width: float, colour: np.ndarray, dashes: tuple[float, float]) -> Strokes:
    """Dashed polylines through chart points `[..., samples]`, dashes and gaps in pixels."""
    placed = diagram.placement(points)                                       # [..., samples] Point-
    return Strokes(placed.reshape(-1, placed.shape[-1]), width, colour, 1.0, dashes, Cap.BUTT)


def areas(diagram: Diagram, corners: Point, colour: np.ndarray, opacity: float) -> Polygons:
    """The areas inside outlines through chart points `[..., corners]`, each closed from its last corner back to its
    first, which it does not repeat."""
    placed = diagram.placement(corners)                                      # [..., corners] Point-
    return Polygons(placed.reshape(-1, placed.shape[-1]), colour, opacity)


def dots(diagram: Diagram, points: Point, colour: np.ndarray) -> list:
    """Dots at chart points `[n]`, each in a white ring."""
    placed = diagram.placement(points)                                       # [n] Point-
    return [Marks(placed, Marker.DISC, 0.0, ring, white, 1.0), Marks(placed, Marker.DISC, 0.0, dot, colour, 1.0)]


class Label(NamedTuple):
    """A name written at a chart point of unit weight, in its colour and size, placed by a Pillow anchor (`"lb"` puts
    the text's left bottom there) and shifted a few pixels right and up."""
    place: Point
    text: str
    colour: np.ndarray
    size: int
    anchor: str
    shift: tuple[float, float]


def ticks(diagram: Diagram, across: np.ndarray, up: np.ndarray) -> tuple[list, list]:
    """Short marks on the axes at values to read off, and their numbers beside them: layers and labels."""
    x0, y0 = diagram.crossing
    on_across = diagram.placement(chart_points(across, y0))                  # [ticks] Point-
    on_up = diagram.placement(chart_points(x0, up))                          # [ticks] Point-
    # Each mark reaches out from its axis: down from the one across, left from the one up.
    marks = concatenate([stack((on_across, on_across - mv.y.dual() * tick_length), axis=-1),
                         stack((on_up, on_up - mv.x.dual() * tick_length), axis=-1)])   # [ticks, 2] Point-
    labels = ([Label(chart_points(value, y0), f"{value:g}", ink, tick_size, "mt", (0, -6)) for value in across]
              + [Label(chart_points(x0, value), f"{value:g}", ink, tick_size, "rm", (-6, 0)) for value in up])
    return [Strokes(marks, axis_width, ink, 1.0, SOLID, Cap.BUTT)], labels


def picture(diagram: Diagram, layers: list, labels: list[Label]) -> np.ndarray:
    """The diagram's axes under the layers, `planar`'s primitives composited in order, with the quantities' names at
    the ends of the axes and the labels written over them: RGB bytes `[rows, columns, 3]`."""
    image = planar.draw(diagram.window, axes(diagram) + layers, white)
    x0, y0 = diagram.crossing
    names = [Label(chart_points(diagram.across[1], y0), diagram.names[0], ink, text_size, "rt", (0, -4)),
             Label(chart_points(x0, diagram.up[1]), diagram.names[1], ink, text_size, "lt", (8, 6))]
    return written(diagram, image, names + labels)


def written(diagram: Diagram, image: np.ndarray, labels: list[Label]) -> np.ndarray:
    """An image of the diagram with the labels written over it."""
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for label in labels:
        right, up = pixel_places(diagram.placement(label.place)) + np.array(label.shift)
        colour = tuple(int(round(255 * channel)) for channel in label.colour)
        # A name with a subscript after an underscore: the subscript at the size of the ticks' numbers and lower, after
        # the rest, the two placed together by the anchor.
        base, _, subscript = label.text.partition("_")
        font, small = fonts[label.size], fonts[tick_size]
        whole = pen.textbbox((right, diagram.rows - up), base, font=font, anchor=label.anchor)
        shift = small.getlength(subscript) if subscript else 0.0
        horizontal = {"l": 0.0, "m": shift / 2, "r": shift}[label.anchor[0]]
        # A thin white halo keeps the text readable over the lines under it.
        pen.text((right - horizontal, diagram.rows - up), base, fill=colour, font=font, anchor=label.anchor,
                 stroke_width=halo, stroke_fill=(255, 255, 255))
        if subscript:
            pen.text((whole[2] - horizontal + 1, whole[3] + 2), subscript, fill=colour, font=small, anchor="lb",
                     stroke_width=halo, stroke_fill=(255, 255, 255))
    return np.asarray(canvas)


class Floating(NamedTuple):
    """A name that finds its own place: among candidate chart points of unit weight `[n]`, the one where the text,
    placed by its Pillow anchor, covers the least of what is drawn; in its colour and size."""
    places: Point
    text: str
    colour: np.ndarray
    size: int
    anchor: str


def along(lines: list[Point], every: int) -> Point:
    """Candidate places along polylines of chart points: every so many of their samples."""
    return concatenate([points[every // 2::every] for points in lines])


# Where each floating name was written the first time its diagram was drawn, so that it holds still across the frames
# of an animation.
places: dict = {}


def floated(diagram: Diagram, image: np.ndarray, names: list[Floating]) -> np.ndarray:
    """An image of the diagram with each floating name written where it covers the least ink, inside the plot and
    clear of the names written before it; at the place found for it the first time, after that."""
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for name in names:
        candidates = pixel_places(diagram.placement(name.places))           # [n, 2]
        key = (diagram, name.text, name.anchor, candidates.tobytes())
        if key in places:
            if places[key] is not None:
                colour = tuple(int(round(255 * channel)) for channel in name.colour)
                pen.text((places[key][0], diagram.rows - places[key][1]), name.text, fill=colour,
                         font=fonts[name.size], anchor=name.anchor, stroke_width=halo, stroke_fill=(255, 255, 255))
            continue
        inked = 255 - np.asarray(canvas).min(axis=-1).astype(float)                       # [rows, columns]
        best, cost = None, np.inf
        for right, up in candidates:
            box = pen.textbbox((right, diagram.rows - up), name.text, font=fonts[name.size], anchor=name.anchor)
            left, top, far, bottom = (int(round(value)) for value in box)
            left, top, far, bottom = left - halo, top - halo, far + halo, bottom + halo
            inside = (left >= left_margin and far <= diagram.columns - right_margin // 2 and top >= top_margin // 2
                      and bottom <= diagram.rows - bottom_margin)
            if inside and inked[top:bottom, left:far].sum() < cost:
                best, cost = (right, up), inked[top:bottom, left:far].sum()
        places[key] = best
        if best is not None:
            colour = tuple(int(round(255 * channel)) for channel in name.colour)
            pen.text((best[0], diagram.rows - best[1]), name.text, fill=colour, font=fonts[name.size],
                     anchor=name.anchor, stroke_width=halo, stroke_fill=(255, 255, 255))
    return np.asarray(canvas)


def beside(images: list[np.ndarray]) -> np.ndarray:
    """Pictures side by side, each on white about the middle of the tallest's height."""
    rows = max(image.shape[0] for image in images)
    padded = [np.pad(image, (((rows - image.shape[0]) // 2, rows - image.shape[0] - (rows - image.shape[0]) // 2),
                             (0, 0), (0, 0)), constant_values=255) for image in images]
    return np.concatenate(padded, axis=1)


def figure(image: np.ndarray):
    """A still of an image at its own pixels."""
    return image_figure(image)


# --- the charts of a gas ---------------------------------------------------------------
class GasPoints(NamedTuple):
    """A gas's states on its two charts: entropy across and temperature up, and volume across and pressure up."""
    entropy_temperature: Point
    volume_pressure: Point


class GasFamily(NamedTuple):
    """A family of a gas's lines on its two charts, each a list of polylines of chart points `[samples]`."""
    entropy_temperature: list
    volume_pressure: list


class GasLines(NamedTuple):
    """A gas's lines of equal volume, pressure, temperature and entropy, each family a `GasFamily`; or, as the levels
    asked for, numbers."""
    isochores: GasFamily
    isobars: GasFamily
    isotherms: GasFamily
    adiabats: GasFamily


def gas_points(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor],
               states: Extensor) -> GasPoints:
    """A gas's states, antivectors of the plane of entropy and volume `[...]`, on its two charts: entropy and volume
    read by pairing, temperature and pressure by the gas's functions."""
    blades = states.context.multivector
    return GasPoints(chart_points((blades.s & states).to_array(), temperature_of(states).to_array()),
                     chart_points((blades.v & states).to_array(), pressure_of(states).to_array()))


def gas_lines(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor], blades,
              entropies: np.ndarray, volumes: np.ndarray, levels: GasLines) -> GasLines:
    """A gas's lines at the levels given, each level a number: lines of equal volume and of equal entropy run straight
    across the plane of states sampled at the entropies and volumes; lines of equal pressure and of equal temperature
    are level lines of the gas's functions over it."""
    states = lambda entropy, volume: blades.s.dual() * entropy + blades.v.dual() * volume
    charted = lambda lines: GasFamily([points.entropy_temperature for points in lines],
                                      [points.volume_pressure for points in lines])
    points = lambda entropy, volume: gas_points(temperature_of, pressure_of, states(entropy, volume))
    grid = states(entropies[None, :], volumes[:, None])                                   # [volumes, entropies] State-
    temperatures, pressures = temperature_of(grid).to_array(), pressure_of(grid).to_array()
    return GasLines(charted([points(entropies, level) for level in levels.isochores]),
                    charted([points(*line.T) for _, line in level_lines(entropies, volumes, pressures,
                                                                       np.asarray(levels.isobars))]),
                    charted([points(*line.T) for _, line in level_lines(entropies, volumes, temperatures,
                                                                       np.asarray(levels.isotherms))]),
                    charted([points(level, volumes) for level in levels.adiabats]))


# --- apparatus -------------------------------------------------------------------------
# A unit up, the direction a piston and what rests on it are carried along by the fill under it.
rise = solids.mv.direction([0, 0, 1])                                        # [] Point-


def box(centre: solids.Point, half: np.ndarray) -> solids.Plane:
    """A box about a centre, of half extents along x, y and z, as the six planes bounding it `[1, 6]`."""
    return solids.boxes(solids.translator(centre - solids.origin), np.asarray(half))[None]


# The thermal ramp, from icy blue for the coldest shown, through a pale middle, to red heat for the hottest.
thermal = np.stack([rgb(colour) for colour in ("#1f6fc5", "#8fd0f7", "#f2eee8", "#f6a04d", "#d01f17")])


def thermal_colour(fraction: float) -> np.ndarray:
    """The thermal ramp's colour a share of the way from the coldest shown to the hottest."""
    place = np.clip(fraction, 0.0, 1.0) * (len(thermal) - 1)
    low = min(int(place), len(thermal) - 2)
    return thermal[low] * (1 - (place - low)) + thermal[low + 1] * (place - low)


def relaxing(gap: float, widest: float, decades: float) -> float:
    """The share along the thermal ramp of a temperature a gap above a reference, positive or negative, which the
    widest gap shown puts at the ramp's ends: half by the gap itself, half by its logarithm over as many decades as
    shown, so that a gap closing exponentially keeps moving along the ramp to the end without its start crowding
    together."""
    share = abs(gap) / widest
    depth = 0.5 * np.clip(share, 0.0, 1.0) + 0.5 * np.clip(np.log10(share + 1e-300) / decades + 1, 0.0, 1.0)
    return 0.5 + 0.5 * np.sign(gap) * depth


# The cylinder's inner half width and depth, its wall thickness, and the piston's thickness.
bore, bore_depth, wall, plate = 0.52, 0.43, 0.055, 0.12
# The piston resting on an empty cylinder, and on it the weight that holds the pressure or the latch that holds the
# volume.
resting_piston = box(solids.mv.point([0, 0, plate / 2]), np.array([bore + 0.06, bore_depth + 0.05, plate / 2]))
resting_weight = box(solids.mv.point([0, 0, plate + 0.14]), np.array([0.31, 0.32, 0.14]))
resting_latch = box(solids.mv.point([bore + 0.12, 0, plate / 2]), np.array([0.14, bore_depth + 0.13, 0.05]))


def cylinder(height: float, maximum: float) -> list:
    """A cylinder's base and walls, tall enough for a fill of `maximum`, and its piston resting on a fill of
    `height`, as parts with their colours."""
    centres = solids.mv.point([[-bore - wall, 0, maximum / 2], [bore + wall, 0, maximum / 2]])
    walls = solids.boxes(solids.translator(centres - solids.origin), np.array([wall, bore_depth + 0.07, maximum / 2]))
    base = box(solids.mv.point([0, 0, -0.08]), np.array([bore + 2 * wall + 0.06, bore_depth + 0.17, 0.08]))
    return [(base, grey), (walls, grey), (solids.translator(rise * height) >> resting_piston, grey * 0.65)]


def load(height: float) -> list:
    """A weight on a piston resting on a fill of `height`: it holds the pressure under the piston fixed."""
    return [(solids.translator(rise * height) >> resting_weight, ink)]


def latch(height: float) -> list:
    """A latch across the piston resting on a fill of `height`: it holds the volume under the piston fixed."""
    return [(solids.translator(rise * height) >> resting_latch, orange)]


def fill(bottom: float, top: float) -> solids.Plane:
    """The contents of the cylinder between two heights, the six planes bounding them `[1, 6]`: the bore's four
    walls, and the planes at the two heights, at least a sliver apart."""
    x, y, z, w = solids.mv.x, solids.mv.y, solids.mv.z, solids.mv.w
    return stack((x - w * bore, -x - w * bore, y - w * bore_depth, -y - w * bore_depth,
                  z - w * max(top, bottom + 1e-6), w * bottom - z))[None]


def cylinder_view(maximum: float, pixels: int) -> solids.View:
    """The view of a cylinder tall enough for a fill of `maximum`, its piston and a weight on it, `pixels` across."""
    corners = solids.mv.point(np.array([[x, y, z] for x in (-0.75, 0.75) for y in (-0.6, 0.6)
                                        for z in (-0.16, maximum + 0.45)]))
    return solids.view([corners], 0.0, np.radians(18), np.radians(-25), 0.08, pixels)


def gas_cylinder(height: float, warmth: float, latched: bool, maximum: float, pixels: int) -> np.ndarray:
    """A cylinder of gas filled to a height, its colour by how warm it is, a share of the way from cold to hot, under
    a piston either latched or loaded with a weight; framed for a fill of `maximum`, `pixels` across."""
    holder = latch(height) if latched else load(height)
    return solids.draw(cylinder(height, maximum) + holder, cylinder_view(maximum, pixels), white, inks=[],
                       translucent=[(fill(0.0, height), thermal_colour(warmth), 0.55)], meshes=[])
