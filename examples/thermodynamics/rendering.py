"""Drawing shared by the thermodynamics examples: diagrams of one quantity against another, drawn on the GPU under a
camera with their names written over them, and the apparatus, a cylinder under a piston traced by `solids`.

A diagram reads its two quantities off the points it shows by two planes, `across & point` across and `up & point` up,
each quantity at its own scale. Where the two are readings of a state, the planes are the state algebra's own and the
states are drawn as they are; where one is a function of the state, the diagram is a chart, a projective plane whose
points carry the two quantities, `mv.x & point` across and `mv.y & point` up. Curves are polylines through the points,
areas are meshes of triangles, the level lines of a function are drawn where it crosses its levels across a grid of
the diagram's points, and the conics of a chart are drawn where their forms vanish; the picture's points are read as
pixels only where Pillow writes text.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import NamedTuple

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont

from numga import Extensor, NumpyContext, concatenate, stack
from numga.algebras import PGA2D
from examples.util import charts
from examples.util.wgsl import contours, image_figure, simplices, solids
from examples.util.wgsl.camera import Camera, read, weighing
from examples.util.wgsl.simplices import Cap, Topology

context = NumpyContext(PGA2D)
mv = context.multivector
Point = PGA2D.gatype.antivector()                                            # Antivector-
Plane = PGA2D.gatype.vector()                                                # Vector+
Scalar = PGA2D.gatype.scalar()                                               # Scalar+
Quadric = PGA2D.gatype((Plane, Point))                                       # Plane+ <- Point-
# A solid stroke: one dash, no gap.
SOLID = (1.0, 0.0)


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
# The axes' width and the length of the strokes of the chevrons at their ends, in pixels: the axes as wide as a chart's
# frame; and how far a tick mark reaches out from its axis.
axis_width, axis_chevron, tick_length = charts.FRAME_PX, 7.0, 4.0
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
# The radius of a dot marking a state, and of the white ring around it, in pixels; the length of the strokes of the
# chevron at an arrow's tip, in widths of its stroke.
dot, ring = 4.0, 5.5
chevron = 4.0
# How far outside a plot a point may read and still count as inside it, in pixels: the rounding of its readings.
slack = 1e-6


# --- diagrams --------------------------------------------------------------------------
def on_numpy(values: Extensor) -> Extensor:
    """Values on the NumPy backend, in their own algebra and of their own type. A diagram draws there, whatever backend
    its values come from: a path a JAX computation gives, whose growing slices an animation draws, then compiles
    nothing a frame."""
    return NumpyContext(values.algebra).include_from_algebra(values)


def chart_points(across: Scalar, up: Scalar) -> Point:
    """The points of a chart's plane at values of its two quantities `[...]`: the origin moved by each value along its
    quantity's direction. Every value a chart shows enters its plane here."""
    return mv.w.dual() + mv.x.dual() * across + mv.y.dual() * up


def chart_directions(across: Scalar, up: Scalar) -> Point:
    """The directions of a chart's plane, points at infinity, along changes of its two quantities `[...]`."""
    return mv.x.dual() * across + mv.y.dual() * up


@dataclass(frozen=True)
class Diagram:
    """A diagram of one quantity across against another up: the range of each shown, the values at which the axes
    cross, the picture's rows and columns of pixels, the names of the two quantities, and the planes that read them off
    the points it shows, a state algebra's where they are readings of a state and the chart's otherwise."""
    across: tuple[float, float]
    up: tuple[float, float]
    crossing: tuple[float, float]
    rows: int
    columns: int
    names: tuple[str, str]
    planes: tuple[Extensor, Extensor] = field(default=(mv.x, mv.y), compare=False)

    @cached_property
    def scales(self) -> np.ndarray:
        """Pixels a unit of each quantity, across and up `[2]`: each range across the plot's width or up its height."""
        width, height = self.columns - left_margin - right_margin, self.rows - bottom_margin - top_margin
        return np.array([width / (self.across[1] - self.across[0]), height / (self.up[1] - self.up[0])])

    def reading(self, across: Extensor, up: Extensor) -> Camera:
        """The camera of the picture reading points by the planes `across` and `up`, each at its quantity's scale, in
        pixels, its bottom left corner the starts of the two ranges less the margins."""
        across, up = on_numpy(across), on_numpy(up)
        weight, unit = weighing(across)
        half = np.array([self.columns, self.rows]) / 2
        corner = np.array([self.across[0], self.up[0]]) * self.scales - np.array([left_margin, bottom_margin])
        return Camera(across * self.scales[0], up * self.scales[1], 0 * across, weight, unit, corner + half, half,
                      self.columns)

    @cached_property
    def camera(self) -> Camera:
        """The camera of the picture reading the diagram's points by its own planes."""
        return self.reading(*self.planes)

    def at(self, across: Scalar, up: Scalar) -> Extensor:
        """The points of the diagram's algebra its planes read as values of its two quantities `[...]`: its origin, the
        point its weight plane reads as one, moved by each value along the direction its plane reads as one, the
        plane's complement over its reading of it."""
        planes = [on_numpy(plane) for plane in self.planes]
        weight, _ = weighing(planes[0])
        steps = [plane.dual() / read(plane, plane.dual()) for plane in planes]
        return weight.dual() + steps[0] * across + steps[1] * up

    def pixels(self, points: Extensor) -> np.ndarray:
        """The pixels right and up from the picture's bottom left corner at which points `[...]` lie, `[..., 2]`, read
        by the camera's planes over their weights: where Pillow writes text and where a polyline leaves the plot."""
        camera, points = self.camera, on_numpy(points)
        weights = read(camera.weight, points) + camera.unit                    # [...] Scalar+
        readings = stack((read(camera.across, points), read(camera.up, points)), axis=-1) / weights[..., None]
        return readings.to_array() - (camera.middle - camera.half)

    def lengths(self, directions: Extensor) -> Extensor:
        """The lengths in pixels of directions `[...]` across the picture: their readings across and up, which no
        weight divides."""
        directions = on_numpy(directions)
        across, up = read(self.camera.across, directions), read(self.camera.up, directions)
        return (across * across + up * up).square_root()

    def inside(self, points: Extensor) -> np.ndarray:
        """Whether points `[...]` lie inside the diagram's ranges, its plot."""
        right, up = np.moveaxis(self.pixels(points), -1, 0)
        return ((right >= left_margin - slack) & (right <= self.columns - right_margin + slack)
                & (up >= bottom_margin - slack) & (up <= self.rows - top_margin + slack))


def strokes(diagram: Diagram, lines: list[Extensor], width: float, colours: np.ndarray, opacity: float,
            dashes: tuple[float, float]) -> list[np.ndarray]:
    """Polylines through points of the diagram's algebra, each its own number of samples and its colour `[lines, 3]`,
    `width` pixels across, dashed `dashes` pixels on and off, `SOLID` for solid. Nothing where there are none."""
    if not lines:
        return []
    cap = Cap.ROUND if dashes == SOLID else Cap.BUTT
    lengths = [points.shape[0] for points in lines]
    # Each edge in its polyline's colour.
    colours = np.repeat(np.broadcast_to(colours, (len(lines), 3)), np.subtract(lengths, 1), axis=0)
    return [simplices.strokes(on_numpy(concatenate(lines)), Topology.chains(lengths), colours, width, diagram.camera,
                              dashes=dashes, cap=cap, opacity=opacity)]


def axes(diagram: Diagram) -> list[np.ndarray]:
    """The two axes, crossing where the diagram says, each ending in a chevron."""
    x0, y0 = diagram.crossing
    corner = diagram.at(x0, y0)
    ends = diagram.at(np.array([diagram.across[1], x0]), np.array([y0, diagram.up[1]]))   # [2] Point-
    return [simplices.strokes(on_numpy(stack((stack((corner, corner)), ends), axis=-1).reshape(-1)), Topology.chains([2, 2]),
                              ink, axis_width, diagram.camera, chevrons=np.ones(2), chevron=axis_chevron)]


def within(diagram: Diagram, lines: Extensor, width: float, colour: np.ndarray, opacity: float) -> list[np.ndarray]:
    """Polylines through points `[curves, samples]`, each in its colour `[curves, 3]`, cut to the runs of samples
    inside the diagram's ranges."""
    return polylines(diagram, [lines[curve] for curve in range(lines.shape[0])], width, colour, opacity, SOLID)


def polylines(diagram: Diagram, lines: list[Extensor], width: float, colour: np.ndarray, opacity: float,
              dashes: tuple[float, float]) -> list[np.ndarray]:
    """Polylines through points of the diagram's algebra, each its own number of samples and its colour, or all in one,
    cut to the runs of samples inside the diagram's ranges; dashed `dashes` pixels on and off, `SOLID` for solid."""
    runs, colours = [], np.broadcast_to(colour, (len(lines), 3))
    for points, run_colour in zip(lines, colours):
        inside = diagram.inside(points)
        # The runs of samples inside, between the steps into and out of the ranges.
        steps = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
        runs += [(points[start:end], run_colour) for start, end in zip(np.flatnonzero(steps == 1),
                                                                       np.flatnonzero(steps == -1)) if end - start > 1]
    return strokes(diagram, [run for run, _ in runs], width, np.array([colour for _, colour in runs]).reshape(-1, 3),
                   opacity, dashes)


def family(diagram: Diagram, lines: list[Extensor], name: str, width: float) -> list[np.ndarray]:
    """A family of lines, polylines of the diagram's points, in the family's colour."""
    return polylines(diagram, lines, width, families[name], 1.0, SOLID)


class Levels(NamedTuple):
    """A function's levels over a diagram: the grid of the diagram's points it is sampled at `[rows, columns]`, its
    values there, not finite where it is not known, and the levels."""
    grid: Extensor
    values: np.ndarray
    levels: np.ndarray


def plotted(diagram: Diagram, picture: np.ndarray) -> np.ndarray:
    """A picture of the diagram `[rows, columns, 4]` clear outside its plot, the pixels within its margins."""
    inside = np.zeros(picture.shape[:2] + (1,))
    inside[top_margin:diagram.rows - bottom_margin, left_margin:diagram.columns - right_margin] = 1.0
    return np.concatenate((picture[..., :3], picture[..., 3:] * inside), axis=-1)


def known(levels: Levels) -> tuple[Topology, np.ndarray]:
    """The grid's triangles where the function is known at every corner, and its values `[vertices]`, nothing where
    it is not known."""
    rows, columns = levels.values.shape
    grid = Topology.grid(rows, columns)
    values = levels.values.reshape(-1)
    triangles = grid.elements[2]
    return Topology(grid.n_vertices, triangles[np.isfinite(values)[triangles].all(axis=-1)]), np.nan_to_num(values)


def level_lines(diagram: Diagram, levels: Levels, colour: np.ndarray, width: float) -> list[np.ndarray]:
    """A function's level lines, `width` pixels wide in a colour, where it crosses each level across the grid's
    triangles, cut to the plot."""
    topology, values = known(levels)
    inks = tuple(contours.Level(level, np.append(colour, 1.0), width) for level in levels.levels)
    return [plotted(diagram, contours.picture(on_numpy(levels.grid).reshape(-1), topology, values, np.zeros(3),
                                              diagram.camera, levels=inks, opacity=0.0))]


def level_family(diagram: Diagram, levels: Levels, name: str, width: float) -> list[np.ndarray]:
    """A family of level lines in the family's colour."""
    return level_lines(diagram, levels, families[name], width)


def sides(diagram: Diagram, levels: Levels, inks: np.ndarray) -> np.ndarray:
    """The two sides of a function's one level, where it is known: where it lies below the level filled in the first
    ink, colour and opacity, and where above in the second `[2, 4]`, cut to the plot."""
    topology, values = known(levels)
    level, = levels.levels
    bands = contours.Bands(level - 1.0, 1.0, inks)
    return plotted(diagram, contours.picture(on_numpy(levels.grid).reshape(-1), topology, values, np.zeros(3),
                                             diagram.camera, bands=(bands,), opacity=0.0))


def crossings(levels: Levels, every: int) -> Extensor:
    """Places on a function's level lines, every so many of them: where it crosses each level between neighbouring
    points of its grid along a row or a column, both known, the share of the way between them its values there give."""
    places = []
    for tails, heads, at_tails, at_heads in [(levels.grid[:, :-1], levels.grid[:, 1:], levels.values[:, :-1],
                                              levels.values[:, 1:]),
                                             (levels.grid[:-1], levels.grid[1:], levels.values[:-1], levels.values[1:])]:
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = (levels.levels - at_tails[..., None]) / (at_heads - at_tails)[..., None]   # [..., levels]
        rows, columns, level = np.nonzero((shares >= 0) & (shares < 1))
        starts, steps = tails[rows, columns], heads[rows, columns] - tails[rows, columns]       # [places] Point-, Point-
        places.append(starts + steps * shares[rows, columns, level])
    return concatenate(places)[every // 2::every]


def furthest(levels: Levels, plane: Extensor) -> list[tuple[float, Extensor]]:
    """Each level a function crosses on its grid, with the place on its level line that a plane of the diagram's
    algebra reads highest."""
    found = [(level, crossings(levels._replace(levels=np.array([level])), 1)) for level in levels.levels]
    return [(level, places[(plane & places).argmax()]) for level, places in found if places.shape[0] > 0]


def tangent_lines(diagram: Diagram, points: Extensor, directions: Extensor, across: np.ndarray) -> Extensor:
    """Lines of the diagram through points along directions `[...]`, sampled where its quantity across takes the values
    `[samples]`, `[..., samples]`: each point moved along its direction by the change of that quantity over the
    direction's, as the diagram's plane across reads them."""
    plane, points, directions = on_numpy(diagram.planes[0]), on_numpy(points), on_numpy(directions)
    weight, unit = weighing(plane)
    # How far each point lies across, its reading over its weight, and how far its direction carries it.
    weights = read(weight, points) + unit                                      # [...] Scalar+
    steps = (weights[..., None] * across - read(plane, points)[..., None]) / read(plane, directions)[..., None]
    return points[..., None] + directions[..., None] * steps


def arrows(diagram: Diagram, starts: Extensor, toward: Extensor, length: float, width: float,
           colour: np.ndarray) -> list[np.ndarray]:
    """Arrows `length` pixels long from points of the diagram `[n]` toward others, each a stroke with a chevron at its
    tip."""
    starts, toward = on_numpy(starts), on_numpy(toward)
    along = toward - starts                                                  # [n] Point-
    tips = starts + along / diagram.lengths(along) * length                  # [n] Point-
    n_arrows = starts.shape[0]
    return [simplices.strokes(stack((starts, tips), axis=-1).reshape(-1), Topology.chains([2] * n_arrows), colour, width,
                              diagram.camera, chevrons=np.ones(n_arrows), chevron=chevron * width)]


def dashed(diagram: Diagram, points: Extensor, width: float, colour: np.ndarray,
           dashes: tuple[float, float]) -> list[np.ndarray]:
    """Dashed polylines through points `[..., samples]`, dashes and gaps in pixels."""
    rows = points.reshape(-1, points.shape[-1])
    return strokes(diagram, [rows[row] for row in range(rows.shape[0])], width, colour, 1.0, dashes)


def area(diagram: Diagram, points: Extensor, topology: Topology, colour: np.ndarray, opacity: float) -> np.ndarray:
    """The area a mesh of triangles over points of the diagram `[vertices]` covers, in one colour."""
    return contours.picture(on_numpy(points), topology, np.zeros(points.shape[0]), colour, diagram.camera, opacity=opacity)


def between(diagram: Diagram, curve: Extensor, feet: Extensor, colour: np.ndarray, opacity: float) -> np.ndarray:
    """The area between a curve of the diagram's points `[samples]` and the points it is dropped onto `[samples]`, such
    as its feet on an axis: a strip of triangles, two between each pair of samples and their feet."""
    return area(diagram, concatenate((curve, feet)), Topology.grid(2, curve.shape[0]), colour, opacity)


def dots(diagram: Diagram, points: Extensor, colour: np.ndarray) -> list[np.ndarray]:
    """Dots at points of the diagram `[n]`, each in a white ring."""
    points = on_numpy(points)
    return [simplices.markers(points, white, ring, diagram.camera),
            simplices.markers(points, colour, dot, diagram.camera)]


class Label(NamedTuple):
    """A name written at a point of a diagram, in its colour and size, placed by a Pillow anchor (`"lb"` puts the
    text's left bottom there) and shifted a few pixels right and up."""
    place: Extensor
    text: str
    colour: np.ndarray
    size: int
    anchor: str
    shift: tuple[float, float]


def ticks(diagram: Diagram, across: np.ndarray, up: np.ndarray) -> tuple[list, list]:
    """Short marks on the axes at values to read off, and their numbers beside them: pictures and labels."""
    x0, y0 = diagram.crossing
    reach = tick_length / diagram.scales                                     # [2]
    # Each mark reaches out from its axis: down from the one across, left from the one up.
    marks = ([stack((diagram.at(value, y0), diagram.at(value, y0 - reach[1]))) for value in across]
             + [stack((diagram.at(x0, value), diagram.at(x0 - reach[0], value))) for value in up])
    labels = ([Label(diagram.at(value, y0), f"{value:g}", ink, tick_size, "mt", (0, -6)) for value in across]
              + [Label(diagram.at(x0, value), f"{value:g}", ink, tick_size, "rm", (-6, 0)) for value in up])
    return strokes(diagram, marks, axis_width, ink, 1.0, SOLID), labels


def picture(diagram: Diagram, pictures: list[np.ndarray], labels: list[Label]) -> np.ndarray:
    """The diagram's axes under the pictures, laid over each other in order over white, with the quantities' names at
    the ends of the axes and the labels written over them: RGB bytes `[rows, columns, 3]`."""
    laid = simplices.composited(axes(diagram) + pictures)
    seen = white * (1 - laid[..., 3:]) + laid[..., :3] * laid[..., 3:]
    image = np.round(np.clip(seen, 0, 1) * 255).astype(np.uint8)
    x0, y0 = diagram.crossing
    names = [Label(diagram.at(diagram.across[1], y0), diagram.names[0], ink, text_size, "rt", (0, -4)),
             Label(diagram.at(x0, diagram.up[1]), diagram.names[1], ink, text_size, "lt", (8, 6))]
    return written(diagram, image, names + labels)


def written(diagram: Diagram, image: np.ndarray, labels: list[Label]) -> np.ndarray:
    """An image of the diagram with the labels written over it."""
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for label in labels:
        right, up = diagram.pixels(label.place) + np.array(label.shift)
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
    """A name that finds its own place: among candidate points of a diagram `[n]`, the one where the text, placed by
    its Pillow anchor, covers the least of what is drawn; in its colour and size."""
    places: Extensor
    text: str
    colour: np.ndarray
    size: int
    anchor: str


def along(lines: list[Extensor], every: int) -> Extensor:
    """Candidate places along polylines of a diagram's points: every so many of their samples."""
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
        candidates = diagram.pixels(name.places)                             # [n, 2]
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


class GasLevels(NamedTuple):
    """A family of a gas's level lines on its two charts, each the `Levels` of a function over the plane of states
    carried onto the chart."""
    entropy_temperature: Levels
    volume_pressure: Levels


class GasLines(NamedTuple):
    """A gas's lines of equal volume, pressure, temperature and entropy, the straight families a `GasFamily` and the
    level lines a `GasLevels`; or, as the levels asked for, numbers."""
    isochores: GasFamily
    isobars: GasLevels
    isotherms: GasLevels
    adiabats: GasFamily


def gas_points(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor],
               states: Extensor) -> GasPoints:
    """A gas's states, antivectors of the plane of entropy and volume `[...]`, on its two charts: entropy and volume
    read by pairing, temperature and pressure by the gas's functions."""
    blades = states.context.multivector
    return GasPoints(chart_points(context.include_from_algebra(blades.s & states), context.include_from_algebra(temperature_of(states))),
                     chart_points(context.include_from_algebra(blades.v & states), context.include_from_algebra(pressure_of(states))))


def gas_lines(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor], blades,
              entropies: np.ndarray, volumes: np.ndarray, levels: GasLines) -> GasLines:
    """A gas's lines at the levels given, each level a number: lines of equal volume and of equal entropy run straight
    across the plane of states sampled at the entropies and volumes; lines of equal pressure and of equal temperature
    are level lines of the gas's functions over it, the grid of its states carried onto either chart."""
    states = lambda entropy, volume: blades.s.dual() * entropy + blades.v.dual() * volume
    charted = lambda lines: GasFamily([points.entropy_temperature for points in lines],
                                      [points.volume_pressure for points in lines])
    points = lambda entropy, volume: gas_points(temperature_of, pressure_of, states(entropy, volume))
    grid = states(entropies[None, :], volumes[:, None])                                   # [volumes, entropies] State-
    temperatures, pressures = np.asarray(temperature_of(grid).to_array()), np.asarray(pressure_of(grid).to_array())
    on_charts = gas_points(temperature_of, pressure_of, grid)                     # [volumes, entropies] Point-, Point-
    leveled = lambda values, asked: GasLevels(Levels(on_charts.entropy_temperature, values, np.asarray(asked)),
                                              Levels(on_charts.volume_pressure, values, np.asarray(asked)))
    return GasLines(charted([points(entropies, level) for level in levels.isochores]),
                    leveled(pressures, levels.isobars),
                    leveled(temperatures, levels.isotherms),
                    charted([points(level, volumes) for level in levels.adiabats]))


# --- apparatus -------------------------------------------------------------------------
# A unit up, the direction a piston and what rests on it are carried along by the fill under it.
rise = solids.mv.direction([0, 0, 1])                                        # [] Point-


def box(centre: solids.Point, half: solids.Scalar) -> solids.Plane:
    """A box about a centre, of half extents along x, y and z, as the six planes bounding it `[1, 6]`."""
    return solids.boxes(solids.translator(centre - solids.origin), half)[None]


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


def cylinder(height: solids.Scalar, maximum: solids.Scalar) -> list:
    """A cylinder's base and walls, tall enough for a fill of `maximum`, and its piston resting on a fill of
    `height`, as parts with their colours."""
    centres = solids.mv.point([[-bore - wall, 0, 0], [bore + wall, 0, 0]]) + rise * (maximum / 2)
    half = stack((solids.mv.scalar([wall]), solids.mv.scalar([bore_depth + 0.07]), maximum / 2))
    walls = solids.boxes(solids.translator(centres - solids.origin), half)
    base = box(solids.mv.point([0, 0, -0.08]), np.array([bore + 2 * wall + 0.06, bore_depth + 0.17, 0.08]))
    return [(base, grey), (walls, grey), (solids.translator(rise * height) >> resting_piston, grey * 0.65)]


def load(height: solids.Scalar) -> list:
    """A weight on a piston resting on a fill of `height`: it holds the pressure under the piston fixed."""
    return [(solids.translator(rise * height) >> resting_weight, ink)]


def latch(height: solids.Scalar) -> list:
    """A latch across the piston resting on a fill of `height`: it holds the volume under the piston fixed."""
    return [(solids.translator(rise * height) >> resting_latch, orange)]


def fill(bottom: solids.Scalar, top: solids.Scalar) -> solids.Plane:
    """The contents of the cylinder between two heights, the six planes bounding them `[1, 6]`: the bore's four
    walls, and the planes at the two heights, at least a sliver apart."""
    x, y, z, w = solids.mv.x, solids.mv.y, solids.mv.z, solids.mv.w
    return stack((x - w * bore, -x - w * bore, y - w * bore_depth, -y - w * bore_depth,
                  z - w * (bottom + (top - bottom).clip(1e-6, np.inf)), w * bottom - z))[None]


def cylinder_view(maximum: solids.Scalar, pixels: int) -> solids.View:
    """The view of a cylinder tall enough for a fill of `maximum`, its piston and a weight on it, `pixels` across."""
    base = solids.mv.point([[x, y, -0.16] for x in (-0.75, 0.75) for y in (-0.6, 0.6)])
    corners = stack((base, base + rise * (maximum + 0.61))).reshape(-1)
    return solids.view([corners], 0.0, np.radians(18), np.radians(-25), 0.08, pixels)


def gas_cylinder(height: solids.Scalar, warmth: float, latched: bool, maximum: solids.Scalar, pixels: int) -> np.ndarray:
    """A cylinder of gas filled to a height, its colour by how warm it is, a share of the way from cold to hot, under
    a piston either latched or loaded with a weight; framed for a fill of `maximum`, `pixels` across."""
    holder = latch(height) if latched else load(height)
    return solids.draw(cylinder(height, maximum) + holder, cylinder_view(maximum, pixels), white, inks=[],
                       translucent=[(fill(0.0, height), thermal_colour(warmth), 0.55)], meshes=[])
