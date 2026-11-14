"""Shared GPU diagram and piston-cylinder apparatus rendering for the thermodynamics examples.

A `Diagram` reads horizontal and vertical quantities from points through two complementary planes,
`across & point` and `up & point`, each scaled to its plot range. When both quantities are direct
readings of a thermodynamic state, `Diagram` uses the state algebra's own planes directly; when
either axis plots a derived potential such as temperature or pressure, states map onto a
two-dimensional projective chart plane whose `x` and `y` planes read the two plotted coordinates.

Curves, shaded strips, contour lines, and filled phase regions render on the GPU through
`numga.render`, converting coordinates to pixels only where Pillow draws axis names and floating
curve labels. At the bottom of the module, `gas_cylinder` ray-traces a three-dimensional cylinder,
translucent gas fill, piston, weight, and volume latch through `numga.render.solids`. Companion
notes on reading states through complementary planes are in `docs/orientation.md`.
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
from numga.topology import Topology
from numga.render import Camera, Cap, contours, drawing, read, simplices, solids
from examples.util.panels import image_figure

context = NumpyContext(PGA2D)
mv = context.multivector
Point = PGA2D.gatype.antivector()                                            # Antivector-
Plane = PGA2D.gatype.vector()                                                # Vector+
Scalar = PGA2D.gatype.scalar()                                               # Scalar+
Quadric = PGA2D.gatype((Plane, Point))                                       # Plane+ <- Point-
# Continuous stroke pattern with unit dash length and zero gap.
SOLID = (1.0, 0.0)


def rgb(colour: str) -> np.ndarray:
    return np.array(to_rgb(colour))


blue, orange, grey, ink, green = rgb("#3196b4"), rgb("#de9439"), rgb("#aab9c1"), rgb("#485961"), rgb("#458875")
# Neutral reference-grid stroke colour and opaque white canvas background.
faint = rgb("#cfd9dc")
white = np.ones(3)
# Fixed palette per thermodynamic curve family, with black reserved for phase boundaries.
families = dict(temperature=rgb("#d9604c"), pressure=rgb("#8a68b8"), volume=rgb("#5aa260"), entropy=rgb("#3f9e96"),
                humidity=rgb("#4a82bd"), enthalpy=rgb("#a8763f"), level=rgb("#8a9ea6"))
boundary = rgb("#1c2326")

# Pixel margins around the plot box for left and bottom axis labels and right and top breathing room.
left_margin, bottom_margin, right_margin, top_margin = 30, 24, 12, 16
# Axis stroke width, terminal chevron arm length, and outward tick-mark length in pixels.
axis_width, axis_chevron, tick_length = charts.FRAME_PX, 7.0, 4.0
# Reference-grid, curve, and highlighted-trajectory stroke widths matched to `charts`.
thin, medium, bold = charts.FRAME_PX, charts.LINE_PX, 1.5 * charts.LINE_PX
# Label and tick-number pixel heights matched to `charts`.
text_size, tick_size = charts.TEXT_PX, charts.TICK_PX
fonts = {size: ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", style="italic")), size)
         for size in (text_size, tick_size)}
# White text-outline thickness in pixels so labels remain legible over grid curves.
halo = 2
# State-marker dot and white-ring radii in pixels, and arrow chevron length in stroke widths.
dot, ring = 4.0, 5.5
chevron = 4.0
# Floating-point pixel tolerance when testing whether projected samples lie inside the plot box.
slack = 1e-6


# --- diagrams --------------------------------------------------------------------------
def on_numpy(values: Extensor) -> Extensor:
    """Re-hosting `values` onto `NumpyContext` in the same algebra and `GAType` lets a diagram draw
    slices of JAX trajectories frame by frame without triggering per-frame XLA compilation."""
    return NumpyContext(values.algebra).include_from_algebra(values)


def chart_points(across: Scalar, up: Scalar) -> Point:
    """Unit-weight points (`[...] Point-`) in the 2D projective chart plane offset the origin by
    `across` along `x` and `up` along `y`."""
    return mv.w.dual() + mv.x.dual() * across + mv.y.dual() * up


def chart_directions(across: Scalar, up: Scalar) -> Point:
    """Ideal directions (`[...] Point-`) in the 2D projective chart plane combine steps `across`
    along `x` and `up` along `y` with zero weight."""
    return mv.x.dual() * across + mv.y.dual() * up


@dataclass(frozen=True)
class Diagram:
    """Two-axis thermodynamic plot specification pairing value ranges `(across, up)`, axis
    intersection `crossing`, canvas size `(rows, columns)`, axis `names`, reading `planes`, and
    point `weighing` `(weight_plane, unit_offset)`."""
    across: tuple[float, float]
    up: tuple[float, float]
    crossing: tuple[float, float]
    rows: int
    columns: int
    names: tuple[str, str]
    planes: tuple[Extensor, Extensor] = field(compare=False)
    weighing: tuple[Extensor, float] = field(compare=False)

    @cached_property
    def scales(self) -> np.ndarray:
        """Horizontal and vertical pixel-per-unit scale factors (`[2]`) across the interior plot
        box."""
        width, height = self.columns - left_margin - right_margin, self.rows - bottom_margin - top_margin
        return np.array([width / (self.across[1] - self.across[0]), height / (self.up[1] - self.up[0])])

    def reading(self, across: Extensor, up: Extensor, weight: Extensor, unit: float) -> Camera:
        """Scales `across` and `up` into pixel readings divided by `weight & point + unit` and
        offsets the `Camera` frame by `(left_margin, bottom_margin)` around the plot ranges."""
        across, up, weight = on_numpy(across), on_numpy(up), on_numpy(weight)
        half = np.array([self.columns, self.rows]) / 2
        corner = np.array([self.across[0], self.up[0]]) * self.scales - np.array([left_margin, bottom_margin])
        return Camera(across * self.scales[0], up * self.scales[1], 0 * across, weight, unit, corner + half, half,
                      self.columns)

    @cached_property
    def camera(self) -> Camera:
        """Pixel-space `Camera` configured from `self.planes` and `self.weighing`."""
        return self.reading(*self.planes, *self.weighing)

    def at(self, across: Scalar, up: Scalar) -> Extensor:
        """Offsets `weight.dual()` along the dual directions of `self.planes` to construct points
        (`[...]`) whose horizontal and vertical readings equal `across` and `up`."""
        planes = [on_numpy(plane) for plane in self.planes]
        weight, _ = self.weighing
        steps = [plane.dual() / read(plane, plane.dual()) for plane in planes]
        return on_numpy(weight).dual() + steps[0] * across + steps[1] * up

    def pixels(self, points: Extensor) -> np.ndarray:
        """Projects `points` (`[...]`) through `self.camera` into `[..., 2]` pixel coordinates
        measured right and up from the canvas's bottom-left corner."""
        camera, points = self.camera, on_numpy(points)
        weights = read(camera.weight, points) + camera.unit                    # [...] Scalar+
        readings = stack((read(camera.across, points), read(camera.up, points)), axis=-1) / weights[..., None]
        return readings.to_array() - (camera.middle - camera.half)

    def lengths(self, directions: Extensor) -> Extensor:
        """Euclidean pixel lengths (`[...] Scalar+`) of unweighted `directions` (`[...]`) across
        `self.camera`."""
        directions = on_numpy(directions)
        across, up = read(self.camera.across, directions), read(self.camera.up, directions)
        return (across * across + up * up).square_root()

    def inside(self, points: Extensor) -> np.ndarray:
        """Boolean mask (`[...]`) marking where `points` project within the interior plot box up to
        `slack`."""
        right, up = np.moveaxis(self.pixels(points), -1, 0)
        return ((right >= left_margin - slack) & (right <= self.columns - right_margin + slack)
                & (up >= bottom_margin - slack) & (up <= self.rows - top_margin + slack))


def strokes(diagram: Diagram, lines: list[Extensor], width: float, colours: np.ndarray, opacity: float,
            dashes: tuple[float, float]) -> list[np.ndarray]:
    """Renders a list of variable-length point chains `lines` as a single RGBA stroke layer of pixel
    `width`, per-line RGB `colours` (`[n_lines, 3]`), and `(on, off)` `dashes` (`SOLID` for solid),
    returning an empty list when `lines` is empty."""
    if not lines:
        return []
    cap = Cap.ROUND if dashes == SOLID else Cap.BUTT
    lengths = [points.shape[0] for points in lines]
    # Broadcast each polyline's colour across its constituent edges.
    colours = np.repeat(np.broadcast_to(colours, (len(lines), 3)), np.subtract(lengths, 1), axis=0)
    return [simplices.strokes(on_numpy(concatenate(lines)), Topology.chains(lengths), colours, width, diagram.camera,
                              dashes=dashes, cap=cap, opacity=opacity)]


def axes(diagram: Diagram) -> list[np.ndarray]:
    """Draws the horizontal and vertical coordinate axes from `diagram.crossing` to the upper ends
    of both ranges, each tipped with a chevron."""
    x0, y0 = diagram.crossing
    corner = diagram.at(x0, y0)
    ends = diagram.at(np.array([diagram.across[1], x0]), np.array([y0, diagram.up[1]]))   # [2] Point-
    return [simplices.strokes(on_numpy(stack((stack((corner, corner)), ends), axis=-1).reshape(-1)), Topology.chains([2, 2]),
                              ink, axis_width, diagram.camera, chevrons=np.ones(2), chevron=axis_chevron)]


def within(diagram: Diagram, lines: Extensor, width: float, colour: np.ndarray, opacity: float) -> list[np.ndarray]:
    """Clips batched curves `lines` (`[n_curves, n_samples]`) to contiguous runs inside `diagram`
    and renders them as solid strokes."""
    return polylines(diagram, [lines[curve] for curve in range(lines.shape[0])], width, colour, opacity, SOLID)


def polylines(diagram: Diagram, lines: list[Extensor], width: float, colour: np.ndarray, opacity: float,
              dashes: tuple[float, float]) -> list[np.ndarray]:
    """Clips each point chain in `lines` to contiguous sample runs inside `diagram`'s plot box and
    renders the visible runs with `strokes`."""
    runs, colours = [], np.broadcast_to(colour, (len(lines), 3))
    for points, run_colour in zip(lines, colours):
        inside = diagram.inside(points)
        # Split each polyline at transitions into and out of the interior plot box.
        steps = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
        runs += [(points[start:end], run_colour) for start, end in zip(np.flatnonzero(steps == 1),
                                                                       np.flatnonzero(steps == -1)) if end - start > 1]
    return strokes(diagram, [run for run, _ in runs], width, np.array([colour for _, colour in runs]).reshape(-1, 3),
                   opacity, dashes)


def family(diagram: Diagram, lines: list[Extensor], name: str, width: float) -> list[np.ndarray]:
    """Renders `lines` as solid clipped polylines in the palette colour `families[name]`."""
    return polylines(diagram, lines, width, families[name], 1.0, SOLID)


class Levels(NamedTuple):
    """Sampled scalar field `values` (`[n_rows, n_columns]`, non-finite outside the valid domain) on
    diagram points `grid` (`[n_rows, n_columns]`) paired with target contour `levels`."""
    grid: Extensor
    values: np.ndarray
    levels: np.ndarray


def plotted(diagram: Diagram, picture: np.ndarray) -> np.ndarray:
    """Masks the alpha channel of `picture` (`[rows, columns, 4]`) to zero outside `diagram`'s
    interior plot margins."""
    inside = np.zeros(picture.shape[:2] + (1,))
    inside[top_margin:diagram.rows - bottom_margin, left_margin:diagram.columns - right_margin] = 1.0
    return np.concatenate((picture[..., :3], picture[..., 3:] * inside), axis=-1)


def known(levels: Levels) -> tuple[Topology, np.ndarray]:
    """Restricts `Topology.grid` over `levels.values` to triangles whose three corners all hold
    finite values, replacing non-finite entries in the flattened value array with zero."""
    rows, columns = levels.values.shape
    grid = Topology.grid(rows, columns)
    values = levels.values.reshape(-1)
    triangles = grid.elements[2]
    return grid.subset(np.isfinite(values)[triangles].all(axis=-1), -1), np.nan_to_num(values)


def level_lines(diagram: Diagram, levels: Levels, colour: np.ndarray, width: float) -> list[np.ndarray]:
    """Draws iso-level contours of `levels` in `colour` at stroke `width`, clipped to `diagram`'s
    interior plot box."""
    topology, values = known(levels)
    inks = tuple(contours.Level(level, np.append(colour, 1.0), width) for level in levels.levels)
    return [plotted(diagram, drawing.composited([
        contours.picture(on_numpy(levels.grid).reshape(-1), topology, values, np.zeros(3), diagram.camera,
                         levels=inks, opacity=0.0)]))]


def level_family(diagram: Diagram, levels: Levels, name: str, width: float) -> list[np.ndarray]:
    """Draws iso-level contours of `levels` in the palette colour `families[name]`."""
    return level_lines(diagram, levels, families[name], width)


def sides(diagram: Diagram, levels: Levels, inks: np.ndarray) -> np.ndarray:
    """Fills the two regions separated by the single threshold in `levels.levels` using the two RGBA
    rows of `inks` (`[2, 4]`), clipped to `diagram`'s interior plot box."""
    topology, values = known(levels)
    level, = levels.levels
    bands = contours.Bands(level - 1.0, 1.0, inks)
    return plotted(diagram, drawing.composited([
        contours.picture(on_numpy(levels.grid).reshape(-1), topology, values, np.zeros(3), diagram.camera,
                         bands=(bands,), opacity=0.0)]))


def crossings(levels: Levels, every: int) -> Extensor:
    """Linearly interpolates grid-edge crossings of `levels.levels` along rows and columns where
    both endpoints are finite, subsampling every `every`-th crossing point."""
    places = []
    for tails, heads, at_tails, at_heads in [(levels.grid[:, :-1], levels.grid[:, 1:], levels.values[:, :-1],
                                              levels.values[:, 1:]),
                                             (levels.grid[:-1], levels.grid[1:], levels.values[:-1], levels.values[1:])]:
        with np.errstate(divide="ignore", invalid="ignore"):
            shares = (levels.levels - at_tails[..., None]) / (at_heads - at_tails)[..., None]
        rows, columns, level = np.nonzero((shares >= 0) & (shares < 1))
        starts, steps = tails[rows, columns], heads[rows, columns] - tails[rows, columns]       # [...] Point-
        places.append(starts + steps * shares[rows, columns, level])
    return concatenate(places)[every // 2::every]


def furthest(levels: Levels, plane: Extensor) -> list[tuple[float, Extensor]]:
    """Pairs each crossed contour value in `levels.levels` with the grid-crossing point maximizing
    `plane & point`."""
    found = [(level, crossings(levels._replace(levels=np.array([level])), 1)) for level in levels.levels]
    return [(level, places[(plane & places).argmax()]) for level, places in found if places.shape[0] > 0]


def tangent_lines(diagram: Diagram, points: Extensor, directions: Extensor, across: np.ndarray) -> Extensor:
    """Advances `points` (`[...]`) along `directions` (`[...]`) to the horizontal coordinates
    `across` (`[n_samples]`), returning sampled tangent lines of shape `[..., n_samples]`."""
    plane, points, directions = on_numpy(diagram.planes[0]), on_numpy(points), on_numpy(directions)
    weight, unit = diagram.weighing
    weight = on_numpy(weight)
    # Solve `(plane & (point + step * direction)) / weight == across` for each horizontal sample value.
    weights = read(weight, points) + unit                                      # [...] Scalar+
    steps = (weights[..., None] * across - read(plane, points)[..., None]) / read(plane, directions)[..., None]
    return points[..., None] + directions[..., None] * steps


def arrows(diagram: Diagram, starts: Extensor, toward: Extensor, length: float, width: float,
           colour: np.ndarray) -> list[np.ndarray]:
    """Draws chevron-tipped arrows of fixed pixel `length` and stroke `width` starting at `starts`
    (`[n_arrows]`) and pointing toward `toward` (`[n_arrows]`)."""
    starts, toward = on_numpy(starts), on_numpy(toward)
    n_arrows = starts.shape[0]
    along = toward - starts                                                    # [n_arrows] Point-
    tips = starts + along / diagram.lengths(along) * length                    # [n_arrows] Point-
    return [simplices.strokes(stack((starts, tips), axis=-1).reshape(-1), Topology.chains([2] * n_arrows), colour, width,
                              diagram.camera, chevrons=np.ones(n_arrows), chevron=chevron * width)]


def dashed(diagram: Diagram, points: Extensor, width: float, colour: np.ndarray,
           dashes: tuple[float, float]) -> list[np.ndarray]:
    """Renders batched polylines `points` (`[..., n_samples]`) with pixel `(on, off)` `dashes`."""
    rows = points.reshape(-1, points.shape[-1])
    return strokes(diagram, [rows[row] for row in range(rows.shape[0])], width, colour, 1.0, dashes)


def area(diagram: Diagram, points: Extensor, topology: Topology, colour: np.ndarray, opacity: float) -> np.ndarray:
    """Fills the triangles of `topology` over `points` (`[V]`) in uniform `colour` and `opacity`."""
    return contours.picture(on_numpy(points), topology, np.zeros(points.shape[0]), colour, diagram.camera, opacity=opacity)


def between(diagram: Diagram, curve: Extensor, feet: Extensor, colour: np.ndarray, opacity: float) -> np.ndarray:
    """Shades the ribbon between `curve` (`[n_samples]`) and matching base points `feet`
    (`[n_samples]`) as a triangulated `2`-by-`n_samples` strip."""
    return area(diagram, concatenate((curve, feet)), Topology.grid(2, curve.shape[0]), colour, opacity)


def dots(diagram: Diagram, points: Extensor, colour: np.ndarray) -> list[np.ndarray]:
    """Draws circular state markers in `colour` ringed with white at `points` (`[n_points]`)."""
    points = on_numpy(points)
    return [simplices.markers(points, white, ring, diagram.camera),
            simplices.markers(points, colour, dot, diagram.camera)]


class Label(NamedTuple):
    """Text annotation anchored at diagram point `place` using Pillow anchor code `anchor` (such as
    `"lb"` for left-bottom) plus pixel offset `shift` `(right, up)`."""
    place: Extensor
    text: str
    colour: np.ndarray
    size: int
    anchor: str
    shift: tuple[float, float]


def ticks(diagram: Diagram, across: np.ndarray, up: np.ndarray) -> tuple[list, list]:
    """Outward tick-mark stroke layers and numeric `Label` annotations mark horizontal values
    `across` and vertical values `up` along `diagram`'s axes."""
    x0, y0 = diagram.crossing
    reach = tick_length / diagram.scales                                     # [2]
    # Extend ticks downward from the horizontal axis and leftward from the vertical axis.
    marks = ([stack((diagram.at(value, y0), diagram.at(value, y0 - reach[1]))) for value in across]
             + [stack((diagram.at(x0, value), diagram.at(x0 - reach[0], value))) for value in up])
    labels = ([Label(diagram.at(value, y0), f"{value:g}", ink, tick_size, "mt", (0, -6)) for value in across]
              + [Label(diagram.at(x0, value), f"{value:g}", ink, tick_size, "rm", (-6, 0)) for value in up])
    return strokes(diagram, marks, axis_width, ink, 1.0, SOLID), labels


def picture(diagram: Diagram, pictures: list[np.ndarray], labels: list[Label]) -> np.ndarray:
    """Composites `diagram`'s axes and `pictures` over white and draws axis titles and `labels`,
    returning a `[rows, columns, 3]` `uint8` RGB image."""
    laid = drawing.composited(axes(diagram) + pictures)
    seen = white * (1 - laid[..., 3:]) + laid[..., :3] * laid[..., 3:]
    image = np.round(np.clip(seen, 0, 1) * 255).astype(np.uint8)
    x0, y0 = diagram.crossing
    names = [Label(diagram.at(diagram.across[1], y0), diagram.names[0], ink, text_size, "rt", (0, -4)),
             Label(diagram.at(x0, diagram.up[1]), diagram.names[1], ink, text_size, "lt", (8, 6))]
    return written(diagram, image, names + labels)


def written(diagram: Diagram, image: np.ndarray, labels: list[Label]) -> np.ndarray:
    """Renders `labels` (formatting `_subscript` suffixes at tick font size) onto `image` with a
    white halo."""
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for label in labels:
        right, up = diagram.pixels(label.place) + np.array(label.shift)
        colour = tuple(int(round(255 * channel)) for channel in label.colour)
        # Split optional `_subscript` suffixes and shift the combined bounding box to honour `label.anchor`.
        base, _, subscript = label.text.partition("_")
        font, small = fonts[label.size], fonts[tick_size]
        whole = pen.textbbox((right, diagram.rows - up), base, font=font, anchor=label.anchor)
        shift = small.getlength(subscript) if subscript else 0.0
        horizontal = {"l": 0.0, "m": shift / 2, "r": shift}[label.anchor[0]]
        # Outline text with a thin white stroke so labels remain legible across curves.
        pen.text((right - horizontal, diagram.rows - up), base, fill=colour, font=font, anchor=label.anchor,
                 stroke_width=halo, stroke_fill=(255, 255, 255))
        if subscript:
            pen.text((whole[2] - horizontal + 1, whole[3] + 2), subscript, fill=colour, font=small, anchor="lb",
                     stroke_width=halo, stroke_fill=(255, 255, 255))
    return np.asarray(canvas)


class Floating(NamedTuple):
    """Curve or region label that selects whichever candidate point in `places` (`[n_candidates]`)
    overlaps the least existing ink inside the plot box."""
    places: Extensor
    text: str
    colour: np.ndarray
    size: int
    anchor: str


def along(lines: list[Extensor], every: int) -> Extensor:
    """Subsamples every `every`-th point along `lines` as candidate `Floating` label positions."""
    return concatenate([points[every // 2::every] for points in lines])


# Cached pixel positions per `(diagram, text, anchor, candidates)` so floating labels stay fixed across animation frames.
places: dict = {}


def floated(diagram: Diagram, image: np.ndarray, names: list[Floating]) -> np.ndarray:
    """Places each `Floating` label in `names` at the candidate point inside the plot box that
    covers the least existing ink (avoiding earlier labels) and caches that position across
    frames."""
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for name in names:
        candidates = diagram.pixels(name.places)                             # [..., 2]
        key = (diagram, name.text, name.anchor, candidates.tobytes())
        if key in places:
            if places[key] is not None:
                colour = tuple(int(round(255 * channel)) for channel in name.colour)
                pen.text((places[key][0], diagram.rows - places[key][1]), name.text, fill=colour,
                         font=fonts[name.size], anchor=name.anchor, stroke_width=halo, stroke_fill=(255, 255, 255))
            continue
        inked = 255 - np.asarray(canvas).min(axis=-1).astype(float)
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
    """Vertically centres `[rows, columns, 3]` RGB `images` on white padding to match the tallest
    panel and concatenates them horizontally."""
    rows = max(image.shape[0] for image in images)
    padded = [np.pad(image, (((rows - image.shape[0]) // 2, rows - image.shape[0] - (rows - image.shape[0]) // 2),
                             (0, 0), (0, 0)), constant_values=255) for image in images]
    return np.concatenate(padded, axis=1)


def figure(image: np.ndarray):
    """Wraps `image` in a borderless matplotlib `Figure` at native pixel dimensions."""
    return image_figure(image)


# --- the charts of a gas ---------------------------------------------------------------
class GasPoints(NamedTuple):
    """Paired chart positions of gas states on the `(entropy, temperature)` and `(volume, pressure)`
    planes."""
    entropy_temperature: Point
    volume_pressure: Point


class GasFamily(NamedTuple):
    """Paired polyline lists (`[n_samples] Point-` per curve) of one curve family on both gas
    charts."""
    entropy_temperature: list
    volume_pressure: list


class GasLevels(NamedTuple):
    """Paired contour `Levels` of one scalar state function mapped onto both gas charts."""
    entropy_temperature: Levels
    volume_pressure: Levels


class GasLines(NamedTuple):
    """Isochores, isobars, isotherms, and adiabats across both gas charts (or the numeric level
    arrays requested for each family when passed into `gas_lines`)."""
    isochores: GasFamily
    isobars: GasLevels
    isotherms: GasLevels
    adiabats: GasFamily


def gas_points(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor],
               states: Extensor) -> GasPoints:
    """Maps `(entropy, volume)` antivector `states` (`[...]`) onto `(entropy, temperature)` and
    `(volume, pressure)` chart points using `temperature_of` and `pressure_of`."""
    blades = states.context.multivector
    return GasPoints(chart_points(context.include_from_algebra(blades.s & states), context.include_from_algebra(temperature_of(states))),
                     chart_points(context.include_from_algebra(blades.v & states), context.include_from_algebra(pressure_of(states))))


def gas_lines(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor], blades,
              entropies: np.ndarray, volumes: np.ndarray, levels: GasLines) -> GasLines:
    """Samples straight isochores and adiabats directly along `entropies` and `volumes` and builds
    isobar and isotherm `GasLevels` from `pressure_of` and `temperature_of` over the product state
    grid."""
    states = lambda entropy, volume: blades.s.dual() * entropy + blades.v.dual() * volume
    charted = lambda lines: GasFamily([points.entropy_temperature for points in lines],
                                      [points.volume_pressure for points in lines])
    points = lambda entropy, volume: gas_points(temperature_of, pressure_of, states(entropy, volume))
    grid = states(entropies[None, :], volumes[:, None])
    temperatures, pressures = np.asarray(temperature_of(grid).to_array()), np.asarray(pressure_of(grid).to_array())
    on_charts = gas_points(temperature_of, pressure_of, grid)
    leveled = lambda values, asked: GasLevels(Levels(on_charts.entropy_temperature, values, np.asarray(asked)),
                                              Levels(on_charts.volume_pressure, values, np.asarray(asked)))
    return GasLines(charted([points(entropies, level) for level in levels.isochores]),
                    leveled(pressures, levels.isobars),
                    leveled(temperatures, levels.isotherms),
                    charted([points(level, volumes) for level in levels.adiabats]))


# --- apparatus -------------------------------------------------------------------------
# Unit upward displacement direction along which the gas fill carries the piston and its load.
rise = solids.mv.direction([0, 0, 1])                                        # [] Point-


def box(centre: solids.Point, half: solids.Scalar) -> solids.Plane:
    """Six bounding planes (`[1, 6] Plane+`) of an axis-aligned box of `half` extents around
    `centre`."""
    return solids.boxes(solids.translator(centre - solids.origin), half)[None]


# Five-stop cold-to-hot RGB colour scale from icy blue through neutral cream to red heat.
thermal = np.stack([rgb(colour) for colour in ("#1f6fc5", "#8fd0f7", "#f2eee8", "#f6a04d", "#d01f17")])


def thermal_colour(fraction: float) -> np.ndarray:
    """Piecewise-linear RGB interpolation along `thermal` at `fraction` in `[0, 1]`."""
    place = np.clip(fraction, 0.0, 1.0) * (len(thermal) - 1)
    low = min(int(place), len(thermal) - 2)
    return thermal[low] * (1 - (place - low)) + thermal[low + 1] * (place - low)


def relaxing(gap: float, widest: float, decades: float) -> float:
    """Blends linear and `decades`-logarithmic normalized magnitude around `0.5` so an exponentially
    decaying temperature `gap` (relative to `widest`) stays visually distinguishable from start to
    finish."""
    share = abs(gap) / widest
    depth = 0.5 * np.clip(share, 0.0, 1.0) + 0.5 * np.clip(np.log10(share + 1e-300) / decades + 1, 0.0, 1.0)
    return 0.5 + 0.5 * np.sign(gap) * depth


# Cylinder inner half-width and half-depth, wall thickness, and piston plate thickness.
bore, bore_depth, wall, plate = 0.52, 0.43, 0.055, 0.12
# Reference boxes at zero fill height for the piston, constant-pressure weight, and constant-volume latch.
resting_piston = box(solids.mv.point([0, 0, plate / 2]), np.array([bore + 0.06, bore_depth + 0.05, plate / 2]))
resting_weight = box(solids.mv.point([0, 0, plate + 0.14]), np.array([0.31, 0.32, 0.14]))
resting_latch = box(solids.mv.point([bore + 0.12, 0, plate / 2]), np.array([0.14, bore_depth + 0.13, 0.05]))


def cylinder(height: solids.Scalar, maximum: solids.Scalar) -> list:
    """Base, side walls sized for `maximum` fill height, and piston translated to `height`, paired
    with their grey RGB colours."""
    centres = solids.mv.point([[-bore - wall, 0, 0], [bore + wall, 0, 0]]) + rise * (maximum / 2)
    half = stack((solids.mv.scalar([wall]), solids.mv.scalar([bore_depth + 0.07]), maximum / 2))
    walls = solids.boxes(solids.translator(centres - solids.origin), half)
    base = box(solids.mv.point([0, 0, -0.08]), np.array([bore + 2 * wall + 0.06, bore_depth + 0.17, 0.08]))
    return [(base, grey), (walls, grey), (solids.translator(rise * height) >> resting_piston, grey * 0.65)]


def load(height: solids.Scalar) -> list:
    """Constant-pressure weight block translated atop a piston at fill `height`."""
    return [(solids.translator(rise * height) >> resting_weight, ink)]


def latch(height: solids.Scalar) -> list:
    """Constant-volume stop pin translated across a piston at fill `height`."""
    return [(solids.translator(rise * height) >> resting_latch, orange)]


def fill(bottom: solids.Scalar, top: solids.Scalar) -> solids.Plane:
    """Six bounding planes (`[1, 6] Plane+`) of the cylinder bore between `bottom` and `top`
    (clamped to at least `1e-6` thickness)."""
    x, y, z, w = solids.mv.x, solids.mv.y, solids.mv.z, solids.mv.w
    return stack((x - w * bore, -x - w * bore, y - w * bore_depth, -y - w * bore_depth,
                  z - w * (bottom + (top - bottom).clip(1e-6, np.inf)), w * bottom - z))[None]


def cylinder_view(maximum: solids.Scalar, pixels: int) -> solids.View:
    """Oblique `solids.View` of width `pixels` framed around a cylinder of `maximum` fill height
    plus its piston and weight."""
    base = solids.mv.point([[x, y, -0.16] for x in (-0.75, 0.75) for y in (-0.6, 0.6)])
    corners = stack((base, base + rise * (maximum + 0.61))).reshape(-1)
    return solids.view([corners], 0.0, np.radians(18), np.radians(-25), 0.08, pixels)


def gas_cylinder(height: solids.Scalar, warmth: float, latched: bool, maximum: solids.Scalar, pixels: int) -> np.ndarray:
    """Ray-traces a gas cylinder filled to `height` with translucent `thermal_colour(warmth)` gas
    under either a volume latch (`latched=True`) or a pressure weight (`latched=False`)."""
    holder = latch(height) if latched else load(height)
    return solids.draw(cylinder(height, maximum) + holder, cylinder_view(maximum, pixels), white, inks=[],
                       translucent=[(fill(0.0, height), thermal_colour(warmth), 0.55)], meshes=[])
