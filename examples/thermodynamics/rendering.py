"""Drawing shared by the thermodynamics examples: diagrams of one quantity against another, drawn on the GPU by
`planar` with their names written over them, and the apparatus, a cylinder under a piston traced by `solids`.

A diagram maps its two quantities linearly onto its pixels, each at its own scale; a point of the diagram is a point of
the plane of its picture, so curves are polylines through such points, areas are polygons, and the lines and conics of
the diagram are lines and conics of that plane.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageDraw, ImageFont
from contourpy import contour_generator

from numga import Extensor
from examples.util.wgsl import image_figure, planar, solids
from examples.util.wgsl.planar import SOLID, Cap, Marker, Marks, Polygons, Strokes


def rgb(colour: str) -> np.ndarray:
    return np.array(to_rgb(colour))


BLUE, ORANGE, GREY, INK, GREEN = rgb("#3196b4"), rgb("#de9439"), rgb("#aab9c1"), rgb("#485961"), rgb("#458875")
# The faint colour of the lines a diagram is read against, isotherms or levels, and the white behind everything.
FAINT = rgb("#cfd9dc")
WHITE = np.ones(3)

# Pixels around a diagram's plot: room for the name of the quantity up on the left, of the one across below, and a
# little on the other two sides.
LEFT, BOTTOM, RIGHT, TOP = 30, 24, 12, 16
# The axes' width and arrowheads, and the size of the names, in pixels.
AXIS_WIDTH, HEAD_LENGTH, HEAD_WIDTH = 1.2, 7.0, 5.0
# The size of the quantities' names, and of the labels on curves and regions.
NAME_SIZE, LABEL_SIZE = 14, 11
FONTS = {size: ImageFont.truetype(findfont(FontProperties(family="DejaVu Sans", style="italic")), size)
         for size in (NAME_SIZE, LABEL_SIZE)}
# The radius of a dot marking a state, and of the white ring around it, in pixels.
DOT, RING = 4.0, 5.5


# --- diagrams --------------------------------------------------------------------------
class Diagram(NamedTuple):
    """A diagram of one quantity across against another up: the range of each shown, the values at which the axes
    cross, the picture's rows and columns of pixels, and the names of the two quantities."""
    across: tuple[float, float]
    up: tuple[float, float]
    crossing: tuple[float, float]
    rows: int
    columns: int
    names: tuple[str, str]

    def places(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Where values of the two quantities lie in the picture, in pixels right and up from its bottom left corner
        `[..., 2]`."""
        x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        right = LEFT + (x - self.across[0]) / (self.across[1] - self.across[0]) * (self.columns - LEFT - RIGHT)
        up = BOTTOM + (y - self.up[0]) / (self.up[1] - self.up[0]) * (self.rows - BOTTOM - TOP)
        return np.stack([right, up], axis=-1)

    def points(self, x: np.ndarray, y: np.ndarray) -> planar.Point:
        """The points of the picture's plane at values of the two quantities `[...]`."""
        return planar.mv.point(self.places(x, y))

    @property
    def window(self) -> planar.Window:
        return planar.window((0, self.columns), (0, self.rows), self.rows)


def axes(diagram: Diagram) -> list:
    """The two axes, crossing where the diagram says, each ending in an arrowhead."""
    x0, y0 = diagram.crossing
    ends = diagram.places([diagram.across[1], x0], [y0, diagram.up[1]])       # [2, 2]
    corner = diagram.places(x0, y0)
    lines = planar.mv.point(np.stack([np.broadcast_to(corner, (2, 2)), ends], axis=1))   # [2, 2] Point
    along = np.array([[1, 0], [0, 1]])
    square = along[:, ::-1]
    heads = planar.mv.point(np.stack([ends + along * HEAD_LENGTH, ends + square * HEAD_WIDTH / 2,
                                      ends - square * HEAD_WIDTH / 2], axis=1))   # [2, 3] Point
    return [Strokes(lines, AXIS_WIDTH, INK, 1.0, SOLID, Cap.BUTT), Polygons(heads, INK, 1.0)]


def curves(diagram: Diagram, x: np.ndarray, y: np.ndarray, width: float, colour: np.ndarray, opacity: float) -> Strokes:
    """Polylines through values of the two quantities `[..., samples]`, `width` pixels across."""
    points = diagram.points(x, y)
    return Strokes(points.reshape(-1, points.shape[-1]), width, colour, opacity, SOLID, Cap.ROUND)


def within(diagram: Diagram, x: np.ndarray, y: np.ndarray, width: float, colour: np.ndarray,
           opacity: float) -> list[Strokes]:
    """Polylines through values of the two quantities `[curves, samples]`, each in its colour `[curves, 3]`, cut to
    the runs of samples inside the diagram's ranges."""
    return [stroke for curve in range(x.shape[0])
            for stroke in polylines(diagram, [(x[curve], y[curve])], width, colour[curve], opacity, SOLID)]


def polylines(diagram: Diagram, lines: list[tuple[np.ndarray, np.ndarray]], width: float, colour: np.ndarray,
              opacity: float, dashes: tuple[float, float]) -> list[Strokes]:
    """Polylines through values of the two quantities, each its own number of samples, cut to the runs of samples
    inside the diagram's ranges; dashed `dashes` pixels on and off, `SOLID` for solid."""
    strokes = []
    for x, y in lines:
        inside = (x >= diagram.across[0]) & (x <= diagram.across[1]) & (y >= diagram.up[0]) & (y <= diagram.up[1])
        # The runs of samples inside, between the steps into and out of the ranges.
        steps = np.diff(np.concatenate([[0], inside.astype(int), [0]]))
        for start, end in zip(np.flatnonzero(steps == 1), np.flatnonzero(steps == -1)):
            if end - start > 1:
                points = diagram.points(x[start:end], y[start:end])[None]
                cap = Cap.ROUND if dashes == SOLID else Cap.BUTT
                strokes.append(Strokes(points, width, colour, opacity, dashes, cap))
    return strokes


def level_lines(across: np.ndarray, up: np.ndarray, values: np.ndarray, levels: np.ndarray) -> list[np.ndarray]:
    """Where a function sampled on a grid, `values[up, across]` at the grid's coordinates `across` and `up`, takes
    each of the levels: polylines `[samples, 2]` of the two coordinates."""
    lines = contour_generator(across, up, values)
    return [line for level in levels for line in lines.lines(level) if len(line) > 1]


def arrows(diagram: Diagram, x: np.ndarray, y: np.ndarray, toward_x: np.ndarray, toward_y: np.ndarray, length: float,
           width: float, colour: np.ndarray) -> list:
    """Arrows `length` pixels long from values of the two quantities `[n]` toward others, each a stroke ending in a
    head."""
    starts = diagram.places(x, y)
    along = diagram.places(toward_x, toward_y) - starts
    along = along / np.linalg.norm(along, axis=-1, keepdims=True)
    ends = starts + along * length
    square = along[..., ::-1] * np.array([1, -1])
    heads = np.stack([ends + along * HEAD_LENGTH, ends + square * HEAD_WIDTH / 2, ends - square * HEAD_WIDTH / 2], axis=-2)
    return [Strokes(planar.mv.point(np.stack([starts, ends], axis=-2)).reshape(-1, 2), width, colour, 1.0, SOLID, Cap.BUTT),
            Polygons(planar.mv.point(heads).reshape(-1, 3), colour, 1.0)]


def ellipses(diagram: Diagram, x: np.ndarray, y: np.ndarray, form: np.ndarray, level: float) -> planar.Quadric:
    """The ellipses about values of the two quantities `[n]` where a quadratic form on their variations,
    `form[n, 2, 2]`, the across and up entries in order, takes the level: conics of the picture's plane, maps from
    points to lines, whose pairing `conic(p) & p` is the form on the variation to the point, less the level."""
    mv, Point = planar.mv, planar.Point
    scale = (np.array([diagram.across[1] - diagram.across[0], diagram.up[1] - diagram.up[0]])
             / np.array([diagram.columns - LEFT - RIGHT, diagram.rows - BOTTOM - TOP]))   # [2] units a pixel
    centres = diagram.places(x, y)
    # The lines through each centre square to across and up, pairing with a point to its offset in pixels.
    offsets = [mv.x - mv.w * centres[:, 0], mv.y - mv.w * centres[:, 1]]                  # [2] of [n] Plane
    conic = -mv.w * (mv.w & Point) * level
    for row in range(2):
        for column in range(2):
            conic = conic + offsets[row] * (offsets[column] & Point) * (form[:, row, column] * scale[row] * scale[column])
    return conic


def dashed(diagram: Diagram, x: np.ndarray, y: np.ndarray, width: float, colour: np.ndarray,
           dashes: tuple[float, float]) -> Strokes:
    """Dashed polylines through values of the two quantities `[..., samples]`, dashes and gaps in pixels."""
    points = diagram.points(x, y)
    return Strokes(points.reshape(-1, points.shape[-1]), width, colour, 1.0, dashes, Cap.BUTT)


def areas(diagram: Diagram, x: np.ndarray, y: np.ndarray, colour: np.ndarray, opacity: float) -> Polygons:
    """The areas inside outlines through values of the two quantities `[..., corners]`, each closed from its last
    corner back to its first, which it does not repeat."""
    corners = diagram.points(x, y)
    return Polygons(corners.reshape(-1, corners.shape[-1]), colour, opacity)


def dots(diagram: Diagram, x: np.ndarray, y: np.ndarray, colour: np.ndarray) -> list:
    """Dots at values of the two quantities `[n]`, each in a white ring."""
    points = diagram.points(np.atleast_1d(x), np.atleast_1d(y))
    return [Marks(points, Marker.DISC, 0.0, RING, WHITE, 1.0), Marks(points, Marker.DISC, 0.0, DOT, colour, 1.0)]


def ticks(diagram: Diagram, across: np.ndarray, up: np.ndarray) -> tuple[list, list]:
    """Short marks on the axes at values to read off, and their numbers beside them: layers and labels."""
    x0, y0 = diagram.crossing
    marks = np.concatenate([np.stack([diagram.places(across, y0), diagram.places(across, y0) + [0, -4]], axis=-2),
                            np.stack([diagram.places(x0, up), diagram.places(x0, up) + [-4, 0]], axis=-2)])
    labels = ([Label(value, y0, f"{value:g}", INK, LABEL_SIZE, "mt", (0, -6)) for value in across]
              + [Label(x0, value, f"{value:g}", INK, LABEL_SIZE, "rm", (-6, 0)) for value in up])
    return [Strokes(planar.mv.point(marks), AXIS_WIDTH, INK, 1.0, SOLID, Cap.BUTT)], labels


class Label(NamedTuple):
    """A name written at values of the two quantities, in its colour and size, placed by a Pillow anchor (`"lb"` puts
    the text's left bottom there) and shifted a few pixels right and up."""
    x: float
    y: float
    text: str
    colour: np.ndarray
    size: int
    anchor: str
    shift: tuple[float, float]


def picture(diagram: Diagram, layers: list, labels: list[Label]) -> np.ndarray:
    """The diagram's axes under the layers, `planar`'s primitives composited in order, with the quantities' names at
    the ends of the axes and the labels written over them: RGB bytes `[rows, columns, 3]`."""
    image = planar.draw(diagram.window, axes(diagram) + layers, WHITE)
    x0, y0 = diagram.crossing
    names = [Label(diagram.across[1], y0, diagram.names[0], INK, NAME_SIZE, "rt", (0, -4)),
             Label(x0, diagram.up[1], diagram.names[1], INK, NAME_SIZE, "lt", (8, 6))]
    return written(diagram, image, names + labels)


def written(diagram: Diagram, image: np.ndarray, labels: list[Label]) -> np.ndarray:
    """An image of the diagram with the labels written over it."""
    canvas = Image.fromarray(image)
    pen = ImageDraw.Draw(canvas)
    for label in labels:
        right, up = diagram.places(label.x, label.y) + np.array(label.shift)
        colour = tuple(int(round(255 * channel)) for channel in label.colour)
        # A name with a subscript after an underscore: the subscript smaller and lower, after the rest, the two placed
        # together by the anchor.
        base, _, subscript = label.text.partition("_")
        font, small = FONTS[label.size], FONTS[LABEL_SIZE if label.size == NAME_SIZE else label.size]
        whole = pen.textbbox((right, diagram.rows - up), base, font=font, anchor=label.anchor)
        shift = small.getlength(subscript) if subscript else 0.0
        horizontal = {"l": 0.0, "m": shift / 2, "r": shift}[label.anchor[0]]
        pen.text((right - horizontal, diagram.rows - up), base, fill=colour, font=font, anchor=label.anchor)
        if subscript:
            pen.text((whole[2] - horizontal + 1, whole[3] + 2), subscript, fill=colour, font=small, anchor="lb")
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
class GasLines(NamedTuple):
    """A gas's lines of equal volume, pressure, temperature and entropy, each a list of polylines of samples of
    entropy, temperature, volume and pressure `[samples, 4]`."""
    isochores: list[np.ndarray]
    isobars: list[np.ndarray]
    isotherms: list[np.ndarray]
    adiabats: list[np.ndarray]


def gas_values(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor],
               states: Extensor) -> np.ndarray:
    """Entropy, temperature, volume and pressure of a gas's states `[..., 4]`, its states antivectors of the plane of
    entropy and volume, read by pairing."""
    mv = states.context.multivector
    return np.stack([(mv.s & states).to_array(), temperature_of(states).to_array(), (mv.v & states).to_array(),
                     pressure_of(states).to_array()], axis=-1)


def gas_lines(temperature_of: Callable[[Extensor], Extensor], pressure_of: Callable[[Extensor], Extensor], mv,
              entropies: np.ndarray, volumes: np.ndarray, levels: GasLines) -> GasLines:
    """A gas's lines at the levels given, each level a number: lines of equal volume and of equal entropy run straight
    across the plane of states sampled at the entropies and volumes; lines of equal pressure and of equal temperature
    are level lines of the gas's functions over it."""
    states = lambda entropy, volume: mv.s.dual() * entropy + mv.v.dual() * volume
    values = lambda entropy, volume: gas_values(temperature_of, pressure_of, states(entropy, volume))
    grid = states(entropies[None, :], volumes[:, None])                                   # [volumes, entropies]
    temperatures, pressures = temperature_of(grid).to_array(), pressure_of(grid).to_array()
    return GasLines(list(values(entropies[None, :], np.asarray(levels.isochores)[:, None])),
                    [values(*line.T) for line in level_lines(entropies, volumes, pressures, np.asarray(levels.isobars))],
                    [values(*line.T) for line in level_lines(entropies, volumes, temperatures, np.asarray(levels.isotherms))],
                    list(values(np.asarray(levels.adiabats)[:, None], volumes[None, :])))


def drawn(diagram: Diagram, lines: list[np.ndarray], x: int, y: int, width: float, colour: np.ndarray,
          dashes: tuple[float, float]) -> list[Strokes]:
    """A gas's lines on a diagram of two of their four values, by index: entropy, temperature, volume, pressure."""
    return polylines(diagram, [(line[:, x], line[:, y]) for line in lines], width, colour, 1.0, dashes)


# --- apparatus -------------------------------------------------------------------------
def box(centre: np.ndarray, half: np.ndarray) -> solids.Plane:
    """A box about a centre, of half extents along x, y and z, as the six planes bounding it `[1, 6]`."""
    return solids.boxes(solids.translator(solids.mv.point(centre) - solids.origin), np.asarray(half))[None]


def thermal_colour(fraction: float) -> np.ndarray:
    """Blue for cold through orange for hot, by a share of the way between the coldest and hottest shown."""
    return BLUE * (1 - fraction) + ORANGE * fraction


# The cylinder's inner half width and depth, its wall thickness, and the piston's thickness.
BORE, DEPTH, WALL, PLATE = 0.52, 0.43, 0.055, 0.12


def cylinder(height: float, maximum: float) -> list:
    """A cylinder's base and walls, tall enough for a fill of `maximum`, and its piston resting on a fill of
    `height`, as parts with their colours."""
    centres = np.array([[-BORE - WALL, 0, maximum / 2], [BORE + WALL, 0, maximum / 2]])
    walls = solids.boxes(solids.translator(solids.mv.point(centres) - solids.origin),
                         np.array([WALL, DEPTH + 0.07, maximum / 2]))
    base = box(np.array([0, 0, -0.08]), np.array([BORE + 2 * WALL + 0.06, DEPTH + 0.17, 0.08]))
    piston = box(np.array([0, 0, height + PLATE / 2]), np.array([BORE + 0.06, DEPTH + 0.05, PLATE / 2]))
    return [(base, GREY), (walls, GREY), (piston, GREY * 0.65)]


def load(height: float) -> list:
    """A weight on a piston resting on a fill of `height`: it holds the pressure under the piston fixed."""
    return [(box(np.array([0, 0, height + PLATE + 0.14]), np.array([0.31, 0.32, 0.14])), INK)]


def latch(height: float) -> list:
    """A latch across the piston resting on a fill of `height`: it holds the volume under the piston fixed."""
    return [(box(np.array([BORE + 0.12, 0, height + PLATE / 2]), np.array([0.14, DEPTH + 0.13, 0.05])), ORANGE)]


def fill(bottom: float, top: float) -> solids.Plane:
    """The contents of the cylinder between two heights."""
    return box(np.array([0, 0, (bottom + top) / 2]), np.array([BORE, DEPTH, max(top - bottom, 1e-6) / 2]))


def cylinder_view(maximum: float, pixels: int) -> solids.View:
    """The view of a cylinder tall enough for a fill of `maximum`, its piston and a weight on it, `pixels` across."""
    corners = solids.mv.point(np.array([[x, y, z] for x in (-0.75, 0.75) for y in (-0.6, 0.6)
                                        for z in (-0.16, maximum + 0.45)]))
    return solids.view([corners], 0.0, np.radians(18), np.radians(-25), 0.08, pixels)


def gas_cylinder(height: float, warmth: float, latched: bool, maximum: float, pixels: int) -> np.ndarray:
    """A cylinder of gas filled to a height, its colour by how warm it is, a share of the way from cold to hot, under
    a piston either latched or loaded with a weight; framed for a fill of `maximum`, `pixels` across."""
    holder = latch(height) if latched else load(height)
    return solids.draw(cylinder(height, maximum) + holder, cylinder_view(maximum, pixels), WHITE, inks=[],
                       translucent=[(fill(0.0, height), thermal_colour(warmth), 0.55)], meshes=[])
