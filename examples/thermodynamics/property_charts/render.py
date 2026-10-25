"""Moist air cooled, dried and reheated, beside its psychrometric chart.

The chart holds the air's temperature across and its water content up, with the lines of equal relative humidity,
blue, the saturation curve black, and the lines of equal enthalpy, brown, each marked with its value where it meets the
saturation curve. Above the saturation curve no air holds that much water as vapour. The air's path runs left at fixed
water content to the saturation curve, down along it while water condenses, and right at fixed water content through
the reheater; the arrows are the directions it moves in at the inlet and at the coil's outlet, solved for one degree
warmer with the held reading unchanged.

The chart's lines are level lines of the air's relative humidity and enthalpy, functions the notebook hands over; this
module only samples them.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids

Function = Callable[[Extensor], Extensor]

# The chart's size in pixels, its ranges, temperature in degrees and water content in grams per kg of dry air, and the
# values read off its axes.
ROWS, COLUMNS = 320, 400
TEMPERATURES, WATER = (5.0, 35.0), (0.0, 25.0)
TEMPERATURE_TICKS, WATER_TICKS = np.array([10, 20, 30]), np.array([5, 10, 15, 20])
# The grid the levels are found on; the relative humidities and enthalpies of the lines.
GRID_TEMPERATURES, GRID_WATER = np.linspace(5.0, 35.0, 150), np.linspace(0.0, 0.025, 150)
HUMIDITIES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
ENTHALPIES = np.arange(20.0, 110.0, 10.0)
# Grams per kilogram; how far a direction's arrow reaches in pixels; line widths.
GRAMS = 1000.0
ARROW = 28.0
# Where the saturation curve's name is written, in degrees and grams per kg.
SATURATION_LABEL = (27.5, 22.5)
THIN, BOLD = 1.0, 2.2
# The apparatus's picture, pixels across.
APPARATUS_PIXELS = 240


def apparatus(fraction: float, warmth: float, liquid: float) -> np.ndarray:
    """The duct: the cooling coil, its tray of collected water, the reheater, and the parcel of air passing them."""
    tray = art.box(np.array([0, 0, 0.1]), np.array([2.35, 0.6, 0.08]))
    coil_centres = np.array([[-0.8 + offset, 0, 0.8] for offset in (-0.25, 0, 0.25)])
    coils = solids.boxes(solids.translator(solids.mv.point(coil_centres) - solids.origin), np.array([0.045, 0.5, 0.55]))
    reheater = art.box(np.array([1.1, 0, 0.8]), np.array([0.12, 0.45, 0.45]))
    parcel = solids.ball(solids.mv.point([-2 + 4 * fraction, -0.65, 0.82]), 0.23)[None, None]
    depth = 0.01 + 0.2 * liquid
    collected = art.box(np.array([-0.8, -0.72, 0.15 + depth / 2]), np.array([0.45, 0.2, depth / 2]))
    corners = solids.mv.point(np.array([[x, y, z] for x in (-2.5, 2.5) for y in (-1.0, 0.65) for z in (0, 1.5)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-10), 0.04, APPARATUS_PIXELS)
    return solids.draw([(tray, art.GREY), (coils, art.BLUE), (reheater, art.ORANGE),
                        (parcel, art.thermal_colour(warmth)), (collected, art.BLUE)], view, art.WHITE, inks=[],
                       translucent=[], meshes=[])


class Chart:
    """The psychrometric chart's lines, sampled once, and the drawing of the air's path on it."""

    def __init__(self, relative_humidity_of: Function, enthalpy_of: Function, mv) -> None:
        self.read = lambda states: np.stack([(mv.t & states).to_array(), GRAMS * (mv.w & states).to_array()], axis=-1)
        self.diagram = art.Diagram(TEMPERATURES, WATER, (TEMPERATURES[0], 0.0), ROWS, COLUMNS, ("T", "w"))
        grid = mv.t.dual() * GRID_TEMPERATURES[None, :] + mv.w.dual() * GRID_WATER[:, None]
        relative = relative_humidity_of(grid).to_array()
        # The enthalpy only where the air holds its water as vapour.
        heat = np.where(relative <= 1, enthalpy_of(grid).to_array(), np.nan)
        scaled = lambda lines: [(line[:, 0], GRAMS * line[:, 1]) for line in lines]
        humidities = [line for _, line in art.level_lines(GRID_TEMPERATURES, GRID_WATER, relative, HUMIDITIES)]
        saturation = [line for _, line in art.level_lines(GRID_TEMPERATURES, GRID_WATER, relative, np.ones(1))]
        enthalpies = [[line for _, line in art.level_lines(GRID_TEMPERATURES, GRID_WATER, heat, np.array([level]))]
                      for level in ENTHALPIES]
        # Above the saturation curve: the region up to the chart's top left corner.
        top = [(np.concatenate([line[:, 0], [TEMPERATURES[1], TEMPERATURES[0]]]),
                np.concatenate([GRAMS * line[:, 1], [WATER[1], WATER[1]]])) for line in saturation]
        tick_layers, self.tick_labels = art.ticks(self.diagram, TEMPERATURE_TICKS, WATER_TICKS)
        self.layers = ([art.areas(self.diagram, x, np.clip(y, *WATER), art.GREY, 0.18) for x, y in top]
                       + art.polylines(self.diagram, scaled(humidities), THIN, art.FAMILIES["humidity"], 1.0, art.SOLID)
                       + art.polylines(self.diagram, [line for lines in enthalpies for line in scaled(lines)], THIN,
                                       art.FAMILIES["enthalpy"], 1.0, art.SOLID)
                       + art.polylines(self.diagram, scaled(saturation), 1.8, art.BOUNDARY, 1.0, art.SOLID)
                       + tick_layers)
        # Each enthalpy line's value at its upper end, where it meets the saturation curve; each humidity line's at its
        # right end.
        self.labels = [art.Label(SATURATION_LABEL[0], SATURATION_LABEL[1], "saturation", art.BOUNDARY, art.LABEL_SIZE, "lb",
                                 (0, 2))]
        for level, lines in zip(ENTHALPIES, enthalpies):
            for x, y in scaled(lines):
                top = np.argmax(y)
                if TEMPERATURES[0] < x[top] and y[top] < WATER[1] - 1:
                    self.labels.append(art.Label(x[top], y[top], f"{level:g}", art.FAMILIES["enthalpy"], art.LABEL_SIZE,
                                                 "rb", (-2, 2)))
        for level in HUMIDITIES:
            for x, y in scaled([line for _, line in art.level_lines(GRID_TEMPERATURES, GRID_WATER, relative,
                                                                     np.array([level]))]):
                right = np.argmax(x)
                if y[right] < WATER[1] - 1:
                    self.labels.append(art.Label(x[right], y[right], f"{level:.0%}", art.FAMILIES["humidity"],
                                                 art.LABEL_SIZE, "rm", (-2, 0)))

    def picture(self, path: np.ndarray, travelled: int, states: np.ndarray, toward: np.ndarray) -> np.ndarray:
        layers = (self.layers
                  + art.polylines(self.diagram, [(path[:, 0], path[:, 1])], THIN, art.ORANGE, 0.6, art.SOLID)
                  + art.polylines(self.diagram, [(path[:travelled + 1, 0], path[:travelled + 1, 1])], BOLD, art.ORANGE,
                                  1.0, art.SOLID)
                  + art.arrows(self.diagram, states[:, 0], states[:, 1], toward[:, 0], toward[:, 1], ARROW, 1.6, art.GREEN)
                  + art.dots(self.diagram, path[travelled, 0], path[travelled, 1], art.ORANGE))
        return art.picture(self.diagram, layers, self.labels + self.tick_labels)


def conditioning(relative_humidity_of: Function, enthalpy_of: Function, path: Extensor, states: Extensor,
                 directions: Extensor, passing: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The apparatus and the chart at each state of the air along its path."""
    first = next(passing)
    chart = Chart(relative_humidity_of, enthalpy_of, first.context.multivector)
    traced, places, toward = chart.read(path), chart.read(states), chart.read(states + directions)
    for travelled, state in enumerate(chain([first], passing)):
        now = chart.read(state)
        warmth = (now[0] - traced[:, 0].min()) / np.ptp(traced[:, 0])
        liquid = (traced[0, 1] - now[1]) / (traced[0, 1] - traced[-1, 1])
        yield art.beside([apparatus(travelled / (len(traced) - 1), warmth, liquid),
                          chart.picture(traced, travelled, places, toward)])


def draw_chart(relative_humidity_of: Function, enthalpy_of: Function, states: Extensor, directions: Extensor):
    """The chart with the directions the air moves in at the inlet and at the coil's outlet."""
    chart = Chart(relative_humidity_of, enthalpy_of, states.context.multivector)
    places = chart.read(states)
    return art.figure(chart.picture(places[:1], 0, places, chart.read(states + directions)))
