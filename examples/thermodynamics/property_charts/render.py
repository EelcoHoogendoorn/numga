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

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids

Function = Callable[[Extensor], Extensor]

# The chart's size in pixels, its ranges, temperature in degrees and water content in grams per kg of dry air, and the
# values read off its axes.
n_rows, n_columns = 300, 360
temperature_range, water_range = (5.0, 35.0), (0.0, 25.0)
temperature_ticks, water_ticks = np.array([10, 20, 30]), np.array([5, 10, 15, 20])
# The grid the levels are found on; the relative humidities and enthalpies of the lines.
grid_temperatures, grid_water = np.linspace(5.0, 35.0, 150), np.linspace(0.0, 0.025, 150)
humidity_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
enthalpy_levels = np.arange(20.0, 110.0, 10.0)
# Grams per kilogram; how far a direction's arrow reaches in pixels; line widths.
grams_per_kilogram = 1000.0
arrow_pixels = 25.0
# Where the saturation curve's name is written, in degrees and grams per kg.
saturation_label = (27.5, 22.5)
thin_width, bold_width = 1.0, 2.0
# The apparatus's picture, pixels across.
apparatus_pixels = 240


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
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-10), 0.04, apparatus_pixels)
    return solids.draw([(tray, art.GREY), (coils, art.BLUE), (reheater, art.ORANGE),
                        (parcel, art.thermal_colour(warmth)), (collected, art.BLUE)], view, art.WHITE, inks=[],
                       translucent=[], meshes=[])


class Chart:
    """The psychrometric chart's lines, sampled once, and the drawing of the air's path on it."""

    def __init__(self, relative_humidity_of: Function, enthalpy_of: Function, mv) -> None:
        self.diagram = art.Diagram(temperature_range, water_range, (temperature_range[0], 0.0), n_rows, n_columns,
                                   ("T", "w"))
        grid = mv.t.dual() * grid_temperatures[None, :] + mv.w.dual() * grid_water[:, None]
        relative = relative_humidity_of(grid).to_array()
        # The enthalpy only where the air holds its water as vapour.
        heat = np.where(relative <= 1, enthalpy_of(grid).to_array(), np.nan)
        scaled = lambda lines: [(line[:, 0], grams_per_kilogram * line[:, 1]) for line in lines]
        humidities = [line for _, line in art.level_lines(grid_temperatures, grid_water, relative, humidity_levels)]
        saturation = [line for _, line in art.level_lines(grid_temperatures, grid_water, relative, np.ones(1))]
        enthalpies = [[line for _, line in art.level_lines(grid_temperatures, grid_water, heat, np.array([level]))]
                      for level in enthalpy_levels]
        # Above the saturation curve: the region up to the chart's top left corner.
        top = [(np.concatenate([line[:, 0], [temperature_range[1], temperature_range[0]]]),
                np.concatenate([grams_per_kilogram * line[:, 1], [water_range[1], water_range[1]]]))
               for line in saturation]
        tick_layers, self.tick_labels = art.ticks(self.diagram, temperature_ticks, water_ticks)
        self.layers = ([art.areas(self.diagram, x, np.clip(y, *water_range), art.GREY, 0.18) for x, y in top]
                       + art.polylines(self.diagram, scaled(humidities), thin_width, art.FAMILIES["humidity"], 1.0,
                                       art.SOLID)
                       + art.polylines(self.diagram, [line for lines in enthalpies for line in scaled(lines)],
                                       thin_width, art.FAMILIES["enthalpy"], 1.0, art.SOLID)
                       + art.polylines(self.diagram, scaled(saturation), 1.8, art.BOUNDARY, 1.0, art.SOLID)
                       + tick_layers)
        # Each enthalpy line's value at its upper end, where it meets the saturation curve; each humidity line's at its
        # right end.
        self.labels = [art.Label(saturation_label[0], saturation_label[1], "saturation", art.BOUNDARY, art.LABEL_SIZE,
                                 "lb", (0, 2))]
        for level, lines in zip(enthalpy_levels, enthalpies):
            for x, y in scaled(lines):
                top = np.argmax(y)
                if temperature_range[0] < x[top] and y[top] < water_range[1] - 1:
                    self.labels.append(art.Label(x[top], y[top], f"{level:g}", art.FAMILIES["enthalpy"], art.LABEL_SIZE,
                                                 "rb", (-2, 2)))
        for level in humidity_levels:
            for x, y in scaled([line for _, line in art.level_lines(grid_temperatures, grid_water, relative,
                                                                     np.array([level]))]):
                right = np.argmax(x)
                if y[right] < water_range[1] - 1:
                    self.labels.append(art.Label(x[right], y[right], f"{level:.0%}", art.FAMILIES["humidity"],
                                                 art.LABEL_SIZE, "rm", (-2, 0)))

    @staticmethod
    def values(states: Extensor) -> np.ndarray:
        """The states' temperatures and water contents in grams per kg `[..., 2]`, read by pairing."""
        mv = states.context.multivector
        return np.asarray((stack((mv.t, mv.w)) & states[..., None]).to_array()) * np.array([1, grams_per_kilogram])

    def picture(self, path: np.ndarray, travelled: int, states: np.ndarray, toward: np.ndarray) -> np.ndarray:
        layers = (self.layers
                  + art.polylines(self.diagram, [(path[:, 0], path[:, 1])], thin_width, art.ORANGE, 0.6, art.SOLID)
                  + art.polylines(self.diagram, [(path[:travelled + 1, 0], path[:travelled + 1, 1])], bold_width,
                                  art.ORANGE, 1.0, art.SOLID)
                  + art.arrows(self.diagram, states[:, 0], states[:, 1], toward[:, 0], toward[:, 1], arrow_pixels, 1.6,
                               art.GREEN)
                  + art.dots(self.diagram, path[travelled, 0], path[travelled, 1], art.ORANGE))
        return art.picture(self.diagram, layers, self.labels + self.tick_labels)


def conditioning(relative_humidity_of: Function, enthalpy_of: Function, path: Extensor, states: Extensor,
                 directions: Extensor, passing: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The apparatus and the chart at each state of the air along its path."""
    first = next(passing)
    chart = Chart(relative_humidity_of, enthalpy_of, first.context.multivector)
    traced, places, toward = chart.values(path), chart.values(states), chart.values(states + directions)
    for travelled, state in enumerate(chain([first], passing)):
        now = chart.values(state)
        warmth = (now[0] - traced[:, 0].min()) / np.ptp(traced[:, 0])
        liquid = (traced[0, 1] - now[1]) / (traced[0, 1] - traced[-1, 1])
        yield art.beside([apparatus(travelled / (len(traced) - 1), warmth, liquid),
                          chart.picture(traced, travelled, places, toward)])


def draw_chart(relative_humidity_of: Function, enthalpy_of: Function, states: Extensor, directions: Extensor):
    """The chart with the directions the air moves in at the inlet and at the coil's outlet."""
    chart = Chart(relative_humidity_of, enthalpy_of, states.context.multivector)
    places = chart.values(states)
    return art.figure(chart.picture(places[:1], 0, places, chart.values(states + directions)))
