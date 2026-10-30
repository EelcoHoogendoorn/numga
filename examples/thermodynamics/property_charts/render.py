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

from numga import Extensor, concatenate, stack, where
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
# Grams per kilogram; how far a direction's arrow reaches in pixels.
grams_per_kilogram = 1000.0
arrow_pixels = 25.0
# Where the saturation curve's name is written, in degrees and grams per kg.
saturation_label = (27.5, 22.5)
# The apparatus's picture, pixels across.
apparatus_pixels = 240


def apparatus(fraction: float, warmth: float, liquid: solids.Scalar) -> np.ndarray:
    """The duct: the cooling coil, its tray of collected water, the reheater, and the parcel of air passing them."""
    mv = solids.mv
    tray = art.box(mv.point([0, 0, 0.1]), np.array([2.35, 0.6, 0.08]))
    coil_centres = mv.point([[-0.8 + offset, 0, 0.8] for offset in (-0.25, 0, 0.25)])
    coils = solids.boxes(solids.translator(coil_centres - solids.origin), np.array([0.045, 0.5, 0.55]))
    reheater = art.box(mv.point([1.1, 0, 0.8]), np.array([0.12, 0.45, 0.45]))
    # The parcel a fraction of the way along the duct, from its entry to its exit.
    entry, far_end = mv.point([-2, -0.65, 0.82]), mv.point([2, -0.65, 0.82])
    parcel = solids.ball(entry + (far_end - entry) * fraction, 0.23)[None, None]
    # The collected water, resting on the tray's floor and as deep as it has filled.
    depth = 0.01 + 0.2 * liquid
    collected = art.box(mv.point([-0.8, -0.72, 0.15]) + art.rise * (depth / 2), stack((mv.scalar([0.45]), mv.scalar([0.2]), depth / 2)))
    corners = mv.point(np.array([[x, y, z] for x in (-2.5, 2.5) for y in (-1.0, 0.65) for z in (0, 1.5)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-10), 0.04, apparatus_pixels)
    return solids.draw([(tray, art.grey), (coils, art.blue), (reheater, art.orange),
                        (parcel, art.thermal_colour(warmth)), (collected, art.blue)], view, art.white, inks=[],
                       translucent=[], meshes=[])


def chart_points(temperatures: art.Scalar, water: art.Scalar) -> art.Point:
    """Places on the chart of temperatures and water contents in kg per kg of dry air `[...]`, the water in grams."""
    return art.chart_points(temperatures, grams_per_kilogram * water)


class Chart:
    """The psychrometric chart's lines, sampled once, and the drawing of the air's path on it."""

    def __init__(self, relative_humidity_of: Function, enthalpy_of: Function, mv) -> None:
        self.diagram = art.Diagram(temperature_range, water_range, (temperature_range[0], 0.0), n_rows, n_columns,
                                   ("T", "w"))
        grid = mv.t.dual() * grid_temperatures[None, :] + mv.w.dual() * grid_water[:, None]
        relative = relative_humidity_of(grid).to_array()
        # The enthalpy only where the air holds its water as vapour.
        heat = np.where(relative <= 1, enthalpy_of(grid).to_array(), np.nan)
        charted = lambda levels: [chart_points(*line.T) for _, line in levels]
        humidities = [charted(art.level_lines(grid_temperatures, grid_water, relative, np.array([level])))
                      for level in humidity_levels]
        saturation = charted(art.level_lines(grid_temperatures, grid_water, relative, np.ones(1)))
        enthalpies = [charted(art.level_lines(grid_temperatures, grid_water, heat, np.array([level])))
                      for level in enthalpy_levels]
        # Above the saturation curve: the region up to the chart's top left corner, the curve's samples past the top
        # dropped onto it, along the water content.
        top_edge = art.mv.y - art.mv.w * water_range[1]                                       # [] Plane+
        corners = art.chart_points(np.array(temperature_range[::-1]), water_range[1])            # [2] Point-
        tops = [concatenate([where((top_edge & curve) > 0, art.planar.unit((curve & art.mv.y.dual()) ^ top_edge),
                                   curve), corners]) for curve in saturation]
        tick_layers, self.tick_labels = art.ticks(self.diagram, temperature_ticks, water_ticks)
        self.layers = ([art.areas(self.diagram, top, art.grey, 0.18) for top in tops]
                       + art.polylines(self.diagram, [curve for curves in humidities for curve in curves], art.thin,
                                       art.families["humidity"], 1.0, art.SOLID)
                       + art.polylines(self.diagram, [curve for curves in enthalpies for curve in curves], art.thin,
                                       art.families["enthalpy"], 1.0, art.SOLID)
                       + art.polylines(self.diagram, saturation, art.medium, art.boundary, 1.0, art.SOLID)
                       + tick_layers)
        # Each enthalpy line's value at its upper end, where it meets the saturation curve; each humidity line's at its
        # right end; each only well inside the chart.
        self.labels = [art.Label(art.chart_points(*saturation_label), "saturation", art.boundary, art.text_size,
                                 "lb", (0, 2))]
        for level, curves in zip(enthalpy_levels, enthalpies):
            for curve in curves:
                top = curve[(art.mv.y & curve).argmax()]                                      # [] Point-
                if temperature_range[0] < (art.mv.x & top) and (art.mv.y & top) < water_range[1] - 1:
                    self.labels.append(art.Label(top, f"{level:g}", art.families["enthalpy"], art.tick_size, "rb",
                                                 (-2, 2)))
        for level, curves in zip(humidity_levels, humidities):
            for curve in curves:
                right = curve[(art.mv.x & curve).argmax()]                                    # [] Point-
                if (art.mv.y & right) < water_range[1] - 1:
                    self.labels.append(art.Label(right, f"{level:.0%}", art.families["humidity"], art.tick_size, "rm",
                                                 (-2, 0)))

    @staticmethod
    def points(states: Extensor) -> art.Point:
        """The states on the chart, their temperatures and water contents read by pairing `[...]`."""
        mv = states.context.multivector
        return chart_points(art.context.include_from_algebra(mv.t & states), art.context.include_from_algebra(mv.w & states))

    def picture(self, path: art.Point, travelled: int, states: art.Point, toward: art.Point) -> np.ndarray:
        layers = (self.layers
                  + art.polylines(self.diagram, [path], art.thin, art.orange, 0.6, art.SOLID)
                  + art.polylines(self.diagram, [path[:travelled + 1]], art.bold, art.orange, 1.0, art.SOLID)
                  + art.arrows(self.diagram, states, toward, arrow_pixels, art.medium, art.green)
                  + art.dots(self.diagram, path[travelled][None], art.orange))
        return art.picture(self.diagram, layers, self.labels + self.tick_labels)


def conditioning(relative_humidity_of: Function, enthalpy_of: Function, path: Extensor, states: Extensor,
                 directions: Extensor, passing: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The apparatus and the chart at each state of the air along its path."""
    first = next(passing)
    mv = first.context.multivector
    chart = Chart(relative_humidity_of, enthalpy_of, mv)
    traced, places, toward = chart.points(path), chart.points(states), chart.points(states + directions)
    temperatures, water = mv.t & path, mv.w & path
    cold, hot = temperatures.min(), temperatures.max()
    for travelled, state in enumerate(chain([first], passing)):
        warmth = float((((mv.t & state) - cold) / (hot - cold)).to_array())
        liquid = solids.context.include_from_algebra((water[0] - (mv.w & state)) / (water[0] - water[-1]))
        yield art.beside([apparatus(travelled / (len(temperatures) - 1), warmth, liquid),
                          chart.picture(traced, travelled, places, toward)])


def draw_chart(relative_humidity_of: Function, enthalpy_of: Function, states: Extensor, directions: Extensor):
    """The chart with the directions the air moves in at the inlet and at the coil's outlet."""
    chart = Chart(relative_humidity_of, enthalpy_of, states.context.multivector)
    places = chart.points(states)
    return art.figure(chart.picture(places[:1], 0, places, chart.points(states + directions)))
