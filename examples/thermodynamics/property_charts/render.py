"""Moist air cooled, dried and reheated, beside its psychrometric chart.

The chart holds the air's temperature across and its water content up, with the lines of equal relative humidity,
blue, the saturation curve black, and the lines of equal enthalpy, brown, each marked with its value where it meets the
saturation curve. Above the saturation curve no air holds that much water as vapour. The air's path runs left at fixed
water content to the saturation curve, down along it while water condenses, and right at fixed water content through
the reheater; the arrows are the directions it moves in at the inlet and at the coil's outlet, solved for one degree
warmer with the held reading unchanged.

The chart's two quantities are readings of the air's state, so the chart draws the states as they are, read by the
planes of temperature and of water content. Its lines are level lines of the air's relative humidity and enthalpy,
functions the notebook hands over; this module only samples them.
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


class Chart:
    """The psychrometric chart's lines, sampled once, and the drawing of the air's path on it."""

    def __init__(self, relative_humidity_of: Function, enthalpy_of: Function, mv) -> None:
        # The air's states read by the planes of temperature and of water content, this in grams per kg of dry air.
        across, up = mv.t, mv.w * grams_per_kilogram                                           # [] Plane+, Plane+
        self.diagram = art.Diagram(temperature_range, water_range, (temperature_range[0], 0.0), n_rows, n_columns,
                                   ("T", "w"), (across, up), (0 * across, 1.0))
        grid = mv.t.dual() * grid_temperatures[None, :] + mv.w.dual() * grid_water[:, None]
        relative = np.asarray(relative_humidity_of(grid).to_array())
        humidities = art.Levels(grid, relative, humidity_levels)
        saturation = art.Levels(grid, relative, np.ones(1))
        # The enthalpy only where the air holds its water as vapour.
        enthalpies = art.Levels(grid, np.where(relative <= 1, np.asarray(enthalpy_of(grid).to_array()), np.nan),
                                enthalpy_levels)
        tick_layers, self.tick_labels = art.ticks(self.diagram, temperature_ticks, water_ticks)
        # Above the saturation curve, where the relative humidity exceeds one, shaded.
        self.layers = ([art.sides(self.diagram, saturation, np.array([np.zeros(4), np.append(art.grey, 0.18)]))]
                       + art.level_family(self.diagram, humidities, "humidity", art.thin)
                       + art.level_family(self.diagram, enthalpies, "enthalpy", art.thin)
                       + art.level_lines(self.diagram, saturation, art.boundary, art.medium)
                       + tick_layers)
        # Each enthalpy line's value at its upper end, where it meets the saturation curve; each humidity line's at its
        # right end; each only well inside the chart.
        self.labels = [art.Label(self.diagram.at(*saturation_label), "saturation", art.boundary, art.text_size,
                                 "lb", (0, 2))]
        for level, top in art.furthest(enthalpies, up):
            if temperature_range[0] < (across & top) and (up & top) < water_range[1] - 1:
                self.labels.append(art.Label(top, f"{level:g}", art.families["enthalpy"], art.tick_size, "rb",
                                             (-2, 2)))
        for level, right in art.furthest(humidities, across):
            if (up & right) < water_range[1] - 1:
                self.labels.append(art.Label(right, f"{level:.0%}", art.families["humidity"], art.tick_size, "rm",
                                             (-2, 0)))

    def picture(self, path: Extensor, travelled: int, states: Extensor, toward: Extensor) -> np.ndarray:
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
    temperatures, water = mv.t & path, mv.w & path
    path, states, toward = (art.on_numpy(values) for values in (path, states, states + directions))
    cold, hot = temperatures.min(), temperatures.max()
    for travelled, state in enumerate(chain([first], passing)):
        warmth = float((((mv.t & state) - cold) / (hot - cold)).to_array())
        liquid = solids.context.include_from_algebra((water[0] - (mv.w & state)) / (water[0] - water[-1]))
        yield art.beside([apparatus(travelled / (len(temperatures) - 1), warmth, liquid),
                          chart.picture(path, travelled, states, toward)])


def draw_chart(relative_humidity_of: Function, enthalpy_of: Function, states: Extensor, directions: Extensor):
    """The chart with the directions the air moves in at the inlet and at the coil's outlet."""
    chart = Chart(relative_humidity_of, enthalpy_of, states.context.multivector)
    return art.figure(chart.picture(states[:1], 0, states, states + directions))
