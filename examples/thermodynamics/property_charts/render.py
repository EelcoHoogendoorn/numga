"""Moist air cooled, dried, and reheated beside its psychrometric chart.

The chart plots temperature horizontally and water content in grams per kilogram of dry air vertically, with blue curves
of equal relative humidity, a black saturation curve, and brown lines of equal enthalpy labelled where they meet
saturation. Air above the shaded saturation boundary cannot hold all that water as vapour. Along the conditioning path,
the state moves left at fixed water content to saturation, slides down the saturation curve while water condenses into
the tray, and moves right at fixed water content through the reheater. Green arrows mark the constrained directions per
degree warmer at the inlet and at the coil outlet.

Because temperature and water content are direct pairings on the state space, the diagram reads `State` values directly
through the covectors `t` and `w` without intermediate chart coordinates, and samples the level curves of relative
humidity and enthalpy across a grid of states.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import stack
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids
from examples.thermodynamics.property_charts import core

mv = core.mv
StateFunction = Callable[[core.State], core.Scalar]

# Chart dimensions in pixels, axis ranges (degrees Celsius and grams of water per kg of dry air), and tick values.
n_rows, n_columns = 300, 360
temperature_range, water_range = (5.0, 35.0), (0.0, 25.0)
temperature_ticks, water_ticks = np.array([10, 20, 30]), np.array([5, 10, 15, 20])
# Sampling grid for contour extraction and level values for relative humidity and enthalpy.
n_grid = 150
grid_temperatures, grid_water = np.linspace(5.0, 35.0, n_grid), np.linspace(0.0, 0.025, n_grid)
humidity_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
enthalpy_levels = np.arange(20.0, 110.0, 10.0)
grams_per_kilogram = 1000.0
arrow_pixels = 25.0
saturation_label = (27.5, 22.5)
n_apparatus_pixels = 240


def apparatus(fraction: float, warmth: float, liquid: solids.Scalar) -> np.ndarray:
    """Rendered duct showing the cooling coil, condensate tray, reheater, and air parcel at `fraction` along the duct."""
    mv = solids.mv
    tray = art.box(mv.point([0, 0, 0.1]), np.array([2.35, 0.6, 0.08]))
    coil_centres = mv.point([[-0.8 + offset, 0, 0.8] for offset in (-0.25, 0, 0.25)])
    coils = solids.boxes(solids.translator(coil_centres - solids.origin), np.array([0.045, 0.5, 0.55]))
    reheater = art.box(mv.point([1.1, 0, 0.8]), np.array([0.12, 0.45, 0.45]))
    # Place the air parcel `fraction` of the way from the duct inlet to its exit.
    entry, far_end = mv.point([-2, -0.65, 0.82]), mv.point([2, -0.65, 0.82])
    parcel = solids.ball(entry + (far_end - entry) * fraction, 0.23)[None, None]
    # Collect condensed liquid water on the floor of the tray below the cooling coil.
    depth = 0.01 + 0.2 * liquid
    collected_water = art.box(mv.point([-0.8, -0.72, 0.15]) + art.rise * (depth / 2),
                              stack((mv.scalar([0.45]), mv.scalar([0.2]), depth / 2)))
    corners = mv.point(np.array([[x, y, z] for x in (-2.5, 2.5) for y in (-1.0, 0.65) for z in (0, 1.5)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-10), 0.04, n_apparatus_pixels)
    return solids.draw([(tray, art.grey), (coils, art.blue), (reheater, art.orange),
                        (parcel, art.thermal_colour(warmth)), (collected_water, art.blue)], view, art.white, inks=[],
                       translucent=[], meshes=[])


class Chart:
    """Psychrometric chart background sampled once, ready to draw the air's path and tangent directions."""

    def __init__(self, relative_humidity_of: StateFunction, enthalpy_of: StateFunction, mv) -> None:
        # Read temperature in degrees Celsius horizontally and water content in grams per kg of dry air vertically.
        across, up = mv.t, mv.w * grams_per_kilogram                                           # [] Potential+, Potential+
        self.diagram = art.Diagram(temperature_range, water_range, (temperature_range[0], 0.0), n_rows, n_columns,
                                   ("T", "w"), (across, up), (0 * across, 1.0))
        grid = mv.t.dual() * grid_temperatures[None, :] + mv.w.dual() * grid_water[:, None]    # [n_grid, n_grid] State-
        relative = np.asarray(relative_humidity_of(grid).to_array())
        humidities = art.Levels(grid, relative, humidity_levels)
        saturation = art.Levels(grid, relative, np.ones(1))
        # Restrict enthalpy contours to unsaturated and saturated vapour states (`relative <= 1`).
        enthalpies = art.Levels(grid, np.where(relative <= 1, np.asarray(enthalpy_of(grid).to_array()), np.nan),
                                enthalpy_levels)
        tick_layers, self.tick_labels = art.ticks(self.diagram, temperature_ticks, water_ticks)
        # Shade the supersaturated region above the saturation curve where relative humidity exceeds one.
        self.layers = ([art.sides(self.diagram, saturation, np.array([np.zeros(4), np.append(art.grey, 0.18)]))]
                       + art.level_family(self.diagram, humidities, "humidity", art.thin)
                       + art.level_family(self.diagram, enthalpies, "enthalpy", art.thin)
                       + art.level_lines(self.diagram, saturation, art.boundary, art.medium)
                       + tick_layers)
        # Label each enthalpy line where it meets the saturation curve and each humidity curve at the right border.
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

    def picture(self, path: core.State, step_index: int, key_states: core.State, toward: core.State) -> np.ndarray:
        layers = (self.layers
                  + art.polylines(self.diagram, [path], art.thin, art.orange, 0.6, art.SOLID)
                  + art.polylines(self.diagram, [path[:step_index + 1]], art.bold, art.orange, 1.0, art.SOLID)
                  + art.arrows(self.diagram, key_states, toward, arrow_pixels, art.medium, art.green)
                  + art.dots(self.diagram, path[step_index][None], art.orange))
        return art.picture(self.diagram, layers, self.labels + self.tick_labels)


def conditioning(relative_humidity_of: StateFunction, enthalpy_of: StateFunction, path: core.State,
                 key_states: core.State, directions: core.State, states: Iterator[core.State]) -> Iterator[np.ndarray]:
    """Yields side-by-side frames of the duct apparatus and psychrometric chart along the air's path."""
    first = next(states)
    chart = Chart(relative_humidity_of, enthalpy_of, mv)
    temperatures, water = mv.t & path, mv.w & path                                             # [...] Scalar+, Scalar+
    path, key_states, toward = (art.on_numpy(values) for values in (path, key_states, key_states + directions))
    cold, hot = temperatures.min(), temperatures.max()
    for step_index, state in enumerate(chain([first], states)):
        warmth = float((((mv.t & state) - cold) / (hot - cold)).to_array())
        liquid = solids.context.include_from_algebra((water[0] - (mv.w & state)) / (water[0] - water[-1]))
        yield art.beside([apparatus(step_index / (len(temperatures) - 1), warmth, liquid),
                          chart.picture(path, step_index, key_states, toward)])


def draw_chart(relative_humidity_of: StateFunction, enthalpy_of: StateFunction, key_states: core.State,
               directions: core.State):
    """Psychrometric chart figure showing the constrained directions at the inlet and coil outlet."""
    chart = Chart(relative_humidity_of, enthalpy_of, mv)
    return art.figure(chart.picture(key_states[:1], 0, key_states, key_states + directions))
