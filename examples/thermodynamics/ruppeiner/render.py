"""The fluctuation metric of a van der Waals fluid, drawn as the ellipses of equal cost about its states.

The left chart is the plane of temperature and volume. Below the black spinodal the uniform fluid is unstable, shaded;
above it, the teal ellipses are the variations of equal cost about a lattice of states, and the grey-blue curves are
the level lines of the metric's scalar curvature, crowding toward the critical point, the dot where the spinodal peaks.
The moving state's ellipse is orange: as it cools toward the critical point, fluctuations of the volume grow cheap and
its ellipse stretches along the volume. The right chart holds the isotherms and the moving state's isotherm with its
tangent, whose slope, the mechanical part of the cost times minus the temperature, flattens.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art
from examples.util.wgsl import planar

Function = Callable[[Extensor], Extensor]

# The charts' size in pixels and their ranges: volume and temperature, volume and pressure.
n_rows, n_columns = 310, 340
volume_range, temperature_range = (0.0, 3.0), (0.7, 2.0)
pressure_range = (0.0, 1.0)
# The grid of states the curvature and the spinodal are found on, and the curvature's levels.
n_grid = 160
grid_volumes, grid_temperatures = np.linspace(0.36, 3.0, n_grid), np.linspace(0.7, 2.0, n_grid)
curvature_levels = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
# The lattice of states whose ellipses are drawn, and the cost the ellipses are drawn at.
lattice_volumes, lattice_temperatures = np.arange(0.5, 2.9, 0.4), np.arange(0.95, 2.0, 0.2)
cost_level = 0.0036
# The isotherms' temperatures; the half span of the tangent in volume; line widths.
isotherms = np.array([0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0])
tangent_span = 0.35
thin_width, bold_width = 1.0, 2.2


def coefficients(metric_of: Function, states: Extensor) -> np.ndarray:
    """The metric's entries on variations of volume and of temperature, in that order `[..., 2, 2]`: the metric's image
    of each variation paired with the other."""
    mv = states.context.multivector
    steps = [mv.v.dual(), mv.t.dual()]
    form = metric_of(states)
    return np.stack([np.stack([(form(row) & column).to_array() for column in steps], axis=-1) for row in steps],
                    axis=-2)


class Charts:
    """The plane's lines, regions and lattice of ellipses, sampled once, and the drawing of a state on both charts."""

    def __init__(self, metric_of: Function, curvature_of: Function, pressure_of: Function, mv) -> None:
        self.metric_of, self.pressure_of, self.mv = metric_of, pressure_of, mv
        self.plane = art.Diagram(volume_range, temperature_range, (0.0, temperature_range[0]), n_rows, n_columns,
                                 ("V", "T"))
        self.isotherm = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"))
        grid = mv.v.dual() * grid_volumes[None, :] + mv.t.dual() * grid_temperatures[:, None]
        mechanical = coefficients(metric_of, grid)[..., 0, 0]
        stable = mechanical > 0
        curvature = np.where(stable, curvature_of(grid).to_array(), np.nan)
        spinodal = [line for _, line in art.level_lines(grid_volumes, grid_temperatures, mechanical, np.zeros(1))]
        unstable = [art.areas(self.plane, line[:, 0], line[:, 1], art.GREY, 0.25) for line in spinodal]
        lattice = mv.v.dual() * lattice_volumes[None, :] + mv.t.dual() * lattice_temperatures[:, None]
        forms = coefficients(metric_of, lattice).reshape(-1, 2, 2)
        kept = np.linalg.eigvalsh(forms).min(axis=-1) > 0
        x, y = np.meshgrid(lattice_volumes, lattice_temperatures)
        conics = art.ellipses(self.plane, x.reshape(-1)[kept], y.reshape(-1)[kept], forms[kept], cost_level)
        self.curvature_lines = [(line[:, 0], line[:, 1]) for _, line in
                                art.level_lines(grid_volumes, grid_temperatures, curvature, curvature_levels)]
        self.lattice = (x.reshape(-1)[kept], y.reshape(-1)[kept])
        self.plane_layers = (unstable
                             + art.polylines(self.plane, self.curvature_lines, thin_width, art.FAMILIES["level"], 1.0,
                                             art.SOLID)
                             + [planar.Curves(conics, 1.0, art.FAMILIES["entropy"], 1.0)]
                             + art.polylines(self.plane, [(line[:, 0], line[:, 1]) for line in spinodal], 1.6,
                                             art.BOUNDARY, 1.0, art.SOLID)
                             + art.dots(self.plane, 1.0, 1.0, art.BOUNDARY))
        volumes = np.linspace(0.38, volume_range[1], 240)
        family = mv.v.dual() * volumes[None, :] + mv.t.dual() * isotherms[:, None]
        pressures = pressure_of(family).to_array()
        shades = np.where((isotherms == 1.0)[:, None], art.BOUNDARY, art.FAMILIES["temperature"])
        self.isotherm_layers = art.within(self.isotherm, np.broadcast_to(volumes, pressures.shape), pressures,
                                          thin_width, shades, 1.0)
        self.volumes = volumes
        self.isotherm_lines = [(volumes, line) for line in pressures]

    def pictures(self, state: Extensor, path: np.ndarray) -> list[np.ndarray]:
        mv = self.mv
        volume, temperature = float((mv.v & state).to_array()), float((mv.t & state).to_array())
        form = coefficients(self.metric_of, state[None])
        conic = art.ellipses(self.plane, np.array([volume]), np.array([temperature]), form, cost_level)
        plane = (self.plane_layers
                 + art.polylines(self.plane, [(path[:, 0], path[:, 1])], bold_width, art.ORANGE, 0.6, art.SOLID)
                 + [planar.Regions(conic, art.ORANGE, 0.25), planar.Curves(conic, 1.8, art.ORANGE, 1.0)]
                 + art.dots(self.plane, volume, temperature, art.ORANGE))
        own = self.pressure_of(mv.v.dual() * self.volumes + mv.t.dual() * temperature).to_array()
        pressure = float(self.pressure_of(state).to_array())
        slope = float(np.interp(volume, self.volumes, np.gradient(own, self.volumes)))
        span = np.array([volume - tangent_span, volume + tangent_span])
        isotherm = (self.isotherm_layers
                    + art.polylines(self.isotherm, [(self.volumes, own)], bold_width, art.INK, 1.0, art.SOLID)
                    + art.polylines(self.isotherm, [(span, pressure + slope * (span - volume))], 1.6, art.ORANGE, 1.0,
                                    art.SOLID)
                    + art.dots(self.isotherm, volume, pressure, art.ORANGE))
        labels = [art.Label(1.0, 1.0, "critical point", art.BOUNDARY, art.LABEL_SIZE, "lt", (6, -4)),
                  art.Label(0.45, 0.75, "unstable", art.INK, art.LABEL_SIZE, "lb", (0, 2))]
        names = [art.Floating(*art.along(self.curvature_lines, 6), "R = const", art.FAMILIES["level"], art.LABEL_SIZE,
                              "lb"),
                 art.Floating(*self.lattice, "equal cost", art.FAMILIES["entropy"], art.LABEL_SIZE, "lt")]
        isotherm_names = [art.Floating(*art.along(self.isotherm_lines, 8), "T = const", art.FAMILIES["temperature"],
                                       art.LABEL_SIZE, "lb")]
        return [art.floated(self.plane, art.picture(self.plane, plane, labels), names),
                art.floated(self.isotherm, art.picture(self.isotherm, isotherm, []), isotherm_names)]


def fluctuations(metric_of: Function, curvature_of: Function, pressure_of: Function,
                 states: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The two charts at each state as it comes."""
    first = next(states)
    charts = Charts(metric_of, curvature_of, pressure_of, first.context.multivector)
    path = []
    for state in chain([first], states):
        path.append([float((charts.mv.v & state).to_array()), float((charts.mv.t & state).to_array())])
        yield art.beside(charts.pictures(state, np.array(path)))


def draw_landscape(metric_of: Function, curvature_of: Function, pressure_of: Function, state: Extensor):
    """The plane of temperature and volume with its ellipses, curvature and spinodal, and a state's ellipse."""
    mv = state.context.multivector
    charts = Charts(metric_of, curvature_of, pressure_of, mv)
    here = np.array([[float((mv.v & state).to_array()), float((mv.t & state).to_array())]])
    return art.figure(charts.pictures(state, here)[0])


def scaling(states: Extensor, curvature_of: Function):
    """The scalar curvature against the distance above the critical temperature, both on logarithmic scales, beside a
    dashed line falling as the inverse square."""
    mv = states.context.multivector
    distance = np.log10((mv.t & states).to_array() - 1)
    curvature = np.log10(curvature_of(states).to_array())
    diagram = art.Diagram((-3.0, 0.0), (-0.5, 5.5), (-3.0, -0.5), n_rows, n_columns, ("log(T - 1)", "log R"))
    across = np.linspace(-3.0, 0.0, 60)
    reference = curvature[0] - 2 * (across - distance[0]) + 0.4
    layers = (art.polylines(diagram, [(across, reference)], 1.2, art.GREY, 1.0, (4.0, 3.0))
              + art.polylines(diagram, [(distance, curvature)], bold_width, art.BLUE, 1.0, art.SOLID))
    labels = [art.Label(-1.0, float(np.interp(-1.0, across, reference)), "slope -2", art.GREY, art.LABEL_SIZE, "lb",
                        (6, 2))]
    return art.figure(art.picture(diagram, layers, labels))
