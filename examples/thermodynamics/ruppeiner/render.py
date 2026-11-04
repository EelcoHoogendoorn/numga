"""The fluctuation metric of a van der Waals fluid, drawn as the ellipses of equal cost about its states.

The left chart is the plane of temperature and volume. Below the black spinodal the uniform fluid is unstable, shaded;
above it, the teal ellipses are the variations of equal cost about a lattice of states, and the grey-blue curves are
the level lines of the metric's scalar curvature, crowding toward the critical point, the dot where the spinodal peaks.
The moving state's ellipse is orange: as it cools toward the critical point, fluctuations of the volume grow cheap and
its ellipse stretches along the volume. The right chart holds the isotherms and the moving state's isotherm with its
tangent, whose slope, the mechanical part of the cost times minus the temperature, flattens.

The left chart's two quantities are readings of the fluid's state, so it draws the states as they are, read by the
planes of volume and of temperature. Its ellipses are conics of a chart, a projective plane, drawn where their forms
vanish under a camera of that plane framed alike: an ellipse about a state needs the weight a projective plane gives
its points. The right chart's pressure is a function of the state, and that chart is a chart.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from numga.backend.jax import derivative
from examples.thermodynamics import rendering as art
from examples.util.wgsl import quadrics

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
# The isotherms' temperatures; the half span of the tangent in volume.
isotherms = np.array([0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0])
tangent_span = 0.35


def chart_points(states: Extensor) -> art.Point:
    """States on the chart of temperature up against volume across, read by pairing `[...]`."""
    mv = states.context.multivector
    return art.chart_points(art.context.include_from_algebra(mv.v & states), art.context.include_from_algebra(mv.t & states))


def cost_quadrics(metric_of: Function, states: Extensor) -> art.Quadric:
    """The quadrics of the chart's plane where the cost of a variation about each state `[...]` is the cost level:
    the chart's two readings less their values at the state pair a chart point with its variation's volume and
    temperature, weighed by the metric's entries, its image of each of those variations paired with the other. Here
    the metric leaves the fluid's algebra for the chart's."""
    mv, Point = states.context.multivector, art.Point
    form = metric_of(states)                                                    # [...] Gradient+ <- State-
    steps = stack((mv.v.dual(), mv.t.dual()))                                 # [quantities] State-
    entries = art.context.include_from_algebra(form[..., None, None](steps[:, None]) & steps[None, :])
    centres = chart_points(states)                                              # [...] Point-
    across = art.mv.x - art.mv.w * (art.mv.x & centres)                         # [...] Plane+
    up = art.mv.y - art.mv.w * (art.mv.y & centres)                             # [...] Plane+
    readings = stack((across, up), axis=-1)                                   # [..., quantities] Plane+
    cost = (readings[..., :, None] * entries * (readings[..., None, :] & Point)).sum(axis=(-2, -1))
    return cost - art.mv.w * cost_level * (art.mv.w & Point)


class Charts:
    """The plane's lines, regions and lattice of ellipses, sampled once, and the drawing of a state on both charts."""

    def __init__(self, metric_of: Function, curvature_of: Function, pressure_of: Function, mv) -> None:
        self.metric_of, self.pressure_of, self.mv = metric_of, pressure_of, mv
        self.plane = art.Diagram(volume_range, temperature_range, (0.0, temperature_range[0]), n_rows, n_columns,
                                 ("V", "T"), (mv.v, mv.t), (0 * mv.v, 1.0))
        # The camera of the chart the ellipses are conics of, framed as the plane's own.
        self.conic_camera = self.plane.reading(art.mv.x, art.mv.y, art.mv.w, 0.0)
        self.isotherm = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"),
                                    (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        grid = mv.v.dual() * grid_volumes[None, :] + mv.t.dual() * grid_temperatures[:, None]
        # The mechanical part of the cost, its value on a variation of the volume alone; the spinodal where it vanishes.
        mechanical = metric_of(grid)(mv.v.dual()) & mv.v.dual()                   # [temperatures, volumes] Scalar+
        spinodal = art.Levels(grid, np.asarray(mechanical.to_array()), np.zeros(1))
        # The curvature, known where the fluid is stable.
        curvature = art.Levels(grid, np.where(mechanical > 0, curvature_of(grid).to_array(), np.nan), curvature_levels)
        # Below the spinodal, where the mechanical part is negative, unstable.
        unstable = [art.sides(self.plane, spinodal, np.array([np.append(art.grey, 0.25), np.zeros(4)]))]
        lattice = (mv.v.dual() * lattice_volumes[None, :]
                   + mv.t.dual() * lattice_temperatures[:, None]).reshape(-1)    # [lattice] State-
        # The lattice's states where every variation costs: the metric, carried back from gradients to variations by
        # the complement, has a positive spectrum there.
        stable = lattice[(metric_of(lattice).dual().eigvalsh() > 0).all(axis=-1)]   # [stable] State-
        conics = cost_quadrics(metric_of, stable)                                       # [stable] Plane+ <- Point-
        self.curvature = curvature
        self.lattice = stable
        self.plane_layers = (unstable
                             + art.level_family(self.plane, curvature, "level", art.thin)
                             + [quadrics.curves(conics, art.thin, art.families["entropy"], self.conic_camera)]
                             + art.level_lines(self.plane, spinodal, art.boundary, art.medium)
                             + art.dots(self.plane, self.plane.at(np.ones(1), np.ones(1)), art.boundary))
        self.volumes = np.linspace(0.38, volume_range[1], 240)
        family = mv.v.dual() * self.volumes[None, :] + mv.t.dual() * isotherms[:, None]   # [isotherms, samples] State-
        isotherm_lines = self.isotherm_points(family)                                   # [isotherms, samples] Point-
        shades = np.where((isotherms == 1.0)[:, None], art.boundary, art.families["temperature"])
        self.isotherm_layers = art.within(self.isotherm, isotherm_lines, art.thin, shades, 1.0)
        self.isotherm_lines = [isotherm_lines[line] for line in range(isotherm_lines.shape[0])]

    def isotherm_points(self, states: Extensor) -> art.Point:
        """States on the chart of pressure up against volume across `[...]`."""
        return art.chart_points(art.context.include_from_algebra(self.mv.v & states), art.context.include_from_algebra(self.pressure_of(states)))

    def pictures(self, state: Extensor, path: list[art.Point]) -> list[np.ndarray]:
        mv = self.mv
        conic = cost_quadrics(self.metric_of, state[None])                              # [1] Plane+ <- Point-
        plane = (self.plane_layers
                 + art.polylines(self.plane, [stack(path)], art.bold, art.orange, 0.6, art.SOLID)
                 + [quadrics.regions(conic[:, None], art.orange, self.conic_camera, 0.25),
                    quadrics.curves(conic, art.medium, art.orange, self.conic_camera)]
                 + art.dots(self.plane, state[None], art.orange))
        # The state's own isotherm, and its tangent there: a unit of volume, and the change of the pressure along it.
        own = self.isotherm_points(mv.v.dual() * self.volumes + mv.t.dual() * (mv.t & state))   # [samples] Point-
        on_isotherm = self.isotherm_points(state)                                        # [] Point-
        tangent = art.chart_directions(1.0, art.context.include_from_algebra(derivative(self.pressure_of)(state)(mv.v.dual())))   # [] Point-
        reach = art.context.include_from_algebra(mv.v & state) + tangent_span * np.array([-1, 1])
        isotherm = (self.isotherm_layers
                    + art.polylines(self.isotherm, [own], art.bold, art.ink, 1.0, art.SOLID)
                    + art.polylines(self.isotherm, [art.tangent_lines(self.isotherm, on_isotherm, tangent, reach)],
                                    art.medium, art.orange, 1.0, art.SOLID)
                    + art.dots(self.isotherm, on_isotherm[None], art.orange))
        labels = [art.Label(self.plane.at(1.0, 1.0), "critical point", art.boundary, art.text_size, "lt", (6, -4)),
                  art.Label(self.plane.at(0.45, 0.75), "unstable", art.ink, art.text_size, "lb", (0, 2))]
        names = [art.Floating(art.crossings(self.curvature, 6), "R = const", art.families["level"], art.text_size,
                              "lb"),
                 art.Floating(self.lattice, "equal cost", art.families["entropy"], art.text_size, "lt")]
        isotherm_names = [art.Floating(art.along(self.isotherm_lines, 8), "T = const", art.families["temperature"],
                                       art.text_size, "lb")]
        return [art.floated(self.plane, art.picture(self.plane, plane, labels), names),
                art.floated(self.isotherm, art.picture(self.isotherm, isotherm, []), isotherm_names)]


def fluctuations(metric_of: Function, curvature_of: Function, pressure_of: Function,
                 states: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The two charts at each state as it comes."""
    first = next(states)
    charts = Charts(metric_of, curvature_of, pressure_of, first.context.multivector)
    path = []
    for state in chain([first], states):
        path.append(art.on_numpy(state))
        yield art.beside(charts.pictures(state, path))


def draw_landscape(metric_of: Function, curvature_of: Function, pressure_of: Function, state: Extensor):
    """The plane of temperature and volume with its ellipses, curvature and spinodal, and a state's ellipse."""
    charts = Charts(metric_of, curvature_of, pressure_of, state.context.multivector)
    return art.figure(charts.pictures(state, [art.on_numpy(state)])[0])


def scaling(states: Extensor, curvature_of: Function):
    """The scalar curvature against the distance above the critical temperature, both on logarithmic scales, beside a
    dashed line falling as the inverse square."""
    mv = states.context.multivector
    distance = np.log10((mv.t & states).to_array() - 1)
    curvature = np.log10(curvature_of(states).to_array())
    diagram = art.Diagram((-3.0, 0.0), (-0.5, 5.5), (-3.0, -0.5), n_rows, n_columns, ("log(T - 1)", "log R"),
                          (art.mv.x, art.mv.y), (art.mv.w, 0.0))
    # The line falling as the inverse square, through a point a little above the curve's first.
    through, falling = art.chart_points(distance[0], curvature[0] + 0.4), art.chart_directions(1.0, -2.0)
    reference = art.tangent_lines(diagram, through, falling, np.linspace(-3.0, 0.0, 60))   # [samples] Point-
    layers = (art.polylines(diagram, [reference], art.medium, art.grey, 1.0, (4.0, 3.0))
              + art.polylines(diagram, [art.chart_points(distance, curvature)], art.bold, art.blue, 1.0, art.SOLID))
    labels = [art.Label(art.tangent_lines(diagram, through, falling, np.array([-1.0]))[0], "slope -2", art.grey,
                        art.text_size, "lb", (6, 2))]
    return art.figure(art.picture(diagram, layers, labels))
