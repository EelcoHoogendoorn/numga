"""Renders the Ruppeiner fluctuation metric of a van der Waals fluid as equal-cost ellipses beside its isotherms.

The left chart shows the temperature–volume plane. Below the black spinodal curve the uniform fluid is unstable and
shaded grey; above it, teal ellipses mark variations of equal cost around a lattice of states, while grey-blue contours
trace the scalar curvature crowding toward the critical point at the peak of the spinodal. As the highlighted orange
state cools toward the critical point, volume fluctuations become cheap and its equal-cost ellipse stretches
horizontally. The right chart shows isotherms on the pressure–volume plane together with the orange tangent to the
moving state's isotherm, whose slope equals minus the temperature times the mechanical entry of the metric.

Because temperature and volume are direct readings of the state, the left diagram reads states in the fluid's own
algebra and draws each equal-cost ellipse as a projective conic on the chart plane.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from numga.render import quadrics
from examples.thermodynamics.ruppeiner import core

mv = core.mv
Function = Callable[[Extensor], Extensor]

n_rows, n_columns = 310, 340
volume_range, temperature_range = (0.0, 3.0), (0.7, 2.0)
pressure_range = (0.0, 1.0)
n_grid = 160
grid_volumes, grid_temperatures = np.linspace(0.36, 3.0, n_grid), np.linspace(0.7, 2.0, n_grid)
curvature_levels = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
lattice_volumes, lattice_temperatures = np.arange(0.5, 2.9, 0.4), np.arange(0.95, 2.0, 0.2)
n_lattice = len(lattice_volumes) * len(lattice_temperatures)
cost_level = 0.0036
isotherms = np.array([0.9, 1.0, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0])
n_isotherms, n_volumes = len(isotherms), 240
tangent_span = 0.35


def chart_points(states: Extensor) -> art.Point:
    """Places fluid states on the temperature–volume chart plane by regressive pairing."""
    return art.chart_points(art.context.include_from_algebra(mv.v & states),
                            art.context.include_from_algebra(mv.t & states))


def cost_quadrics(metric_of: Function, states: Extensor) -> art.Quadric:
    """Builds projective conics on the chart plane where variations around each state reach the fixed cost level."""
    Point = art.Point
    form = metric_of(states)                                                    # [...] Potential+ <- State-
    steps = stack((mv.v.dual(), mv.t.dual()))                                   # [2] State-
    entries = art.context.include_from_algebra(form[..., None, None](steps[:, None]) & steps[None, :])
    centres = chart_points(states)                                              # [...] Point-
    across = art.mv.x - art.mv.w * (art.mv.x & centres)                         # [...] Plane+
    up = art.mv.y - art.mv.w * (art.mv.y & centres)                             # [...] Plane+
    readings = stack((across, up), axis=-1)                                     # [..., 2] Plane+
    cost = (readings[..., :, None] * entries * (readings[..., None, :] & Point)).sum(axis=(-2, -1))
    return cost - art.mv.w * cost_level * (art.mv.w & Point)


class Charts:
    """Samples the spinodal, curvature contours, equal-cost lattice and isotherms once for composite rendering."""

    def __init__(self, metric_of: Function, curvature_of: Function, pressure_of: Function, mv) -> None:
        self.metric_of, self.pressure_of, self.mv = metric_of, pressure_of, mv
        self.plane = art.Diagram(volume_range, temperature_range, (0.0, temperature_range[0]), n_rows, n_columns,
                                 ("V", "T"), (mv.v, mv.t), (0 * mv.v, 1.0))
        self.conic_camera = self.plane.reading(art.mv.x, art.mv.y, art.mv.w, 0.0)
        self.isotherm = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"),
                                    (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        grid = mv.v.dual() * grid_volumes[None, :] + mv.t.dual() * grid_temperatures[:, None]
        # The spinodal curve marks where the mechanical fluctuation entry vanishes.
        with np.errstate(divide="ignore", invalid="ignore"):
            mechanical = metric_of(grid)(mv.v.dual()) & mv.v.dual()             # [n_grid, n_grid] Scalar+
            curvature_values = np.where(mechanical > 0, curvature_of(grid).to_array(), np.nan)
        spinodal = art.Levels(grid, np.asarray(mechanical.to_array()), np.zeros(1))
        curvature = art.Levels(grid, curvature_values, curvature_levels)
        unstable_region = [art.sides(self.plane, spinodal, np.array([np.append(art.grey, 0.25), np.zeros(4)]))]
        lattice = (mv.v.dual() * lattice_volumes[None, :]
                   + mv.t.dual() * lattice_temperatures[:, None]).reshape(-1)   # [n_lattice] State-
        # Retain only thermodynamically stable lattice sites where every metric eigenvalue is positive.
        positive_mask = (metric_of(lattice).dual().eigvalsh() > 0).all(axis=-1)
        stable_states = lattice[positive_mask]
        n_stable = stable_states.shape[0]
        conics = cost_quadrics(metric_of, stable_states)                        # [n_stable] Plane+ <- Point-
        self.curvature = curvature
        self.lattice = stable_states                                            # [n_stable] State-
        self.plane_layers = (unstable_region
                             + art.level_family(self.plane, curvature, "level", art.thin)
                             + [quadrics.curves(conics, art.thin, art.families["entropy"], self.conic_camera)]
                             + art.level_lines(self.plane, spinodal, art.boundary, art.medium)
                             + art.dots(self.plane, self.plane.at(np.ones(1), np.ones(1)), art.boundary))
        self.volumes = np.linspace(0.38, volume_range[1], n_volumes)
        family = mv.v.dual() * self.volumes[None, :] + mv.t.dual() * isotherms[:, None]   # [n_isotherms, n_volumes] State-
        isotherm_lines = self.isotherm_points(family)                           # [n_isotherms, n_volumes] Point-
        shades = np.where((isotherms == 1.0)[:, None], art.boundary, art.families["temperature"])
        self.isotherm_layers = art.within(self.isotherm, isotherm_lines, art.thin, shades, 1.0)
        self.isotherm_lines = [isotherm_lines[line] for line in range(isotherm_lines.shape[0])]

    def isotherm_points(self, states: Extensor) -> art.Point:
        """Places fluid states on the pressure–volume chart plane."""
        return art.chart_points(art.context.include_from_algebra(self.mv.v & states),
                                art.context.include_from_algebra(self.pressure_of(states)))

    def pictures(self, state: Extensor, path: list[art.Point]) -> list[np.ndarray]:
        mv = self.mv
        conic = cost_quadrics(self.metric_of, state[None])                      # [1] Plane+ <- Point-
        plane = (self.plane_layers
                 + art.polylines(self.plane, [stack(path)], art.bold, art.orange, 0.6, art.SOLID)
                 + [quadrics.regions(conic[:, None], art.orange, self.conic_camera, 0.25),
                    quadrics.curves(conic, art.medium, art.orange, self.conic_camera)]
                 + art.dots(self.plane, state[None], art.orange))
        state_isotherm = self.isotherm_points(
            mv.v.dual() * self.volumes + mv.t.dual() * (mv.t & state))          # [n_volumes] Point-
        state_point = self.isotherm_points(state)                               # [] Point-
        # Isothermal pressure slope equals minus the temperature times the mechanical metric entry.
        pressure_slope = -(mv.t & state) * (self.metric_of(state)(mv.v.dual()) & mv.v.dual())   # [] Scalar+
        tangent = art.chart_directions(1.0, art.context.include_from_algebra(pressure_slope))   # [] Point-
        reach = art.context.include_from_algebra(mv.v & state) + tangent_span * np.array([-1, 1])
        isotherm = (self.isotherm_layers
                    + art.polylines(self.isotherm, [state_isotherm], art.bold, art.ink, 1.0, art.SOLID)
                    + art.polylines(self.isotherm, [art.tangent_lines(self.isotherm, state_point, tangent, reach)],
                                    art.medium, art.orange, 1.0, art.SOLID)
                    + art.dots(self.isotherm, state_point[None], art.orange))
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
    """Streams side-by-side frames of the state-space fluctuation ellipse and the pressure–volume isotherm."""
    first = next(states)
    charts = Charts(metric_of, curvature_of, pressure_of, mv)
    path = []
    for state in chain([first], states):
        path.append(art.on_numpy(state))
        yield art.beside(charts.pictures(state, path))


def draw_landscape(metric_of: Function, curvature_of: Function, pressure_of: Function, state: Extensor):
    """Draws the temperature–volume plane with its spinodal, curvature contours, equal-cost lattice and one state."""
    charts = Charts(metric_of, curvature_of, pressure_of, mv)
    return art.figure(charts.pictures(state, [art.on_numpy(state)])[0])


def draw_scaling(states: Extensor, curvature_of: Function):
    """Plots scalar curvature against temperature offset above the critical point on logarithmic axes."""
    n_reference = 60
    distance = np.log10((mv.t & states).to_array() - 1)
    curvature = np.log10(curvature_of(states).to_array())
    diagram = art.Diagram((-3.0, 0.0), (-0.5, 5.5), (-3.0, -0.5), n_rows, n_columns, ("log(T - 1)", "log R"),
                          (art.mv.x, art.mv.y), (art.mv.w, 0.0))
    anchor = art.chart_points(distance[0], curvature[0] + 0.4)
    reference_direction = art.chart_directions(1.0, -2.0)
    reference_line = art.tangent_lines(
        diagram, anchor, reference_direction, np.linspace(-3.0, 0.0, n_reference))              # [n_reference] Point-
    layers = (art.polylines(diagram, [reference_line], art.medium, art.grey, 1.0, (4.0, 3.0))
              + art.polylines(diagram, [art.chart_points(distance, curvature)], art.bold, art.blue, 1.0, art.SOLID))
    labels = [art.Label(art.tangent_lines(diagram, anchor, reference_direction, np.array([-1.0]))[0], "slope -2",
                        art.grey, art.text_size, "lb", (6, 2))]
    return art.figure(art.picture(diagram, layers, labels))
