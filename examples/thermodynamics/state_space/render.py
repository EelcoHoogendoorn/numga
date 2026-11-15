"""Renders two gas cylinders heated side by side, one latched at fixed volume and one loaded at fixed pressure, beside
three thermodynamic charts.

The temperature–entropy chart draws isochores in green and isobars in purple; reversible heating takes in heat equal
to the area under its path down to zero temperature, so equal added heat shades equal areas. The pressure–volume chart
draws isotherms in red and adiabats in teal, and the loaded piston does work equal to the area under its path. On both
charts, an arrow at each cylinder's state points along its heating direction. The third chart plots each cylinder's
temperature against the cumulative heat added as straight lines whose slopes equal the reciprocal heat capacities.

Level curves on the charts are contoured by sampling the gas's temperature and pressure callables over a grid of states.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from numga.render import solids
from examples.thermodynamics.state_space import core

mv = core.mv
Observable = Callable[[Extensor], Extensor]

# Pixel dimensions and axis ranges for the entropy–temperature and volume–pressure charts.
n_rows, n_columns = 300, 320
entropy_range, temperature_range = (-0.4, 2.0), (0.0, 3.0)
volume_range, pressure_range = (0.0, 2.4), (0.0, 3.0)
# Contour levels for constant volume, pressure, temperature, and entropy.
isochores = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
isobars = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8])
isotherms = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
adiabats = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
# Sample grid of entropies and logarithmically spaced volumes over state space.
state_entropies = np.linspace(-2.5, 3.0, 120)
state_volumes = np.geomspace(0.2, 4.0, 120)
# Cylinder colours, panel width in pixels, and piston height per unit volume.
case_colours = np.stack([art.blue, art.orange])
n_cylinders = len(case_colours)
cylinder_pixels, height_per_volume = 130, 0.9
# Step along each heating direction in state units and arrow length in pixels.
arrow_step, arrow_pixels = 0.05, 26.0
# Horizontal range of cumulative heat on the third chart.
heat_range = (0.0, 2.6)


def feet(points: art.Point) -> art.Point:
    """Projections of chart points onto the horizontal axis where the vertical coordinate vanishes, met along vertical
    lines through each point."""
    return (points & art.mv.y.dual()) ^ art.mv.y


class Charts:
    """Samples the state-space contour families once and draws cylinder trajectories on the three charts."""

    def __init__(self, temperature_of: Observable, pressure_of: Observable, blades) -> None:
        self.temperature_of, self.pressure_of = temperature_of, pressure_of
        self.entropy_temperature = art.Diagram(entropy_range, temperature_range, (entropy_range[0], 0.0), n_rows,
                                               n_columns, ("S", "T"), (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        self.volume_pressure = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"),
                                           (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        lines = art.gas_lines(temperature_of, pressure_of, blades, state_entropies, state_volumes,
                              art.GasLines(isochores, isobars, isotherms, adiabats))
        self.lines = lines
        self.temperature_lines = (art.family(self.entropy_temperature, lines.isochores.entropy_temperature, "volume",
                                             art.thin)
                                  + art.level_family(self.entropy_temperature, lines.isobars.entropy_temperature,
                                                     "pressure", art.thin))
        self.pressure_lines = (art.level_family(self.volume_pressure, lines.isotherms.volume_pressure, "temperature",
                                                art.thin)
                               + art.family(self.volume_pressure, lines.adiabats.volume_pressure, "entropy", art.thin))
        self.heat_temperature = art.Diagram(heat_range, temperature_range, (0.0, 0.0), n_rows, n_columns, ("Q", "T"),
                                            (art.mv.x, art.mv.y), (art.mv.w, 0.0))

    def points(self, states: Extensor) -> art.GasPoints:
        return art.gas_points(self.temperature_of, self.pressure_of, states)

    def pictures(self, path: list[art.GasPoints], toward: art.GasPoints, show_arrows: bool) -> list[np.ndarray]:
        """Renders the temperature–entropy and pressure–volume charts with the cylinders' trajectories, shaded heat and
        work areas, and optional arrows along their heating directions."""
        layers = []
        for diagram, lines, steps, aims in [
                (self.entropy_temperature, self.temperature_lines, [step.entropy_temperature for step in path],
                 toward.entropy_temperature),
                (self.volume_pressure, self.pressure_lines, [step.volume_pressure for step in path],
                 toward.volume_pressure)]:
            trajectory = stack(steps)                                         # [..., n_cylinders] Point-
            now = trajectory[-1]                                              # [n_cylinders] Point-
            # Shade the area under each curve down to zero for heat on S-T and piston work on V-p.
            areas = [art.between(diagram, trajectory[:, case], feet(trajectory[:, case]), case_colours[case], 0.16)
                     for case in range(n_cylinders) if len(path) > 1]
            paths = [stroke for case in range(n_cylinders) for stroke in
                     art.polylines(diagram, [trajectory[:, case]], art.bold, case_colours[case], 1.0, art.SOLID)]
            layers.append(areas + lines + paths
                          + (art.arrows(diagram, now, aims, arrow_pixels, art.medium, art.ink) if show_arrows else [])
                          + art.dots(diagram, now, case_colours))
        lines = self.lines
        temperature_names = [art.Floating(art.along(lines.isochores.entropy_temperature, 8), "V = const",
                                          art.families["volume"], art.text_size, "lb"),
                             art.Floating(art.crossings(lines.isobars.entropy_temperature, 8), "p = const",
                                          art.families["pressure"], art.text_size, "lb")]
        pressure_names = [art.Floating(art.crossings(lines.isotherms.volume_pressure, 8), "T = const",
                                       art.families["temperature"], art.text_size, "lb"),
                          art.Floating(art.along(lines.adiabats.volume_pressure, 8), "S = const",
                                       art.families["entropy"], art.text_size, "lb")]
        return [art.floated(self.entropy_temperature, art.picture(self.entropy_temperature, layers[0], []),
                            temperature_names),
                art.floated(self.volume_pressure, art.picture(self.volume_pressure, layers[1], []), pressure_names)]

    def heating_curves(self, heats: np.ndarray, temperatures: art.Scalar) -> np.ndarray:
        """Renders each cylinder's temperature against the cumulative heat added."""
        diagram = self.heat_temperature
        curves = art.chart_points(heats[:, None], temperatures)               # [..., n_cylinders] Point-
        now = curves[-1]                                                      # [n_cylinders] Point-
        layers = []
        for case in range(n_cylinders):
            layers += art.polylines(diagram, [curves[:, case]], art.bold, case_colours[case], 1.0, art.SOLID)
        layers += art.dots(diagram, now, case_colours)
        labels = [art.Label(now[0], "fixed volume", case_colours[0], art.text_size, "rb", (-6, 6)),
                  art.Label(now[1], "fixed pressure", case_colours[1], art.text_size, "rt", (-6, -8))]
        return art.picture(diagram, layers, labels)


def heating(temperature_of: Observable, pressure_of: Observable,
            stages: Iterator[tuple[Extensor, Extensor, float]]) -> Iterator[np.ndarray]:
    """Yields animation frames placing the two cylinders beside the three thermodynamic charts at each stage."""
    first = next(stages)
    charts, path, heats, temperatures = Charts(temperature_of, pressure_of, mv), [], [], []
    for states, directions, heat in chain([first], stages):
        path.append(charts.points(states))
        heats.append(heat)
        temperatures.append(art.context.include_from_algebra(temperature_of(states)))
        toward = charts.points(states + directions * arrow_step)
        heights = solids.context.include_from_algebra(mv.v & states) * height_per_volume
        pictures = [art.gas_cylinder(heights[case], float((temperatures[-1][case] / temperature_range[1]).to_array()), case == 0,
                                     solids.mv.scalar([volume_range[1] * height_per_volume]), cylinder_pixels)
                    for case in range(n_cylinders)]
        yield art.beside(pictures + charts.pictures(path, toward, True)
                         + [charts.heating_curves(np.array(heats), stack(temperatures))])


def draw_charts(temperature_of: Observable, pressure_of: Observable, states: Extensor):
    """Draws the temperature–entropy and pressure–volume charts with the cylinders' states marked."""
    charts = Charts(temperature_of, pressure_of, mv)
    here = charts.points(states)
    return art.figure(art.beside(charts.pictures([here], here, False)))


def draw_directions(temperature_of: Observable, pressure_of: Observable, states: Extensor, directions: Extensor):
    """Draws the temperature–entropy and pressure–volume charts with arrows along the cylinders' heating directions."""
    charts = Charts(temperature_of, pressure_of, mv)
    return art.figure(art.beside(charts.pictures([charts.points(states)],
                                                 charts.points(states + directions * arrow_step), True)))
