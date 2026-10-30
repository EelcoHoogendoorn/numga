"""Two cylinders heated alike, one latched at fixed volume and one under a weight at fixed pressure, beside the gas's
two charts.

The temperature–entropy chart holds lines of equal volume, green, and of equal pressure, purple; the heat a
reversible change takes in is the area under its path down to zero temperature, so equal heat shades equal areas. The
pressure–volume chart holds isotherms, red, and adiabats, lines of equal entropy, teal; the work
the loaded piston does is the area under its path. On both, an arrow at each cylinder's state points along its heating
direction. The third chart is each cylinder's temperature against the heat it has taken in: straight lines whose
slopes are one over the heat capacities.

The charts' lines are where the gas's temperature or pressure, functions the notebook hands over, take their levels
over the plane of states; this module only samples them.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, concatenate, stack
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids

Reading = Callable[[Extensor], Extensor]

# The charts' size in pixels and their ranges: entropy and temperature, volume and pressure.
n_rows, n_columns = 300, 320
entropy_range, temperature_range = (-0.4, 2.0), (0.0, 3.0)
volume_range, pressure_range = (0.0, 2.4), (0.0, 3.0)
# The levels of the charts' families: volumes, pressures, temperatures and entropies.
isochores = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
isobars = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8])
isotherms = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
adiabats = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
# The plane of states sampled for the levels: entropies, and volumes by equal ratios.
state_entropies = np.linspace(-2.5, 3.0, 120)
state_volumes = np.geomspace(0.2, 4.0, 120)
# The two cylinders' colours on the charts; the cylinders' pictures, pixels across, and the height of a unit of volume.
case_colours = np.stack([art.blue, art.orange])
cylinder_pixels, height_per_volume = 130, 0.9
# How far along a heating direction its arrow looks, in the state's units, and its length in pixels.
arrow_step, arrow_pixels = 0.05, 26.0
# The heat taken in, across the third chart.
heat_range = (0.0, 2.6)


def feet(points: art.Point) -> art.Point:
    """The feet of chart points `[...]` on the plane where the quantity up vanishes: the line through each along the
    quantity up, met with it."""
    return art.planar.unit((points & art.mv.y.dual()) ^ art.mv.y)


class Charts:
    """The two charts' families of lines, sampled once from the gas's functions, and the drawing of states on them."""

    def __init__(self, temperature_of: Reading, pressure_of: Reading, blades) -> None:
        self.temperature_of, self.pressure_of = temperature_of, pressure_of
        self.entropy_temperature = art.Diagram(entropy_range, temperature_range, (entropy_range[0], 0.0), n_rows,
                                               n_columns, ("S", "T"))
        self.volume_pressure = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"))
        lines = art.gas_lines(temperature_of, pressure_of, blades, state_entropies, state_volumes,
                              art.GasLines(isochores, isobars, isotherms, adiabats))
        self.lines = lines
        self.temperature_lines = (art.family(self.entropy_temperature, lines.isochores.entropy_temperature, "volume",
                                             art.thin)
                                  + art.family(self.entropy_temperature, lines.isobars.entropy_temperature, "pressure",
                                               art.thin))
        self.pressure_lines = (art.family(self.volume_pressure, lines.isotherms.volume_pressure, "temperature",
                                          art.thin)
                               + art.family(self.volume_pressure, lines.adiabats.volume_pressure, "entropy", art.thin))
        self.heat_temperature = art.Diagram(heat_range, temperature_range, (0.0, 0.0), n_rows, n_columns, ("Q", "T"))

    def points(self, states: Extensor) -> art.GasPoints:
        return art.gas_points(self.temperature_of, self.pressure_of, states)

    def pictures(self, path: list[art.GasPoints], toward: art.GasPoints, pointed: bool) -> list[np.ndarray]:
        """The two charts with the cylinders' paths so far, each step's states `[cylinders]` on both charts, the heat
        and the work they took in shaded, and, if pointed, arrows from their present states toward where their heating
        directions lead `[cylinders]`."""
        layers = []
        for diagram, lines, steps, aims in [
                (self.entropy_temperature, self.temperature_lines, [step.entropy_temperature for step in path],
                 toward.entropy_temperature),
                (self.volume_pressure, self.pressure_lines, [step.volume_pressure for step in path],
                 toward.volume_pressure)]:
            traced = stack(steps)                                            # [steps, cylinders] Point-
            now = traced[-1]                                                 # [cylinders] Point-
            # The area under each path down to zero: the heat on the temperature–entropy chart, the work on the other.
            under = concatenate([traced, feet(now)[None], feet(traced[0])[None]])   # [steps + 2, cylinders] Point-
            shaded = [art.areas(diagram, under[:, case], case_colours[case], 0.16)
                      for case in range(2) if len(path) > 1]
            paths = [stroke for case in range(2) for stroke in
                     art.polylines(diagram, [traced[:, case]], art.bold, case_colours[case], 1.0, art.SOLID)]
            layers.append(shaded + lines + paths
                          + (art.arrows(diagram, now, aims, arrow_pixels, art.medium, art.ink) if pointed else [])
                          + art.dots(diagram, now, case_colours))
        lines = self.lines
        temperature_names = [art.Floating(art.along(lines.isochores.entropy_temperature, 8), "V = const",
                                          art.families["volume"], art.text_size, "lb"),
                             art.Floating(art.along(lines.isobars.entropy_temperature, 8), "p = const",
                                          art.families["pressure"], art.text_size, "lb")]
        pressure_names = [art.Floating(art.along(lines.isotherms.volume_pressure, 8), "T = const",
                                       art.families["temperature"], art.text_size, "lb"),
                          art.Floating(art.along(lines.adiabats.volume_pressure, 8), "S = const",
                                       art.families["entropy"], art.text_size, "lb")]
        return [art.floated(self.entropy_temperature, art.picture(self.entropy_temperature, layers[0], []),
                            temperature_names),
                art.floated(self.volume_pressure, art.picture(self.volume_pressure, layers[1], []), pressure_names)]

    def heating_curves(self, heats: np.ndarray, temperatures: art.Scalar) -> np.ndarray:
        """Each cylinder's temperature against the heat it has taken in so far `[steps, cylinders]`."""
        diagram = self.heat_temperature
        curves = art.chart_points(heats[:, None], temperatures)               # [steps, cylinders] Point-
        now = curves[-1]                                                     # [cylinders] Point-
        layers = []
        for case in range(2):
            layers += art.polylines(diagram, [curves[:, case]], art.bold, case_colours[case], 1.0, art.SOLID)
        layers += art.dots(diagram, now, case_colours)
        labels = [art.Label(now[0], "fixed volume", case_colours[0], art.text_size, "rb", (-6, 6)),
                  art.Label(now[1], "fixed pressure", case_colours[1], art.text_size, "rt", (-6, -8))]
        return art.picture(diagram, layers, labels)


def heating(temperature_of: Reading, pressure_of: Reading,
            stages: Iterator[tuple[Extensor, Extensor, float]]) -> Iterator[np.ndarray]:
    """Each stage of the heating as it comes, the two cylinders' states `[cylinders]`, latched then loaded, their
    heating directions and the heat each has taken in: the cylinders beside the three charts."""
    first = next(stages)
    blades = first[0].context.multivector
    charts, path, heats, temperatures = Charts(temperature_of, pressure_of, blades), [], [], []
    for states, directions, heat in chain([first], stages):
        path.append(charts.points(states))
        heats.append(heat)
        temperatures.append(art.context.include_from_algebra(temperature_of(states)))
        toward = charts.points(states + directions * arrow_step)
        heights = solids.context.include_from_algebra(blades.v & states) * height_per_volume
        pictures = [art.gas_cylinder(heights[case], float((temperatures[-1][case] / temperature_range[1]).to_array()), case == 0,
                                     solids.mv.scalar([volume_range[1] * height_per_volume]), cylinder_pixels) for case in range(2)]
        yield art.beside(pictures + charts.pictures(path, toward, True)
                         + [charts.heating_curves(np.array(heats), stack(temperatures))])


def draw_charts(temperature_of: Reading, pressure_of: Reading, states: Extensor):
    """The two charts with the cylinders' states."""
    charts = Charts(temperature_of, pressure_of, states.context.multivector)
    here = charts.points(states)
    return art.figure(art.beside(charts.pictures([here], here, False)))


def draw_directions(temperature_of: Reading, pressure_of: Reading, states: Extensor, directions: Extensor):
    """The two charts with the cylinders' states and arrows along their heating directions."""
    charts = Charts(temperature_of, pressure_of, states.context.multivector)
    return art.figure(art.beside(charts.pictures([charts.points(states)],
                                                 charts.points(states + directions * arrow_step), True)))
