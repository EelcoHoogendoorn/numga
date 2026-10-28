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

from numga import Extensor
from examples.thermodynamics import rendering as art

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
case_colours = np.stack([art.BLUE, art.ORANGE])
cylinder_pixels, height_per_volume = 130, 0.9
# How far along a heating direction its arrow looks, in the state's units, and its length in pixels; line widths.
arrow_step, arrow_pixels = 0.05, 26.0
thin_width, bold_width = 1.0, 2.2
# The heat taken in, across the third chart.
heat_range = (0.0, 2.6)


class Charts:
    """The two charts' families of lines, sampled once from the gas's functions, and the drawing of states on them."""

    def __init__(self, temperature_of: Reading, pressure_of: Reading, mv) -> None:
        self.temperature_of, self.pressure_of = temperature_of, pressure_of
        self.entropy_temperature = art.Diagram(entropy_range, temperature_range, (entropy_range[0], 0.0), n_rows,
                                               n_columns, ("S", "T"))
        self.volume_pressure = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"))
        lines = art.gas_lines(temperature_of, pressure_of, mv, state_entropies, state_volumes,
                              art.GasLines(isochores, isobars, isotherms, adiabats))
        self.lines = lines
        self.temperature_lines = (art.drawn(self.entropy_temperature, lines.isochores, 0, 1, thin_width, "volume")
                                  + art.drawn(self.entropy_temperature, lines.isobars, 0, 1, thin_width, "pressure"))
        self.pressure_lines = (art.drawn(self.volume_pressure, lines.isotherms, 2, 3, thin_width, "temperature")
                               + art.drawn(self.volume_pressure, lines.adiabats, 2, 3, thin_width, "entropy"))
        self.heat_temperature = art.Diagram(heat_range, temperature_range, (0.0, 0.0), n_rows, n_columns, ("Q", "T"))

    def values(self, states: Extensor) -> np.ndarray:
        return art.gas_values(self.temperature_of, self.pressure_of, states)

    def pictures(self, path: np.ndarray, toward: np.ndarray, pointed: bool) -> list[np.ndarray]:
        """The two charts with the cylinders' paths so far `[steps, cylinders, 4]`, the heat and the work they took in
        shaded, and, if pointed, arrows from their present states toward where their heating directions lead
        `[cylinders, 4]`."""
        now = path[-1]
        layers = []
        for (diagram, lines), (x, y) in zip([(self.entropy_temperature, self.temperature_lines),
                                            (self.volume_pressure, self.pressure_lines)], [(0, 1), (2, 3)]):
            # The area under each path: the heat on the temperature–entropy chart, the work on the other.
            shaded = [art.areas(diagram, np.concatenate([path[:, case, x], [now[case, x], path[0, case, x]]]),
                                np.concatenate([path[:, case, y], [0.0, 0.0]]), case_colours[case], 0.16)
                      for case in range(2) if len(path) > 1]
            paths = [stroke for case in range(2) for stroke in
                     art.polylines(diagram, [(path[:, case, x], path[:, case, y])], bold_width, case_colours[case],
                                   1.0, art.SOLID)]
            layers.append(shaded + lines + paths
                          + (art.arrows(diagram, now[:, x], now[:, y], toward[:, x], toward[:, y], arrow_pixels, 1.6,
                                        art.INK)
                             if pointed else [])
                          + art.dots(diagram, now[:, x], now[:, y], case_colours))
        lines = self.lines
        temperature_names = [art.Floating(*art.gas_along(lines.isochores, 0, 1, 8), "V = const", art.FAMILIES["volume"],
                                          art.LABEL_SIZE, "lb"),
                             art.Floating(*art.gas_along(lines.isobars, 0, 1, 8), "p = const", art.FAMILIES["pressure"],
                                          art.LABEL_SIZE, "lb")]
        pressure_names = [art.Floating(*art.gas_along(lines.isotherms, 2, 3, 8), "T = const",
                                       art.FAMILIES["temperature"], art.LABEL_SIZE, "lb"),
                          art.Floating(*art.gas_along(lines.adiabats, 2, 3, 8), "S = const", art.FAMILIES["entropy"],
                                       art.LABEL_SIZE, "lb")]
        return [art.floated(self.entropy_temperature, art.picture(self.entropy_temperature, layers[0], []),
                            temperature_names),
                art.floated(self.volume_pressure, art.picture(self.volume_pressure, layers[1], []), pressure_names)]

    def heating_curves(self, heats: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Each cylinder's temperature against the heat it has taken in so far."""
        diagram, now = self.heat_temperature, path[-1]
        layers = []
        for case in range(2):
            layers += art.polylines(diagram, [(heats, path[:, case, 1])], bold_width, case_colours[case], 1.0,
                                    art.SOLID)
        layers += art.dots(diagram, np.full(2, heats[-1]), now[:, 1], case_colours)
        labels = [art.Label(heats[-1], now[0, 1], "fixed volume", case_colours[0], art.LABEL_SIZE, "rb", (-6, 6)),
                  art.Label(heats[-1], now[1, 1], "fixed pressure", case_colours[1], art.LABEL_SIZE, "rt", (-6, -8))]
        return art.picture(diagram, layers, labels)


def heating(temperature_of: Reading, pressure_of: Reading,
            stages: Iterator[tuple[Extensor, Extensor, float]]) -> Iterator[np.ndarray]:
    """Each stage of the heating as it comes, the two cylinders' states `[cylinders]`, latched then loaded, their
    heating directions and the heat each has taken in: the cylinders beside the three charts."""
    first = next(stages)
    charts, path, heats = Charts(temperature_of, pressure_of, first[0].context.multivector), [], []
    for states, directions, heat in chain([first], stages):
        path.append(charts.values(states))
        heats.append(heat)
        toward = charts.values(states + directions * arrow_step)
        now = path[-1]
        pictures = [art.gas_cylinder(now[case, 2] * height_per_volume, now[case, 1] / temperature_range[1], case == 0,
                                     volume_range[1] * height_per_volume, cylinder_pixels) for case in range(2)]
        yield art.beside(pictures + charts.pictures(np.array(path), toward, True)
                         + [charts.heating_curves(np.array(heats), np.array(path))])


def draw_charts(temperature_of: Reading, pressure_of: Reading, states: Extensor):
    """The two charts with the cylinders' states."""
    charts = Charts(temperature_of, pressure_of, states.context.multivector)
    values = charts.values(states)
    return art.figure(art.beside(charts.pictures(values[None], values, False)))


def draw_directions(temperature_of: Reading, pressure_of: Reading, states: Extensor, directions: Extensor):
    """The two charts with the cylinders' states and arrows along their heating directions."""
    charts = Charts(temperature_of, pressure_of, states.context.multivector)
    return art.figure(art.beside(charts.pictures(charts.values(states)[None],
                                                 charts.values(states + directions * arrow_step), True)))
