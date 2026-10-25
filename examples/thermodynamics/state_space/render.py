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
ROWS, COLUMNS = 300, 320
ENTROPIES, TEMPERATURES = (-0.4, 2.0), (0.0, 3.0)
VOLUMES, PRESSURES = (0.0, 2.4), (0.0, 3.0)
# The levels of the charts' families: volumes, pressures, temperatures and entropies.
ISOCHORES = np.array([0.5, 0.7, 1.0, 1.4, 2.0])
ISOBARS = np.array([0.5, 0.7, 1.0, 1.4, 2.0, 2.8])
ISOTHERMS = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
ADIABATS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
# The plane of states sampled for the levels: entropies, and volumes by equal ratios.
STATE_ENTROPIES = np.linspace(-2.5, 3.0, 120)
STATE_VOLUMES = np.geomspace(0.2, 4.0, 120)
# The two cylinders' colours on the charts; the cylinders' pictures, pixels across, and the height of a unit of volume.
CASES = np.stack([art.BLUE, art.ORANGE])
CYLINDER_PIXELS, HEIGHT_PER_VOLUME = 130, 0.9
# How far along a heating direction its arrow looks, in the state's units, and its length in pixels; line widths.
REACH, ARROW = 0.05, 26.0
THIN, BOLD = 1.0, 2.2
# The heat taken in, across the third chart.
HEATS = (0.0, 2.6)


class Charts:
    """The two charts' families of lines, sampled once from the gas's functions, and the drawing of states on them."""

    def __init__(self, temperature_of: Reading, pressure_of: Reading, mv) -> None:
        self.temperature_of, self.pressure_of = temperature_of, pressure_of
        self.entropy_temperature = art.Diagram(ENTROPIES, TEMPERATURES, (ENTROPIES[0], 0.0), ROWS, COLUMNS, ("S", "T"))
        self.volume_pressure = art.Diagram(VOLUMES, PRESSURES, (0.0, 0.0), ROWS, COLUMNS, ("V", "p"))
        lines = art.gas_lines(temperature_of, pressure_of, mv, STATE_ENTROPIES, STATE_VOLUMES,
                              art.GasLines(ISOCHORES, ISOBARS, ISOTHERMS, ADIABATS))
        self.temperature_lines = (art.drawn(self.entropy_temperature, lines.isochores, 0, 1, THIN, "volume")
                                  + art.drawn(self.entropy_temperature, lines.isobars, 0, 1, THIN, "pressure"))
        self.pressure_lines = (art.drawn(self.volume_pressure, lines.isotherms, 2, 3, THIN, "temperature")
                               + art.drawn(self.volume_pressure, lines.adiabats, 2, 3, THIN, "entropy"))
        self.heat_temperature = art.Diagram(HEATS, TEMPERATURES, (0.0, 0.0), ROWS, COLUMNS, ("Q", "T"))

    def values(self, states: Extensor) -> np.ndarray:
        return art.gas_values(self.temperature_of, self.pressure_of, states)

    def pictures(self, path: np.ndarray, toward: np.ndarray, pointed: bool) -> list[np.ndarray]:
        """The two charts with the cylinders' paths so far `[steps, 2, 4]`, the heat and the work they took in shaded,
        and, if pointed, arrows from their present states toward where their heating directions lead `[2, 4]`."""
        now = path[-1]
        layers, labels = [], []
        for (diagram, lines), (x, y) in zip([(self.entropy_temperature, self.temperature_lines),
                                            (self.volume_pressure, self.pressure_lines)], [(0, 1), (2, 3)]):
            # The area under each path: the heat on the temperature–entropy chart, the work on the other.
            shaded = [art.areas(diagram, np.concatenate([path[:, case, x], [now[case, x], path[0, case, x]]]),
                                np.concatenate([path[:, case, y], [0.0, 0.0]]), CASES[case], 0.16)
                      for case in range(2) if len(path) > 1]
            paths = [stroke for case in range(2) for stroke in
                     art.polylines(diagram, [(path[:, case, x], path[:, case, y])], BOLD, CASES[case], 1.0, art.SOLID)]
            layers.append(shaded + lines + paths
                          + (art.arrows(diagram, now[:, x], now[:, y], toward[:, x], toward[:, y], ARROW, 1.6, art.INK)
                             if pointed else [])
                          + art.dots(diagram, now[:, x], now[:, y], CASES))
        temperature_labels = [art.Label(ENTROPIES[1], 0.3 * TEMPERATURES[1], "V = const", art.FAMILIES["volume"],
                                        art.LABEL_SIZE, "rb", (0, 0)),
                              art.Label(ENTROPIES[0] + 0.1, 0.62 * TEMPERATURES[1], "p = const",
                                        art.FAMILIES["pressure"], art.LABEL_SIZE, "lb", (0, 0))]
        pressure_labels = [art.Label(VOLUMES[1], 0.2 * PRESSURES[1], "T = const", art.FAMILIES["temperature"],
                                     art.LABEL_SIZE, "rb", (0, 0)),
                           art.Label(VOLUMES[1], 0.04 * PRESSURES[1], "S = const", art.FAMILIES["entropy"],
                                     art.LABEL_SIZE, "rb", (0, 0))]
        return [art.picture(self.entropy_temperature, layers[0], temperature_labels),
                art.picture(self.volume_pressure, layers[1], pressure_labels)]

    def heating_curves(self, heats: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Each cylinder's temperature against the heat it has taken in so far."""
        diagram, now = self.heat_temperature, path[-1]
        layers = []
        for case in range(2):
            layers += art.polylines(diagram, [(heats, path[:, case, 1])], BOLD, CASES[case], 1.0, art.SOLID)
        layers += art.dots(diagram, np.full(2, heats[-1]), now[:, 1], CASES)
        labels = [art.Label(heats[-1], now[0, 1], "fixed volume", CASES[0], art.LABEL_SIZE, "rb", (-6, 6)),
                  art.Label(heats[-1], now[1, 1], "fixed pressure", CASES[1], art.LABEL_SIZE, "lt", (6, -6))]
        return art.picture(diagram, layers, labels)


def heating(temperature_of: Reading, pressure_of: Reading,
            stages: Iterator[tuple[Extensor, Extensor, float]]) -> Iterator[np.ndarray]:
    """Each stage of the heating as it comes, the two cylinders' states `[2]`, latched then loaded, their heating
    directions and the heat each has taken in: the cylinders beside the three charts."""
    first = next(stages)
    charts, path, heats = Charts(temperature_of, pressure_of, first[0].context.multivector), [], []
    for states, directions, heat in chain([first], stages):
        path.append(charts.values(states))
        heats.append(heat)
        toward = charts.values(states + directions * REACH)
        now = path[-1]
        pictures = [art.gas_cylinder(now[case, 2] * HEIGHT_PER_VOLUME, now[case, 1] / TEMPERATURES[1], case == 0,
                                     VOLUMES[1] * HEIGHT_PER_VOLUME, CYLINDER_PIXELS) for case in range(2)]
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
    return art.figure(art.beside(charts.pictures(charts.values(states)[None], charts.values(states + directions * REACH),
                                                 True)))
