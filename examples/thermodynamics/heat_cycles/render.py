"""A piston taken around a Carnot cycle beside the gas's two charts.

The pressure–volume chart holds isotherms, red, and adiabats, lines of equal entropy, teal; the cycle runs along two of
each, and the area it encloses is the net work. The temperature–entropy chart holds lines of equal volume, green, and of
equal pressure, purple; the cycle is a rectangle there, and the area it encloses is the net heat, the same as the
work. Each stroke has its colour: orange while the gas takes in heat from the hot reservoir, blue while it gives heat
to the cold one, grey while it is insulated.

The charts' lines are level lines of the gas's temperature and pressure, functions the notebook hands over; this
module only samples them.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art

Reading = Callable[[Extensor], Extensor]

# The charts' size in pixels and their ranges: volume and pressure, entropy and temperature.
n_rows, n_columns = 300, 320
volume_range, pressure_range = (0.0, 3.2), (0.0, 3.2)
entropy_range, temperature_range = (-0.6, 1.6), (0.0, 2.0)
# The levels of the charts' families, and the plane of states sampled for them.
isochores = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
isobars = np.array([0.4, 0.7, 1.0, 1.5, 2.2, 3.0])
isotherms = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
adiabats = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
state_entropies = np.linspace(-2.5, 3.0, 120)
state_volumes = np.geomspace(0.2, 5.0, 120)
# The strokes' colours: heated, insulated, cooled, insulated; the enclosed area's.
stroke_colours = np.stack([art.ORANGE, art.GREY, art.BLUE, art.GREY])
enclosed_colour = art.rgb("#c5d9a7")
# The cylinder's picture, pixels across, and the height of a unit of volume.
cylinder_pixels, height_per_volume = 150, 0.7


class Charts:
    """The two charts' lines and the cycle on them, sampled once, and the drawing of the gas's state on them."""

    def __init__(self, temperature_of: Reading, pressure_of: Reading, cycle: Extensor) -> None:
        self.temperature_of, self.pressure_of = temperature_of, pressure_of
        self.volume_pressure = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"))
        self.entropy_temperature = art.Diagram(entropy_range, temperature_range, (entropy_range[0], 0.0), n_rows,
                                               n_columns, ("S", "T"))
        lines = art.gas_lines(temperature_of, pressure_of, cycle.context.multivector, state_entropies, state_volumes,
                              art.GasLines(isochores, isobars, isotherms, adiabats))
        self.lines = lines
        self.pressure_lines = (art.drawn(self.volume_pressure, lines.isotherms, 2, 3, art.THIN, "temperature")
                               + art.drawn(self.volume_pressure, lines.adiabats, 2, 3, art.THIN, "entropy"))
        self.temperature_lines = (art.drawn(self.entropy_temperature, lines.isochores, 0, 1, art.THIN, "volume")
                                  + art.drawn(self.entropy_temperature, lines.isobars, 0, 1, art.THIN, "pressure"))
        self.cycle = self.values(cycle)                                                    # [steps, 4]
        self.n_steps = (len(self.cycle) - 1) // 4

    def values(self, states: Extensor) -> np.ndarray:
        return art.gas_values(self.temperature_of, self.pressure_of, states)

    def picture(self, diagram: art.Diagram, lines: list, x: int, y: int, travelled: int, labels: list,
                names: list) -> np.ndarray:
        """A chart with its lines, the area the cycle encloses, the strokes faint and, as far as travelled, bold."""
        layers = [art.areas(diagram, self.cycle[:-1, x], self.cycle[:-1, y], enclosed_colour, 0.45)] + lines
        for stroke in range(4):
            part = self.cycle[stroke * self.n_steps: (stroke + 1) * self.n_steps + 1]
            layers += art.polylines(diagram, [(part[:, x], part[:, y])], art.THIN, stroke_colours[stroke], 1.0,
                                    art.SOLID)
            done = part[: max(0, min(travelled - stroke * self.n_steps + 1, self.n_steps + 1))]
            layers += art.polylines(diagram, [(done[:, x], done[:, y])], art.BOLD, stroke_colours[stroke], 1.0,
                                    art.SOLID)
        now = self.cycle[travelled]
        return art.floated(diagram, art.picture(diagram, layers + art.dots(diagram, now[x], now[y], art.INK), labels),
                           names)

    def pictures(self, travelled: int) -> list[np.ndarray]:
        middle = self.cycle[:-1].mean(axis=0)
        hot, cold = self.cycle[0, 1], self.cycle[2 * self.n_steps, 1]
        pressure_labels = [art.Label(middle[2], middle[3], "W", art.INK, art.TEXT_SIZE, "mm", (0, 0))]
        temperature_labels = [art.Label(middle[0], middle[1], "W", art.INK, art.TEXT_SIZE, "mm", (0, 0)),
                              art.Label(entropy_range[0], hot, "hot", art.ORANGE, art.TEXT_SIZE, "lb", (4, 2)),
                              art.Label(entropy_range[0], cold, "cold", art.BLUE, art.TEXT_SIZE, "lt", (4, -2))]
        lines = self.lines
        pressure_names = [art.Floating(*art.gas_along(lines.isotherms, 2, 3, 8), "T = const",
                                       art.FAMILIES["temperature"], art.TEXT_SIZE, "lb"),
                          art.Floating(*art.gas_along(lines.adiabats, 2, 3, 8), "S = const", art.FAMILIES["entropy"],
                                       art.TEXT_SIZE, "lb")]
        temperature_names = [art.Floating(*art.gas_along(lines.isochores, 0, 1, 8), "V = const", art.FAMILIES["volume"],
                                          art.TEXT_SIZE, "lb"),
                             art.Floating(*art.gas_along(lines.isobars, 0, 1, 8), "p = const", art.FAMILIES["pressure"],
                                          art.TEXT_SIZE, "lb")]
        return [self.picture(self.volume_pressure, self.pressure_lines, 2, 3, travelled, pressure_labels, pressure_names),
                self.picture(self.entropy_temperature, self.temperature_lines, 0, 1, travelled, temperature_labels,
                             temperature_names)]


def animate_cycle(temperature_of: Reading, pressure_of: Reading, cycle: Extensor,
                  running: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The piston and the two charts at each state of the gas as it runs around the cycle."""
    charts = Charts(temperature_of, pressure_of, cycle)
    largest = charts.cycle[:, 2].max()
    temperatures = charts.cycle[:, 1]
    for travelled, state in enumerate(running):
        now = charts.values(state)
        warmth = (now[1] - temperatures.min()) / np.ptp(temperatures)
        piston = art.gas_cylinder(now[2] * height_per_volume, warmth, False, largest * height_per_volume,
                                  cylinder_pixels)
        yield art.beside([piston] + charts.pictures(travelled))


def draw_cycle(temperature_of: Reading, pressure_of: Reading, cycle: Extensor):
    """The cycle on the gas's two charts."""
    charts = Charts(temperature_of, pressure_of, cycle)
    return art.figure(art.beside(charts.pictures(0)))
