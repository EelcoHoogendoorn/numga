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
stroke_colours = np.stack([art.orange, art.grey, art.blue, art.grey])
enclosed_colour = art.rgb("#c5d9a7")
# The cylinder's picture, pixels across, and the height of a unit of volume.
cylinder_pixels, height_per_volume = 150, 0.7


class Charts:
    """The two charts' lines and the cycle on them, sampled once, and the drawing of the gas's state on them."""

    def __init__(self, temperature_of: Reading, pressure_of: Reading, cycle: Extensor) -> None:
        self.volume_pressure = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"))
        self.entropy_temperature = art.Diagram(entropy_range, temperature_range, (entropy_range[0], 0.0), n_rows,
                                               n_columns, ("S", "T"))
        lines = art.gas_lines(temperature_of, pressure_of, cycle.context.multivector, state_entropies, state_volumes,
                              art.GasLines(isochores, isobars, isotherms, adiabats))
        self.lines = lines
        self.pressure_lines = (art.family(self.volume_pressure, lines.isotherms.volume_pressure, "temperature",
                                          art.thin)
                               + art.family(self.volume_pressure, lines.adiabats.volume_pressure, "entropy", art.thin))
        self.temperature_lines = (art.family(self.entropy_temperature, lines.isochores.entropy_temperature, "volume",
                                             art.thin)
                                  + art.family(self.entropy_temperature, lines.isobars.entropy_temperature, "pressure",
                                               art.thin))
        self.cycle = art.gas_points(temperature_of, pressure_of, cycle)                   # [steps] Point-, Point-
        self.n_steps = (cycle.shape[0] - 1) // 4

    def picture(self, diagram: art.Diagram, cycle: art.Point, lines: list, travelled: int, labels: list,
                names: list) -> np.ndarray:
        """A chart with its lines, the area the cycle `[steps]` encloses, the strokes faint and, as far as travelled,
        bold."""
        layers = [art.areas(diagram, cycle[:-1], enclosed_colour, 0.45)] + lines
        for stroke in range(4):
            part = cycle[stroke * self.n_steps: (stroke + 1) * self.n_steps + 1]
            layers += art.polylines(diagram, [part], art.thin, stroke_colours[stroke], 1.0, art.SOLID)
            done = part[: max(0, min(travelled - stroke * self.n_steps + 1, self.n_steps + 1))]
            layers += art.polylines(diagram, [done], art.bold, stroke_colours[stroke], 1.0, art.SOLID)
        now = cycle[travelled][None]
        return art.floated(diagram, art.picture(diagram, layers + art.dots(diagram, now, art.ink), labels), names)

    def pictures(self, travelled: int) -> list[np.ndarray]:
        pressure_cycle, temperature_cycle = self.cycle.volume_pressure, self.cycle.entropy_temperature
        # The net work written amid the area the cycle encloses, at the mean of its samples.
        pressure_labels = [art.Label(pressure_cycle[:-1].mean(axis=0), "W", art.ink, art.text_size, "mm", (0, 0))]
        # The reservoirs named at the chart's left edge, level with the strokes at their temperatures: the line
        # through a corner of the cycle along the entropy, met with that edge.
        left_edge = art.mv.x - art.mv.w * entropy_range[0]                                # [] Plane+
        reservoirs = temperature_cycle[np.array([0, 2 * self.n_steps])]                  # [2] Point-
        hot, cold = art.planar.unit((reservoirs & art.mv.x.dual()) ^ left_edge)          # [] Point-, Point-
        temperature_labels = [art.Label(temperature_cycle[:-1].mean(axis=0), "W", art.ink, art.text_size, "mm",
                                        (0, 0)),
                              art.Label(hot, "hot", art.orange, art.text_size, "lb", (4, 2)),
                              art.Label(cold, "cold", art.blue, art.text_size, "lt", (4, -2))]
        lines = self.lines
        pressure_names = [art.Floating(art.along(lines.isotherms.volume_pressure, 8), "T = const",
                                       art.families["temperature"], art.text_size, "lb"),
                          art.Floating(art.along(lines.adiabats.volume_pressure, 8), "S = const",
                                       art.families["entropy"], art.text_size, "lb")]
        temperature_names = [art.Floating(art.along(lines.isochores.entropy_temperature, 8), "V = const",
                                          art.families["volume"], art.text_size, "lb"),
                             art.Floating(art.along(lines.isobars.entropy_temperature, 8), "p = const",
                                          art.families["pressure"], art.text_size, "lb")]
        return [self.picture(self.volume_pressure, pressure_cycle, self.pressure_lines, travelled, pressure_labels,
                             pressure_names),
                self.picture(self.entropy_temperature, temperature_cycle, self.temperature_lines, travelled,
                             temperature_labels, temperature_names)]


def animate_cycle(temperature_of: Reading, pressure_of: Reading, cycle: Extensor,
                  running: Iterator[Extensor]) -> Iterator[np.ndarray]:
    """The piston and the two charts at each state of the gas as it runs around the cycle."""
    charts = Charts(temperature_of, pressure_of, cycle)
    blades = cycle.context.multivector
    largest = np.asarray((blades.v & cycle).to_array()).max()
    temperatures = np.asarray(temperature_of(cycle).to_array())
    for travelled, state in enumerate(running):
        warmth = (float(temperature_of(state).to_array()) - temperatures.min()) / np.ptp(temperatures)
        piston = art.gas_cylinder(float((blades.v & state).to_array()) * height_per_volume, warmth, False,
                                  largest * height_per_volume, cylinder_pixels)
        yield art.beside([piston] + charts.pictures(travelled))


def draw_cycle(temperature_of: Reading, pressure_of: Reading, cycle: Extensor):
    """The cycle on the gas's two charts."""
    charts = Charts(temperature_of, pressure_of, cycle)
    return art.figure(art.beside(charts.pictures(0)))
