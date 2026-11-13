"""A piston taken around a Carnot cycle beside the gas's pressure-volume and temperature-entropy charts.

The pressure-volume chart draws isotherms in red and adiabats (lines of equal entropy) in teal; the cycle runs along two
of each, and the enclosed area equals the net work. The temperature-entropy chart draws isochores (lines of equal
volume) in green and isobars in purple; the cycle forms a rectangle there whose enclosed area equals the net heat,
matching the net work. Each stroke carries its thermal colour: orange while the gas absorbs heat from the hot reservoir,
blue while it rejects heat to the cold reservoir, and grey along the insulated strokes.

Both charts draw level curves from the temperature and pressure functions handed over by the caller.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np

from examples.thermodynamics import rendering as art
from examples.thermodynamics.heat_cycles import core
from examples.util.wgsl import simplices, solids

mv = core.mv
StateFunction = Callable[[core.State], core.Scalar]

# Chart pixel sizes and axis ranges for volume, pressure, entropy, and temperature.
n_rows, n_columns = 300, 320
volume_range, pressure_range = (0.0, 3.2), (0.0, 3.2)
entropy_range, temperature_range = (-0.6, 1.6), (0.0, 2.0)
# Reference curve levels and the state grid sampled to contour them.
isochores = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
isobars = np.array([0.4, 0.7, 1.0, 1.5, 2.2, 3.0])
isotherms = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
adiabats = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
state_entropies = np.linspace(-2.5, 3.0, 120)
state_volumes = np.geomspace(0.2, 5.0, 120)
# Stroke colours (heated, insulated, cooled, insulated), enclosed area tint, and interior grid row count.
stroke_colours = np.stack([art.orange, art.grey, art.blue, art.grey])
enclosed_colour = art.rgb("#c5d9a7")
n_fill = 16
# Cylinder panel width in pixels and height per unit volume.
cylinder_pixels, height_per_volume = 150, 0.7


class Charts:
    """Reference curves, enclosed cycle region, and frame renderer for the two thermodynamic charts."""

    def __init__(self, temperature_of: StateFunction, pressure_of: StateFunction, cycle: core.State) -> None:
        cycle_batch = cycle.batch()                                                       # [4 * n_steps] State-
        n_steps = cycle_batch.shape[0] // 4
        self.n_steps = n_steps
        self.volume_pressure = art.Diagram(volume_range, pressure_range, (0.0, 0.0), n_rows, n_columns, ("V", "p"),
                                           (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        self.entropy_temperature = art.Diagram(entropy_range, temperature_range, (entropy_range[0], 0.0), n_rows,
                                               n_columns, ("S", "T"), (art.mv.x, art.mv.y), (art.mv.w, 0.0))
        lines = art.gas_lines(temperature_of, pressure_of, mv, state_entropies, state_volumes,
                              art.GasLines(isochores, isobars, isotherms, adiabats))
        self.lines = lines
        self.pressure_lines = (art.level_family(self.volume_pressure, lines.isotherms.volume_pressure, "temperature",
                                                art.thin)
                               + art.family(self.volume_pressure, lines.adiabats.volume_pressure, "entropy", art.thin))
        self.temperature_lines = (art.family(self.entropy_temperature, lines.isochores.entropy_temperature, "volume",
                                             art.thin)
                                  + art.level_family(self.entropy_temperature, lines.isobars.entropy_temperature,
                                                     "pressure", art.thin))
        self.cycle = art.gas_points(temperature_of, pressure_of, cycle_batch)             # [4 * n_steps] Point-, Point-
        # Closed loop of edges connecting each cycle state to the next.
        self.around = art.Topology.loops([cycle_batch.shape[0]])
        # Sample the interior region by interpolating along each adiabat from the hot stroke to the cold stroke.
        heated = cycle_batch[:n_steps + 1]                                                # [n_steps + 1] State-
        cooled = cycle_batch[2 * n_steps:3 * n_steps + 1][::-1]                           # [n_steps + 1] State-
        shares = np.linspace(0, 1, n_fill)[:, None]
        enclosed = (heated * (1 - shares) + cooled * shares).reshape(-1)                  # [n_fill * (n_steps + 1)] State-
        self.enclosed = art.gas_points(temperature_of, pressure_of, enclosed)             # [n_fill * (n_steps + 1)] Point-, Point-
        self.grid = art.Topology.grid(n_fill, n_steps + 1)

    def picture(self, diagram: art.Diagram, cycle: art.Point, enclosed: art.Point, lines: list, travelled: int,
                labels: list, names: list) -> np.ndarray:
        """Render one chart with its reference curves, shaded cycle area, faint full strokes, and bold travelled path."""
        n_steps = self.n_steps
        layers = [art.area(diagram, enclosed, self.grid, enclosed_colour, 0.45)] + lines
        points = art.on_numpy(cycle)                                                      # [4 * n_steps] Point-
        for stroke in range(4):
            # Edges belonging to this stroke, and the prefix already traversed up to the current step.
            part = np.arange(stroke * n_steps, (stroke + 1) * n_steps)
            done = part[: max(0, min(travelled - stroke * n_steps, n_steps))]
            layers += [simplices.strokes(points, self.around.subset(part, -1), stroke_colours[stroke], art.thin,
                                         diagram.camera),
                       simplices.strokes(points, self.around.subset(done, -1), stroke_colours[stroke], art.bold,
                                         diagram.camera)]
        now = cycle[travelled][None]
        return art.floated(diagram, art.picture(diagram, layers + art.dots(diagram, now, art.ink), labels), names)

    def pictures(self, travelled: int) -> list[np.ndarray]:
        pressure_cycle, temperature_cycle = self.cycle.volume_pressure, self.cycle.entropy_temperature
        # Place the net-work label at the centroid of the sampled cycle points.
        pressure_labels = [art.Label(pressure_cycle.mean(axis=0), "W", art.ink, art.text_size, "mm", (0, 0))]
        # Intersect horizontal lines through the hot and cold corners with the left chart edge to place reservoir labels.
        left_edge = art.mv.x - art.mv.w * entropy_range[0]                                # [] Plane+
        reservoirs = temperature_cycle[np.array([0, 2 * self.n_steps])]                   # [2] Point-
        hot, cold = (reservoirs & art.mv.x.dual()) ^ left_edge                            # [] Point-, Point-
        temperature_labels = [art.Label(temperature_cycle.mean(axis=0), "W", art.ink, art.text_size, "mm",
                                        (0, 0)),
                              art.Label(hot, "hot", art.orange, art.text_size, "lb", (4, 2)),
                              art.Label(cold, "cold", art.blue, art.text_size, "lt", (4, -2))]
        lines = self.lines
        pressure_names = [art.Floating(art.crossings(lines.isotherms.volume_pressure, 8), "T = const",
                                       art.families["temperature"], art.text_size, "lb"),
                          art.Floating(art.along(lines.adiabats.volume_pressure, 8), "S = const",
                                       art.families["entropy"], art.text_size, "lb")]
        temperature_names = [art.Floating(art.along(lines.isochores.entropy_temperature, 8), "V = const",
                                          art.families["volume"], art.text_size, "lb"),
                             art.Floating(art.crossings(lines.isobars.entropy_temperature, 8), "p = const",
                                          art.families["pressure"], art.text_size, "lb")]
        return [self.picture(self.volume_pressure, pressure_cycle, self.enclosed.volume_pressure, self.pressure_lines,
                             travelled, pressure_labels, pressure_names),
                self.picture(self.entropy_temperature, temperature_cycle, self.enclosed.entropy_temperature,
                             self.temperature_lines, travelled, temperature_labels, temperature_names)]


def animate_cycle(temperature_of: StateFunction, pressure_of: StateFunction, cycle: core.State,
                  states: Iterator[core.State]) -> Iterator[np.ndarray]:
    """Yield side-by-side cylinder and chart frames as the gas traverses the cycle."""
    cycle_batch = cycle.batch()                                                           # [...] State-
    charts = Charts(temperature_of, pressure_of, cycle_batch)
    volumes = mv.v & cycle_batch                                                          # [...] Scalar+
    largest = solids.context.include_from_algebra(volumes.max())
    temperatures = temperature_of(cycle_batch)                                            # [...] Scalar+
    cold, hot = temperatures.min(), temperatures.max()
    for travelled, state in enumerate(states):
        warmth = float(((temperature_of(state) - cold) / (hot - cold)).to_array())
        piston = art.gas_cylinder(solids.context.include_from_algebra(mv.v & state) * height_per_volume, warmth, False,
                                  largest * height_per_volume, cylinder_pixels)
        yield art.beside([piston] + charts.pictures(travelled))


def draw_cycle(temperature_of: StateFunction, pressure_of: StateFunction, cycle: core.State):
    """Draw the Carnot cycle on the pressure-volume and temperature-entropy charts."""
    charts = Charts(temperature_of, pressure_of, cycle)
    return art.figure(art.beside(charts.pictures(0)))
