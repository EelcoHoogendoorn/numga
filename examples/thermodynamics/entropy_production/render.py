"""Two bodies sharing heat, beside the plane of their energies and the entropy along the line of fixed total energy.

The bodies are coloured by how far each is from the common temperature they approach, half by that gap and half by its
logarithm, so that the closing gap keeps changing their colours to the end without crowding the start. On the first chart, the bodies' energies,
with the entropy's level lines, teal; the state runs along the line of fixed total energy, straight
in the direction of the transfer, and its arrow is the heat flow. It comes to rest where the entropy's level line,
black, touches the line. On the second, the entropy along that line against body a's energy: a hill whose top is
equilibrium, which the state climbs; the tangent's slope is how much entropy a unit of energy moved to body a would
produce, and the flow runs uphill. On the third, the two temperatures against time, closing on the common one.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids

# The charts' size in pixels and their ranges: the two energies, and body a's energy against the entropy.
n_rows, n_columns = 300, 310
energy_range = (0.0, 4.3)
entropy_range = (-1.6, 1.3)
# The entropy's levels, the grid of energies they are found on, and the samples along the line of fixed energy.
entropy_levels = np.linspace(-0.4, 1.6, 11)
grid_energies = np.linspace(0.04, 4.3, 160)
n_line = 240
# How far an arrow reaches in pixels for a unit of flow, and the tangent's half length in units of energy.
arrow_pixels_per_flow, tangent_half_length = 40.0, 0.8
# The bodies' picture, pixels across, and the decades of the gap to the common temperature their colours span; the
# third chart's ranges, time and temperature.
block_pixels = 240
colour_decades = 2.5
time_range, temperature_range = (0.0, 5.8), (0.0, 3.0)


def bodies(temperatures: np.ndarray, capacities: np.ndarray, widest: float) -> np.ndarray:
    """The two bodies, each as large as its heat capacity and coloured by its gap to the common temperature, the
    widest gap shown at the thermal ramp's ends, joined by a bar."""
    sizes = 0.5 * np.cbrt(capacities)
    centres = np.array([[-1.0, 0, sizes[0]], [1.0, 0, sizes[1]]])
    blocks = solids.boxes(solids.translator(solids.mv.point(centres) - solids.origin), np.repeat(sizes[:, None], 3, axis=1))
    common = (capacities * temperatures).sum() / capacities.sum()
    colours = np.stack([art.thermal_colour(art.relaxing(temperature - common, widest, colour_decades))
                        for temperature in temperatures])
    bar = art.box(np.array([0, 0, 0.45]), np.array([0.5, 0.12, 0.1]))
    corners = solids.mv.point(np.array([[x, y, z] for x in (-1.75, 1.75) for y in (-0.75, 0.75) for z in (0, 1.6)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-16), 0.06, block_pixels)
    return solids.draw([(blocks, colours), (bar, art.GREY)], view, art.WHITE, inks=[], translucent=[], meshes=[])


class Charts:
    """The two charts' fixed lines, sampled once from the entropy, and the drawing of the state on them."""

    def __init__(self, entropy_of: Callable[[Extensor], Extensor], transfer: Extensor, state: Extensor) -> None:
        mv = state.context.multivector
        self.read = lambda states: (stack((mv.a, mv.b)) & states[..., None]).to_array()
        self.energies = art.Diagram(energy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("U_a", "U_b"))
        self.cooling = art.Diagram(time_range, temperature_range, (0.0, 0.0), n_rows, n_columns, ("t", "T"))
        self.hill = art.Diagram(energy_range, entropy_range, (0.0, entropy_range[0]), n_rows, n_columns, ("U_a", "S"))
        grid = mv.a.dual() * grid_energies[None, :] + mv.b.dual() * grid_energies[:, None]
        levels = art.level_lines(grid_energies, grid_energies, entropy_of(grid).to_array(), entropy_levels)
        # The line of fixed total energy through the state, along the transfer, from body a holding almost none of the
        # energy to almost all of it.
        start = self.read(state)
        along = np.linspace(0.02, 0.98, n_line) * start.sum()
        line = state + transfer * ((along - start[0]) / self.read(transfer)[0])
        self.line = self.read(line)
        self.line_entropy = entropy_of(line).to_array()
        top = np.argmax(self.line_entropy)
        self.top = (self.line[top], self.line_entropy[top])
        # The entropy's level line through the top of the hill touches the line of fixed energy there.
        touching = art.level_lines(grid_energies, grid_energies, entropy_of(grid).to_array(),
                                   np.array([self.line_entropy.max()]))
        self.level_lines = [(line[:, 0], line[:, 1]) for _, line in levels]
        self.energy_layers = (art.family(self.energies, [(level, line[:, 0], line[:, 1]) for level, line in levels],
                                         "entropy", art.THIN)
                              + art.polylines(self.energies, [(line[:, 0], line[:, 1]) for _, line in touching], art.LINE,
                                              art.BOUNDARY, 1.0, art.SOLID)
                              + art.polylines(self.energies, [(self.line[:, 0], self.line[:, 1])], art.LINE, art.INK, 1.0,
                                              art.SOLID)
                              + art.dots(self.energies, self.top[0][0], self.top[0][1], art.GREEN))
        self.hill_layers = (art.polylines(self.hill, [(self.line[:, 0], self.line_entropy)], art.LINE, art.INK, 1.0, art.SOLID)
                            + art.dots(self.hill, self.top[0][0], self.top[1], art.GREEN))

    def pictures(self, path: np.ndarray, path_entropy: np.ndarray, flow: np.ndarray) -> list[np.ndarray]:
        now, now_entropy = path[-1], path_entropy[-1]
        energy = (self.energy_layers
                  + art.polylines(self.energies, [(path[:, 0], path[:, 1])], art.BOLD, art.ORANGE, 1.0, art.SOLID)
                  + (art.arrows(self.energies, now[:1], now[1:], now[:1] + flow[0], now[1:] + flow[1],
                                arrow_pixels_per_flow * np.linalg.norm(flow), art.LINE, art.INK)
                     if np.linalg.norm(flow) > 1e-3 else [])
                  + art.dots(self.energies, now[0], now[1], art.ORANGE))
        # The hill's slope at the state, from its samples on either side.
        slope = np.interp(now[0], self.line[:, 0], np.gradient(self.line_entropy, self.line[:, 0]))
        tangent = np.array([now[0] - tangent_half_length, now[0] + tangent_half_length])
        hill = (self.hill_layers
                + art.polylines(self.hill, [(path[:, 0], path_entropy)], art.BOLD, art.ORANGE, 1.0, art.SOLID)
                + art.polylines(self.hill, [(tangent, now_entropy + slope * (tangent - now[0]))], art.LINE, art.ORANGE, 1.0,
                                art.SOLID)
                + art.dots(self.hill, now[0], now_entropy, art.ORANGE))
        names = [art.Floating(*art.along(self.level_lines, 8), "S = const", art.FAMILIES["entropy"], art.TEXT_SIZE, "lb")]
        return [art.floated(self.energies, art.picture(self.energies, energy, []), names), art.picture(self.hill, hill, [])]

    def temperatures(self, times: np.ndarray, temperatures: np.ndarray, common: float) -> np.ndarray:
        """The two temperatures against time so far `[steps, 2]`, the hot body's red and the cold one's blue, over
        the common temperature, dashed."""
        diagram = self.cooling
        colours = [art.thermal_colour(1.0), art.thermal_colour(0.0)]
        layers = [art.dashed(diagram, np.array(time_range), np.full(2, common), art.LINE, art.INK, (4.0, 3.0))]
        for body in range(2):
            layers += art.polylines(diagram, [(times, temperatures[:, body])], art.BOLD, colours[body], 1.0,
                                    art.SOLID)
            layers += art.dots(diagram, times[-1], temperatures[-1, body], colours[body])
        labels = [art.Label(time_range[1], common, "common", art.INK, art.TEXT_SIZE, "rb", (0, 3))]
        return art.picture(diagram, layers, labels)


def exchange(entropy_of: Callable[[Extensor], Extensor], temperatures_of: Callable[[Extensor], Extensor],
             transfer: Extensor, capacities: np.ndarray,
             states: Iterator[tuple[Extensor, Extensor, float]]) -> Iterator[np.ndarray]:
    """The bodies and the three charts at each state of the exchange, with its heat flow and its time."""
    first = next(states)
    charts = Charts(entropy_of, transfer, first[0])
    path, path_entropy, times, history = [], [], [], []
    for state, flow, time in chain([first], states):
        path.append(charts.read(state))
        path_entropy.append(float(entropy_of(state).to_array()))
        times.append(time)
        history.append(temperatures_of(state).to_array())
        temperatures = np.array(history)
        common = (capacities * temperatures[0]).sum() / capacities.sum()
        widest = np.abs(temperatures[0] - common).max()
        yield art.beside([bodies(temperatures[-1], capacities, widest)]
                         + charts.pictures(np.array(path), np.array(path_entropy), charts.read(flow))
                         + [charts.temperatures(np.array(times), temperatures, common)])


def draw(entropy_of: Callable[[Extensor], Extensor], temperatures_of: Callable[[Extensor], Extensor],
         transfer: Extensor, capacities: np.ndarray, state: Extensor, flow: Extensor):
    """The bodies and the three charts at a state, with its heat flow."""
    return art.figure(next(exchange(entropy_of, temperatures_of, transfer, capacities, iter([(state, flow, 0.0)]))))
