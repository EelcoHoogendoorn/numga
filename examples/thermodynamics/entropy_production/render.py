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
from numga.backend.jax import derivative
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
    # Each block rests on the floor beneath its place: its centre half its size up.
    centres = solids.mv.point([[-1, 0, 0], [1, 0, 0]]) + art.rise * sizes                # [bodies] Point-
    blocks = solids.boxes(solids.translator(centres - solids.origin), np.repeat(sizes[:, None], 3, axis=1))
    common = (capacities * temperatures).sum() / capacities.sum()
    colours = np.stack([art.thermal_colour(art.relaxing(temperature - common, widest, colour_decades))
                        for temperature in temperatures])
    bar = art.box(solids.mv.point([0, 0, 0.45]), np.array([0.5, 0.12, 0.1]))
    corners = solids.mv.point(np.array([[x, y, z] for x in (-1.75, 1.75) for y in (-0.75, 0.75) for z in (0, 1.6)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-16), 0.06, block_pixels)
    return solids.draw([(blocks, colours), (bar, art.grey)], view, art.white, inks=[], translucent=[], meshes=[])


class Charts:
    """The two charts' fixed lines, sampled once from the entropy, and the drawing of the state on them."""

    def __init__(self, entropy_of: Callable[[Extensor], Extensor], transfer: Extensor, state: Extensor) -> None:
        mv = state.context.multivector
        self.entropy_of, self.transfer, self.mv = entropy_of, transfer, mv
        self.energies = art.Diagram(energy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("U_a", "U_b"))
        self.cooling = art.Diagram(time_range, temperature_range, (0.0, 0.0), n_rows, n_columns, ("t", "T"))
        self.hill = art.Diagram(energy_range, entropy_range, (0.0, entropy_range[0]), n_rows, n_columns, ("U_a", "S"))
        grid = mv.a.dual() * grid_energies[None, :] + mv.b.dual() * grid_energies[:, None]
        grid_entropy = entropy_of(grid).to_array()
        levels = art.level_lines(grid_energies, grid_energies, grid_entropy, entropy_levels)
        # The line of fixed total energy through the state, along the transfer, from body a holding almost none of the
        # energy to almost all of it.
        held_by_a = ((mv.a + mv.b) & state) * np.linspace(0.02, 0.98, n_line)         # [line] Scalar+
        line = state + transfer * ((held_by_a - (mv.a & state)) / (mv.a & transfer))    # [line] State-
        line_entropy = entropy_of(line)                                                 # [line] Scalar+
        top = line_entropy.argmax()
        energy_line, hill_line = self.energy_points(line), self.hill_points(line)       # [line] Point-, Point-
        # The entropy's level line through the top of the hill touches the line of fixed energy there.
        touching = art.level_lines(grid_energies, grid_energies, grid_entropy, line_entropy[top].to_array()[None])
        self.level_lines = [art.chart_points(*level_line.T) for _, level_line in levels]
        self.energy_layers = (art.family(self.energies, self.level_lines, "entropy", art.thin)
                              + art.polylines(self.energies,
                                              [art.chart_points(*level_line.T) for _, level_line in touching],
                                              art.medium, art.boundary, 1.0, art.SOLID)
                              + art.polylines(self.energies, [energy_line], art.medium, art.ink, 1.0, art.SOLID)
                              + art.dots(self.energies, energy_line[top][None], art.green))
        self.hill_layers = (art.polylines(self.hill, [hill_line], art.medium, art.ink, 1.0, art.SOLID)
                            + art.dots(self.hill, hill_line[top][None], art.green))

    def energy_points(self, states: Extensor) -> art.Point:
        """States on the chart of the two bodies' energies, read by pairing."""
        return art.chart_points(art.context.include_from_algebra(self.mv.a & states), art.context.include_from_algebra(self.mv.b & states))

    def hill_points(self, states: Extensor) -> art.Point:
        """States on the chart of the entropy against body a's energy."""
        return art.chart_points(art.context.include_from_algebra(self.mv.a & states), art.context.include_from_algebra(self.entropy_of(states)))

    def pictures(self, state: Extensor, flow: Extensor, path: list[art.Point], hill_path: list[art.Point]
                 ) -> list[np.ndarray]:
        """The two charts with the paths so far, each step's state on either chart, and the state's heat flow."""
        now, hill_now = path[-1], hill_path[-1]                                         # [] Point-, Point-
        # The flow's arrow: as long as the flow, on the chart, times so many pixels.
        ahead = self.energy_points(state + flow)                                       # [] Point-
        flow_size = (ahead - now).dual().norm()
        energy = (self.energy_layers
                  + art.polylines(self.energies, [stack(path)], art.bold, art.orange, 1.0, art.SOLID)
                  + (art.arrows(self.energies, now[None], ahead[None], arrow_pixels_per_flow * flow_size, art.medium,
                                art.ink)
                     if flow_size > 1e-3 else [])
                  + art.dots(self.energies, now[None], art.orange))
        # The hill's tangent at the state: the change along the transfer, of body a's energy and of the entropy.
        mv = self.mv
        tangent = art.chart_directions(art.context.include_from_algebra(mv.a & self.transfer),
                                     art.context.include_from_algebra(derivative(self.entropy_of)(state)(self.transfer)))   # [] Point-
        reach = art.context.include_from_algebra(mv.a & state) + tangent_half_length * np.array([-1, 1])
        hill = (self.hill_layers
                + art.polylines(self.hill, [stack(hill_path)], art.bold, art.orange, 1.0, art.SOLID)
                + art.polylines(self.hill, [art.tangent_lines(hill_now, tangent, reach)], art.medium, art.orange, 1.0,
                                art.SOLID)
                + art.dots(self.hill, hill_now[None], art.orange))
        names = [art.Floating(art.along(self.level_lines, 8), "S = const", art.families["entropy"], art.text_size,
                              "lb")]
        return [art.floated(self.energies, art.picture(self.energies, energy, []), names),
                art.picture(self.hill, hill, [])]

    def temperatures(self, times: np.ndarray, temperatures: art.Scalar, common: art.Scalar) -> np.ndarray:
        """The two temperatures against time so far `[steps, 2]`, the hot body's red and the cold one's blue, over
        the common temperature, dashed."""
        diagram = self.cooling
        colours = [art.thermal_colour(1.0), art.thermal_colour(0.0)]
        curves = art.chart_points(times[:, None], temperatures)                        # [steps, bodies] Point-
        layers = [art.dashed(diagram, art.chart_points(np.array(time_range), common), art.medium, art.ink, (4.0, 3.0))]
        for body in range(2):
            layers += art.polylines(diagram, [curves[:, body]], art.bold, colours[body], 1.0, art.SOLID)
            layers += art.dots(diagram, curves[-1, body][None], colours[body])
        labels = [art.Label(art.chart_points(time_range[1], common), "common", art.ink, art.text_size, "rb", (0, 3))]
        return art.picture(diagram, layers, labels)


def exchange(entropy_of: Callable[[Extensor], Extensor], temperatures_of: Callable[[Extensor], Extensor],
             transfer: Extensor, capacities: np.ndarray,
             states: Iterator[tuple[Extensor, Extensor, float]]) -> Iterator[np.ndarray]:
    """The bodies and the three charts at each state of the exchange, with its heat flow and its time."""
    first = next(states)
    charts = Charts(entropy_of, transfer, first[0])
    path, hill_path, times, history = [], [], [], []
    for state, flow, time in chain([first], states):
        path.append(charts.energy_points(state))
        hill_path.append(charts.hill_points(state))
        times.append(time)
        history.append(art.context.include_from_algebra(temperatures_of(state)))
        temperatures = stack(history)
        common = (capacities * temperatures[0]).sum() / capacities.sum()
        gaps = (temperatures[0] - common).abs()
        widest = gaps.max()
        yield art.beside([bodies(temperatures[-1].to_array(), capacities, float(widest.to_array()))]
                         + charts.pictures(state, flow, path, hill_path)
                         + [charts.temperatures(np.array(times), temperatures, common)])


def draw(entropy_of: Callable[[Extensor], Extensor], temperatures_of: Callable[[Extensor], Extensor],
         transfer: Extensor, capacities: np.ndarray, state: Extensor, flow: Extensor):
    """The bodies and the three charts at a state, with its heat flow."""
    return art.figure(next(exchange(entropy_of, temperatures_of, transfer, capacities, iter([(state, flow, 0.0)]))))
