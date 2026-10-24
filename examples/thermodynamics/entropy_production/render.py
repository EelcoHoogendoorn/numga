"""Two bodies sharing heat, beside the plane of their energies and the entropy along the line of fixed total energy.

On the left chart, the bodies' energies, with the entropy's level lines faint; the state runs along the line of fixed
total energy, straight in the direction of the transfer, and its arrow is the heat flow. It comes to rest where an
entropy level line touches the line. On the right, the entropy along that line against body a's energy: a hill whose
top is equilibrium, which the state climbs; the tangent's slope is how much entropy a unit of energy moved to body a
would produce, and the flow runs uphill.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids

# The charts' size in pixels and their ranges: the two energies, and body a's energy against the entropy.
ROWS, COLUMNS = 250, 260
ENERGIES = (0.0, 4.3)
ENTROPIES = (-1.6, 1.3)
# The entropy's levels, the grid of energies they are found on, and the samples along the line of fixed energy.
LEVELS = np.linspace(-0.4, 1.6, 11)
GRID = np.linspace(0.04, 4.3, 160)
N_LINE = 240
# How far an arrow reaches in pixels for a unit of flow, and the tangent's half length in units of energy.
ARROW_SCALE, TANGENT = 40.0, 0.8
# The bodies' picture, pixels across, and its temperature colours' range; line widths.
BLOCK_PIXELS = 220
COLDEST, HOTTEST = 0.6, 2.8
THIN, BOLD = 1.0, 2.2


def bodies(temperatures: np.ndarray, capacities: np.ndarray) -> np.ndarray:
    """The two bodies, each as large as its heat capacity and coloured by its temperature, joined by a bar."""
    sizes = 0.5 * np.cbrt(capacities)
    centres = np.array([[-1.0, 0, sizes[0]], [1.0, 0, sizes[1]]])
    blocks = solids.boxes(solids.translator(solids.mv.point(centres) - solids.origin), np.repeat(sizes[:, None], 3, axis=1))
    colours = np.stack([art.thermal_colour((temperature - COLDEST) / (HOTTEST - COLDEST)) for temperature in temperatures])
    bar = art.box(np.array([0, 0, 0.45]), np.array([0.5, 0.12, 0.1]))
    corners = solids.mv.point(np.array([[x, y, z] for x in (-1.75, 1.75) for y in (-0.75, 0.75) for z in (0, 1.6)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-16), 0.06, BLOCK_PIXELS)
    return solids.draw([(blocks, colours), (bar, art.GREY)], view, art.WHITE, inks=[], translucent=[], meshes=[])


class Charts:
    """The two charts' fixed lines, sampled once from the entropy, and the drawing of the state on them."""

    def __init__(self, entropy_of: Callable[[Extensor], Extensor], transfer: Extensor, state: Extensor) -> None:
        mv = state.context.multivector
        self.read = lambda states: np.stack([(mv.a & states).to_array(), (mv.b & states).to_array()], axis=-1)
        self.energies = art.Diagram(ENERGIES, ENERGIES, (0.0, 0.0), ROWS, COLUMNS, ("U_a", "U_b"))
        self.hill = art.Diagram(ENERGIES, ENTROPIES, (0.0, ENTROPIES[0]), ROWS, COLUMNS, ("U_a", "S"))
        grid = mv.a.dual() * GRID[None, :] + mv.b.dual() * GRID[:, None]
        levels = art.level_lines(GRID, GRID, entropy_of(grid).to_array(), LEVELS)
        # The line of fixed total energy through the state, along the transfer, from body a holding almost none of the
        # energy to almost all of it.
        start = self.read(state)
        along = np.linspace(0.02, 0.98, N_LINE) * start.sum()
        line = state + transfer * ((along - start[0]) / self.read(transfer)[0])
        self.line = self.read(line)
        self.line_entropy = entropy_of(line).to_array()
        top = np.argmax(self.line_entropy)
        self.top = (self.line[top], self.line_entropy[top])
        # The entropy's level line through the top of the hill touches the line of fixed energy there.
        touching = art.level_lines(GRID, GRID, entropy_of(grid).to_array(), np.array([self.line_entropy.max()]))
        self.energy_layers = (art.polylines(self.energies, [(level[:, 0], level[:, 1]) for level in levels], THIN,
                                            art.FAINT, 1.0, art.SOLID)
                              + art.polylines(self.energies, [(level[:, 0], level[:, 1]) for level in touching], 1.2,
                                              art.GREEN, 1.0, art.SOLID)
                              + art.polylines(self.energies, [(self.line[:, 0], self.line[:, 1])], 1.4, art.INK, 1.0,
                                              art.SOLID)
                              + art.dots(self.energies, self.top[0][0], self.top[0][1], art.GREEN))
        self.hill_layers = (art.polylines(self.hill, [(self.line[:, 0], self.line_entropy)], 1.4, art.INK, 1.0, art.SOLID)
                            + art.dots(self.hill, self.top[0][0], self.top[1], art.GREEN))
        self.entropy_of = entropy_of

    def pictures(self, path: np.ndarray, path_entropy: np.ndarray, flow: np.ndarray) -> list[np.ndarray]:
        now, now_entropy = path[-1], path_entropy[-1]
        energy = (self.energy_layers
                  + art.polylines(self.energies, [(path[:, 0], path[:, 1])], BOLD, art.ORANGE, 1.0, art.SOLID)
                  + (art.arrows(self.energies, now[:1], now[1:], now[:1] + flow[0], now[1:] + flow[1],
                                ARROW_SCALE * np.linalg.norm(flow), 1.6, art.INK) if np.linalg.norm(flow) > 1e-3 else [])
                  + art.dots(self.energies, now[0], now[1], art.ORANGE))
        # The hill's slope at the state, from its samples on either side.
        slope = np.interp(now[0], self.line[:, 0], np.gradient(self.line_entropy, self.line[:, 0]))
        tangent = np.array([now[0] - TANGENT, now[0] + TANGENT])
        hill = (self.hill_layers
                + art.polylines(self.hill, [(path[:, 0], path_entropy)], BOLD, art.ORANGE, 1.0, art.SOLID)
                + art.polylines(self.hill, [(tangent, now_entropy + slope * (tangent - now[0]))], 1.4, art.ORANGE, 1.0,
                                art.SOLID)
                + art.dots(self.hill, now[0], now_entropy, art.ORANGE))
        return [art.picture(self.energies, energy, []), art.picture(self.hill, hill, [])]


def exchange(entropy_of: Callable[[Extensor], Extensor], temperatures_of: Callable, transfer: Extensor,
             capacities: np.ndarray, states: Iterator[tuple[Extensor, Extensor]]) -> Iterator[np.ndarray]:
    """The bodies and the two charts at each state of the exchange, with its heat flow."""
    first = next(states)
    charts = Charts(entropy_of, transfer, first[0])
    path, path_entropy = [], []
    for state, flow in chain([first], states):
        path.append(charts.read(state))
        path_entropy.append(float(entropy_of(state).to_array()))
        temperatures = np.array([float(temperature.to_array()) for temperature in temperatures_of(state)])
        yield art.beside([bodies(temperatures, capacities)]
                         + charts.pictures(np.array(path), np.array(path_entropy), charts.read(flow)))


def draw(entropy_of: Callable[[Extensor], Extensor], temperatures_of: Callable, transfer: Extensor,
         capacities: np.ndarray, state: Extensor, flow: Extensor):
    """The bodies and the two charts at a state, with its heat flow."""
    return art.figure(next(exchange(entropy_of, temperatures_of, transfer, capacities, iter([(state, flow)]))))
