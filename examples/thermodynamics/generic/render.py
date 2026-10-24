"""A damped oscillator warming its damper, beside its phase portrait and its energies over time.

The phase portrait holds the ellipses of equal mechanical energy, kinetic and spring; the state spirals inward across
them. At the state, the reversible motion, blue, runs along its ellipse, and the irreversible motion, green, crosses
the ellipses inward, the momentum it takes going into the store. The energies over time are stacked: the store's at
the bottom, then the spring's, then the kinetic; their sum stays flat.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids, strokes

# The charts' size in pixels and their ranges: position and momentum, time and energy.
ROWS, COLUMNS = 250, 260
POSITIONS, MOMENTA = (-2.6, 2.6), (-4.6, 4.6)
TIMES, ENERGIES = (0.0, 12.5), (0.0, 11.5)
# The mechanical energies of the ellipses, and the grid of the phase plane they are found on.
LEVELS = np.array([0.5, 1.5, 3.0, 5.0, 8.0, 11.0])
GRID_POSITIONS, GRID_MOMENTA = np.linspace(-2.6, 2.6, 121), np.linspace(-4.6, 4.6, 121)
# The bands' colours: the store, the spring, the motion; how far ahead the rates' arrows look, in seconds, at most and
# in pixels; line widths.
BANDS = np.stack([art.ORANGE, art.rgb("#6bbcaf"), art.BLUE])
LOOK, ARROW_MOST = 0.25, 40.0
THIN, BOLD = 1.0, 2.2
# The oscillator's picture, pixels across.
APPARATUS_PIXELS = 230


def oscillator(position: float, warmth: float) -> np.ndarray:
    """The mass on its floor, held to the wall by a spring and a damper, the damper coloured by its temperature."""
    mv = solids.mv
    body = art.box(np.array([position, 0, 0.65]), np.array([0.3, 0.4, 0.38]))
    floor = art.box(np.array([0, 0, 0.18]), np.array([2.9, 0.62, 0.05]))
    wall = art.box(np.array([-2.75, 0, 0.78]), np.array([0.08, 0.58, 0.55]))
    damper = art.box(np.array([-2.0, 0, 0.4]), np.array([0.5, 0.17, 0.15]))
    rod = solids.rods(mv.point([-1.5, 0, 0.4]), mv.point([position - 0.3, 0, 0.4]), 0.045)[None]
    phase = np.linspace(0, 12 * np.pi, 160)
    coil = np.stack([np.linspace(-2.67, position - 0.3, len(phase)), 0.13 * np.sin(phase), 0.98 + 0.13 * np.cos(phase)], axis=-1)
    corners = mv.point(np.array([[x, y, z] for x in (-2.95, 2.5) for y in (-0.65, 0.65) for z in (0.1, 1.4)]))
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-12), 0.04, APPARATUS_PIXELS)
    spring = strokes.polylines([(mv.point(coil), art.INK)], 1.6, view)
    return solids.draw([(floor, art.GREY), (wall, art.GREY), (body, art.BLUE), (rod, art.GREY),
                        (damper, art.thermal_colour(warmth))], view, art.WHITE, inks=[], translucent=[],
                       meshes=[(spring, 1.0)])


class Charts:
    """The phase portrait's ellipses, sampled once, and the drawing of the motion so far on both charts."""

    def __init__(self, energies_of: Callable, mv) -> None:
        self.read = lambda states: np.stack([(mv.q & states).to_array(), (mv.p & states).to_array()], axis=-1)
        self.phase = art.Diagram(POSITIONS, MOMENTA, (0.0, 0.0), ROWS, COLUMNS, ("q", "p"))
        self.energy = art.Diagram(TIMES, ENERGIES, (0.0, 0.0), ROWS, COLUMNS, ("t", "E"))
        grid = mv.q.dual() * GRID_POSITIONS[None, :] + mv.p.dual() * GRID_MOMENTA[:, None]
        kinetic, spring, _ = energies_of(grid)
        ellipses = art.level_lines(GRID_POSITIONS, GRID_MOMENTA, (kinetic + spring).to_array(), LEVELS)
        self.ellipses = art.polylines(self.phase, [(line[:, 0], line[:, 1]) for line in ellipses], THIN, art.FAINT,
                                      1.0, art.SOLID)

    def pictures(self, path: np.ndarray, rates: np.ndarray, times: np.ndarray, energies: np.ndarray) -> list[np.ndarray]:
        """The spiral so far with the rates' arrows at its end `[2, 2]`, and the stacked energies so far
        `[steps, 3]`."""
        now = path[-1]
        arrows = []
        for rate, colour in zip(rates, [art.BLUE, art.GREEN]):
            reach = np.linalg.norm(rate * LOOK / (np.array([np.ptp(POSITIONS), np.ptp(MOMENTA)]) / COLUMNS))
            if reach > 3:
                arrows += art.arrows(self.phase, now[:1], now[1:], now[:1] + rate[0], now[1:] + rate[1],
                                     min(reach, ARROW_MOST), 1.8, colour)
        phase = (self.ellipses + art.polylines(self.phase, [(path[:, 0], path[:, 1])], BOLD, art.ORANGE, 1.0, art.SOLID)
                 + arrows + art.dots(self.phase, now[0], now[1], art.ORANGE))
        stacked = np.cumsum(energies, axis=1)                                                    # [steps, 3]
        floors = np.concatenate([np.zeros((len(times), 1)), stacked[:, :-1]], axis=1)
        bands = [art.areas(self.energy, np.concatenate([times, times[::-1]]),
                           np.concatenate([stacked[:, band], floors[::-1, band]]), BANDS[band], 0.75)
                 for band in range(3) if len(times) > 1]
        total = art.dashed(self.energy, np.array(TIMES), np.full(2, stacked[0, -1]), 1.2, art.INK, (4.0, 3.0))
        labels = [art.Label(times[-1], (floors[-1, band] + stacked[-1, band]) / 2, name, art.INK, art.LABEL_SIZE,
                            "lm", (4, 0))
                  for band, name in enumerate(["U", "V", "K"]) if stacked[-1, band] - floors[-1, band] > 0.6]
        return [art.picture(self.phase, phase, []), art.picture(self.energy, bands + [total], labels)]


def damping(energies_of: Callable, rates_of: Callable, states: Iterator[Extensor], frame_time: float
            ) -> Iterator[np.ndarray]:
    """The oscillator and the two charts at each state as it comes, the states a frame time apart."""
    first = next(states)
    charts = Charts(energies_of, first.context.multivector)
    path, energies = [], []
    for state in chain([first], states):
        path.append(charts.read(state))
        # The store's energy, then the spring's and the kinetic, stacked from the bottom up.
        energies.append([float(energy.to_array()) for energy in energies_of(state)][::-1])
        rates = np.stack([charts.read(rate) for rate in rates_of(state)])
        # The damper's colour: the share of the mechanical energy it has taken in.
        start, now = np.array(energies[0]), np.array(energies[-1])
        warmth = (now[0] - start[0]) / (start.sum() - start[0])
        times = np.arange(len(path)) * frame_time
        yield art.beside([oscillator(path[-1][0], warmth)]
                         + charts.pictures(np.array(path), rates, times, np.array(energies)))


def draw(energies_of: Callable, rates_of: Callable, state: Extensor):
    """The oscillator and the two charts at a state."""
    return art.figure(next(damping(energies_of, rates_of, iter([state]), 0.0)))
