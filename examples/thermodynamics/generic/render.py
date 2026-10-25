"""A damped oscillator warming its damper, drawn around the entropy the damping forces up.

At fixed total energy the store's entropy is a function of the position and momentum alone, the store holding all the
energy they leave over; its level lines on the phase portrait are the ellipses of equal mechanical energy, teal, each
marked with its entropy. The reversible motion, the Poisson map of the energy's covector, runs along them, keeping the
entropy; the irreversible motion, the friction mobility of the entropy's covector, crosses them inward, raising it.
The state spirals in, up the entropy. Beside it, the entropy against time, rising in steps: fast while the mass moves
fast, flat at its turning points, never falling. Last, the entropy against the mechanical energy left, along the
fixed total energy: the state slides to the curve's top, where the store holds everything.

The damper is coloured from icy blue to red heat by how far it still has to warm, half as it is and half on its
logarithm, so that its last warming stays visible without crowding its first.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids, strokes

# The charts' size in pixels and their ranges: position and momentum, time and entropy, mechanical energy and entropy.
ROWS, COLUMNS = 300, 310
POSITIONS, MOMENTA = (-2.6, 2.6), (-4.6, 4.6)
TIMES, ENTROPIES = (0.0, 12.5), (0.0, 3.6)
MECHANICAL = (0.0, 8.6)
# The entropies of the level lines, and the grid of the phase plane they are found on.
LEVELS = np.array([0.0, 1.0, 1.8, 2.3, 2.7, 3.0, 3.15])
GRID_POSITIONS, GRID_MOMENTA = np.linspace(-2.6, 2.6, 121), np.linspace(-4.6, 4.6, 121)
# How far ahead the rates' arrows look, in seconds, at most in pixels, and their colours; line widths.
LOOK, ARROW_MOST = 0.25, 44.0
REVERSIBLE, IRREVERSIBLE = art.rgb("#5b6fd6"), art.rgb("#d0402f")
THIN, BOLD = 1.0, 2.2
# The oscillator's picture, pixels across, and the decades of the damper's remaining warming its colour spans.
APPARATUS_PIXELS = 230
DECADES = 2.0


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
    """The level lines of the entropy at fixed energy, and its curve against the mechanical energy, sampled once; and
    the drawing of the motion so far on the three charts."""

    def __init__(self, energies_of: Callable, surface_entropy_of: Callable, mv) -> None:
        self.read = lambda states: np.stack([(mv.q & states).to_array(), (mv.p & states).to_array()], axis=-1)
        self.phase = art.Diagram(POSITIONS, MOMENTA, (0.0, 0.0), ROWS, COLUMNS, ("q", "p"))
        self.rising = art.Diagram(TIMES, ENTROPIES, (0.0, 0.0), ROWS, COLUMNS, ("t", "S"))
        self.hill = art.Diagram(MECHANICAL, ENTROPIES, (0.0, 0.0), ROWS, COLUMNS, ("E_mech", "S"))
        grid = mv.q.dual() * GRID_POSITIONS[None, :] + mv.p.dual() * GRID_MOMENTA[:, None]
        entropy = surface_entropy_of(grid).to_array()
        levels = art.level_lines(GRID_POSITIONS, GRID_MOMENTA, np.where(np.isfinite(entropy), entropy, -10.0), LEVELS)
        self.levels = art.family(self.phase, [(level, line[:, 0], line[:, 1]) for level, line in levels], "entropy", THIN)
        self.level_lines = [(line[:, 0], line[:, 1]) for _, line in levels]
        # Each level's entropy written at its upper right, where it crosses the diagonal of the chart.
        self.level_labels = []
        for level, line in levels:
            corner = line[np.argmax(line[:, 0] / np.ptp(POSITIONS) + line[:, 1] / np.ptp(MOMENTA))]
            if corner[1] < MOMENTA[1] - 0.3:
                self.level_labels.append(art.Label(corner[0], corner[1], f"{level:g}", art.FAMILIES["entropy"],
                                                   art.LABEL_SIZE, "lb", (2, 2)))
        # The entropy at fixed energy against the mechanical energy, along the positions at rest.
        rest = mv.q.dual() * np.linspace(0, 2.0, 200)
        kinetic, spring, _ = energies_of(rest)
        self.curve = ((kinetic + spring).to_array(), surface_entropy_of(rest).to_array())

    def pictures(self, path: np.ndarray, rates: np.ndarray, times: np.ndarray, entropies: np.ndarray,
                 mechanical: np.ndarray) -> list[np.ndarray]:
        """The spiral so far over the entropy's level lines, with the rates' arrows at its end `[2, 2]`; the entropy
        against time; and the entropy against the mechanical energy left."""
        now = path[-1]
        arrows, arrow_labels = [], []
        for rate, colour, name in zip(rates, [REVERSIBLE, IRREVERSIBLE], ["reversible", "dissipative"]):
            reach = np.linalg.norm(rate * LOOK / (np.array([np.ptp(POSITIONS), np.ptp(MOMENTA)]) / COLUMNS))
            if reach > 3:
                arrows += art.arrows(self.phase, now[:1], now[1:], now[:1] + rate[0], now[1:] + rate[1],
                                     min(reach, ARROW_MOST), 1.8, colour)
        phase = (self.levels + art.polylines(self.phase, [(path[:, 0], path[:, 1])], BOLD, art.ORANGE, 1.0, art.SOLID)
                 + arrows + art.dots(self.phase, now[0], now[1], art.ORANGE))
        phase_labels = self.level_labels + [
            art.Label(POSITIONS[0], MOMENTA[1], "reversible", REVERSIBLE, art.LABEL_SIZE, "lt", (4, 0)),
            art.Label(POSITIONS[0], MOMENTA[1] - 0.8, "dissipative", IRREVERSIBLE, art.LABEL_SIZE, "lt", (4, 0))]
        rising = (art.polylines(self.rising, [(times, entropies)], BOLD, art.ORANGE, 1.0, art.SOLID)
                  + art.dots(self.rising, times[-1], entropies[-1], art.ORANGE))
        hill = (art.polylines(self.hill, [self.curve], 1.6, art.FAMILIES["entropy"], 1.0, art.SOLID)
                + art.polylines(self.hill, [(mechanical, entropies)], BOLD, art.ORANGE, 1.0, art.SOLID)
                + art.dots(self.hill, mechanical[-1], entropies[-1], art.ORANGE))
        hill_labels = [art.Label(MECHANICAL[1], self.curve[1][-1], "E = const", art.FAMILIES["entropy"], art.LABEL_SIZE,
                                 "rb", (0, 4))]
        names = [art.Floating(*art.along(self.level_lines, 6), "S = const", art.FAMILIES["entropy"], art.LABEL_SIZE, "lt")]
        return [art.floated(self.phase, art.picture(self.phase, phase, phase_labels), names),
                art.picture(self.rising, rising, []),
                art.picture(self.hill, hill, hill_labels)]


def damping(energies_of: Callable, surface_entropy_of: Callable, rates_of: Callable, states: Iterator[Extensor],
            frame_time: float) -> Iterator[np.ndarray]:
    """The oscillator and the three charts at each state as it comes, the states a frame time apart."""
    first = next(states)
    mv = first.context.multivector
    charts = Charts(energies_of, surface_entropy_of, mv)
    path, entropies, mechanical = [], [], []
    start = np.array([float(energy.to_array()) for energy in energies_of(first)])
    for state in chain([first], states):
        kinetic, spring, internal = (float(energy.to_array()) for energy in energies_of(state))
        path.append(charts.read(state))
        entropies.append(float((mv.s & state).to_array()))
        mechanical.append(kinetic + spring)
        rates = np.stack([charts.read(rate) for rate in rates_of(state)])
        # The share of its warming the damper has still to do, half as it is and half on its logarithm.
        remaining = max(start.sum() - internal, 1e-300) / (start.sum() - start[2])
        warmth = 1.0 - 0.5 * remaining - 0.5 * np.clip(np.log10(remaining) / DECADES + 1, 0.0, 1.0)
        times = np.arange(len(path)) * frame_time
        yield art.beside([oscillator(path[-1][0], warmth)]
                         + charts.pictures(np.array(path), rates, times, np.array(entropies), np.array(mechanical)))


def draw(energies_of: Callable, surface_entropy_of: Callable, rates_of: Callable, state: Extensor):
    """The oscillator and the three charts at a state."""
    return art.figure(next(damping(energies_of, surface_entropy_of, rates_of, iter([state]), 0.0)))
