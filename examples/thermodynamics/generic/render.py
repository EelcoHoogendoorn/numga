"""A damped oscillator warming its damper, drawn around the entropy the damping forces up.

At fixed total energy the store's entropy is a function of the position and momentum alone, the store holding all the
energy they leave over; its level lines on the phase portrait are the ellipses of equal mechanical energy, teal, each
marked with its entropy. The reversible motion, the Poisson map of the energy's gradient, runs along them, keeping the
entropy; the irreversible motion, the friction mobility of the entropy's gradient, crosses them inward, raising it.
The state spirals in, up the entropy. The chart beside it draws the entropy against time, rising in steps: fast while
the mass moves fast, flat at its turning points, never falling. The last chart draws the entropy against the
mechanical energy left, along the fixed total energy: the state slides to the curve's top, where the store holds
everything.

The damper is coloured from icy blue to red heat by how far it still has to warm, half as it is and half on its
logarithm, so that its last warming stays visible without crowding its first.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import chain

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from examples.util.wgsl import solids, strokes

# The charts' size in pixels and their ranges: position and momentum, time and entropy, mechanical energy and entropy.
chart_rows, chart_columns = 300, 310
position_range, momentum_range = (-2.6, 2.6), (-4.6, 4.6)
time_range, entropy_range = (0.0, 12.5), (0.0, 3.6)
mechanical_range = (0.0, 8.6)
# The entropies of the level lines, and the grid of the phase plane they are found on.
entropy_levels = np.array([0.0, 1.0, 1.8, 2.3, 2.7, 3.0, 3.15])
grid_positions, grid_momenta = np.linspace(-2.6, 2.6, 121), np.linspace(-4.6, 4.6, 121)
# How far ahead the rates' arrows look, in seconds, at most in pixels, and their colours.
look_ahead, arrow_most = 0.25, 44.0
reversible_colour, irreversible_colour = art.rgb("#5b6fd6"), art.rgb("#d0402f")
# The oscillator's picture, pixels across, and the decades of the damper's remaining warming its colour spans.
apparatus_pixels = 230
decades = 2.0


def phase_points(states: Extensor) -> art.Point:
    """States on the phase portrait: their position and momentum, read by pairing with `mv.q` and `mv.p`."""
    mv = states.context.multivector
    return art.chart_points(art.context.include_from_algebra(mv.q & states), art.context.include_from_algebra(mv.p & states))


def oscillator(position: solids.Scalar, warmth: float) -> np.ndarray:
    """The mass on its floor, held to the wall by a spring and a damper, the damper coloured by its temperature."""
    mv = solids.mv
    # The mass, and the hook on its near side where the rod and the spring hold it, carried along by its position.
    carried = solids.translator(mv.direction([1, 0, 0]) * position)                   # [] Motor+
    body = carried >> art.box(mv.point([0, 0, 0.65]), np.array([0.3, 0.4, 0.38]))
    floor = art.box(mv.point([0, 0, 0.18]), np.array([2.9, 0.62, 0.05]))
    wall = art.box(mv.point([-2.75, 0, 0.78]), np.array([0.08, 0.58, 0.55]))
    damper = art.box(mv.point([-2.0, 0, 0.4]), np.array([0.5, 0.17, 0.15]))
    rod = solids.rods(mv.point([-1.5, 0, 0.4]), carried >> mv.point([-0.3, 0, 0.4]), 0.045)[None]
    # The spring's coil: points along its axis from the wall to the hook, each with the coil's radius turned about
    # that axis.
    coil_angles = np.linspace(0, 12 * np.pi, 160)
    fixed, hook = mv.point([-2.67, 0, 0.98]), carried >> mv.point([-0.3, 0, 0.98])       # [] Point-, Point-
    centres = fixed + (hook - fixed) * np.linspace(0, 1, len(coil_angles))               # [coil] Point-
    coil = centres + ((mv.yz * (coil_angles / 2)).exp() >> mv.direction([0, 0, 0.13]))   # [coil] Point-
    corners = mv.point([[x, y, z] for x in (-2.95, 2.5) for y in (-0.65, 0.65) for z in (0.1, 1.4)])
    view = solids.view([corners], 0.0, np.radians(18), np.radians(-12), 0.04, apparatus_pixels)
    spring = strokes.polylines([(coil, art.ink)], 1.6, view)
    return solids.draw([(floor, art.grey), (wall, art.grey), (body, art.blue), (rod, art.grey),
                        (damper, art.thermal_colour(warmth))], view, art.white, inks=[], translucent=[],
                       meshes=[(spring, 1.0)])


class Charts:
    """The level lines of the entropy at fixed energy, and its curve against the mechanical energy, sampled once; and
    the drawing of the motion so far on the three charts."""

    def __init__(self, energies_of: Callable, surface_entropy_of: Callable, mv) -> None:
        self.phase = art.Diagram(position_range, momentum_range, (0.0, 0.0), chart_rows, chart_columns, ("q", "p"))
        self.rising = art.Diagram(time_range, entropy_range, (0.0, 0.0), chart_rows, chart_columns, ("t", "S"))
        self.hill = art.Diagram(mechanical_range, entropy_range, (0.0, 0.0), chart_rows, chart_columns, ("E_mech", "S"))
        grid = mv.q.dual() * grid_positions[None, :] + mv.p.dual() * grid_momenta[:, None]
        entropy = surface_entropy_of(grid).to_array()
        levels = art.level_lines(grid_positions, grid_momenta, np.where(np.isfinite(entropy), entropy, -10.0),
                                 entropy_levels)
        self.level_lines = [art.chart_points(*line.T) for _, line in levels]
        self.levels = art.family(self.phase, self.level_lines, "entropy", art.thin)
        # Each level's entropy written at its upper right, its point furthest along the chart's diagonal.
        diagonal = art.mv.x / np.ptp(position_range) + art.mv.y / np.ptp(momentum_range)   # [] Plane+
        self.level_labels = []
        for (level, _), points in zip(levels, self.level_lines):
            corner = points[(diagonal & points).argmax()]                                # [] Point-
            if (art.mv.y & corner) < momentum_range[1] - 0.3:
                self.level_labels.append(art.Label(corner, f"{level:g}", art.families["entropy"], art.tick_size, "lb",
                                                   (2, 2)))
        # The entropy at fixed energy against the mechanical energy, along the positions at rest.
        rest = mv.q.dual() * np.linspace(0, 2.0, 200)
        kinetic, spring, _ = energies_of(rest)
        self.curve = art.chart_points(art.context.include_from_algebra(kinetic + spring), art.context.include_from_algebra(surface_entropy_of(rest)))

    def pictures(self, path: list[art.Point], rates: list[art.Point], times: np.ndarray, entropies: art.Scalar,
                 mechanical: art.Scalar) -> list[np.ndarray]:
        """The spiral so far over the entropy's level lines, with the rates' arrows at its end, from the state toward
        where each rate alone would carry it a look ahead on; the entropy against time; and the entropy against the
        mechanical energy left."""
        now = path[-1]                                                                   # [] Point-
        arrows = []
        for toward, colour in zip(rates, [reversible_colour, irreversible_colour]):
            arrow_length = (self.phase.placement(toward) - self.phase.placement(now)).dual().norm()
            if arrow_length > 3:
                arrows += art.arrows(self.phase, now[None], toward[None], arrow_length.clip(0, arrow_most), art.medium,
                                     colour)
        phase = (self.levels + art.polylines(self.phase, [stack(path)], art.bold, art.orange, 1.0, art.SOLID)
                 + arrows + art.dots(self.phase, now[None], art.orange))
        phase_labels = self.level_labels + [
            art.Label(art.chart_points(position_range[0], momentum_range[1]), "reversible", reversible_colour,
                      art.text_size, "lt", (4, 0)),
            art.Label(art.chart_points(position_range[0], momentum_range[1] - 0.8), "dissipative", irreversible_colour,
                      art.text_size, "lt", (4, 0))]
        rising_path = art.chart_points(times, entropies)                                 # [steps] Point-
        hill_path = art.chart_points(mechanical, entropies)                              # [steps] Point-
        rising = (art.polylines(self.rising, [rising_path], art.bold, art.orange, 1.0, art.SOLID)
                  + art.dots(self.rising, rising_path[-1:], art.orange))
        hill = (art.polylines(self.hill, [self.curve], art.medium, art.families["entropy"], 1.0, art.SOLID)
                + art.polylines(self.hill, [hill_path], art.bold, art.orange, 1.0, art.SOLID)
                + art.dots(self.hill, hill_path[-1:], art.orange))
        # The curve's name at the chart's right edge, level with the curve's end: the line through that end along the
        # mechanical energy, met with the edge.
        right_edge = art.mv.x - art.mv.w * mechanical_range[1]                           # [] Plane+
        curve_name = art.planar.unit((self.curve[-1] & art.mv.x.dual()) ^ right_edge)    # [] Point-
        hill_labels = [art.Label(curve_name, "E = const", art.families["entropy"], art.text_size, "rb", (0, 4))]
        names = [art.Floating(art.along(self.level_lines, 6), "S = const", art.families["entropy"], art.text_size,
                              "lt")]
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
    start = stack(energies_of(first))
    for state in chain([first], states):
        kinetic, spring, internal = energies_of(state)
        path.append(phase_points(state))
        entropies.append(art.context.include_from_algebra(mv.s & state))
        mechanical.append(art.context.include_from_algebra(kinetic + spring))
        rates = [phase_points(state + rate * look_ahead) for rate in rates_of(state)]
        # The share of its warming the damper has still to do, half as it is and half on its logarithm.
        remaining = float(((start.sum() - internal).clip(1e-300, np.inf) / (start.sum() - start[2])).to_array())
        warmth = 1.0 - 0.5 * remaining - 0.5 * np.clip(np.log10(remaining) / decades + 1, 0.0, 1.0)
        times = np.arange(len(path)) * frame_time
        yield art.beside([oscillator(solids.context.include_from_algebra(mv.q & state), warmth)]
                         + charts.pictures(path, rates, times, stack(entropies), stack(mechanical)))


def draw(energies_of: Callable, surface_entropy_of: Callable, rates_of: Callable, state: Extensor):
    """The oscillator and the three charts at a state."""
    return art.figure(next(damping(energies_of, surface_entropy_of, rates_of, iter([state]), 0.0)))
