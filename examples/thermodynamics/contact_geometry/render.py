"""The Legendre change from the energy as a function of the entropy to the Helmholtz energy as a function of the
temperature, as tangent lines.

On the left, the energy against the entropy at several volumes, green, the section the example follows bold, and its
tangent lines, grey-blue, enveloping it. The moving tangent's slope is the temperature, and where it crosses zero entropy, its
height is the Helmholtz energy. On the right, the Helmholtz energy against the temperature, traced as the tangent moves,
with its own tangent: its slope is minus the entropy and its height at zero temperature the energy again.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Extensor
from examples.thermodynamics import rendering as art

# The charts' size in pixels and their ranges: entropy and energy, temperature and Helmholtz energy.
ROWS, COLUMNS = 300, 330
ENTROPIES, ENERGIES = (-0.8, 1.25), (0.0, 3.4)
TEMPERATURES, FREE_ENERGIES = (0.0, 2.2), (0.0, 3.4)
# Every so many samples of the section, a faint tangent; the samples along each tangent line; line widths.
EVERY, N_LINE = 3, 60
THIN, BOLD = 1.0, 2.2


def readings(points: Extensor) -> np.ndarray:
    """The entropy and the energy of points, read by pairing `[..., 2]`; after the Legendre change, the temperature and
    the Helmholtz energy."""
    mv = points.context.multivector
    return np.stack([(mv.s & points).to_array(), (mv.u & points).to_array()], axis=-1)


def tangent_lines(points: np.ndarray, slopes: np.ndarray, across: tuple[float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lines through points `[n, 2]` with slopes `[n]`, sampled across a range."""
    x = np.linspace(*across, N_LINE)
    return [(x, point[1] + slope * (x - point[0])) for point, slope in zip(points, slopes)]


class Chart:
    """One of the two charts: its curves, the faint tangents enveloping its section, and the moving tangent."""

    def __init__(self, diagram: art.Diagram, family: Extensor, points: Extensor, tangents: Extensor,
                 intercept_name: str) -> None:
        self.diagram, self.intercept_name = diagram, intercept_name
        self.family, self.section = readings(family), readings(points)                   # [volumes, samples, 2], [samples, 2]
        steps = readings(tangents)
        self.slopes = steps[:, 1] / steps[:, 0]
        self.layers = (art.polylines(diagram, tangent_lines(self.section[::EVERY], self.slopes[::EVERY], diagram.across),
                                     THIN, art.FAMILIES["level"], 0.6, art.SOLID)
                       + art.polylines(diagram, [(curve[:, 0], curve[:, 1]) for curve in self.family], THIN,
                                       art.FAMILIES["volume"], 1.0, art.SOLID))

    def picture(self, index: int) -> np.ndarray:
        point, slope = self.section[index], self.slopes[index]
        intercept = point[1] - slope * point[0]
        layers = (self.layers
                  + art.polylines(self.diagram, [(self.section[:, 0], self.section[:, 1])], THIN, art.INK, 1.0, art.SOLID)
                  + art.polylines(self.diagram, [(self.section[:index + 1, 0], self.section[:index + 1, 1])], BOLD,
                                  art.INK, 1.0, art.SOLID)
                  + art.polylines(self.diagram, tangent_lines(point[None], np.array([slope]), self.diagram.across), 1.6,
                                  art.ORANGE, 1.0, art.SOLID)
                  + art.dots(self.diagram, 0.0, intercept, art.GREEN)
                  + art.dots(self.diagram, point[0], point[1], art.ORANGE))
        last = self.family[-1][-1]
        labels = [art.Label(0.0, intercept, self.intercept_name, art.GREEN, art.NAME_SIZE, "rm", (-8, 0)),
                  art.Label(last[0], last[1], "V = const", art.FAMILIES["volume"], art.LABEL_SIZE, "rb", (-2, 4))]
        return art.picture(self.diagram, layers, labels)


def charts(family: Extensor, changed_family: Extensor, points: Extensor, tangents: Extensor, changed: Extensor,
           changed_tangents: Extensor) -> tuple[Chart, Chart]:
    energy = art.Diagram(ENTROPIES, ENERGIES, (0.0, 0.0), ROWS, COLUMNS, ("S", "U"))
    free = art.Diagram(TEMPERATURES, FREE_ENERGIES, (0.0, 0.0), ROWS, COLUMNS, ("T", "F"))
    return Chart(energy, family, points, tangents, "F"), Chart(free, changed_family, changed, changed_tangents, "U")


def legendre(family: Extensor, changed_family: Extensor, points: Extensor, tangents: Extensor, changed: Extensor,
             changed_tangents: Extensor, moving: Iterator) -> Iterator[np.ndarray]:
    """The two charts as the point of tangency moves along the section."""
    left, right = charts(family, changed_family, points, tangents, changed, changed_tangents)
    for index, _ in enumerate(moving):
        yield art.beside([left.picture(index), right.picture(index)])


def draw(family: Extensor, changed_family: Extensor, points: Extensor, tangents: Extensor, changed: Extensor,
         changed_tangents: Extensor):
    """The two charts with the tangent at the middle of the section."""
    left, right = charts(family, changed_family, points, tangents, changed, changed_tangents)
    middle = points.shape[0] // 2
    return art.figure(art.beside([left.picture(middle), right.picture(middle)]))


def draw_energy(family: Extensor, points: Extensor, tangents: Extensor):
    """The energy chart alone, with the tangent at the middle of the section."""
    energy = art.Diagram(ENTROPIES, ENERGIES, (0.0, 0.0), ROWS, COLUMNS, ("S", "U"))
    return art.figure(Chart(energy, family, points, tangents, "F").picture(points.shape[0] // 2))
