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

from numga import Extensor, stack
from examples.thermodynamics import rendering as art

# The charts' size in pixels and their ranges: entropy and energy, temperature and Helmholtz energy.
n_rows, n_columns = 300, 330
entropy_range, energy_range = (-0.8, 1.25), (0.0, 3.4)
temperature_range, free_energy_range = (0.0, 2.2), (0.0, 3.4)
# Every so many samples of the section, a faint tangent; the samples along each tangent line.
tangent_every, n_line_samples = 3, 60


def readings(points: Extensor) -> np.ndarray:
    """The entropy and the energy of points, read by pairing `[..., 2]`; after the Legendre change, the temperature and
    the Helmholtz energy."""
    mv = points.context.multivector
    return (stack((mv.s, mv.u)) & points[..., None]).to_array()


def tangent_lines(points: np.ndarray, slopes: np.ndarray, across: tuple[float, float]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Lines through points `[lines, 2]` with slopes `[lines]`, sampled across a range."""
    x = np.linspace(*across, n_line_samples)
    return [(x, point[1] + slope * (x - point[0])) for point, slope in zip(points, slopes)]


class Chart:
    """One of the two charts: its curves, the faint tangents enveloping its section, and the moving tangent."""

    def __init__(self, diagram: art.Diagram, family: Extensor, points: Extensor, tangents: Extensor,
                 intercept_name: str) -> None:
        self.diagram, self.intercept_name = diagram, intercept_name
        self.family, self.section = readings(family), readings(points)                   # [volumes, samples, 2], [samples, 2]
        steps = readings(tangents)
        slopes = steps[:, 1] / steps[:, 0]
        self.layers = (art.polylines(diagram, tangent_lines(self.section[::tangent_every], slopes[::tangent_every],
                                                            diagram.across),
                                     art.THIN, art.FAMILIES["level"], 0.6, art.SOLID)
                       + art.polylines(diagram, [(curve[:, 0], curve[:, 1]) for curve in self.family], art.THIN,
                                       art.FAMILIES["volume"], 1.0, art.SOLID)
                       + art.polylines(diagram, [(self.section[:, 0], self.section[:, 1])], art.THIN, art.INK, 1.0,
                                       art.SOLID))

    def picture(self, point: Extensor, tangent: Extensor, trace: np.ndarray) -> np.ndarray:
        """The chart with the tangent at a point of the section, and the section traced up to it `[samples, 2]` bold."""
        position, step = readings(point), readings(tangent)
        slope = step[1] / step[0]
        intercept = position[1] - slope * position[0]
        layers = (self.layers
                  + art.polylines(self.diagram, [(trace[:, 0], trace[:, 1])], art.BOLD, art.INK, 1.0, art.SOLID)
                  + art.polylines(self.diagram, tangent_lines(position[None], np.array([slope]), self.diagram.across),
                                  art.LINE, art.ORANGE, 1.0, art.SOLID)
                  + art.dots(self.diagram, 0.0, intercept, art.GREEN)
                  + art.dots(self.diagram, position[0], position[1], art.ORANGE))
        labels = [art.Label(0.0, intercept, self.intercept_name, art.GREEN, art.TEXT_SIZE, "rm", (-8, 0))]
        names = [art.Floating(*art.along([(curve[:, 0], curve[:, 1]) for curve in self.family], 4), "V = const",
                              art.FAMILIES["volume"], art.TEXT_SIZE, "lb")]
        return art.floated(self.diagram, art.picture(self.diagram, layers, labels), names)


def charts(family: Extensor, changed_family: Extensor, points: Extensor, tangents: Extensor,
           changed_points: Extensor, changed_tangents: Extensor) -> tuple[Chart, Chart]:
    energy = art.Diagram(entropy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("S", "U"))
    free = art.Diagram(temperature_range, free_energy_range, (0.0, 0.0), n_rows, n_columns, ("T", "F"))
    return (Chart(energy, family, points, tangents, "F"),
            Chart(free, changed_family, changed_points, changed_tangents, "U"))


def legendre(family: Extensor, changed_family: Extensor, points: Extensor, tangents: Extensor,
             changed_points: Extensor, changed_tangents: Extensor,
             moving: Iterator[tuple[Extensor, Extensor, Extensor, Extensor]]
             ) -> Iterator[np.ndarray]:
    """The two charts as the point of tangency moves along the section, before and after the change, each section
    traced bold behind it."""
    left, right = charts(family, changed_family, points, tangents, changed_points, changed_tangents)
    trace, changed_trace = [], []
    for point, tangent, changed_point, changed_tangent in moving:
        trace.append(readings(point))
        changed_trace.append(readings(changed_point))
        yield art.beside([left.picture(point, tangent, np.stack(trace)),
                          right.picture(changed_point, changed_tangent, np.stack(changed_trace))])


def draw(family: Extensor, changed_family: Extensor, points: Extensor, tangents: Extensor,
         changed_points: Extensor, changed_tangents: Extensor):
    """The two charts with the tangent at the middle of the section."""
    left, right = charts(family, changed_family, points, tangents, changed_points, changed_tangents)
    middle = points.shape[0] // 2
    return art.figure(art.beside([
        left.picture(points[middle], tangents[middle], left.section[:middle + 1]),
        right.picture(changed_points[middle], changed_tangents[middle], right.section[:middle + 1])]))


def draw_energy(family: Extensor, points: Extensor, tangents: Extensor):
    """The energy chart alone, with the tangent at the middle of the section."""
    energy = art.Diagram(entropy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("S", "U"))
    chart = Chart(energy, family, points, tangents, "F")
    middle = points.shape[0] // 2
    return art.figure(chart.picture(points[middle], tangents[middle], chart.section[:middle + 1]))
