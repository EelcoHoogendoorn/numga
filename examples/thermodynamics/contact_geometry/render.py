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


def chart_points(points: Extensor) -> art.Point:
    """Points on a chart of their entropy across and their energy up, read by pairing `[...]`; after the Legendre
    change, of the temperature and the Helmholtz energy."""
    mv = points.context.multivector
    return art.chart_points(art.context.include_from_algebra(mv.s & points), art.context.include_from_algebra(mv.u & points))


def chart_directions(tangents: Extensor) -> art.Point:
    """The directions on the chart of changes `[...]`: their changes of entropy and of energy, read by pairing."""
    mv = tangents.context.multivector
    return art.chart_directions(art.context.include_from_algebra(mv.s & tangents), art.context.include_from_algebra(mv.u & tangents))


class Chart:
    """One of the two charts: its curves, the faint tangents enveloping its section, and the moving tangent."""

    def __init__(self, diagram: art.Diagram, family: Extensor, points: Extensor, tangents: Extensor,
                 intercept_name: str) -> None:
        self.diagram, self.intercept_name = diagram, intercept_name
        curves = chart_points(family)                                        # [volumes, samples] Point-
        self.family = [curves[volume] for volume in range(curves.shape[0])]
        self.section = chart_points(points)                                  # [samples] Point-
        self.crossings = np.linspace(*diagram.across, n_line_samples)
        faint = art.tangent_lines(self.section[::tangent_every], chart_directions(tangents[::tangent_every]),
                                  self.crossings)                            # [tangents, samples] Point-
        self.layers = (art.polylines(diagram, [faint[line] for line in range(faint.shape[0])], art.thin,
                                     art.families["level"], 0.6, art.SOLID)
                       + art.polylines(diagram, self.family, art.thin, art.families["volume"], 1.0, art.SOLID)
                       + art.polylines(diagram, [self.section], art.thin, art.ink, 1.0, art.SOLID))

    def picture(self, point: Extensor, tangent: Extensor, trace: art.Point) -> np.ndarray:
        """The chart with the tangent at a point of the section, and the section traced up to it `[samples]` bold."""
        position, direction = chart_points(point), chart_directions(tangent)          # [] Point-, Point-
        # Where the tangent crosses zero across, its height is the other potential: the tangent line met with the axis
        # up.
        intercept = art.planar.unit((position & direction) ^ art.mv.x)               # [] Point-
        layers = (self.layers
                  + art.polylines(self.diagram, [trace], art.bold, art.ink, 1.0, art.SOLID)
                  + art.polylines(self.diagram, [art.tangent_lines(position, direction, self.crossings)], art.medium,
                                  art.orange, 1.0, art.SOLID)
                  + art.dots(self.diagram, intercept[None], art.green)
                  + art.dots(self.diagram, position[None], art.orange))
        labels = [art.Label(intercept, self.intercept_name, art.green, art.text_size, "rm", (-8, 0))]
        names = [art.Floating(art.along(self.family, 4), "V = const", art.families["volume"], art.text_size, "lb")]
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
        trace.append(chart_points(point))
        changed_trace.append(chart_points(changed_point))
        yield art.beside([left.picture(point, tangent, stack(trace)),
                          right.picture(changed_point, changed_tangent, stack(changed_trace))])


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
