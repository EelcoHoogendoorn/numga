"""Renders the Legendre transformation between internal energy as a function of entropy and Helmholtz free energy as a
function of temperature through their enveloping tangent lines.

The left chart plots internal energy against entropy for several isochores in green, highlighting one constant-volume
section in bold ink surrounded by faint grey-blue tangent lines. A moving orange tangent has the temperature as its
slope and meets zero entropy at the Helmholtz free energy. The right chart plots the transformed curve of Helmholtz free
energy against temperature, where the moving tangent has minus the entropy as its slope and meets zero temperature at
the internal energy.

Because both charts plot coordinates of points in the five-dimensional contact space directly, each diagram reads the
ambient points through the planes `s` and `u` before and after the Legendre map.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Extensor, stack
from examples.thermodynamics import rendering as art
from examples.thermodynamics.contact_geometry import core

mv = core.mv
# Pixel dimensions and axis ranges for the entropy–energy and temperature–Helmholtz charts.
n_rows, n_columns = 300, 330
entropy_range, energy_range = (-0.8, 1.25), (0.0, 3.4)
temperature_range, free_energy_range = (0.0, 2.2), (0.0, 3.4)
# Stride between faint envelope tangents along the section, and sample count along each tangent line.
tangent_stride, n_line_samples = 3, 60


class Chart:
    """Stores the static curve families and envelope tangents of one chart and renders each tangent frame."""

    def __init__(self, diagram: art.Diagram, family: Extensor, points: Extensor, tangents: Extensor,
                 intercept_name: str) -> None:
        self.diagram, self.intercept_name = diagram, intercept_name
        self.family = [family[volume] for volume in range(family.shape[0])]
        self.section = points                                                 # [...] Ambient-
        self.crossings = np.linspace(*diagram.across, n_line_samples)
        envelope = art.tangent_lines(diagram, self.section[::tangent_stride], tangents[::tangent_stride],
                                     self.crossings)                          # [..., n_line_samples] Ambient-
        self.layers = (art.polylines(diagram, [envelope[line] for line in range(envelope.shape[0])], art.thin,
                                     art.families["level"], 0.6, art.SOLID)
                       + art.polylines(diagram, self.family, art.thin, art.families["volume"], 1.0, art.SOLID)
                       + art.polylines(diagram, [self.section], art.thin, art.ink, 1.0, art.SOLID))

    def picture(self, point: Extensor, tangent: Extensor, trace: Extensor) -> np.ndarray:
        """Renders the chart with the active tangent at `point` and the bold section curve traced so far."""
        # Meeting the tangent line with zero horizontal coordinate reads the dual thermodynamic potential vertically.
        intercept = art.tangent_lines(self.diagram, point, tangent, np.zeros(1))   # [1] Ambient-
        layers = (self.layers
                  + art.polylines(self.diagram, [trace], art.bold, art.ink, 1.0, art.SOLID)
                  + art.polylines(self.diagram, [art.tangent_lines(self.diagram, point, tangent, self.crossings)],
                                  art.medium, art.orange, 1.0, art.SOLID)
                  + art.dots(self.diagram, intercept, art.green)
                  + art.dots(self.diagram, point[None], art.orange))
        labels = [art.Label(intercept[0], self.intercept_name, art.green, art.text_size, "rm", (-8, 0))]
        names = [art.Floating(art.along(self.family, 4), "V = const", art.families["volume"], art.text_size, "lb")]
        return art.floated(self.diagram, art.picture(self.diagram, layers, labels), names)


def charts(family: Extensor, transformed_family: Extensor, points: Extensor, tangents: Extensor,
           transformed_points: Extensor, transformed_tangents: Extensor) -> tuple[Chart, Chart]:
    """Builds the paired internal-energy and Helmholtz-free-energy charts."""
    energy = art.Diagram(entropy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("S", "U"), (mv.s, mv.u),
                         (0 * mv.s, 1.0))
    free = art.Diagram(temperature_range, free_energy_range, (0.0, 0.0), n_rows, n_columns, ("T", "F"), (mv.s, mv.u),
                       (0 * mv.s, 1.0))
    return (Chart(energy, family, points, tangents, "F"),
            Chart(free, transformed_family, transformed_points, transformed_tangents, "U"))


def legendre(family: Extensor, transformed_family: Extensor, points: Extensor, tangents: Extensor,
             transformed_points: Extensor, transformed_tangents: Extensor,
             stages: Iterator[tuple[Extensor, Extensor, Extensor, Extensor]]
             ) -> Iterator[np.ndarray]:
    """Yields side-by-side chart frames as the point of tangency advances along the section and its Legendre image."""
    left, right = charts(family, transformed_family, points, tangents, transformed_points, transformed_tangents)
    trace, transformed_trace = [], []
    for point, tangent, transformed_point, transformed_tangent in stages:
        trace.append(point)
        transformed_trace.append(transformed_point)
        yield art.beside([left.picture(point, tangent, stack(trace)),
                          right.picture(transformed_point, transformed_tangent, stack(transformed_trace))])


def draw(family: Extensor, transformed_family: Extensor, points: Extensor, tangents: Extensor,
         transformed_points: Extensor, transformed_tangents: Extensor):
    """Draws both charts side by side at the midpoint of the section."""
    left, right = charts(family, transformed_family, points, tangents, transformed_points, transformed_tangents)
    middle = points.shape[0] // 2
    return art.figure(art.beside([
        left.picture(points[middle], tangents[middle], left.section[:middle + 1]),
        right.picture(transformed_points[middle], transformed_tangents[middle], right.section[:middle + 1])]))


def draw_energy(family: Extensor, points: Extensor, tangents: Extensor):
    """Draws the internal-energy chart at the midpoint of the section."""
    energy = art.Diagram(entropy_range, energy_range, (0.0, 0.0), n_rows, n_columns, ("S", "U"), (mv.s, mv.u),
                         (0 * mv.s, 1.0))
    chart = Chart(energy, family, points, tangents, "F")
    middle = points.shape[0] // 2
    return art.figure(chart.picture(points[middle], tangents[middle], chart.section[:middle + 1]))
