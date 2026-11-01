"""Polarization and transmitted intensity through two identical optical stacks, drawn on the GPU in spacetime.

The diagram lays its glyphs out side by side in the plane of the transverse directions x and y, each a small view
of that plane: the beam runs across it from component to component, the polarizers' and plates' axes are drawn
across their components, and above the beam the field between them along the direction it sweeps through time."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import panel
from examples.optics.black_sandwich import core

background_colour = "#fbfaf7"
beam_colour, outline_colour, polarizer_colour, plate_colour, faint_colour = (
    to_rgb(colour) for colour in ("#2b8cbe", "#8a949c", "#35434d", "#db663d", "#d9dcda"))
component_positions = np.array([1.2, 3.2, 5.2, 8.2, 10.2, 12.2])
field_positions = np.array([0.1, 2.2, 4.2, 6.7, 9.2, 11.2, 13.7])
field_height = 1.25
component_radius = 0.42
field_radius = 0.38
# Widths in points, drawn at the figure's dots per inch: the beam's axis, the beam, the components' outlines, the
# axes across them and the fields, and the fields' circles.
axis_width, beam_width, outline_width, glyph_width, circle_width = 1.0, 4.0, 1.0, 2.2, 0.8

# --- plumbing -------------------------------------------------------------------------
def _across(centres: Extensor, halves: Extensor) -> tuple[Extensor, simplices.Topology]:
    """Strokes across glyphs `[glyphs]`, either way from each centre by its half `[glyphs]`: their ends, and an
    edge from each glyph's first end to its second."""
    n_glyphs = len(centres)
    ends = concatenate([centres - halves, centres + halves])                 # [2 * glyphs]
    return ends, simplices.Topology(2 * n_glyphs, np.stack([np.arange(n_glyphs), n_glyphs + np.arange(n_glyphs)], axis=-1))


class Diagram:
    """The diagram, laid out once in the algebra of the axes it is given: the beam's axis, the components and the
    fields' circles drawn once; and its picture at each state, the beam, the plates' axis and the fields."""

    def __init__(self, input_axis: core.Direction, output_axis: core.Direction, dpi: float) -> None:
        mv = input_axis.context.multivector
        self.points = dpi / 72
        # The planes the transverse directions sweep in the observer's time, which read a field as it is, its parts
        # along x and y.
        self.swept_planes = stack((mv.x.dual(), mv.y.dual())) & -mv.t.dual()  # [2] Bivector-
        # The glyphs' places in the diagram: the components on the beam; the fields above it, as the fields these
        # planes read there; and the beam's ends and the components between them.
        self.components = mv.x.dual() * component_positions                  # [components] Direction-
        along, rise = self.swept_planes.dual() / (self.swept_planes & self.swept_planes.dual())   # [] Field+, Field+
        self.field_places = along * field_positions + rise * field_height     # [stages] Field+
        self.boundaries = mv.x.dual() * np.r_[field_positions[0] - 0.3, component_positions,
                                              field_positions[-1] + 0.3]      # [stages + 1] Direction-
        self.figure = plt.figure(figsize=(9.2, 2.2), dpi=dpi, facecolor=background_colour)
        ax = self.figure.add_axes((0, 0, 1, 1), facecolor=background_colour)
        ax.set(xlim=(-0.5, 14.3), ylim=(-0.75, 1.85), aspect="equal")
        ax.set_axis_off()
        self.camera = panel(ax, mv.x, mv.y)
        self.field_camera = panel(ax, *self.swept_planes)
        # Each polarizer's axis drawn either way across its middle, its glyph placed on the polarizer.
        polarizer_axes = stack((input_axis, output_axis, input_axis, output_axis))          # [polarizers] Direction-
        polarizers = _across(self.components[[0, 2, 3, 5]], polarizer_axes * (0.8 * component_radius))
        # The components' discs, in the background's colour within their outlines, and the fields' circles alike:
        # each a dot over a larger dot.
        radius, field_ring = component_radius / self.camera.pixel, field_radius / self.camera.pixel
        self.under = [simplices.strokes(self.boundaries, simplices.Topology.chains([len(self.boundaries)]), faint_colour,
                                        axis_width * self.points, self.camera, cap=simplices.Cap.BUTT)]
        self.over = [
            simplices.markers(self.components, outline_colour, radius, self.camera),
            simplices.markers(self.components, to_rgb(background_colour), radius - outline_width * self.points,
                              self.camera),
            simplices.strokes(*polarizers, polarizer_colour, glyph_width * self.points, self.camera),
            simplices.markers(self.field_places, faint_colour, field_ring, self.field_camera),
            simplices.markers(self.field_places, to_rgb(background_colour), field_ring - circle_width * self.points,
                              self.field_camera)]
        self.image = simplices.overlay(ax, self.under + self.over)

    def picture(self, fields: core.Field, plate_axis: core.Direction) -> np.ndarray:
        """The diagram with the fields between components `[stages]` and the plates' axis."""
        # The beam's strength between components, from the field's square size there, its readings by the planes the
        # transverse directions sweep in time: each stretch of the beam as strong.
        readings = fields[..., None] & self.swept_planes                      # [stages, 2] Scalar+
        strengths = np.clip((readings * readings).sum(axis=-1).to_array(), 0, 1)
        beam = [simplices.strokes(self.boundaries[stage:stage + 2], simplices.Topology.chains([2]), beam_colour,
                                  beam_width * self.points, self.camera, cap=simplices.Cap.BUTT, opacity=strength)
                for stage, strength in enumerate(strengths)]
        # The field drawn either way across its glyph's middle, above the beam, as it is, read by the planes the
        # transverse directions sweep in time, its ends square, so that no field draws nothing; and the plates' axis
        # on each plate.
        field_glyphs = _across(self.field_places, fields * 0.34)
        plates = _across(self.components[[1, 4]], stack((plate_axis, plate_axis)) * (0.8 * component_radius))
        return simplices.composited(self.under + beam + self.over + [
            simplices.strokes(*plates, plate_colour, glyph_width * self.points, self.camera),
            simplices.strokes(*field_glyphs, beam_colour, glyph_width * self.points, self.field_camera,
                              cap=simplices.Cap.BUTT)])


def frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
           output_axis: core.Direction) -> Iterator[np.ndarray]:
    """The plates' axes turn together while the second stack's outgoing beam stays dark; above the beam,
    the field between components."""
    return sized_frames(fields, plate_axes, input_axis, output_axis, 100)


def sized_frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
                 output_axis: core.Direction, dpi: float) -> Iterator[np.ndarray]:
    """As `frames`, the figure drawn at `dpi` dots per inch."""
    diagram = Diagram(input_axis, output_axis, dpi)
    for field, plate_axis in zip(fields, plate_axes):
        diagram.image.set_data(diagram.picture(field, plate_axis))
        yield capture(diagram.figure)
    plt.close(diagram.figure)
