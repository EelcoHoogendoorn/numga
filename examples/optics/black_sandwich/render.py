"""Spacetime GPU diagrams display beam intensity and stage polarizations across two identical optical stacks."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import Extensor, concatenate, stack
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import drawing, simplices
from examples.util.panels import offset_panel, overlay
from examples.optics.black_sandwich import core

mv = core.mv
background_colour = "#fbfaf7"
beam_colour, outline_colour, polarizer_colour, plate_colour, faint_colour = (
    to_rgb(colour) for colour in ("#2b8cbe", "#8a949c", "#35434d", "#db663d", "#d9dcda"))
component_positions = np.array([1.2, 3.2, 5.2, 8.2, 10.2, 12.2])
field_positions = np.array([0.1, 2.2, 4.2, 6.7, 9.2, 11.2, 13.7])
field_height = 1.25
component_radius = 0.42
field_radius = 0.38
# Stroke widths in points at the figure DPI for the optical axis, beam, component outlines, axis glyphs, and field rings.
axis_width, beam_width, outline_width, glyph_width, circle_width = 1.0, 4.0, 1.0, 2.2, 0.8


# --- plumbing -------------------------------------------------------------------------
def glyph_strokes(centres: Extensor, half_spans: Extensor) -> tuple[Extensor, Topology]:
    """Symmetric segments centred at `centres` (`[n_glyphs]`) and extending by `half_spans` (`[n_glyphs]`)
    form `[2 * n_glyphs]` endpoints paired by a 1D edge `Topology`."""
    n_glyphs = len(centres)
    endpoints = concatenate([centres - half_spans, centres + half_spans])      # [2 * n_glyphs]
    return endpoints, Topology(2 * n_glyphs, np.stack([np.arange(n_glyphs), n_glyphs + np.arange(n_glyphs)], axis=-1))


class Diagram:
    """Reusable figure layout holding static polarizer and ring overlays across animation frames."""

    def __init__(self, input_axis: core.Direction, output_axis: core.Direction, dpi: float) -> None:
        self.points = dpi / 72
        # Worldsheets swept by transverse `x` and `y` directions read a spacetime electric field directly on screen.
        self.swept_planes = stack((mv.x.dual(), mv.y.dual())) & -mv.t.dual()   # [2] Antibivector-
        self.components = mv.x.dual() * component_positions                    # [6] Direction-
        horizontal_field, vertical_field = (
            self.swept_planes.dual() / (self.swept_planes & self.swept_planes.dual())
        )                                                                      # [] Field+, Field+
        self.field_places = horizontal_field * field_positions + vertical_field * field_height  # [7] Field+
        self.boundaries = mv.x.dual() * np.r_[field_positions[0] - 0.3, component_positions,
                                              field_positions[-1] + 0.3]       # [8] Direction-
        self.figure = plt.figure(figsize=(9.2, 2.2), dpi=dpi, facecolor=background_colour)
        ax = self.figure.add_axes((0, 0, 1, 1), facecolor=background_colour)
        ax.set(xlim=(-0.5, 14.3), ylim=(-0.75, 1.85), aspect="equal")
        ax.set_axis_off()
        self.camera = offset_panel(ax, mv.x, mv.y)
        self.field_camera = offset_panel(ax, *self.swept_planes)
        polarizer_axes = stack((input_axis, output_axis, input_axis, output_axis))  # [4] Direction-
        polarizers = glyph_strokes(self.components[[0, 2, 3, 5]], polarizer_axes * (0.8 * component_radius))
        radius, field_ring = component_radius / self.camera.pixel, field_radius / self.camera.pixel
        self.under = [simplices.strokes(self.boundaries, Topology.chains([len(self.boundaries)]), faint_colour,
                                        axis_width * self.points, self.camera, cap=simplices.Cap.BUTT)]
        self.over = [
            simplices.markers(self.components, outline_colour, radius, self.camera),
            simplices.markers(self.components, to_rgb(background_colour), radius - outline_width * self.points,
                              self.camera),
            simplices.strokes(*polarizers, polarizer_colour, glyph_width * self.points, self.camera),
            simplices.markers(self.field_places, faint_colour, field_ring, self.field_camera),
            simplices.markers(self.field_places, to_rgb(background_colour), field_ring - circle_width * self.points,
                              self.field_camera)]
        self.image = overlay(ax, self.under + self.over)

    def picture(self, stage_fields: core.Field, plate_axis: core.Direction) -> np.ndarray:
        """Composited RGBA layer for the `[7] Field+` `stage_fields` and fast `plate_axis`."""
        readings = stage_fields[..., None] & self.swept_planes                 # [7, 2] Scalar+
        strengths = np.clip((readings * readings).sum(axis=-1).to_array(), 0, 1)
        beam = [simplices.strokes(self.boundaries[stage:stage + 2], Topology.chains([2]), beam_colour,
                                  beam_width * self.points, self.camera, cap=simplices.Cap.BUTT, opacity=strength)
                for stage, strength in enumerate(strengths)]
        field_glyphs = glyph_strokes(self.field_places, stage_fields * 0.34)
        plates = glyph_strokes(self.components[[1, 4]], stack((plate_axis, plate_axis)) * (0.8 * component_radius))
        return drawing.composited(self.under + beam + self.over + [
            simplices.strokes(*plates, plate_colour, glyph_width * self.points, self.camera),
            simplices.strokes(*field_glyphs, beam_colour, glyph_width * self.points, self.field_camera,
                              cap=simplices.Cap.BUTT)])


def sized_sandwich_animation(
    input_axis: core.Direction, output_axis: core.Direction, dpi: float,
    states: Iterable[tuple[core.Field, core.Direction, core.OpticalMap]],
) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB frames (starting at the rest position, swinging out, and swinging back)
    of the two-stack optical diagram rendered at `dpi`."""
    diagram = Diagram(input_axis, output_axis, dpi)
    swing_frames = []
    for stage_fields, plate_axis, _ in states:
        diagram.image.set_data(diagram.picture(stage_fields, plate_axis))
        frame = capture(diagram.figure)
        swing_frames.append(frame)
        yield frame
    plt.close(diagram.figure)
    for frame in swing_frames[-2:0:-1]:
        yield frame


def sandwich_animation(
    input_axis: core.Direction, output_axis: core.Direction,
    states: Iterable[tuple[core.Field, core.Direction, core.OpticalMap]],
) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB frames (forward and back) of the two-stack optical diagram at `100` DPI."""
    return sized_sandwich_animation(input_axis, output_axis, 100, states)
