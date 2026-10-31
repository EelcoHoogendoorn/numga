"""Polarization and transmitted intensity through two identical optical stacks, drawn on the GPU in spacetime.

The diagram lays its glyphs out side by side in the plane of the transverse directions x and y, each a small view
of that plane: the beam runs across it from component to component, the polarizers' and plates' axes are drawn
across their components, and above the beam the field between them along the direction it sweeps through time."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import Camera, panel
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

mv = core.mv
# The glyphs' places in the diagram: the components on the beam, the fields above it, and the beam's ends and the
# components between them.
components = mv.x.dual() * component_positions                               # [components] Direction-
field_places = mv.x.dual() * field_positions + mv.y.dual() * field_height    # [stages] Direction-
boundaries = mv.x.dual() * np.r_[field_positions[0] - 0.3, component_positions, field_positions[-1] + 0.3]   # [stages + 1] Direction-


# --- plumbing -------------------------------------------------------------------------
def _across(centres: core.Direction, halves: core.Direction) -> tuple[core.Direction, simplices.Mesh]:
    """Strokes across glyphs `[glyphs]`, either way from each centre by its half `[glyphs]`: their ends, and an
    edge from each glyph's first end to its second."""
    n_glyphs = len(centres)
    ends = concatenate([centres - halves, centres + halves])                 # [2 * glyphs] Direction-
    return ends, simplices.Mesh(2 * n_glyphs, np.stack([np.arange(n_glyphs), n_glyphs + np.arange(n_glyphs)], axis=-1))


def _scene(input_axis: core.Direction, output_axis: core.Direction, dpi: float):
    figure = plt.figure(figsize=(9.2, 2.2), dpi=dpi, facecolor=background_colour)
    ax = figure.add_axes((0, 0, 1, 1), facecolor=background_colour)
    ax.set(xlim=(-0.5, 14.3), ylim=(-0.75, 1.85), aspect="equal")
    ax.set_axis_off()
    camera = panel(ax, mv.x, mv.y)
    points = dpi / 72
    # Each polarizer's axis drawn either way across its middle, its glyph placed on the polarizer.
    polarizer_axes = stack((input_axis, output_axis, input_axis, output_axis))          # [polarizers] Direction-
    polarizers = _across(components[[0, 2, 3, 5]], polarizer_axes * (0.8 * component_radius))
    # The components' discs, in the background's colour, with their outlines; the fields' circles open.
    radius, field_ring = component_radius / camera.pixel, field_radius / camera.pixel
    under = [simplices.strokes(boundaries, simplices.Mesh.chains([len(boundaries)]), faint_colour, axis_width * points,
                               camera, cap=simplices.Cap.BUTT)]
    over = [simplices.markers(components, to_rgb(background_colour), radius, camera),
            simplices.markers(components, outline_colour, radius, camera, inner=radius - outline_width * points),
            simplices.strokes(*polarizers, polarizer_colour, glyph_width * points, camera),
            simplices.markers(field_places, faint_colour, field_ring, camera,
                              inner=field_ring - circle_width * points)]
    image = simplices.overlay(ax, under + over)
    return figure, image, camera, under, over


def _picture(camera: Camera, under: list[np.ndarray], over: list[np.ndarray], fields: core.Field,
             plate_axis: core.Direction, dpi: float) -> np.ndarray:
    points = dpi / 72
    # The beam's strength between components, from the field's square size there, its readings by the planes the
    # transverse directions sweep in time: each stretch of the beam as strong.
    readings = (fields[..., None] & (stack((mv.x.dual(), mv.y.dual())) & core.time)).to_array()   # [stages, 2]
    strengths = np.clip(np.sum(readings * readings, axis=-1), 0, 1)
    beam = [simplices.strokes(boundaries[stage:stage + 2], simplices.Mesh.chains([2]), beam_colour, beam_width * points,
                              camera, cap=simplices.Cap.BUTT, opacity=strength)
            for stage, strength in enumerate(strengths)]
    # The field drawn either way across its glyph's middle, above the beam, along the direction it sweeps through
    # time: the plane it is the field along, `field.dual_inverse()`, met with the observer's space, its ends square,
    # so that no field draws nothing; and the plates' axis on each plate.
    swept = fields.dual_inverse() ^ mv.t                                      # [stages] Direction-
    field_glyphs = _across(field_places, swept * 0.34)
    plates = _across(components[[1, 4]], stack((plate_axis, plate_axis)) * (0.8 * component_radius))
    return simplices.composited(under + beam + over + [
        simplices.strokes(*plates, plate_colour, glyph_width * points, camera),
        simplices.strokes(*field_glyphs, beam_colour, glyph_width * points, camera, cap=simplices.Cap.BUTT)])


def frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
           output_axis: core.Direction) -> Iterator[np.ndarray]:
    """The plates' axes turn together while the second stack's outgoing beam stays dark; above the beam,
    the field between components."""
    return sized_frames(fields, plate_axes, input_axis, output_axis, 100)


def sized_frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
                 output_axis: core.Direction, dpi: float) -> Iterator[np.ndarray]:
    """As `frames`, the figure drawn at `dpi` dots per inch."""
    figure, image, camera, under, over = _scene(input_axis, output_axis, dpi)
    for field, plate_axis in zip(fields, plate_axes):
        image.set_data(_picture(camera, under, over, field, plate_axis, dpi))
        yield capture(figure)
    plt.close(figure)
