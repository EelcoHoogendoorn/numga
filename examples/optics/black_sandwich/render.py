"""Polarization and transmitted intensity through two identical optical stacks."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle

from numga import stack
from examples.util.animation import capture
from examples.optics.black_sandwich import core

background_colour = "#fbfaf7"
beam_colour = "#2b8cbe"
outline_colour = "#8a949c"
polarizer_colour = "#35434d"
plate_colour = "#db663d"
faint_colour = "#d9dcda"
component_positions = np.array([1.2, 3.2, 5.2, 8.2, 10.2, 12.2])
field_positions = np.array([0.1, 2.2, 4.2, 6.7, 9.2, 11.2, 13.7])
field_height = 1.25
component_radius = 0.42


# --- plumbing -------------------------------------------------------------------------
def _coordinates(directions: core.Direction) -> np.ndarray:
    """Directions in the transverse plane, read by pairing with the planes across the axes."""
    mv = directions.context.multivector
    return (stack((mv.x, mv.y)) & directions[..., None]).to_array()


def _components(fields: core.Field) -> np.ndarray:
    """Fields in the transverse plane, read by pairing with the planes the axes' directions sweep in time."""
    mv = fields.context.multivector
    sweeps = stack((mv.x.dual(), mv.y.dual())) & -mv.t.dual()
    return (fields[..., None] & sweeps).to_array()


def _scene(input_axis: core.Direction, output_axis: core.Direction, dpi: float) -> tuple[plt.Figure, LineCollection, LineCollection, LineCollection]:
    figure = plt.figure(figsize=(9.2, 2.2), dpi=dpi, facecolor=background_colour)
    ax = figure.add_axes((0, 0, 1, 1), facecolor=background_colour)
    ax.set(xlim=(-0.5, 14.3), ylim=(-0.75, 1.85), aspect="equal")
    ax.set_axis_off()

    boundaries = np.r_[field_positions[0] - 0.3, component_positions, field_positions[-1] + 0.3]
    beam_segments = np.stack((np.column_stack((boundaries[:-1], np.zeros(len(field_positions)))),
                              np.column_stack((boundaries[1:], np.zeros(len(field_positions))))), axis=1)
    ax.add_collection(LineCollection(beam_segments, colors=faint_colour, linewidths=1, zorder=1))
    beam = LineCollection(beam_segments, linewidths=4, capstyle="butt", zorder=2)
    ax.add_collection(beam)

    for position in component_positions:
        ax.add_patch(Circle((position, 0), component_radius, facecolor=background_colour, edgecolor=outline_colour, linewidth=1, zorder=3))
    polarizer_centres = np.column_stack((component_positions[[0, 2, 3, 5]], np.zeros(4)))
    # Each polarizer's axis drawn either way across its middle, its glyph placed on the polarizer.
    polarizer_axes = stack((input_axis, output_axis, input_axis, output_axis))          # [polarizers] Direction-
    polarizer_ends = stack((-polarizer_axes, polarizer_axes), axis=-1) * (0.8 * component_radius)   # [polarizers, ends] Direction-
    ax.add_collection(LineCollection(polarizer_centres[:, None] + _coordinates(polarizer_ends),
                                     colors=polarizer_colour, linewidths=2.2, capstyle="round", zorder=4))
    plate_lines = LineCollection([], colors=plate_colour, linewidths=2.2, capstyle="round", zorder=4)
    ax.add_collection(plate_lines)

    for position in field_positions:
        ax.add_patch(Circle((position, field_height), 0.38, fill=False, edgecolor=faint_colour, linewidth=0.8))
    field_lines = LineCollection([], colors=beam_colour, linewidths=2.2, capstyle="round", zorder=4)
    ax.add_collection(field_lines)
    return figure, beam, field_lines, plate_lines


def _update(beam: LineCollection, field_lines: LineCollection, plate_lines: LineCollection,
            fields: core.Field, plate_axis: core.Direction) -> None:
    readings = _components(fields)                                               # [stages, 2]
    # The beam's strength between components, from the field's square size there.
    colours = np.tile(to_rgba(beam_colour), (len(readings), 1))
    colours[:, 3] = np.clip(np.sum(readings * readings, axis=-1), 0, 1)
    beam.set_color(colours)
    # The field drawn either way across its glyph's middle, above the beam, and the plates' axis on each plate.
    field_ends = stack((-fields, fields), axis=-1) * 0.34                        # [stages, ends] Field+
    plate_ends = stack((-plate_axis, plate_axis)) * (0.8 * component_radius)     # [ends] Direction-
    field_centres = np.column_stack((field_positions, np.full(len(readings), field_height)))
    field_lines.set_segments(field_centres[:, None] + _components(field_ends))
    plate_centres = np.column_stack((component_positions[[1, 4]], np.zeros(2)))
    plate_lines.set_segments(plate_centres[:, None] + _coordinates(plate_ends)[None])


def frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
           output_axis: core.Direction) -> Iterator[np.ndarray]:
    """The plates' axes turn together while the second stack's outgoing beam stays dark; above the beam,
    the field between components."""
    return sized_frames(fields, plate_axes, input_axis, output_axis, 100)


def sized_frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
                 output_axis: core.Direction, dpi: float) -> Iterator[np.ndarray]:
    """As `frames`, the figure drawn at `dpi` dots per inch."""
    figure, beam, field_lines, plate_lines = _scene(input_axis, output_axis, dpi)
    for field, plate_axis in zip(fields, plate_axes):
        _update(beam, field_lines, plate_lines, field, plate_axis)
        yield capture(figure)
    plt.close(figure)
