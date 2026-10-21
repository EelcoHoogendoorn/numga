"""Polarization and transmitted intensity through two identical optical stacks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle

from numga import stack
from examples.animation import capture
from examples.optics.black_sandwich import core

BACKGROUND = "#fbfaf7"
BEAM = "#2b8cbe"
OUTLINE = "#8a949c"
POLARIZER = "#35434d"
PLATE = "#db663d"
FAINT = "#d9dcda"
COMPONENT_POSITIONS = np.array([1.2, 3.2, 5.2, 8.2, 10.2, 12.2])
FIELD_POSITIONS = np.array([0.1, 2.2, 4.2, 6.7, 9.2, 11.2, 13.7])
FIELD_HEIGHT = 1.25
RADIUS = 0.42


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


def _segments(centres: np.ndarray, directions: np.ndarray, length: float) -> np.ndarray:
    return np.stack((centres - length * directions, centres + length * directions), axis=-2)


def _scene(input_axis: np.ndarray, output_axis: np.ndarray) -> tuple[plt.Figure, LineCollection, LineCollection, LineCollection]:
    figure = plt.figure(figsize=(9.2, 2.2), dpi=100, facecolor=BACKGROUND)
    ax = figure.add_axes((0, 0, 1, 1), facecolor=BACKGROUND)
    ax.set(xlim=(-0.5, 14.3), ylim=(-0.75, 1.85), aspect="equal")
    ax.set_axis_off()

    boundaries = np.r_[FIELD_POSITIONS[0] - 0.3, COMPONENT_POSITIONS, FIELD_POSITIONS[-1] + 0.3]
    beam_segments = np.stack((np.column_stack((boundaries[:-1], np.zeros(len(FIELD_POSITIONS)))),
                              np.column_stack((boundaries[1:], np.zeros(len(FIELD_POSITIONS))))), axis=1)
    ax.add_collection(LineCollection(beam_segments, colors=FAINT, linewidths=1, zorder=1))
    beam = LineCollection(beam_segments, linewidths=4, capstyle="butt", zorder=2)
    ax.add_collection(beam)

    for position in COMPONENT_POSITIONS:
        ax.add_patch(Circle((position, 0), RADIUS, facecolor=BACKGROUND, edgecolor=OUTLINE, linewidth=1, zorder=3))
    polarizer_centres = np.column_stack((COMPONENT_POSITIONS[[0, 2, 3, 5]], np.zeros(4)))
    polarizer_axes = np.stack((input_axis, output_axis, input_axis, output_axis))
    ax.add_collection(LineCollection(_segments(polarizer_centres, polarizer_axes, 0.8 * RADIUS),
                                     colors=POLARIZER, linewidths=2.2, capstyle="round", zorder=4))
    plate_lines = LineCollection([], colors=PLATE, linewidths=2.2, capstyle="round", zorder=4)
    ax.add_collection(plate_lines)

    for position in FIELD_POSITIONS:
        ax.add_patch(Circle((position, FIELD_HEIGHT), 0.38, fill=False, edgecolor=FAINT, linewidth=0.8))
    field_lines = LineCollection([], colors=BEAM, linewidths=2.2, capstyle="round", zorder=4)
    ax.add_collection(field_lines)
    return figure, beam, field_lines, plate_lines


def _update(beam: LineCollection, field_lines: LineCollection, plate_lines: LineCollection,
            fields: np.ndarray, plate_axis: np.ndarray) -> None:
    colours = np.tile(to_rgba(BEAM), (len(fields), 1))
    colours[:, 3] = np.clip(np.sum(fields * fields, axis=-1), 0, 1)
    beam.set_color(colours)
    field_centres = np.column_stack((FIELD_POSITIONS, np.full(len(fields), FIELD_HEIGHT)))
    field_lines.set_segments(_segments(field_centres, fields, 0.34))
    plate_centres = np.column_stack((COMPONENT_POSITIONS[[1, 4]], np.zeros(2)))
    plate_lines.set_segments(_segments(plate_centres, plate_axis, 0.8 * RADIUS))


def frames(fields: core.Field, plate_axes: core.Direction, input_axis: core.Direction,
           output_axis: core.Direction) -> list[np.ndarray]:
    """The plates' axes turn together while the second stack's outgoing beam stays dark; above the beam,
    the field between components."""
    figure, beam, field_lines, plate_lines = _scene(_coordinates(input_axis), _coordinates(output_axis))
    field_values = _components(fields)                                           # [frames, stages, 2]
    axis_values = _coordinates(plate_axes)                                       # [frames, 2]
    images = []
    for field, plate_axis in zip(field_values, axis_values):
        _update(beam, field_lines, plate_lines, field, plate_axis)
        images.append(capture(figure))
    plt.close(figure)
    return images
