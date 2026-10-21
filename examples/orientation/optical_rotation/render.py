"""Polarizations along a beam, out to a reflector and back, unfolded into one straight path."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.orientation.optical_rotation.core import Polarization, mv

leg_colours = ("#df7246", "#238c9b")
line_half_width = 0.45                                     # drawn half-length of a polarization line
# An oblique view of the plane across the beam: x drawn upward, y receding up and to the right.
receding_offsets = np.array([[0.0, 1.0], [0.45, 0.3]])        # screen offsets of x and y


def _segment(ax: plt.Axes, position: float, vector: np.ndarray, half_width: float, **style) -> None:
    """A polarization line across the beam at a position along it."""
    offset = vector[:2] @ receding_offsets * half_width
    ax.plot([position - offset[0], position + offset[0]], [-offset[1], offset[1]], **style)


def draw_isolator(polarizations: Polarization, depths: np.ndarray) -> plt.Figure:
    """`[media, setups, legs, depths]` polarizations: one panel per medium and setup, the beam running
    left to right, out to the reflector and back, with the entering polarization dashed at both ends."""
    # Each polarization read by pairing it with the planes across x, y and z.
    vectors = (stack((mv.x, mv.y, mv.z), axis=-1) & polarizations[..., None]).to_array()   # [media, setups, legs, depths, 3]
    length = depths[-1]
    along = np.stack([depths, length + depths])                                   # [legs, depths]
    media, setups = vectors.shape[:2]

    figure, panels = plt.subplots(setups, media, figsize=(11, 4.4), dpi=100)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04, hspace=0.04)
    reflector = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * line_half_width * 1.4 @ receding_offsets
    for medium in range(media):
        for setup in range(setups):
            ax = panels[setup, medium]
            ax.add_patch(plt.Polygon(reflector + [length, 0], color="#aab4bf", alpha=0.45, zorder=0))
            ax.plot([0, 2 * length], [0, 0], color="#75808a", linewidth=1, zorder=1)
            entering = vectors[medium, setup, 0, 0]
            for end in (0, 2 * length):
                _segment(ax, end, entering, line_half_width * 1.3, color="#262b32", linestyle="--", linewidth=1.2, zorder=2)
            for leg, colour in enumerate(leg_colours):
                for position, vector in zip(along[leg], vectors[medium, setup, leg]):
                    _segment(ax, position, vector, line_half_width, color=colour, linewidth=2.2, zorder=3)
            ax.set(xlim=(-0.6, 2 * length + 0.6), ylim=(-0.75, 0.75), aspect="equal")
            ax.set_axis_off()
    return figure
