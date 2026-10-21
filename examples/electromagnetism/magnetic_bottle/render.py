"""Drawing for the magnetic bottle: the particles' tracks seen from the side, over the field's strength in
the plane through the axis, with the coils' wire where it crosses that plane."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from examples.animation import capture
from examples.electromagnetism.magnetic_bottle import core

mv = core.mv
# The colours of the background, the field's strength, the coils' wire, and the particles by their angle
# to the axis.
background_colour = "#0b1020"
strength_colours = "Blues_r"
wire_colour = "#f59e0b"
particle_colours = plt.get_cmap("plasma")
# The steps of each track drawn behind its particle.
n_trail = 60


# --- plumbing -------------------------------------------------------------------------
def side(points: core.Antivector) -> np.ndarray:
    """Points seen from the side, on the last axis: along the bottle's axis, and across it."""
    return np.stack([(mv.z & points).to_array(), (mv.x & points).to_array()], axis=-1)


def animate_flights(events: core.Antivector, pitches: np.ndarray, points: core.Antivector, strength: core.Scalar,
                    radius: float, heights: np.ndarray, n_steps_per_frame: int) -> list[np.ndarray]:
    """The particles' tracks `[steps, particles]`, coloured by their angle to the axis, over the field's
    strength at the points of a grid in the plane through the axis `[across, along]`."""
    tracks = side(events)                                                     # [steps, particles, 2]
    grid = side(points)                                                       # [across, along, 2]
    extent = (grid[0, 0, 0], grid[0, -1, 0], grid[0, 0, 1], grid[-1, 0, 1])
    colours = particle_colours(0.15 + 0.7 * (pitches - pitches.min()) / np.ptp(pitches))
    images = []
    for end in range(n_steps_per_frame, len(tracks) + 1, n_steps_per_frame):
        figure = plt.figure(figsize=(8.0, 8.0 * (extent[3] - extent[2]) / (extent[1] - extent[0])), dpi=100,
                            facecolor=background_colour)
        ax = figure.add_axes((0, 0, 1, 1))
        ax.imshow(np.log(strength.to_array()), origin="lower", extent=extent, cmap=strength_colours, alpha=0.5,
                  aspect="auto")
        trail = np.moveaxis(tracks[max(0, end - n_trail):end], 0, 1)          # [particles, trail, 2]
        ax.add_collection(LineCollection(trail, colors=colours, linewidths=1.2))
        ax.scatter(*tracks[end - 1].T, color=colours, s=18, zorder=3)
        ax.scatter(np.repeat(heights, 2), np.tile([-radius, radius], len(heights)), color=wire_colour, s=60, zorder=4)
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
        ax.set_axis_off()
        images.append(capture(figure))
        plt.close(figure)
    return images
