"""Planar GPU rendering draws two mirrored cranks, their circular handle paths, handle velocities, and applied forces."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from numga.topology import Topology
from numga.render import drawing, simplices
from examples.util.panels import overlay, panel
from examples.orientation.mechanical_work.core import Point

velocity_colour, force_colour, crank_colour, guide_colour, mirror_colour = (
    to_rgb(colour) for colour in ("#218c9d", "#ce713f", "#424c56", "#d4dce1", "#aebfca"))
# Width and height of the rendered figure in pixels.
figure_pixels = (420, 210)
dpi = 100
# Stroke widths, dash lengths, arrow chevron size, and marker radii in pixels.
mirror_width, mirror_dashes, guide_width, crank_width = 1.2, (4.5, 5.5), 1.0, 3.6
arrow_width, arrow_chevron = 2.5, 10.0
pivot_radius, rim_width, handle_radius = 4.0, 1.3, 2.7


# --- plumbing -------------------------------------------------------------------------
def pairs(n_pairs: int) -> Topology:
    """Each of `n_pairs` directed edges connects point `i` to point `i + n_pairs` in a `2 * n_pairs`-vertex topology."""
    return Topology(2 * n_pairs, np.stack([np.arange(n_pairs), n_pairs + np.arange(n_pairs)], axis=-1))


class WorkView:
    """A planar GPU view holds the static mirror plane, circular handle paths, and pivots while updating the cranks and
    arrows at each frame."""

    def __init__(self, pivots: Point, paths: Point) -> None:
        n_steps, n_cranks = paths.shape
        self.pivots = pivots.normalized()                                       # [n_cranks] Point-
        # Longest crank radius from pivot-to-handle joins sets the view margin.
        radii = (self.pivots & paths.normalized()).norm().reshape(-1)           # [n_steps * n_cranks] Scalar+
        margin = radii.max() * 0.95                                             # [] Scalar+
        mv = paths.context.multivector
        readings = stack((mv.x, mv.y))[:, None] & paths.normalized().reshape(-1)  # [2, n_steps * n_cranks] Scalar+
        lower = readings.min(axis=-1) - margin                                  # [2] Scalar+
        upper = readings.max(axis=-1) + margin                                  # [2] Scalar+

        self.figure, ax = plt.subplots(figsize=np.array(figure_pixels) / dpi, dpi=dpi, layout="constrained")
        x_limits, y_limits = stack([lower, upper], axis=-1).to_array()
        ax.set(xlim=x_limits, ylim=y_limits, aspect="equal")
        ax.set_axis_off()
        self.camera = panel(ax, mv.x, mv.y, mv.w)
        # Endpoints on the mirror plane x = 0 spanning the vertical view range.
        mirror_ends = mv.w.dual() + mv.y.dual() * stack([lower[1], upper[1]])   # [2] Point-
        trails = paths[np.arange(n_steps), np.arange(n_cranks)[:, None]].reshape(-1)  # [n_cranks * n_steps] Point-
        self.guides = [simplices.strokes(mirror_ends, Topology.chains([2]), mirror_colour, mirror_width, self.camera,
                                         dashes=mirror_dashes, cap=simplices.Cap.BUTT),
                       simplices.strokes(trails, Topology.loops([n_steps] * n_cranks), guide_colour,
                                         guide_width, self.camera)]
        self.pivot_marks = [simplices.markers(self.pivots, np.ones(3), pivot_radius + rim_width, self.camera),
                            simplices.markers(self.pivots, crank_colour, pivot_radius, self.camera)]
        self.image = overlay(ax, self.guides + self.pivot_marks)

    def frame(self, handles: Point, velocities: Point, forces: Point) -> plt.Figure:
        """Draws the cranks, handles, velocity arrows, and force arrows for one state (`[n_cranks] Point-`)."""
        n_cranks = len(handles)
        normalized_handles = handles.normalized()                               # [n_cranks] Point-
        radii = (self.pivots & normalized_handles).norm()                       # [n_cranks] Scalar+
        max_radius = radii.max()                                                # [] Scalar+
        # Outward radial displacement from each pivot, scaled by the longest crank radius.
        radial_offsets = (normalized_handles - self.pivots) / max_radius        # [n_cranks] Point-
        speeds = (normalized_handles & velocities).norm()                       # [n_cranks] Scalar+
        arrow_scale = max_radius * 0.55 / speeds.max()                          # [] Scalar+
        glyph_offset = max_radius * 0.055                                       # [] Scalar+
        # Offset the velocity and force arrows to either side of each handle so parallel arrows remain distinct.
        arrows = []
        for colour, side, directions in ((velocity_colour, -1, velocities), (force_colour, 1, forces)):
            tails = normalized_handles + side * glyph_offset * radial_offsets   # [n_cranks] Point-
            tips = tails + directions * arrow_scale                             # [n_cranks] Point-
            arrows.append(simplices.strokes(concatenate([tails, tips]), pairs(n_cranks), colour, arrow_width,
                                            self.camera, chevrons=np.ones(n_cranks), chevron=arrow_chevron))
        self.image.set_data(drawing.composited(self.guides + [
            simplices.strokes(concatenate([self.pivots, normalized_handles]), pairs(n_cranks), crank_colour,
                              crank_width, self.camera),
            *self.pivot_marks,
            simplices.markers(normalized_handles, crank_colour, handle_radius, self.camera),
            *arrows]))
        return self.figure


def work_figure(pivots: Point, handles: Point, velocities: Point, forces: Point, paths: Point) -> plt.Figure:
    """Renders a still figure of the cranks (`[n_cranks] Point-`), velocities, and forces along `paths`
    (`[n_steps, n_cranks] Point-`)."""
    return WorkView(pivots, paths).frame(handles, velocities, forces)


def work_animation(pivots: Point, handles: Point, velocities: Point, forces: Point) -> Iterator[np.ndarray]:
    """Yields RGB frames as the cranks turn through the sampled states (`[n_steps, n_cranks] Point-`)."""
    view = WorkView(pivots, handles)
    try:
        for handle, velocity, force in zip(handles, velocities, forces):
            yield capture(view.frame(handle, velocity, force))
    finally:
        plt.close(view.figure)
