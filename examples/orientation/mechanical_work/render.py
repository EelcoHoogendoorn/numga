"""A rotating crank, its applied force and handle velocity beside a mirror."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import panel
from examples.orientation.mechanical_work.core import Point

velocity_colour, force_colour, crank_colour, guide_colour, mirror_colour = (
    to_rgb(colour) for colour in ("#218c9d", "#ce713f", "#424c56", "#d4dce1", "#aebfca"))
# The figure's width and height in pixels.
figure_pixels = (420, 210)
dpi = 100
# The widths of the mirror, its dashes and gaps, the handles' paths, the cranks and the arrows, the arrows' heads,
# and the radii of the pivots, their rims and the handles, in pixels.
mirror_width, mirror_dashes, guide_width, crank_width = 1.2, (4.5, 5.5), 1.0, 3.6
arrow_width, arrow_head = 2.5, 12.0
pivot_radius, rim_width, handle_radius = 4.0, 1.3, 2.7


# --- plumbing -------------------------------------------------------------------------
def pairs(n_pairs: int) -> simplices.Mesh:
    """An edge from each of the first `n_pairs` points to the one `n_pairs` after it."""
    return simplices.Mesh(2 * n_pairs, np.stack([np.arange(n_pairs), n_pairs + np.arange(n_pairs)], axis=-1))


class WorkView:
    """The mirror, the handles' paths and the pivots, drawn once; the cranks, their handles and the arrows at them,
    drawn for each state, on the GPU in the cranks' own algebra."""

    def __init__(self, pivots: Point, paths: Point) -> None:
        self.pivots = pivots.normalized()                                       # [cranks] Point-
        # The longest crank: the distance from a pivot to its handle is the norm of their join.
        radii = (self.pivots & paths.normalized()).norm().reshape(-1)
        margin = radii.max() * 0.95
        mv = paths.context.multivector
        readings = stack((mv.x, mv.y))[:, None] & paths.normalized().reshape(-1)
        lower = (readings.min(axis=-1) - margin).to_array()
        upper = (readings.max(axis=-1) + margin).to_array()

        self.figure, ax = plt.subplots(figsize=np.array(figure_pixels) / dpi, dpi=dpi, layout="constrained")
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), aspect="equal")
        ax.set_axis_off()
        self.camera = panel(ax, mv.x, mv.y)
        # The mirror, the plane x = 0, from the bottom of the view to its top.
        mirror = mv.point([[0, lower[1]], [0, upper[1]]])                       # [ends] Point-
        # Each handle's path, closed: its places in turn and the first again, a chain of them.
        n_steps, n_cranks = paths.shape
        steps = np.r_[np.arange(n_steps), 0]
        trails = paths[steps, np.arange(n_cranks)[:, None]].reshape(-1)         # [cranks * (steps + 1)] Point-
        self.guides = [simplices.strokes(mirror, simplices.Mesh.chains([2]), mirror_colour, mirror_width, self.camera,
                                         dashes=mirror_dashes, cap=simplices.Cap.BUTT),
                       simplices.strokes(trails, simplices.Mesh.chains([n_steps + 1] * n_cranks), guide_colour,
                                         guide_width, self.camera)]
        self.pivot_marks = [simplices.markers(self.pivots, np.ones(3), pivot_radius + rim_width, self.camera),
                            simplices.markers(self.pivots, crank_colour, pivot_radius, self.camera)]
        self.image = simplices.overlay(ax, self.guides + self.pivot_marks)

    def update(self, handles: Point, velocities: Point, forces: Point) -> plt.Figure:
        handles = handles.normalized()                                          # [cranks] Point-
        radii = (self.pivots & handles).norm()
        radius = radii.max()
        # The direction out from each pivot to its handle, per length of the longest crank.
        radial = (handles - self.pivots) / radius                               # [cranks] Direction-
        # The fastest handle's velocity reaches over half a crank: a direction's size is the norm of its join with a
        # unit point.
        speeds = (handles & velocities).norm()
        arrow_scale = radius * 0.55 / speeds.max()
        offset = radius * 0.055
        n_cranks = len(handles)
        # Separate the glyphs slightly so equal directions remain visible: the velocity's arrows and the force's,
        # each on one side of the handles.
        arrows = []
        for colour, side, directions in ((velocity_colour, -1, velocities), (force_colour, 1, forces)):
            tails = handles + side * offset * radial                            # [cranks] Point-
            arrows.append(simplices.strokes(concatenate([tails, tails + directions * arrow_scale]), pairs(n_cranks),
                                            colour, arrow_width, self.camera, heads=np.ones(n_cranks),
                                            head=arrow_head))
        self.image.set_data(simplices.composited(self.guides + [
            simplices.strokes(concatenate([self.pivots, handles]), pairs(n_cranks), crank_colour, crank_width,
                              self.camera),
            *self.pivot_marks,
            simplices.markers(handles, crank_colour, handle_radius, self.camera),
            *arrows]))
        return self.figure


def draw_work(pivots: Point, handles: Point, velocities: Point, forces: Point, paths: Point) -> plt.Figure:
    return WorkView(pivots, paths).update(handles, velocities, forces)


def animate(pivots: Point, handles: Point, velocities: Point, forces: Point) -> Iterator[np.ndarray]:
    """The handles turn through the supplied geometric states."""
    view = WorkView(pivots, handles)
    for handle, velocity, force in zip(handles, velocities, forces):
        yield capture(view.update(handle, velocity, force))
    plt.close(view.figure)
