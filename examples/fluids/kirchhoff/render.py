"""Drawing for Kirchhoff's equations: the added mass of spheroids by aspect, and bodies gliding through
the fluid, seen from above, each case in its own lane with the line of its impulse."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import flat_panel
from examples.fluids.kirchhoff import core

Topology = simplices.Topology

mv = core.mv
# The colours of the three coefficients: shifting along the axis of symmetry, shifting across it, and
# turning it. The colours of each case's body, its outline and its path, and of the impulse.
coefficient_colours = ["#0369a1", "#f97316", "#16a34a"]
body_colours = ["#cbd5e1", "#7dd3fc", "#fdba74"]
edge_colours = ["#64748b", "#0369a1", "#c2410c"]
path_colours = ["#94a3b8", "#0284c7", "#ea580c"]
impulse_colour = "#475569"
# The free space between the lanes, and around the bodies at the ends of the view.
gap = 1.0
# Widths, dashes and radii in pixels: the impulse's dashed line, the paths and the bodies' outlines, and the noses.
impulse_width, impulse_dashes = 1.1, (8.3, 5.6)
path_width, outline_width, nose_radius = 1.95, 1.95, 3.5


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """A named colour's red, green and blue."""
    return np.array(to_rgb(colour))


@charts.style()
def draw_coefficients(aspects: np.ndarray, ratios: core.Scalar) -> plt.Figure:
    """The fluid's kinetic energy over the solid's for spheroids of each aspect `[aspects, 3]`, against
    the aspect: discs on the left, rods on the right."""
    values = ratios.to_array()                                                # [aspects, 3]
    # Half the column: three smooth curves, and room for the two long axis names.
    figure, ax = charts.subplots(420, 280)
    for column, colour in enumerate(coefficient_colours):
        ax.plot(aspects, values[:, column], color=colour)
    ax.axvline(1, color="#cbd5e1", linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set_xscale("log")
    # The ratios from their zero, read off at the ticks; the aspect without numbers: discs on the left, rods on the
    # right, the sphere at the grey line.
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlabel("axis of symmetry over the other semi-axes")
    ax.set_ylabel("kinetic energy, fluid over solid")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def animate_gliding(impulse: core.Forque,
                    frames: Iterable[tuple[core.Point, core.Point, core.Point]]) -> Iterator[np.ndarray]:
    """Each frame of every case's outline `[cases, outline + 1]`, centre `[cases]` and nose `[cases]`: the
    bodies seen from above, each in its own lane, with its path so far and its impulse."""
    frames = [tuple(points.normalized() for points in frame) for frame in frames]
    outlines = stack([outline for outline, _, _ in frames])                  # [frames, cases, outline + 1] Point-
    centres = stack([middle for _, middle, _ in frames])                     # [frames, cases] Point-
    noses = stack([nose for _, _, nose in frames])                           # [frames, cases] Point-
    count = centres.shape[1]
    # Each lane as wide as the bodies ever reach across it, and the view as long as they reach along it.
    widths = (mv.y & outlines).abs().reshape(-1)
    lane = 2 * widths.max() + gap
    reach = (mv.x & outlines).reshape(-1)
    extent = stack((reach.min() - gap / 2, reach.max() + gap / 2))
    # Each case moved across into its own lane, the first furthest.
    lanes = core.directions[1] * (lane * (count - 1 - np.arange(count)))     # [cases] Direction-
    # The ends of the impulse's line within the extent: where it meets the planes square to x there.
    ends = (impulse ^ (mv.x - extent * mv.w)).normalized()                    # [2] Point-
    lines = (ends + lanes[:, None]).reshape(-1)                               # [cases * 2] Point-
    shapes = outlines + lanes[:, None]                                       # [frames, cases, outline + 1] Point-
    # Each case's path through the frames, case by case.
    tracks = stack([(centres + lanes)[:, case] for case in range(count)])    # [cases, frames] Point-
    tips = noses + lanes                                                     # [frames, cases] Point-
    # The body's triangles, the same in every frame, and its outline a closed chain.
    body = Topology.triangulate(shapes[0, 0, :-1])
    rim = Topology.chains([shapes.shape[-1]])
    height = lane * count
    # The lanes and their impulses set up once; each body, its path so far and its nose move.
    aspect = float((height / (extent[1] - extent[0])).to_array())
    figure = plt.figure(figsize=(4.0, 4.0 * aspect), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.set_xlim(*extent.to_array())
    ax.set_ylim(*stack((-lane / 2, height - lane / 2)).to_array())
    ax.set_aspect("equal")
    ax.set_axis_off()
    camera = flat_panel(ax, core.ga)
    impulses = simplices.strokes(lines, Topology.chains([2] * count), rgb(impulse_colour), impulse_width, camera,
                                 dashes=impulse_dashes, cap=simplices.Cap.BUTT)
    image = simplices.overlay(ax, [impulses])
    for index in range(len(frames)):
        path_ink = np.repeat([rgb(colour) for colour in path_colours], index, axis=0)   # [cases * index, 3]
        bodies = [picture for case in range(count) for picture in (
            contours.picture(shapes[index, case, :-1], body, np.zeros(body.n_vertices), rgb(body_colours[case]),
                             camera),
            simplices.strokes(shapes[index, case], rim, rgb(edge_colours[case]), outline_width, camera))]
        image.set_data(simplices.composited([
            impulses,
            simplices.strokes(tracks[:, :index + 1].reshape(-1), Topology.chains([index + 1] * count), path_ink,
                              path_width, camera),
            *bodies,
            simplices.markers(tips[index], np.array([rgb(colour) for colour in edge_colours]), nose_radius, camera)]))
        yield capture(figure)
    plt.close(figure)
