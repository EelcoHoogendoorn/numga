"""Drawing for Kirchhoff's equations: the added-mass ratios of spheroids by aspect ratio, and bodies gliding through ideal fluid seen from above in separate lanes alongside their impulse lines."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import dim, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.shapes import delaunay
from numga.topology import Topology
from numga.render import contours, drawing, simplices
from examples.util.panels import overlay, panel
from examples.fluids.kirchhoff import core


mv = core.mv
# Curve colours for shifting along the symmetry axis, shifting across it, and turning it; lane colours for bodies, outlines, paths, and impulse lines.
coefficient_colours = ["#0369a1", "#f97316", "#16a34a"]
body_colours = ["#cbd5e1", "#7dd3fc", "#fdba74"]
edge_colours = ["#64748b", "#0369a1", "#c2410c"]
path_colours = ["#94a3b8", "#0284c7", "#ea580c"]
impulse_colour = "#475569"
# Clearance between lanes and around the bodies at the ends of the view.
gap = mv.scalar(1.0, unit=dim.length)
# Pixel widths, dash pattern, and marker radius for impulse lines, centre paths, outlines, and nose markers.
impulse_width, impulse_dashes = mv.scalar(1.1, unit=dim.pixel), (8.3, 5.6)
path_width = mv.scalar(1.95, unit=dim.pixel)
outline_width = mv.scalar(1.95, unit=dim.pixel)
nose_radius = mv.scalar(3.5, unit=dim.pixel)


# --- plumbing -------------------------------------------------------------------------
def rgb(colour: str) -> np.ndarray:
    """Red, green, and blue components of a named colour."""
    return np.array(to_rgb(colour))


@charts.style()
def draw_coefficients(aspects: np.ndarray, ratios: core.Scalar) -> plt.Figure:
    """Fluid-to-solid kinetic energy ratios across spheroid aspect ratios from flat discs to slender rods."""
    values = ratios.to_array()                                                # [..., 3]
    # Half the column width: three smooth curves with room for both axis labels.
    figure, ax = charts.subplots(420, 280)
    for column, colour in enumerate(coefficient_colours):
        ax.plot(aspects, values[:, column], color=colour)
    ax.axvline(1, color="#cbd5e1", linewidth=charts.FRAME_WIDTH, zorder=0)
    ax.set_xscale("log")
    # Ratios start at zero on the vertical axis; the sphere sits at the vertical reference line.
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.set_xlabel("axis of symmetry over the other semi-axes")
    ax.set_ylabel("kinetic energy, fluid over solid")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def animate_gliding(impulse: core.Forque, body_outline: core.Point, nose: core.Point,
                    states: Iterable[core.Motor]) -> Iterator[np.ndarray]:
    """Yield top-down animation frames of each gliding case in its own lane with its centre path and impulse line."""
    motor_frames = list(states)
    n_frames = len(motor_frames)
    motors = stack(motor_frames)                                              # [n_frames, n_cases] Motor+
    n_cases = motors.shape[1]
    n_outline = body_outline.shape[0]
    outlines = (motors[:, :, None] >> body_outline).normalized()              # [n_frames, n_cases, n_outline] Point-
    centres = (motors >> core.centre).normalized()                            # [n_frames, n_cases] Point-
    noses = (motors >> nose).normalized()                                     # [n_frames, n_cases] Point-
    # Size each lane to clear the widest transverse excursion, and span the full longitudinal travel.
    widths = (mv.y & outlines).abs().reshape(-1)                              # [...] Scalar+
    lane = 2 * widths.max() + gap                                             # [] Scalar+
    reach = (mv.x & outlines).reshape(-1)                                     # [...] Scalar+
    extent = stack((reach.min() - gap / 2, reach.max() + gap / 2))            # [2] Scalar+
    # Offset each case transversely into its own lane, placing the first case at the top.
    lanes = core.directions[1] * (lane * (n_cases - 1 - np.arange(n_cases)))  # [n_cases] Direction-
    # Intersect the impulse line with the left and right boundary planes of the view.
    ends = (impulse ^ (mv.x - extent * mv.w)).normalized()                    # [2] Point-
    impulse_segments = (ends + lanes[:, None]).reshape(-1)                    # [...] Point-
    lane_outlines = outlines + lanes[:, None]                                 # [n_frames, n_cases, n_outline] Point-
    tracks = stack([(centres + lanes)[:, case] for case in range(n_cases)])   # [n_cases, n_frames] Point-
    tips = noses + lanes                                                      # [n_frames, n_cases] Point-
    # Triangulate the reference ellipse once from its planar readout and form its closed boundary loop.
    body_triangles, _ = delaunay((stack((mv.x, mv.y)) & lane_outlines[0, 0].batch()[..., None]).to_array())
    V, E, F = body_triangles.n_elements
    rim = Topology.loops([n_outline])
    height = lane * n_cases                                                   # [] Scalar+
    aspect = float((height / (extent[1] - extent[0])).to_array())
    figure = plt.figure(figsize=(4.0, 4.0 * aspect), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ylim = stack((-lane / 2, height - lane / 2))
    ax.set_aspect("equal")
    ax.set_axis_off()
    camera = panel(ax, mv.x, mv.y, mv.w, xlim=extent, ylim=ylim)
    impulses = simplices.strokes(impulse_segments, Topology.chains([2] * n_cases), rgb(impulse_colour), impulse_width,
                                 camera, dashes=impulse_dashes, cap=simplices.Cap.BUTT)
    image = overlay(ax, [impulses])
    for index in range(n_frames):
        path_ink = np.repeat([rgb(colour) for colour in path_colours], index, axis=0)
        bodies = [picture for case in range(n_cases) for picture in (
            contours.picture(lane_outlines[index, case], body_triangles, np.zeros(V),
                             rgb(body_colours[case]), camera),
            simplices.strokes(lane_outlines[index, case], rim, rgb(edge_colours[case]), outline_width, camera))]
        image.set_data(drawing.composited([
            impulses,
            simplices.strokes(tracks[:, :index + 1].reshape(-1), Topology.chains([index + 1] * n_cases), path_ink,
                              path_width, camera),
            *bodies,
            simplices.markers(tips[index], np.array([rgb(colour) for colour in edge_colours]), nose_radius, camera)]))
        yield capture(figure)
    plt.close(figure)
