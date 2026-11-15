"""Planar Josephson junctions render as an oblique chain of mechanical pendula, animated tunnelling-current maps, and a Lorentz-contraction chart.

Camera planes pair directly with junction vertices to read spatial coordinates, while pairing tunnelling-current bivectors or dualized phase displacements reads signed supercurrent amplitudes into a diverging red-blue palette. Each pendulum joins a pivot along the strip axis to a tip offset by the dualized phase direction under an oblique camera that maps the zero-phase axis downward.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.electromagnetism.josephson import core
from examples.util.charts import style, subplots
from numga import concatenate, stack
from numga.extensor import Extensor
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology

mv = core.mv
current_map = colormaps["RdBu_r"]
measured_colour, lorentz_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#c2410c"))
# Pixel count across the junction's longer side and fraction of the diverging colormap used at unit current.
n_pixels, current_share = 300, 0.9
chart_width, chart_height = 300, 220
# Pendulum rod length, oblique camera skew, pixel width, stroke widths, and axis colour.
pendulum_length, skew, pendulum_pixels = 3.0, 0.5, 320
pendulum_width, axis_width = 1.5, 1.2
axis_colour = np.array(to_rgb("#6b7280"))


def white_canvas(picture: np.ndarray) -> np.ndarray:
    """Alpha-compositing RGBA coverage (`[n_rows, n_columns, 4]`) over white yields 8-bit RGB pixels (`[n_rows, n_columns, 3]`)."""
    rgb_canvas = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(rgb_canvas, 0, 1) * 255).astype(np.uint8)


def junction_camera(vertices: Extensor) -> Camera:
    """A top-down camera frames the triangulated junction (`Point-[V]`) with `n_pixels` across its longer side."""
    points = vertices.batch()                                                 # [...] Point-
    coordinates = (stack((mv.x, mv.y)) & points[..., None]).to_array()
    half_extents = np.abs(coordinates).max(axis=0)
    width_pixels = round(n_pixels * half_extents[0] / half_extents.max())
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), half_extents, width_pixels)


def current_frames(topology: Topology, vertices: Extensor, currents: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """Yields RGB frames colouring the triangulated junction by its signed tunnelling current (`Rate+[V]`)."""
    camera = junction_camera(vertices)
    points = vertices.batch()                                                 # [...] Point-
    current_dual = mv.ab.dual()                                               # [] Antibivector-
    for current in currents:
        values = (current.batch() & current_dual).to_array()
        colours = current_map(0.5 + current_share * np.clip(values, -1, 1) / 2)[:, :3]
        yield white_canvas(drawing.composited([contours.picture(points, topology, values, colours, camera)]))


@style()
def contraction_figure(speeds: np.ndarray, initial_strengths: Extensor, final_strengths: Extensor) -> Figure:
    """Inverse peak field strengths (`[n_rapidities] Scalar+[F]`) at the start and end of the strip run trace the Lorentz contraction curve against speed."""
    figure, ax = subplots(chart_width, chart_height)
    smooth_speeds = np.linspace(0, 0.999, 200)
    ax.plot(smooth_speeds, np.sqrt(1 - smooth_speeds ** 2), color=lorentz_colour)
    for strengths, marker in ((initial_strengths, "o"), (final_strengths, "s")):
        peaks = strengths.batch().max(axis=-1).square_root()                  # [...] Scalar+
        ax.plot(speeds, (peaks[0] / peaks).to_array(), marker, color=measured_colour, fillstyle="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("speed / Swihart velocity")
    ax.set_ylabel("width / rest width")
    figure.tight_layout(pad=0.4)
    return figure


def pendulum_frames(pivots: Extensor, phases_stream: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """Yields oblique RGB frames of pendula (`[n_pendula] Phase+`) swinging from `pivots` (`[n_pendula] Point-`) and coloured by tunnelling current."""
    n_pendula = len(pivots)
    x_positions = (mv.x & pivots).to_array()
    half_extents = np.array([np.ptp(x_positions) / 2 + skew * pendulum_length, pendulum_length * 1.1])
    camera = Camera(mv.x + skew * mv.b, -mv.a, 0 * mv.x, 0 * mv.x, 1.0,
                    np.array([x_positions.mean(), 0.0]), half_extents, pendulum_pixels)
    rod_topology = Topology(2 * n_pendula, np.stack([np.arange(n_pendula), n_pendula + np.arange(n_pendula)], axis=-1))
    axis_stroke = simplices.strokes(pivots, Topology.chains([n_pendula]), axis_colour, axis_width, camera)
    for phases in phases_stream:
        displacements = phases.dual()                                         # [n_pendula] Point-
        tips = pivots + pendulum_length * displacements                       # [n_pendula] Point-
        current_values = (mv.b & displacements).to_array()
        colours = 0.15 + 0.85 * current_map(0.5 + current_share * np.clip(current_values, -1, 1) / 2)[:, :3]
        rod_stroke = simplices.strokes(concatenate([pivots, tips]), rod_topology, colours * 0.85, pendulum_width, camera)
        yield white_canvas(drawing.composited([rod_stroke, axis_stroke]))
