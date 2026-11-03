"""Drawing for fluxons in a planar Josephson junction: a row of pendula along a narrow junction, the tunnelling current
over a wide one as an animation, and the boosted fluxons' contraction against their speed as a chart.

The junction's places are read by the planes x and y. The tunnelling current, the bias direction wedged with the
phase, is red where it runs one way across the barrier and blue the other: a fluxon line carries red on one side and
blue on the other, an antifluxon the reverse. A pendulum is a pivot on the junction's axis and a tip displaced from it
by the phase direction, `pivot + length * phase.dual()`, seen obliquely: across by the plane `mv.x + skew * mv.b` and up
by `-mv.a`, so that a resting pendulum hangs down.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga import concatenate
from numga.extensor import Extensor
from examples.util.charts import style, subplots
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
current_map = colormaps["RdBu_r"]
measured_colour, lorentz_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#c2410c"))
# A junction's picture: the pixels across its longer side; the share of the colour map used at full current.
n_pixels, current_share = 300, 0.9
# The chart's pixels.
chart_width, chart_height = 300, 220
# The pendula: their length and the oblique view's skew, in penetration depths, the picture's pixels across, the
# strokes' widths, and the axis's colour.
pendulum_length, skew, pendulum_pixels = 3.0, 0.5, 320
pendulum_width, axis_width = 1.5, 1.2
axis_colour = np.array(to_rgb("#6b7280"))


def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def junction_camera(vertices: Extensor) -> Camera:
    """The camera looking down on the junction, framing it, `n_pixels` across its longer side."""
    mv = vertices.context.multivector
    points = vertices.batch()
    across, up = (mv.x & points).to_array(), (mv.y & points).to_array()
    half = np.array([np.abs(across).max(), np.abs(up).max()])
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), half, round(n_pixels * half[0] / half.max()))


def animate(topology: Topology, vertices: Extensor, currents: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The junction at every frame of its tunnelling current `Rate+[places]` as it comes, red one way and blue the
    other."""
    camera = junction_camera(vertices)
    points = vertices.batch()
    for current in currents:
        values = np.asarray(current.batch().kernel)[..., 0]
        colours = current_map(0.5 + current_share * np.clip(values, -1, 1) / 2)[:, :3]
        yield on_white(contours.picture(points, topology, values, colours, camera))


@style()
def draw_contraction(speeds: np.ndarray, before: Extensor, after: Extensor) -> Figure:
    """The boosted fluxons' width relative to a resting one's, against their speed `[rapidities]`: their field's
    strength at its peak relative to the resting one's, from the squared strength in each triangle
    `[rapidities] Scalar+[faces]`, at the start and after their run, and the Lorentz contraction."""
    figure, ax = subplots(chart_width, chart_height)
    smooth = np.linspace(0, speeds.max(), 100)
    ax.plot(smooth, np.sqrt(1 - smooth ** 2), color=lorentz_colour)
    for strengths, marker in ((before, "o"), (after, "s")):
        peaks = np.sqrt(strengths.to_array().max(axis=-1))
        ax.plot(speeds, peaks[0] / peaks, marker, color=measured_colour, fillstyle="none")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("speed / Swihart velocity")
    ax.set_ylabel("width / resting width")
    figure.tight_layout(pad=0.4)
    return figure


def animate_pendula(pivots: Extensor, frames: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The row of pendula at every frame of their phase directions `Phase+[pendula]` as it comes, each from its pivot
    on the junction's axis `[pendula] Point-` to its tip, coloured by the tunnelling current, under the axis."""
    mv = pivots.context.multivector
    n_pendula = len(pivots)
    along = (mv.x & pivots).to_array()
    camera = Camera(mv.x + skew * mv.b, -mv.a, 0 * mv.x, 0 * mv.x, 1.0, np.array([along.mean(), 0.0]),
                    np.array([np.ptp(along) / 2 + skew * pendulum_length, pendulum_length * 1.1]), pendulum_pixels)
    rods = Topology(2 * n_pendula, np.stack([np.arange(n_pendula), n_pendula + np.arange(n_pendula)], axis=-1))
    axis = simplices.strokes(pivots, Topology.chains([n_pendula]), axis_colour, axis_width, camera)
    for phases in frames:
        tips = pivots + pendulum_length * phases.dual()                       # [pendula] Point-
        currents = np.asarray((mv.a ^ phases).kernel)[..., 0]
        colours = 0.15 + 0.85 * current_map(0.5 + current_share * np.clip(currents, -1, 1) / 2)[:, :3]
        yield on_white(simplices.composited([
            simplices.strokes(concatenate([pivots, tips]), rods, colours * 0.85, pendulum_width, camera), axis]))
