"""Drawing for a rotating Bose-Einstein condensate: its density over the trap as it settles, deep blue where the
atoms crowd and white where there are none, so that its vortices show as white holes; and its vortices' count against
the radius they lie within as a chart, beside Feynman's rule.

The trap's places are read by the planes x and y; a place is a direction from the trap's middle, so its weight is one.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from numga.extensor import Extensor
from examples.util.charts import style, subplots
from examples.util.wgsl import contours, simplices
from examples.util.wgsl.camera import Camera

Topology = simplices.Topology
density_map = colormaps["Blues"]
counted_colour, feynman_colour = np.array(to_rgb("#1d4ed8")), np.array(to_rgb("#c2410c"))
# The picture's pixels across and the radius it frames, in oscillator lengths; the share of the colour map at the
# densest place of each frame.
n_pixels, framed = 300, 7.5
deepest = 0.95
# The chart's pixels.
chart_width, chart_height = 300, 220


def on_white(picture: np.ndarray) -> np.ndarray:
    """Colour and coverage `[rows, columns, 4]` laid over white, as bytes `[rows, columns, 3]`."""
    shown = picture[..., :3] + (1 - picture[..., 3:])
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def animate(topology: Topology, places: Extensor, densities: Iterable[Extensor]) -> Iterator[np.ndarray]:
    """The trap at every frame of the condensate's density `Scalar+[places]` as it comes, from white for none to deep
    blue at the frame's densest place."""
    mv = places.context.multivector
    camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.array([framed, framed]), n_pixels)
    points = places.batch()
    for density in densities:
        values = density.to_array()
        colours = density_map(deepest * values / values.max())[:, :3]
        yield on_white(contours.picture(points, topology, values, colours, camera))


@style()
def draw_count(radii: np.ndarray, counted: np.ndarray, feynman: np.ndarray) -> Figure:
    """The vortices counted within each radius `[radii]` of the trap's middle, as steps, and Feynman's rule, the
    rotation rate over pi times the area within the radius."""
    figure, ax = subplots(chart_width, chart_height)
    ax.plot(radii, feynman, color=feynman_colour)
    ax.plot(radii, counted, color=counted_colour, drawstyle="steps-post")
    ax.set_xlim(0, radii.max())
    ax.set_ylim(0, None)
    ax.set_xlabel("radius / oscillator length")
    ax.set_ylabel("vortices within")
    figure.tight_layout(pad=0.4)
    return figure
