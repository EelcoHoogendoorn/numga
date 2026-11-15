"""Matplotlib charts across the examples share a single pixel typography and stroke weight calibrated
to a notebook column `DESIGN_WIDTH` (`800` pixels) wide.

At `DPI` (`100` pixels per inch), font heights (`TEXT_PX` for labels, titles, legends, and
annotations; `TICK_PX` for tick labels) and stroke, marker, and frame widths convert to fixed
matplotlib point sizes. Each chart chooses its own pixel width and height from its content, between a
third of the column and the full column, and notebook display scales every saved image to
`width_px / DESIGN_WIDTH` of the column so text and line weights appear uniform across figures.
Drawing functions run inside `style()` and allocate figures with `subplots(width_px, height_px)`
rather than overriding `dpi` or font sizes locally.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Design width in pixels of the notebook column against which figures are scaled.
DESIGN_WIDTH = 800
DPI = 100
TEXT_PX, TICK_PX = 13, 11
LINE_PX, MARKER_PX, FRAME_PX = 1.5, 5, 1


def points(pixels: float) -> float:
    """Pixel sizes convert to matplotlib points (`72` per inch) at `DPI` pixels per inch."""
    return pixels * 72 / DPI


# Matplotlib rc parameters take font sizes, line widths, and marker sizes in points.
LINE_WIDTH, MARKER_SIZE, FRAME_WIDTH = points(LINE_PX), points(MARKER_PX), points(FRAME_PX)
RC = {
    "figure.dpi": DPI,
    "savefig.dpi": "figure",
    "font.size": points(TEXT_PX),
    "axes.labelsize": points(TEXT_PX),
    "axes.titlesize": points(TEXT_PX),
    "legend.fontsize": points(TEXT_PX),
    "legend.title_fontsize": points(TEXT_PX),
    "figure.titlesize": points(TEXT_PX),
    "figure.labelsize": points(TEXT_PX),
    "xtick.labelsize": points(TICK_PX),
    "ytick.labelsize": points(TICK_PX),
    "lines.linewidth": LINE_WIDTH,
    "patch.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "axes.linewidth": FRAME_WIDTH,
    "grid.linewidth": FRAME_WIDTH,
    "xtick.major.width": FRAME_WIDTH,
    "ytick.major.width": FRAME_WIDTH,
    "xtick.minor.width": FRAME_WIDTH,
    "ytick.minor.width": FRAME_WIDTH,
    "xtick.major.size": points(4),
    "ytick.major.size": points(4),
    "xtick.minor.size": points(2),
    "ytick.minor.size": points(2),
}


def style():
    """Matplotlib reads `RC` as each artist is created, so wrapping a drawing function in this
    context manager or decorator applies the shared typography and stroke widths throughout."""
    return mpl.rc_context(RC)


def subplots(width_px: int, height_px: int, **options) -> tuple[plt.Figure, object]:
    """Figure dimensions `width_px` and `height_px` divide by `DPI` to size a `plt.subplots` figure
    and axes grid in exact pixels."""
    return plt.subplots(figsize=(width_px / DPI, height_px / DPI), dpi=DPI, **options)
