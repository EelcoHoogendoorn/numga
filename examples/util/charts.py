"""The one style of the examples' charts: curves of a quantity against a parameter, drawn at fixed pixels.

A chart is drawn for a notebook column DESIGN_WIDTH pixels wide, and shown at its own width's share of that column,
so that its text keeps one size on the page whatever the chart's size. At DPI pixels an inch, the text is a fixed
number of pixels tall: TEXT_PX for labels, legends, titles and annotations, TICK_PX for tick labels; curves, markers
and frames have fixed pixel widths too. A chart takes its width from its content, from a third of the column to all
of it, and never more; its height from its content too. A draw function runs under `style()` and builds its figure
with `subplots`, and never sets a dpi or a font size of its own.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# The width in pixels of the notebook column a picture is designed for.
DESIGN_WIDTH = 800
DPI = 100
TEXT_PX, TICK_PX = 13, 11
LINE_PX, MARKER_PX, FRAME_PX = 1.5, 5, 1


def points(pixels: float) -> float:
    """A size in pixels as matplotlib's points, at DPI pixels an inch."""
    return pixels * 72 / DPI


# Line widths, marker sizes and font sizes are given to matplotlib in points.
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
    """The charts' settings, as a context or a decorator: matplotlib reads them as each artist is made, so a draw
    function runs under them from start to end."""
    return mpl.rc_context(RC)


def subplots(width_px: int, height_px: int, **options) -> tuple[plt.Figure, object]:
    """A figure `width_px` by `height_px` pixels at DPI, with `plt.subplots`' grid of axes and options."""
    return plt.subplots(figsize=(width_px / DPI, height_px / DPI), dpi=DPI, **options)
