"""Drawing for the smoke ring flow: the Hashimoto surface swept by a curve, as shaded strips between the
curve at successive times; the tubes swept by several rings, cut open to show how they pass through each
other; curves flowing, as lines shaded by their depth; and the turning angle along a curve through time.
Curves are polylines; a closed one repeats its first vertex at the end."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from examples.animation import capture
from examples.fluids.smoke_rings import core

# The surface's colour from the first time to the last, the curve's colour from far to near, the curve's
# width, the view as (elevation, azimuth) in degrees, and the margin around the drawn extent.
EARLY, LATE = np.array([0.93, 0.88, 0.78]), np.array([0.55, 0.69, 0.80])
FAR, NEAR = np.array([0.75, 0.80, 0.86]), np.array([0.11, 0.21, 0.34])
# The colours of the rings' swept tubes, one per ring.
TUBES = np.array([[0.85, 0.52, 0.32], [0.38, 0.58, 0.78]])
WIDTH = 2.4
# The share of the largest motion above which a vertex counts as moving, for framing a curve's trail, and
# the trail's opacity, so that the curve shows through it.
MOVING = 0.01
TRAIL_OPACITY = 0.6
ELEVATION, AZIMUTH = 50.0, -50.0
MARGIN = 1.05


def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def bounds(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest corner `[3]` of the box around points `[..., 3]`, with the margin."""
    low, high = points.reshape(-1, 3).min(axis=0), points.reshape(-1, 3).max(axis=0)
    middle, reach = (low + high) / 2, MARGIN * (high - low) / 2
    return middle - reach, middle + reach


def frame(ax, points: np.ndarray) -> None:
    """Limits around points `[..., 3]` in a box of their shape, at one scale along every axis, and the
    view."""
    low, high = bounds(points)
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(tuple(high - low), zoom=1.25)
    ax.set_proj_type("ortho")
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def draw_surface(curves: core.Point) -> plt.Figure:
    """The surface swept by a curve through time `[times, n]`: a quad between each edge at one time and
    the same edge at the next, coloured from early to late."""
    points = coordinates(curves)                                               # [times, n, 3]
    before, after = points[:, :-1], points[:, 1:]
    quads = np.stack([before[:-1], after[:-1], after[1:], before[1:]], axis=-2)   # [times - 1, n - 1, 4, 3]
    share = np.linspace(0, 1, len(quads))[:, None, None] * np.ones(quads.shape[:2] + (1,))
    colours = ((1 - share) * EARLY + share * LATE).reshape(-1, 3)
    figure = plt.figure(figsize=(6.0, 6.0), dpi=110)
    ax = figure.add_axes((0, 0, 1, 1), projection="3d")
    ax.add_collection3d(Poly3DCollection(quads.reshape(-1, 4, 3), facecolors=colours, shade=True, linewidths=0))
    frame(ax, points)
    return figure


def toward() -> np.ndarray:
    """The unit vector towards the viewer."""
    elevation, azimuth = np.radians(ELEVATION), np.radians(AZIMUTH)
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def draw_swept(curves: core.Point) -> plt.Figure:
    """The tubes swept by closed curves through time `[times, rings, n]`, each in its own colour, with the
    half of every tube nearer the viewer cut away, so that the tubes inside show."""
    points = coordinates(curves)                                               # [times, rings, n, 3]
    before, after = points[:, :, :-1], points[:, :, 1:]
    quads = np.stack([before[:-1], after[:-1], after[1:], before[1:]], axis=-2)   # [times - 1, rings, n - 1, 4, 3]
    # Cut by the plane through the axis facing the viewer: keep the far half of each tube.
    centres = quads.mean(axis=-2)                                              # [times - 1, rings, n - 1, 3]
    far = centres[..., :2] @ toward()[:2] < 0                                  # [times - 1, rings, n - 1]
    colours = np.broadcast_to(TUBES[:, None, :], centres.shape)               # [times - 1, rings, n - 1, 3]
    figure = plt.figure(figsize=(6.0, 6.0), dpi=110)
    ax = figure.add_axes((0, 0, 1, 1), projection="3d")
    ax.add_collection3d(Poly3DCollection(quads[far], facecolors=colours[far], shade=True, linewidths=0))
    frame(ax, points)
    return figure


def animate_curves(curves: core.Point) -> list[np.ndarray]:
    """Frames of curves flowing `[times, ..., n]`, all from one view and to one scale, the curves shaded
    from far to near."""
    points = coordinates(curves)                                               # [times, ..., n, 3]
    depth = points @ toward()
    low, high = depth.min(), depth.max()
    images = []
    for curve, near in zip(points, depth):
        segments = np.stack([curve[..., :-1, :], curve[..., 1:, :]], axis=-2).reshape(-1, 2, 3)   # [segments, 2, 3]
        share = (((near[..., :-1] + near[..., 1:]) / 2 - low) / (high - low)).reshape(-1)
        # Far segments first, so that near ones are drawn over them.
        order = np.argsort(share)
        figure = plt.figure(figsize=(5.0, 5.0), dpi=90)
        ax = figure.add_axes((0, 0, 1, 1), projection="3d")
        ax.computed_zorder = False
        ax.add_collection3d(Line3DCollection(segments[order], colors=(1 - share[order, None]) * FAR + share[order, None] * NEAR,
                                             linewidths=WIDTH, capstyle="round"))
        frame(ax, points)
        images.append(capture(figure))
        plt.close(figure)
    return images


def animate_trail(curves: core.Point) -> list[np.ndarray]:
    """Frames of a curve flowing `[times, n]`, each over the surface the curve has swept so far, coloured
    from early to late, and framed on the part of space where the curve moves."""
    points = coordinates(curves)                                               # [times, n, 3]
    before, after = points[:, :-1], points[:, 1:]
    quads = np.stack([before[:-1], after[:-1], after[1:], before[1:]], axis=-2)   # [times - 1, n - 1, 4, 3]
    share = np.linspace(0, 1, len(quads))[:, None]
    colours = np.broadcast_to(((1 - share) * EARLY + share * LATE)[:, None], quads.shape[:2] + (3,))
    # The vertices that move, and the box around their paths; what lies outside is left out.
    travel = np.linalg.norm(points - points[0], axis=-1).max(axis=0)          # [n]
    moving = points[:, travel > MOVING * travel.max()]                         # [times, moving, 3]
    low, high = bounds(moving)
    inside = np.all((points >= low) & (points <= high), axis=-1)              # [times, n]
    quads_inside = inside[:-1, :-1] & inside[:-1, 1:] & inside[1:, :-1] & inside[1:, 1:]   # [times - 1, n - 1]
    segments_inside = inside[:, :-1] & inside[:, 1:]                           # [times, n - 1]
    images = []
    for time in range(len(points)):
        figure = plt.figure(figsize=(6.0, 6.0), dpi=90)
        ax = figure.add_axes((0, 0, 1, 1), projection="3d")
        ax.computed_zorder = False
        ax.add_collection3d(Poly3DCollection(quads[:time][quads_inside[:time]], facecolors=colours[:time][quads_inside[:time]],
                                             shade=True, linewidths=0, alpha=TRAIL_OPACITY))
        segments = np.stack([before[time], after[time]], axis=-2)[segments_inside[time]]   # [segments, 2, 3]
        ax.add_collection3d(Line3DCollection(segments, colors=NEAR, linewidths=WIDTH, capstyle="round"))
        frame(ax, moving)
        images.append(capture(figure))
        plt.close(figure)
    return images


def draw_turns(turns: core.Rotor) -> plt.Figure:
    """The turning angle at every inner vertex of a curve through time `[times, n]`: vertices across,
    times down, the angle as darkness."""
    parts = turns.cast(core.ga.subspace("1 yz zx xy")).kernel                  # [times, n, 4]
    angles = np.arctan2(np.linalg.norm(parts[..., 1:], axis=-1), parts[..., 0])   # [times, n]
    figure = plt.figure(figsize=(6.0, 4.0), dpi=110)
    ax = figure.add_axes((0.1, 0.12, 0.86, 0.84))
    ax.imshow(angles, cmap="Greys", origin="upper", aspect="auto", interpolation="nearest",
              extent=(0.5, angles.shape[1] + 0.5, angles.shape[0] - 0.5, -0.5))
    ax.set(xlabel="vertex", ylabel="step")
    return figure
