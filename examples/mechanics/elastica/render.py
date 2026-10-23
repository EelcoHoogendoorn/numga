"""Drawing for the elastica: the curled strip from straight to a double ring, Euler's elastica between their
pins on the line of thrust, the sweep through them, and Kirchhoff's helices with a ribbon showing the twist of
the cross-sections."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from examples.util.animation import capture
from examples.mechanics.elastica import core
from examples.util.wgsl import image_figure, meshes, solids, strokes

ga = core.ga
# The colours of the rods by case, of the line of thrust and the pins, and of the ribbon; the margin around
# the drawn rods, the panels of Euler's plate per row, the spacing of the helices as a share of the widest,
# and their view as (elevation, azimuth), in radians, azimuth zero looking along +y.
ROD_MAP = "viridis"
ROD, THRUST, RIBBON = np.array(to_rgb("#1f6f99")), "0.55", np.array(to_rgb("#7cc4e4"))
MARGIN = 0.08
COLUMNS = 4
SPACING = 1.4
VIEW = np.radians([12.0, 10.0])
# The uncoiling wire's view: its elevation, and the azimuths the camera turns through over the frames.
UNCOIL_ELEVATION = np.radians(16.0)
UNCOIL_AZIMUTHS = np.radians([-30.0, 150.0])
# The telephone cord's view as (elevation, azimuth), and the colour of its clamps.
CORD_VIEW = np.radians([20.0, 26.0])
CLAMP = np.full(3, 0.2)
# The pixels across the helices, the uncoiling wire and the cord; the width of a rod in pixels, the radius of a
# clamp, and the background.
N_HELIX_PIXELS, N_UNCOIL_PIXELS, N_CORD_PIXELS = 660, 460, 900
ROD_WIDTH, CLAMP_RADIUS = 3.0, 5.0
BACKGROUND = np.ones(3)


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of points, on the last axis."""
    kernel = points.cast(ga.subspace("ywz wxz xwy xyz")).kernel
    return kernel[..., :3] / kernel[..., 3:]


def plane(points: core.Point) -> np.ndarray:
    """Points of the plane of bending, along the unbent rod and up, on the last axis."""
    return coordinates(points)[..., [2, 1]]


def frame_plane(ax: plt.Axes, low: np.ndarray, high: np.ndarray) -> None:
    ax.set(xlim=(low[0] - MARGIN, high[0] + MARGIN), ylim=(low[1] - MARGIN, high[1] + MARGIN), aspect="equal")
    ax.set_axis_off()


def draw_curls(centres: core.Point) -> plt.Figure:
    """The strip curled by each couple `[curls, pieces + 1]`, from straight, darkest, to a double ring, with
    its clamp."""
    curves = plane(centres)                                                  # [curls, pieces + 1, 2]
    colours = plt.get_cmap(ROD_MAP)(np.linspace(0.05, 0.9, len(curves)))
    figure, ax = plt.subplots(figsize=(7.5, 4.0), dpi=110)
    for curve, colour in zip(curves, colours):
        ax.plot(*curve.T, color=colour, lw=2.2)
    low, high = curves.reshape(-1, 2).min(axis=0), curves.reshape(-1, 2).max(axis=0)
    ax.plot([0, 0], [low[1] - MARGIN, high[1] + MARGIN], color="0.2", lw=4, solid_capstyle="butt")
    frame_plane(ax, low, high)
    figure.tight_layout()
    return figure


def draw_rod(ax: plt.Axes, curve: np.ndarray, low: np.ndarray, high: np.ndarray) -> tuple[Line2D, Line2D]:
    """One of Euler's elastica: the line of thrust, the rod, and its pins at its two ends. The rod and the pins,
    for an animation to move."""
    ax.axhline(0, color=THRUST, lw=0.8)
    rod, = ax.plot(*curve.T, color=ROD, lw=2.2)
    pins, = ax.plot(*curve[[0, -1]].T, "^", color="0.2", markersize=6)
    frame_plane(ax, low, high)
    return rod, pins


def draw_plate(centres: core.Point) -> plt.Figure:
    """Euler's elastica at each end angle `[angles, pieces + 1]`, each in its own panel, all to one scale."""
    curves = plane(centres)                                                  # [angles, pieces + 1, 2]
    low, high = curves.reshape(-1, 2).min(axis=0), curves.reshape(-1, 2).max(axis=0)
    rows = -(-len(curves) // COLUMNS)
    figure, axes = plt.subplots(rows, COLUMNS, figsize=(2.6 * COLUMNS, 2.0 * rows), dpi=110)
    for ax, curve in zip(axes.reshape(-1), curves):
        draw_rod(ax, curve, low, high)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.05, hspace=0.05)
    return figure


def animate_sweep(centres: core.Point) -> Iterator[np.ndarray]:
    """Euler's elastica as the end angle grows `[frames, pieces + 1]`, all to one scale, in one figure whose rod and
    pins move."""
    curves = plane(centres)                                                  # [frames, pieces + 1, 2]
    low, high = curves.reshape(-1, 2).min(axis=0), curves.reshape(-1, 2).max(axis=0)
    figure, ax = plt.subplots(figsize=(5.0, 5.0 * (high[1] - low[1] + 2 * MARGIN) / (high[0] - low[0] + 2 * MARGIN)),
                              dpi=100)
    rod, pins = draw_rod(ax, curves[0], low, high)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for curve in curves:
        rod.set_data(*curve.T)
        pins.set_data(*curve[[0, -1]].T)
        yield capture(figure)
    plt.close(figure)


def draw_helices(centres: core.Point, edges: core.Point) -> plt.Figure:
    """The wires coiled by their couples `[tilts, pieces + 1]`, side by side in one view of space, each with a
    ribbon across its cross-sections `[tilts, pieces + 1, 2]` showing how they twist."""
    curves, ribbons = coordinates(centres), coordinates(edges)               # [tilts, pieces + 1, 3], [..., 2, 3]
    # The wires set side by side across, each a little more than the widest wire apart.
    spacing = SPACING * np.ptp(ribbons[..., :2], axis=(1, 2)).max()
    across = spacing * np.arange(len(curves))[:, None, None] * [1, 0, 0]       # [tilts, 1, 3]
    curves, ribbons = curves + across, ribbons + across[:, :, None]
    view = solids.view([solids.mv.point(ribbons)], 0.0, *VIEW, MARGIN, N_HELIX_PIXELS)
    return image_figure(scene(curves, ribbons, view))


def animate_uncoiling(centres: core.Point, edges: core.Point) -> Iterator[np.ndarray]:
    """The wire through the sweep of its tilt `[frames, pieces + 1]`, with its ribbon `[frames, pieces + 1, 2]`,
    the camera turning slowly around it, all to one scale."""
    curves, ribbons = coordinates(centres), coordinates(edges)               # [frames, pieces + 1, 3], [..., 2, 3]
    points = ribbons.reshape(-1, 3)
    low, high = points.min(axis=0), points.max(axis=0)
    # The upright cylinder about the coil's axis holding every frame, seen alike from every azimuth: its radius
    # across, and up its height foreshortened by the elevation plus its round top and bottom tilted into view.
    middle = (low + high) / 2
    radius = np.linalg.norm(points[:, :2] - middle[:2], axis=-1).max()
    half = np.array([radius, (high[2] - low[2]) / 2 * np.cos(UNCOIL_ELEVATION) + radius * np.sin(UNCOIL_ELEVATION)])
    half = half + MARGIN * half.max()
    azimuths = np.linspace(*UNCOIL_AZIMUTHS, len(curves), endpoint=False)
    for curve, ribbon, azimuth in zip(curves, ribbons, azimuths):
        yield scene(curve[None], ribbon[None], solids.View(solids.aim(solids.mv.point(middle), UNCOIL_ELEVATION, azimuth),
                                                           half, N_UNCOIL_PIXELS))


def animate_cord(centres: core.Point, edges: core.Point) -> Iterator[np.ndarray]:
    """The telephone cord at each frame `[frames, drawn]`, as a ribbon `[frames, drawn, 2]` with its base and its
    handset, all from one fixed view and to one scale."""
    curves, ribbons = coordinates(centres), coordinates(edges)               # [frames, drawn, 3], [..., 2, 3]
    view = solids.view([solids.mv.point(ribbons)], 0.0, *CORD_VIEW, MARGIN, N_CORD_PIXELS)
    # The clamps at the base and the handset: small balls at the cord's two ends.
    clamps = solids.ball(solids.mv.point(curves[:, [0, -1]]), CLAMP_RADIUS * view.pixel)[..., None]   # [frames, 2, 1] Quadric
    for curve, ribbon, ends in zip(curves, ribbons, clamps):
        yield solids.draw([(ends, CLAMP)], view, BACKGROUND, meshes=wires(curve[None], ribbon[None], view))


def wires(curves: np.ndarray, ribbons: np.ndarray, view: solids.View) -> list[tuple[meshes.Samples, float]]:
    """Rods along curves `[rods, n, 3]` as strokes, through their ribbons `[rods, n, 2, 3]`: strips between the
    edges, two triangles to each piece of rod; each opaque, sample by sample."""
    index = np.arange(ribbons.size // 3).reshape(ribbons.shape[:3])         # [rods, n, 2]
    corners = np.stack([index[:, :-1, 0], index[:, 1:, 0], index[:, 1:, 1], index[:, :-1, 1]], axis=-1)
    faces = corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    edges = solids.mv.point(ribbons.reshape(-1, 3))
    strips = meshes.samples(edges, faces, RIBBON, meshes.smooth(edges, faces), view)
    rods = strokes.polylines([(curve, ROD) for curve in curves], ROD_WIDTH, view)
    return [(strips, 1.0), (rods, 1.0)]


def scene(curves: np.ndarray, ribbons: np.ndarray, view: solids.View) -> np.ndarray:
    """Rods and their ribbons, as `wires`, on the background, `[rows, columns, 3]` bytes."""
    return solids.draw([], view, BACKGROUND, meshes=wires(curves, ribbons, view))
