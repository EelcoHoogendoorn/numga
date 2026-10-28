"""Pair coherence across the energy band and the gap it produces. The arrows of the quench are drawn on the GPU,
the rest by matplotlib."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure

from examples.util.animation import capture
from examples.quantum.superconductivity import core
from examples.util.wgsl import fitted, labels, meshes, show, solids, strokes
from examples.util.wgsl.scene import Scene

sm = solids.mv
# The patches a field's planes xy and yz are read across: the joins of the two directions in each plane.
xy_patch = core.mv.xy.dual()
yz_patch = core.mv.yz.dual()
INK = "#263b51"
PAIR = "#198b87"
FIXED = "#d17c24"
DISPLAY_LEVELS = 33
DISPLAY_CUTOFF = 3.0
# The chart's axes and the circles of the largest pair coherence, a half, at every few levels, faint; the opacity of
# the path through the arrows' tips.
AXIS_COLOUR, CIRCLE_COLOUR, CIRCLE_OPACITY, PATH_OPACITY = np.full(3, 0.55), np.full(3, 0.75), 0.4, 0.3
CIRCLE_RADIUS, CIRCLE_EVERY = 0.5, 8
# The chart's scene: the length of one unit of energy along x, where the energy axis starts and ends, in units of
# energy, and how far the axes of pair coherence reach from the start of the energy axis.
ENERGY_SCALE, ENERGY_START, ENERGY_END, COHERENCE_REACH = 0.4, -3.4, 3.7, 0.68
# The view's elevation and azimuth in radians, azimuth zero looking along +y; the margin around the chart as a share
# of the larger side, and the pixels across it, until the view is fitted to its axes.
ELEVATION, AZIMUTH = np.radians(24.0), np.radians(-58.0 + 90.0)
MARGIN, N_PIXELS = 0.08, 470
# The points around a circle; the widths of the axes' shafts, the circles, the path and the arrows' shafts, the
# arrowheads' lengths and widths, and the radius of the tips, in pixels.
N_CIRCLE = 100
AXIS_WIDTH, CIRCLE_WIDTH, PATH_WIDTH, STEM_WIDTH = 0.8, 0.64, 0.8, 1.43
AXIS_HEAD_LENGTH, AXIS_HEAD_WIDTH, STEM_HEAD_LENGTH, STEM_HEAD_WIDTH = 5.6, 3.2, 5.2, 4.0
TIP_RADIUS = 1.6
# The resolution of the quantity plots and of the animation, in pixels per inch.
DPI, ANIMATION_DPI = 75, 60


# --- plumbing -------------------------------------------------------------------------
def plain_axes(axis: plt.Axes, across: str, up: str) -> None:
    """Light axes named by what runs along them, the quantity's zero as the horizontal one, without numbers: every
    value is in units of the energy scale, and the shapes are the point."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("0.8")
    axis.spines["bottom"].set_position(("data", 0))
    axis.set(xticks=[], yticks=[], xlabel=across, ylabel=up)
    # The horizontal axis's name below the plot, wherever its zero lies.
    axis.xaxis.set_label_coords(0.5, -0.03)
    axis.xaxis.label.set_va("top")


def coherences(spins: core.Spin, planes: tuple[core.Field, ...]) -> np.ndarray:
    """The pseudospins' parts, read by pairing with field planes, `[..., levels, planes]`."""
    return np.stack([(plane & spins).to_array() for plane in planes], axis=-1)


def readings(fields: core.Field, patch: core.Spin) -> np.ndarray:
    """A field's part in one plane, read across the patch square to it."""
    return (fields & patch).to_array()


def draw_equilibrium(dispersion: core.Field, spins: core.Spin) -> Figure:
    energies = readings(dispersion, xy_patch)
    coherence = coherences(spins, (core.mv.yz, core.mv.zx, core.mv.xy))
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 2.7), dpi=DPI, layout="constrained")
    axes[0].plot(energies, coherence[:, 2] + 0.5, color=INK, linewidth=2)
    axes[0].set(ylim=(0, 1.04))
    axes[1].plot(energies, coherence[:, 0], color=PAIR, linewidth=2)
    axes[1].set(ylim=(0, 0.54))
    for axis, up in zip(axes, ("occupation per electron state", r"pair coherence in yz, $s_{yz}$")):
        axis.set(xlim=(energies[0], energies[-1]))
        axis.axvline(0, color="0.85", linewidth=0.8, zorder=0)
        plain_axes(axis, r"energy from chemical potential, $\xi$", up)
    return figure


def chart(energies: np.ndarray, axis: plt.Axes
          ) -> tuple[solids.View, list[tuple[solids.Quadric, np.ndarray]], solids.Point]:
    """The view of the chart, fitted to the matplotlib axes it is shown in, its axes as arrows from the start of the
    energy axis, the energy along +x and the pair coherence in yz and zx along +y and +z, and the circles of the
    largest coherence about the energy axis at every few levels `[circles, circle + 1]`."""
    start, end = ENERGY_SCALE * ENERGY_START, ENERGY_SCALE * ENERGY_END
    starts = np.full((3, 3), [start, 0, 0])
    ends = np.array([[end, 0, 0], [start, COHERENCE_REACH, 0], [start, 0, COHERENCE_REACH]])
    # The circles about the energy axis, x, at every few levels, each through the point the largest coherence above it.
    levels = solids.origin + sm.x.dual() * (ENERGY_SCALE * energies[::CIRCLE_EVERY])        # [circles] Point-
    circles = meshes.circles(levels + sm.z.dual() * CIRCLE_RADIUS, sm.yz, N_CIRCLE)        # [circles, circle + 1] Point-
    view = fitted(solids.view([circles, sm.point(starts), sm.point(ends)], 0.0, ELEVATION, AZIMUTH, MARGIN, N_PIXELS),
                  axis)
    shafts, heads = solids.arrows(sm.point(starts), sm.point(ends), AXIS_WIDTH / 2 * view.pixel,
                                  AXIS_HEAD_LENGTH * view.pixel, AXIS_HEAD_WIDTH / 2 * view.pixel)
    return view, [(shafts, AXIS_COLOUR), (heads, AXIS_COLOUR)], circles


def label_chart(axis: plt.Axes, view: solids.View) -> None:
    """The chart's axes named past their tips."""
    start, end = ENERGY_SCALE * ENERGY_START, ENERGY_SCALE * ENERGY_END
    names = np.array([[end + 0.12, 0, 0], [start, COHERENCE_REACH + 0.1, 0], [start, 0, COHERENCE_REACH + 0.1]])
    labels(axis, view, sm.point(names), [r"$\xi$", r"$s_{yz}$", r"$s_{zx}$"], ha="center", va="center", fontsize=11,
           color=AXIS_COLOUR)


def animate_quench(
    dispersion: core.Field, spins: core.Spin, gaps: core.Field, times: np.ndarray,
) -> Iterator[np.ndarray]:
    """Arrows in each energy level's pairing plane, beside their collective gap; the chart's axes and circles set
    up once."""
    energies = readings(dispersion, xy_patch)
    transverse = coherences(spins, (core.mv.yz, core.mv.zx))
    amplitudes = gaps.norm().to_array()
    visible = np.flatnonzero(np.abs(energies) < DISPLAY_CUTOFF)
    indices = visible[np.linspace(0, len(visible) - 1, DISPLAY_LEVELS, dtype=int)]
    energies, transverse = energies[indices], transverse[:, indices]
    origins = np.stack([ENERGY_SCALE * energies, np.zeros_like(energies), np.zeros_like(energies)], axis=-1)
    colours = plt.colormaps["coolwarm"](np.linspace(0.08, 0.92, len(energies)))[:, :3]

    figure = plt.figure(figsize=(9.0, 3.7), dpi=ANIMATION_DPI)
    axis = figure.add_axes((0.01, 0.08, 0.57, 0.85))
    curve = figure.add_axes((0.68, 0.22, 0.29, 0.64))
    view, axes, circles = chart(energies, axis)
    rings = (strokes.polylines([(circle, CIRCLE_COLOUR) for circle in circles], CIRCLE_WIDTH, view), CIRCLE_OPACITY)
    scene = Scene(view, np.ones(3), axes, [rings], inks=[], translucent=[])

    def arrows(coherence: np.ndarray) -> np.ndarray:
        """Each level's arrow from the energy axis to its pair coherence, and its tip, over the path through the
        tips."""
        tips = origins + np.concatenate([np.zeros((len(coherence), 1)), coherence], axis=-1)
        shafts, heads = solids.arrows(sm.point(origins), sm.point(tips), STEM_WIDTH / 2 * view.pixel,
                                      STEM_HEAD_LENGTH * view.pixel, STEM_HEAD_WIDTH / 2 * view.pixel)
        stems = [(shafts, colours), (heads, colours),
                 (solids.ball(sm.point(tips), TIP_RADIUS * view.pixel)[:, None], np.array(to_rgb(PAIR)))]
        path = strokes.polylines([(sm.point(tips), np.array(to_rgb(INK)))], PATH_WIDTH, view)
        return scene.draw(stems, inks=[], translucent=[], meshes=[(path, PATH_OPACITY)])

    image = show(axis, arrows(transverse[0]), view)
    label_chart(axis, view)
    gap_line, = curve.plot([], [], color=PAIR, linewidth=2)
    gap_tip, = curve.plot([], [], "o", color=PAIR, markersize=6)
    # The gap's axis reaches past its value right after the quench, which it falls from.
    curve.set(xlim=(times[0], times[-1]), ylim=(0, amplitudes[0] * 1.12))
    plain_axes(curve, "time", r"$|\Delta|$")

    for index, coherence in enumerate(transverse):
        image.set_data(arrows(coherence))
        gap_line.set_data(times[:index + 1], amplitudes[:index + 1])
        gap_tip.set_data([times[index]], [amplitudes[index]])
        yield capture(figure)
    plt.close(figure)


def draw_gap(times: np.ndarray, gaps: core.Field, equilibrium_gap: core.Field) -> Figure:
    amplitudes = gaps.norm().to_array()
    equilibrium = equilibrium_gap.norm().to_array()
    figure, axis = plt.subplots(figsize=(7.5, 2.8), dpi=DPI, layout="constrained")
    axis.plot(times, amplitudes, color=PAIR, linewidth=2, label="isolated condensate")
    axis.axhline(equilibrium, color=INK, linestyle="--", linewidth=1, label="final ground-state gap")
    axis.set(xlim=(times[0], times[-1]), ylim=(0, amplitudes.max() * 1.05))
    axis.legend(frameon=False, ncol=2, loc="lower center")
    plain_axes(axis, "time", r"$|\Delta|$")
    return figure


def draw_response(
    times: np.ndarray, exact: core.Field, predicted: core.Field, names: tuple[str, str],
) -> Figure:
    exact_amplitude = readings(exact, yz_patch)
    predicted_amplitude = readings(predicted, yz_patch)
    figure, axis = plt.subplots(figsize=(7.5, 3.0), dpi=DPI, layout="constrained")
    axis.plot(times, exact_amplitude, color=INK, linewidth=3, alpha=0.5, label="nonlinear evolution")
    for values, name, colour in zip(predicted_amplitude.T, names, (PAIR, FIXED)):
        axis.plot(times, values, color=colour, linewidth=1.5, linestyle="--", label=name)
    axis.set(xlim=(times[0], times[-1]))
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1), fontsize=9)
    plain_axes(axis, "time", r"$\delta\Delta_{yz}$")
    return figure
