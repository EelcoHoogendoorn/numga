"""Drawing for hydrostatic stability: the heeled hull in space with its wet surface by depth and its two
forces; the righting arm curves; the floating cube's map over density and heel, with cubes at rest; and the
released hull rolling back upright or capsizing, beside its righting arm curve."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from examples.animation import capture
from examples.fluids.floating import core

ga, mv = core.ga, core.mv
# The colours of the hull, of its wet surface by depth, of the water, of the weight and of the buoyancy;
# the colour maps of the loads and of the cube's map.
HULL, DECK, WATER, WEIGHT, BUOYANCY = "#e7e2d6", "#c9bfa8", "#5aa9c9", "#d06c36", "#1f6f99"
WET_MAP, LOAD_MAP, ARM_MAP = "Blues", "viridis", "RdBu"
# The light falling on the hull, the depth at which its wet surface is darkest, the reach of the water
# drawn around it, the drawn length of the forces, the view in space, and the heels over which each curve's
# tangent is drawn.
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])
DEEPEST = 0.6
SEA = (3.0, 1.8)
FORCE = 0.9
VIEW = (18, -55)
TANGENT = 25.0
# The windows of the end views of the rolling hull and of the hull drawn whole, across and up, and the size
# of the cube drawn above the map.
END = ((-1.6, 1.6), (-1.0, 1.6))
SHIP_END = ((-1.1, 1.1), (-0.8, 1.0))
CUBE_VIEW = 0.95


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of points, on the last axis."""
    kernel = points.cast(ga.subspace("ywz wxz xwy xyz")).kernel
    return kernel[..., :3] / kernel[..., 3:]


def degrees(rolls: core.Twist) -> np.ndarray:
    """The angle of turns about the axis of heel, in degrees."""
    return np.degrees(rolls.cast(ga.subspace("yz")).kernel[..., 0])


def arms(couples: core.Forque, weight: core.Scalar) -> np.ndarray:
    """The righting arm of each couple: its turn back about the axis of heel, over the weight."""
    return scalars(-(couples & core.heel_axis) / weight)


def shade(triangles: np.ndarray) -> np.ndarray:
    """How much light each triangle `[..., 3, 3]` faces, between a third and one."""
    normals = np.cross(triangles[..., 1, :] - triangles[..., 0, :], triangles[..., 2, :] - triangles[..., 0, :])
    lit = np.abs(normals @ LIGHT) / np.maximum(np.linalg.norm(normals, axis=-1), 1e-12)
    return 1 / 3 + 2 / 3 * lit


def forces_end_on(ax: plt.Axes, placed: core.Motor, push: core.Forque, centre: core.Point) -> None:
    """The weight from the centre of gravity and the buoyancy from the water surface, as arrows on their
    vertical lines across an end view, and the righting arm between them at the centre's height."""
    gravity = coordinates(placed >> centre)[1:]
    buoyant = coordinates((placed >> push) ^ core.water)[1:]
    for start, colour, sense in ((buoyant, BUOYANCY, 1), (gravity, WEIGHT, -1)):
        ax.plot([start[0], start[0]], END[1], color=colour, lw=0.8, linestyle=(0, (4, 3)), zorder=5)
        ax.annotate("", xy=start + [0, sense * FORCE * 0.6], xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.2), zorder=6)
    ax.plot([gravity[0], buoyant[0]], [gravity[1], gravity[1]], color="0.1", lw=2.5, solid_capstyle="butt", zorder=7)
    ax.plot(*gravity, "o", color=WEIGHT, markeredgecolor="white", markersize=7, zorder=8)


def draw_ship(hull: core.Point, placed: core.Motor, push: core.Forque, centre: core.Point) -> plt.Figure:
    """The heeled hull: in space, its dry surface pale and its wet surface darker with depth, in the water;
    and seen along its axis of heel, with its weight, its buoyancy and the righting arm between them."""
    triangles = coordinates(placed >> hull)                                  # [triangles, 3, 3]
    depth = np.clip(-triangles[..., 2].mean(axis=-1) / DEEPEST, 0, 1)        # [triangles]
    wet = plt.get_cmap(WET_MAP)(0.35 + 0.6 * depth)[:, :3]
    colours = np.where((depth > 0)[:, None], wet, plt.matplotlib.colors.to_rgb(HULL)) * shade(triangles)[:, None]
    figure = plt.figure(figsize=(11, 4.4), dpi=110)
    grid = figure.add_gridspec(1, 2, width_ratios=(1.6, 1), wspace=0.02)
    ax = figure.add_subplot(grid[0, 0], projection="3d")
    ax.add_collection3d(Poly3DCollection(triangles, facecolors=colours, edgecolors="none"))
    reach, breadth = SEA
    ax.add_collection3d(Poly3DCollection([[[-reach, -breadth, 0], [reach, -breadth, 0], [reach, breadth, 0],
                                           [-reach, breadth, 0]]], facecolors=WATER, alpha=0.18, edgecolors="none"))
    ax.set(xlim=(-reach, reach), ylim=(-breadth, breadth), zlim=(-1.2, 1.2))
    ax.set_box_aspect((2 * reach, 2 * breadth, 2.4), zoom=1.6)
    ax.view_init(*VIEW)
    ax.set_axis_off()
    ax = figure.add_subplot(grid[0, 1])
    end_view(ax, triangles)
    forces_end_on(ax, placed, push, centre)
    ax.set(xlim=SHIP_END[0], ylim=SHIP_END[1])
    figure.subplots_adjust(left=0, right=0.99, bottom=0.02, top=0.98)
    return figure


def draw_curves(heels: np.ndarray, couples: core.Forque, weight: core.Scalar, stiffness: core.Stiffness) -> plt.Figure:
    """The righting arm of each load over the heels `[heels, loads]`, with the tangent at upright that the
    stiffness gives, and the line of no righting."""
    righting = arms(couples, weight)                                         # [heels, loads]
    slopes = scalars(stiffness(mv.yz, mv.yz) / weight)                       # [loads]
    colours = plt.get_cmap(LOAD_MAP)(np.linspace(0.05, 0.85, righting.shape[1]))
    figure, ax = plt.subplots(figsize=(8, 4.2), dpi=110)
    ax.axhline(0, color="0.6", lw=0.8)
    near = heels[heels <= TANGENT]
    for curve, slope, colour in zip(righting.T, slopes, colours):
        ax.plot(heels, curve, color=colour, lw=2)
        ax.plot(near, slope * np.radians(near), color=colour, lw=1, linestyle=(0, (4, 3)))
    ax.set_xticks([0, 45, 90, 135, 180])
    ax.set_xlim(heels[0], heels[-1])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    figure.tight_layout()
    return figure


def end_view(ax: plt.Axes, triangles: np.ndarray) -> None:
    """A body seen along its axis of heel: its triangles `[triangles, 3, 3]` across and up, the water over its
    lower part."""
    ax.add_collection(PolyCollection(triangles[..., 1:], facecolors=HULL, edgecolors=HULL, linewidths=0.3))
    ax.axhspan(-10, 0, color=WATER, alpha=0.45, lw=0, zorder=3)
    ax.axhline(0, color=WATER, lw=1.2, zorder=4)
    ax.set_aspect("equal")
    ax.set_axis_off()


def draw_cubes(densities: np.ndarray, heels: np.ndarray, couples: core.Forque, fluid_weight: float, cube: core.Point,
               shown: core.Motor, shown_densities: list[float], rests: np.ndarray) -> plt.Figure:
    """Above, cubes of a few densities at rest; below, the cube's righting couple over density and heel, blue
    turning it back upright and red over, its rests where it turns from over to back drawn thick, its
    balances the other way thin, and the cubes above marked on them."""
    moments = scalars(-(couples & core.heel_axis) / fluid_weight)            # [densities, heels]
    # A rest is stable where the couple turns from pushing over to pushing back as the heel grows.
    rising = np.gradient(moments, heels, axis=1) > 0
    figure = plt.figure(figsize=(8, 6.6), dpi=110)
    grid = figure.add_gridspec(2, shown.shape[0], height_ratios=(1, 2.6), hspace=0.05, wspace=0.05)
    for column, placed in enumerate(shown):
        ax = figure.add_subplot(grid[0, column])
        end_view(ax, coordinates(placed >> cube))
        ax.set(xlim=(-CUBE_VIEW, CUBE_VIEW), ylim=(-CUBE_VIEW, CUBE_VIEW))
    ax = figure.add_subplot(grid[1, :])
    largest = np.abs(moments).max()
    ax.imshow(moments.T, origin="lower", aspect="auto", cmap=ARM_MAP, vmin=-largest, vmax=largest, interpolation="bilinear",
              extent=(densities[0], densities[-1], heels[0], heels[-1]))
    for stable, width in ((True, 3), (False, 0.8)):
        ax.contour(densities, heels, np.where(rising == stable, moments, np.nan).T, levels=[0], colors="0.1",
                   linewidths=width)
    ax.plot(shown_densities, rests, "o", color="white", markeredgecolor="0.1", markersize=7)
    ax.set(xlim=(densities[0], densities[-1]), ylim=(-3, 93), yticks=[0, 45, 90])
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.98)
    return figure


def animate_release(hull: core.Point, centres: core.Point, frames: list[tuple[core.Twist, core.Motor, core.Forque, core.Forque]],
                    heels: np.ndarray, couples: core.Forque, weight: core.Scalar) -> list[np.ndarray]:
    """Each frame: the released hull with each load seen along its axis of heel, its weight from its centre of
    gravity and its buoyancy from the water surface, as arrows on their lines, beside the righting arm curves
    of the loads, each with the hull's heel so far."""
    curves = arms(couples, weight)                                           # [heels, loads]
    # Past half a turn the righting arm repeats reversed: heeled the other way round, it pushes the other way.
    beyond = 360 - heels[::-1][1:]
    extended, extended_curves = np.concatenate([heels, beyond]), np.concatenate([curves, -curves[::-1][1:]])
    states = np.array([[degrees(roll), arms(push + weighing, weight)] for roll, _, push, weighing in frames])   # [frames, 2, loads]
    colours = plt.get_cmap(LOAD_MAP)(np.array([0.3, 0.85]))
    images = []
    for index, (_, placed, push, weighing) in enumerate(frames):
        figure = plt.figure(figsize=(11, 3.8), dpi=90)
        grid = figure.add_gridspec(1, 3, width_ratios=(1, 1, 1.4), wspace=0.12)
        for case, colour in enumerate(colours):
            ax = figure.add_subplot(grid[0, case])
            end_view(ax, coordinates(placed[case] >> hull))
            forces_end_on(ax, placed[case], push[case], centres[case])
            ax.set(xlim=END[0], ylim=END[1])
            ax.plot(0.9 * END[0][0], 0.85 * END[1][1], "o", color=colour, markersize=9)
        ax = figure.add_subplot(grid[0, 2])
        ax.axhline(0, color="0.6", lw=0.8)
        for case, colour in enumerate(colours):
            ax.plot(extended, extended_curves[:, case], color=colour, lw=1.6, alpha=0.5)
            ax.plot(*states[:index + 1, :, case].T, color=colour, lw=2)
            ax.plot(*states[index, :, case], "o", color=colour, markersize=7)
        ax.set_xticks([0, 90, 180])
        ax.set_xlim(-30, 215)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        figure.subplots_adjust(left=0.01, right=0.98, bottom=0.1, top=0.97)
        images.append(capture(figure))
        plt.close(figure)
    return images
