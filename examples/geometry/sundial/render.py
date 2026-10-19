"""The face of a horizontal sundial, and the sun's rays past the nodus onto it."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.geometry.sundial import core

# The part of the dial drawn, east and north of the point below the nodus.
EAST = (-4.5, 4.5)
NORTH = (-2.0, 4.5)
# How far inside the edge of the dial the hour numerals sit.
INSET = 0.25
HOUR_COLOUR = "0.45"
SEASON_MAP = "plasma"
DAY_COLOUR = "#d06c36"
GNOMON_COLOUR = "0.15"
DIAL_COLOUR = "#efe9dc"
RAY_COLOUR = "#e8a33d"
# How far each ray is drawn beyond the nodus towards the sun.
SUNWARD = 2.0
ELEVATION, AZIMUTH = 16.0, -125.0


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """East, north and up of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def bearings(offsets: core.Point) -> np.ndarray:
    """The bearing of each offset across the dial, its angle from north towards east."""
    xyz = coordinates(offsets)
    return np.arctan2(xyz[..., 0], xyz[..., 1])


def shadows(nodus: core.Point, suns: core.Point) -> np.ndarray:
    """East and north of the nodus's shadow for each sun, and nothing where the sun is below the horizon
    or the shadow falls off the dial."""
    tips = coordinates(core.shadow(nodus, suns))[..., :2]
    inside = (scalars(core.dial & suns) > 0) & (tips[..., 0] > EAST[0]) & (tips[..., 0] < EAST[1]) \
        & (tips[..., 1] > NORTH[0]) & (tips[..., 1] < NORTH[1])
    return np.where(inside[..., None], tips, np.nan)


def season_colours(seasons: np.ndarray) -> np.ndarray:
    """A colour per season by how far it lies from the equinox: a summer and a winter season as far from it
    alike."""
    return plt.get_cmap(SEASON_MAP)(0.15 + 0.7 * np.abs(seasons) / np.abs(seasons).max())


def draw_face(ax: plt.Axes, nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point,
              suns: core.Point, hours: np.ndarray, faint: bool) -> None:
    """The dial's face: each season's shadow path, each hour line through the shadows at that hour from
    solstice to solstice, numbered where it leaves the dial, and the gnomon's edge along the noon line."""
    paths = shadows(nodus, days)                                               # [seasons, samples, 2]
    ax.add_collection(LineCollection(paths, colors=season_colours(seasons), linewidths=1.6, alpha=0.35 if faint else 1.0))
    lines = shadows(nodus, suns)                                               # [hours, samples, 2]
    ax.add_collection(LineCollection(lines, colors=HOUR_COLOUR, linewidths=0.8))
    for hour, line in zip(hours, lines):
        drawn = line[~np.isnan(line[:, 0])]
        outward = drawn[0] - drawn[-1]
        place = drawn[0] - INSET * outward / np.linalg.norm(outward)
        ax.text(*place, f"{hour}", fontsize=8, color=HOUR_COLOUR, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    # The gnomon's edge seen from above: from the foot of the style to the point below the nodus.
    base = np.stack([coordinates(foot), coordinates(nodus) * [1, 1, 0]])[:, :2]
    ax.plot(*base.T, color=GNOMON_COLOUR, linewidth=3, solid_capstyle="butt")
    ax.set(xlim=EAST, ylim=NORTH, aspect="equal")
    ax.set_axis_off()


def draw_dial(nodus: core.Point, foot: core.Point, seasons: np.ndarray, days: core.Point, suns: core.Point,
              hours: np.ndarray) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(6.0, 4.6), layout="constrained")
    draw_face(ax, nodus, foot, seasons, days, suns, hours, faint=False)
    return figure


def draw_sky(ax: plt.Axes, nodus: core.Point, foot: core.Point, rays: core.Point, day: core.Point) -> None:
    """The dial in space with its gnomon, and the day's rays of sunlight past the nodus onto the dial: the
    cone they sweep about the pole, on both sides of the nodus, and its cut with the dial, the shadow path."""
    corners = np.array([[EAST[0], NORTH[0], 0], [EAST[1], NORTH[0], 0], [EAST[1], NORTH[1], 0], [EAST[0], NORTH[1], 0]])
    ax.add_collection3d(Poly3DCollection([corners], facecolor=DIAL_COLOUR, edgecolor="0.7", linewidth=0.6, zorder=1))
    tip, base = coordinates(nodus), coordinates(foot)
    ax.add_collection3d(Poly3DCollection([np.array([base, tip, tip * [1, 1, 0]])], facecolor=GNOMON_COLOUR, alpha=0.85, zorder=2))
    # The rays whose shadow falls on the dial, from beyond the nodus towards the sun down to the shadow.
    landed = ~np.isnan(shadows(nodus, rays)[:, 0])                             # [hours]
    ends = coordinates(core.shadow(nodus, rays))[landed]                       # [landed, 3]
    sunward = tip + SUNWARD * coordinates(rays)[landed]                        # [landed, 3]
    ax.add_collection3d(Line3DCollection(np.stack([sunward, ends], axis=1), colors=RAY_COLOUR, linewidths=0.6, alpha=0.7, zorder=3))
    path = shadows(nodus, day)                                                 # [samples, 2]
    ax.plot(path[:, 0], path[:, 1], np.zeros(len(path)), color=DAY_COLOUR, linewidth=2, zorder=4)
    ax.set(xlim=EAST, ylim=NORTH, zlim=(0, 1 + SUNWARD))
    ax.set_box_aspect((EAST[1] - EAST[0], NORTH[1] - NORTH[0], 1 + SUNWARD), zoom=1.3)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def animate_year(declinations: np.ndarray, rays: core.Point, days: core.Point, nodus: core.Point, foot: core.Point,
                 seasons: np.ndarray, season_days: core.Point, suns: core.Point, hours: np.ndarray) -> list[np.ndarray]:
    """Per frame of the year: the day's rays past the nodus in space, and the day's shadow path over the
    dial's face."""

    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(10.0, 4.6))
        # Drawn in order, dial, gnomon, rays, rather than sorted by depth.
        sky = figure.add_axes((0.0, 0.0, 0.5, 1.0), projection="3d", computed_zorder=False)
        draw_sky(sky, nodus, foot, rays[index], days[index])
        face = figure.add_axes((0.52, 0.04, 0.46, 0.92))
        draw_face(face, nodus, foot, seasons, season_days, suns, hours, faint=True)
        face.add_collection(LineCollection(shadows(nodus, days[index])[None], colors=DAY_COLOUR, linewidths=2.2))
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(len(declinations))]
