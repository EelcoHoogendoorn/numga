"""Bob tracks on the floor, and the earth turning a pendulum's floor underneath it."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import numpy as np

from examples.animation import capture
from examples.mechanics.foucault import core

TRACK_MAP = "viridis"
SWING_COLOUR = "#d06c36"
GLOBE_COLOUR = "0.85"
CAP_COLOUR = "#287fa0"
FLOOR_COLOUR = "#efe9dc"
COMPASS_COLOUR = "0.35"
# The half width of the floor drawn on the globe, and the samples of the globe and its circles.
FLOOR = 0.3
GLOBE_MESH = 40
CIRCLE = 200
ELEVATION, AZIMUTH = 22.0, -35.0


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def on_floor(points: core.Point, north: core.Point, east: core.Point) -> np.ndarray:
    """East and north of points on a floor, on the last axis."""
    return np.stack([scalars(~points | east), scalars(~points | north)], axis=-1)


def turned(directions: core.Point, north: core.Point, east: core.Point) -> np.ndarray:
    """How far each direction on its floor has turned from north, clockwise seen from above."""
    floor = on_floor(directions, north, east)
    return np.arctan2(floor[..., 0], floor[..., 1])


def draw_track(ax: plt.Axes, track: np.ndarray) -> None:
    """A bob's track on its floor, coloured from the start of the day to its end, and a tick to the north."""
    segments = np.stack([track[:-1], track[1:]], axis=1)                     # [samples - 1, ends, 2]
    ax.add_collection(LineCollection(segments, cmap=TRACK_MAP, array=np.linspace(0, 1, len(segments)), linewidths=0.7))
    ax.annotate("N", (0, 1.22), ha="center", va="center", fontsize=8, color=COMPASS_COLOUR)
    ax.plot([0, 0], [1.05, 1.13], color=COMPASS_COLOUR, linewidth=1)
    ax.set(xlim=(-1.15, 1.15), ylim=(-1.15, 1.3), aspect="equal")
    ax.set_axis_off()


def draw_rosettes(tracks: core.Point, north: core.Point, east: core.Point) -> plt.Figure:
    """For each latitude, the bob's track over a day on its floor."""
    floors = on_floor(tracks, north[:, None], east[:, None])                   # [latitudes, samples, 2]
    figure, axes = plt.subplots(1, len(floors), figsize=(3.0 * len(floors), 3.2), layout="constrained")
    for ax, track in zip(axes, floors):
        draw_track(ax, track)
    return figure


def draw_globe(ax: plt.Axes, earth: core.Rotor, vertical: core.Point, swing: core.Point, north: core.Point,
               east: core.Point) -> None:
    """The earth turned by a rotor: its equator and the site's latitude circle, the polar cap that circle
    bounds, and the site's floor with the line the pendulum swings along."""
    longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, GLOBE_MESH), np.linspace(-np.pi / 2, np.pi / 2, GLOBE_MESH))
    ax.plot_surface(np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                    color=GLOBE_COLOUR, alpha=0.25, linewidth=0, shade=False, zorder=1)
    site = coordinates(vertical)
    angles = np.linspace(0, 2 * np.pi, CIRCLE)
    ring = np.stack([np.hypot(*site[:2]) * np.cos(angles), np.hypot(*site[:2]) * np.sin(angles), np.full(CIRCLE, site[2])], axis=-1)
    cap_longitude, cap_latitude = np.meshgrid(np.linspace(0, 2 * np.pi, GLOBE_MESH), np.linspace(np.arcsin(site[2]), np.pi / 2, 12))
    ax.plot_surface(np.cos(cap_latitude) * np.cos(cap_longitude), np.cos(cap_latitude) * np.sin(cap_longitude),
                    np.sin(cap_latitude), color=CAP_COLOUR, alpha=0.3, linewidth=0, shade=False, zorder=2)
    ax.plot(np.cos(angles), np.sin(angles), 0 * angles, color="0.6", linewidth=0.8, zorder=2)
    ax.plot(*ring.T, color=CAP_COLOUR, linewidth=1.2, zorder=3)
    centre = coordinates(earth >> vertical)
    across, along = coordinates(earth >> east) * FLOOR, coordinates(earth >> north) * FLOOR
    corners = centre + np.array([-1, 1, 1, -1])[:, None] * across + np.array([-1, -1, 1, 1])[:, None] * along
    ax.add_collection3d(Poly3DCollection([corners], facecolor=FLOOR_COLOUR, edgecolor="0.5", linewidth=0.6, zorder=4))
    line = coordinates(earth >> swing) * FLOOR
    ax.add_collection3d(Line3DCollection([np.stack([centre - line, centre + line])], colors=SWING_COLOUR, linewidths=2.5, zorder=5))
    ax.add_collection3d(Line3DCollection([np.stack([centre + along, centre + 1.4 * along])], colors=COMPASS_COLOUR, linewidths=1, zorder=5))
    ax.set(xlim=(-1.1, 1.1), ylim=(-1.1, 1.1), zlim=(-1.1, 1.1))
    ax.set_box_aspect((1, 1, 1), zoom=1.3)
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_axis_off()


def animate_day(turns: np.ndarray, earth: core.Rotor, swings: core.Point, vertical: core.Point, north: core.Point,
                east: core.Point, track: core.Point) -> list[np.ndarray]:
    """Per frame of the day: the earth turned so far with the pendulum's floor and swing line, and the bob's
    track on the floor up to then."""
    floor = on_floor(track, north, east)                                       # [samples, 2]
    reached = np.round(turns / turns[-1] * (len(floor) - 1)).astype(int)        # [frames + 1]

    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(9.0, 4.4))
        # Drawn in order, globe, cap, circles, floor, swing, rather than sorted by depth.
        globe = figure.add_axes((0.0, 0.0, 0.5, 1.0), projection="3d", computed_zorder=False)
        draw_globe(globe, earth[index], vertical, swings[index], north, east)
        face = figure.add_axes((0.52, 0.05, 0.46, 0.9))
        draw_track(face, floor[:reached[index] + 1])
        line = on_floor(swings[index], north, east)
        face.plot(*np.stack([-line, line]).T, color=SWING_COLOUR, linewidth=2.5)
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(len(turns) - 1)]
