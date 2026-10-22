"""Bob tracks on the floor, and the earth turning a pendulum's floor underneath it, on the GPU: the globe and its
polar cap translucent, the equator and the latitude circle tubes, the floor a tile on the globe with the swing and
north on it as rods."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
import numpy as np

from numga import stack
from examples.animation import capture
from examples.mechanics.foucault import core
from examples.wgsl import compose, meshes, show, solids

TRACK_MAP = "viridis"
SWING_COLOUR = "#d06c36"
GLOBE_COLOUR = "0.85"
CAP_COLOUR = "#287fa0"
FLOOR_COLOUR = "#efe9dc"
COMPASS_COLOUR = "0.35"
FLOOR_EDGE_COLOUR = "0.5"
EQUATOR_COLOUR = "0.6"
# The half width of the floor drawn on the globe, how far north its compass reaches from the middle, and the
# samples of the globe's circles.
FLOOR = 0.3
COMPASS = 1.4
CIRCLE = 200
# The view of the globe, elevation and azimuth in radians, azimuth zero looking along +y; the margin around the
# globe as a share of its radius, and the pixels across a picture.
ELEVATION, AZIMUTH = np.radians([22.0, -35.0 + 90.0])
MARGIN = 0.15
PIXELS = 440
# The opacities of the globe and of the cap; the width of the circles, the radii of the floor's edges, of the swing
# and of the compass, in pixels, and the points around a circle's tube.
GLOBE_OPACITY, CAP_OPACITY = 0.25, 0.3
CIRCLE_WIDTH, EDGE_RADIUS, SWING_RADIUS, COMPASS_RADIUS = 2.0, 0.8, 2.0, 1.0
N_AROUND = 8


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(core.ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def on_floor(points: core.Point, north: core.Point, east: core.Point) -> np.ndarray:
    """East and north of points on a floor, on the last axis."""
    return np.stack([scalars(points.dot(east)), scalars(points.dot(north))], axis=-1)


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
    view = solids.view([solids.origin], 1.0, ELEVATION, AZIMUTH, MARGIN, PIXELS)
    # The equator and the site's latitude circle: a direction on each turned once around the axis.
    around = core.turn(core.equator, np.linspace(0, 2 * np.pi, CIRCLE))[:, None]   # [circle, 1] Rotor
    equator, ring = np.moveaxis(coordinates(around >> stack([core.x.dual(), vertical])), 1, 0)   # [circle, 3] each
    circles = meshes.tubes([(equator, np.array(to_rgb(EQUATOR_COLOUR))), (ring, np.array(to_rgb(CAP_COLOUR)))],
                           CIRCLE_WIDTH, view, N_AROUND)
    # The cap: the globe on the far side of the plane of the latitude circle from the equator.
    height = coordinates(vertical)[2]
    globe = solids.ball(solids.origin, 1.0)                                   # [] Quadric
    cap = stack([globe, solids.half_space(solids.mv.w * height - solids.mv.z)])   # [2] Quadric
    shells = [solids.layer([(solid, np.array(to_rgb(colour)))], view)
              for solid, colour in ((globe[None, None], GLOBE_COLOUR), (cap[None], CAP_COLOUR))]
    # The floor at the site, turned with the earth: a tile across the vertical, north and east along its sides,
    # lifted off the globe by the latitude circle's thickness, so that the circle through the site passes under it.
    site = vertical * (1 + CIRCLE_WIDTH * view.pixel)                          # [] Point
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * FLOOR             # [corners, 2]
    swing_start, swing_end, compass_start, compass_end, *corners = coordinates(earth >> stack(
        [site - swing * FLOOR, site + swing * FLOOR, site + north * FLOOR, site + north * (COMPASS * FLOOR)]
        + [site + east * across + north * along for across, along in signs]))   # [3] each
    tile = solids.mv.point(np.array(corners))                                  # [corners] Point
    square = np.array([[0, 1, 2], [0, 2, 3]])
    floor = meshes.layer(tile, square, np.array(to_rgb(FLOOR_COLOUR)), meshes.flat(tile, square), view)
    # The swing through the middle of the floor, both ways, and north from its edge outwards.
    swing_rod = solids.rods(solids.mv.point(swing_start), solids.mv.point(swing_end), SWING_RADIUS * view.pixel)
    compass = solids.rods(solids.mv.point(compass_start), solids.mv.point(compass_end), COMPASS_RADIUS * view.pixel)
    edges, joints = solids.edges(tile, tile[[1, 2, 3, 0]], EDGE_RADIUS * view.pixel)
    swing_colour, compass_colour, edge_colour = (np.array(to_rgb(colour))
                                                 for colour in (SWING_COLOUR, COMPASS_COLOUR, FLOOR_EDGE_COLOUR))
    lines = solids.layer([(swing_rod[None], swing_colour), (compass[None], compass_colour), (edges, edge_colour),
                          (joints, edge_colour)], view)
    image = compose([(*circles, 1.0), (*floor, 1.0), (*lines, 1.0), (*shells[0], GLOBE_OPACITY),
                     (*shells[1], CAP_OPACITY)], np.ones(3))
    show(ax, image, view)


def animate_day(turns: np.ndarray, earth: core.Rotor, swings: core.Point, vertical: core.Point, north: core.Point,
                east: core.Point, track: core.Point) -> list[np.ndarray]:
    """Per frame of the day: the earth turned so far with the pendulum's floor and swing line, and the bob's
    track on the floor up to then."""
    floor = on_floor(track, north, east)                                       # [samples, 2]
    reached = np.round(turns / turns[-1] * (len(floor) - 1)).astype(int)        # [frames + 1]

    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(9.0, 4.4))
        draw_globe(figure.add_axes((0.0, 0.0, 0.5, 1.0)), earth[index], vertical, swings[index], north, east)
        face = figure.add_axes((0.52, 0.05, 0.46, 0.9))
        draw_track(face, floor[:reached[index] + 1])
        line = on_floor(swings[index], north, east)
        face.plot(*np.stack([-line, line]).T, color=SWING_COLOUR, linewidth=2.5)
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(len(turns) - 1)]
