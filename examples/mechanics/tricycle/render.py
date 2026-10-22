"""Drawing for the tricycle: the plane from above and space in perspective, with the tricycle's frame and
wheels, each axle drawn on to the turning centre, and the trails of the middle of the rear axle and of the
front wheel."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import NumpyContext, concatenate
from examples.animation import capture
from examples.wgsl import image_figure, meshes, solids

sm = solids.mv

FRAME = "#2c3e50"
WHEEL = "#222222"
AXLE = "#c0392b"
CENTRE = "#c0392b"
TRAIL = "#7f8c8d"
FRONT_TRAIL = "#2980b9"
WHEEL_RADIUS = 0.25
# How far each axle is drawn past the wheels when the turning centre lies at infinity.
REACH = 2.0
# Half the width of the view, which follows the middle of the rear axle.
VIEW = 3.0
BACKGROUND = np.ones(3)
# Space: the view's elevation and azimuth in radians, azimuth zero looking along +y, the margin around the square
# of ground it follows as a share of the larger side, and the pixels across it; the dashes along each axle, the points
# around a rim and around a tube; the widths in pixels of the axles' dashes, the frame, the post, the rims and the
# trail.
ELEVATION, AZIMUTH = np.radians(35.0), np.radians(25.0)
MARGIN, N_PIXELS = 0.04, 600
N_DASHES, N_RIM, N_AROUND = 16, 48, 8
DASH_WIDTH, FRAME_WIDTH, POST_WIDTH, RIM_WIDTH, TRAIL_WIDTH = 3.0, 6.0, 4.0, 4.0, 4.0


def ground(points) -> np.ndarray:
    """The coordinates along x and y of points on the ground: their pairings with the planes x = 0 and
    y = 0 over their weights; infinite for points at infinity."""
    mv = NumpyContext(points.algebra).multivector
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([((plane & points) / (mv.w & points)).to_array() for plane in (mv.x, mv.y)], axis=-1)


def axle_ends(contacts: np.ndarray, forward: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """For each wheel, the far end of its axle as drawn: the turning centre, or a fixed reach sideways
    when the centre lies at infinity. [wheels, 2]."""
    heading = forward - contacts
    sideways = np.stack([-heading[:, 1], heading[:, 0]], axis=-1)
    sideways /= np.linalg.norm(sideways, axis=-1, keepdims=True)
    return np.where(np.isfinite(centre).all(), centre, contacts + REACH * sideways)


def plane(contacts, forward, centre, middle) -> plt.Figure:
    """The plane from above, around the tricycle."""
    wheels, ahead, middle = ground(contacts), ground(forward), ground(middle)
    extent = following(middle)
    centre = ground(centre)
    figure, ax = plt.subplots(figsize=(5, 5))
    ax.plot(*middle.T, color=TRAIL, linewidth=1.0)
    ends = axle_ends(wheels, ahead, centre)
    for start, end in zip(wheels, ends):
        ax.plot(*np.stack([start, end]).T, color=AXLE, linewidth=0.8, linestyle="--")
    if np.isfinite(centre).all():
        ax.scatter(*centre, color=CENTRE, s=30, zorder=4)
    ax.plot(*np.stack([wheels[0], wheels[1]]).T, color=FRAME, linewidth=2.5)
    ax.plot(*np.stack([(wheels[0] + wheels[1]) / 2, wheels[2]]).T, color=FRAME, linewidth=2.5)
    for start, end in zip(wheels, ahead):
        step = (end - start) / np.linalg.norm(end - start) * WHEEL_RADIUS
        ax.plot(*np.stack([start - step, start + step]).T, color=WHEEL, linewidth=5, solid_capstyle="round")
    ax.set(xlim=extent[:, 0], ylim=extent[:, 1], aspect="equal")
    ax.axis("off")
    figure.tight_layout()
    return figure


def space(contacts, forward, centre, middle) -> plt.Figure:
    """Space around the tricycle, drawn on the GPU: the frame and the axles' dashes rods, the wheels' rims and the
    trail tubes, and the turning centre a vertical post, met with the ground where it is drawn. The view follows
    the middle of the rear axle."""
    mv = NumpyContext(centre.algebra).multivector
    wheels, ahead, middle = ground(contacts), ground(forward), ground(middle)
    foot = ground(centre ^ mv.z)
    view = space_view(middle[-1])
    # Each axle's dashes, at the height of the hubs: stretches along it from the wheel to its far end.
    ends = axle_ends(wheels, ahead, foot)
    marks = np.linspace(0, 1, 2 * N_DASHES)
    along = wheels[:, None] + marks[:, None] * (ends - wheels)[:, None]                 # [wheels, 2 * dashes, 2]
    dashes = solids.rods(at_height(along[:, 0::2], WHEEL_RADIUS), at_height(along[:, 1::2], WHEEL_RADIUS),
                         DASH_WIDTH / 2 * view.pixel)
    hubs = at_height(wheels, WHEEL_RADIUS)                                               # [wheels] Point
    frame = solids.rods(concatenate([hubs[:1], (hubs[:1] + hubs[1:2]) / 2]), concatenate([hubs[1:2], hubs[2:]]),
                        FRAME_WIDTH / 2 * view.pixel)
    parts = [(dashes.reshape(-1, 3), to_rgb(AXLE)), (frame, to_rgb(FRAME))]
    if np.isfinite(foot).all():
        post = solids.rods(at_height(foot, 0.0), at_height(foot, 4 * WHEEL_RADIUS), POST_WIDTH / 2 * view.pixel)
        parts.append((post[None], to_rgb(CENTRE)))
    # Each rim: a circle about its hub, square to the axle, across the way the wheel rolls.
    heading = (ahead - wheels) / np.linalg.norm(ahead - wheels, axis=-1, keepdims=True)
    sideways = np.stack([-heading[:, 1], heading[:, 0], np.zeros(len(wheels))], axis=-1)
    rims = meshes.circles(np.concatenate([wheels, np.full((len(wheels), 1), WHEEL_RADIUS)], axis=-1),
                          np.full(len(wheels), WHEEL_RADIUS), sideways, N_RIM)          # [wheels, rim + 1, 3]
    tyres = meshes.tube_samples([(rim, to_rgb(WHEEL)) for rim in rims], RIM_WIDTH, view, N_AROUND)
    trail = meshes.tube_samples([(np.concatenate([middle, np.zeros((len(middle), 1))], axis=-1), to_rgb(TRAIL))],
                                TRAIL_WIDTH, view, N_AROUND)
    return image_figure(solids.draw(parts, view, BACKGROUND, meshes=[(*tyres, 1.0), (*trail, 1.0)]))


def at_height(places: np.ndarray, height: float) -> solids.Point:
    """Points above places on the ground `[..., 2]`, at a height."""
    return sm.point(np.concatenate([places, np.full(places.shape[:-1] + (1,), height)], axis=-1))


def space_view(place: np.ndarray) -> solids.View:
    """A view of the square of ground following a place on it `[2]`, up to the top of the turning centre's post."""
    corners = place + VIEW * np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    return solids.view([at_height(corners, 0.0), at_height(corners, 4 * WHEEL_RADIUS)], 0.0, ELEVATION, AZIMUTH, MARGIN,
                       N_PIXELS)


def following(trail: np.ndarray) -> np.ndarray:
    """The square view centred on the last point of a trail: [corners, 2]."""
    return trail[-1] + np.array([[-VIEW, -VIEW], [VIEW, VIEW]])


def animate(draw, scenes) -> list[np.ndarray]:
    """One frame per scene, drawn by the given view."""
    frames = []
    for scene in scenes:
        figure = draw(*scene)
        frames.append(capture(figure))
        plt.close(figure)
    return frames
