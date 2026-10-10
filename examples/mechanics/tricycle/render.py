"""Drawing for the tricycle: the plane from above and space in perspective, with the tricycle's frame and
wheels, each axle drawn on to the turning centre, and the trails of the middle of the rear axle and of the
front wheel."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from examples.animation import capture

FRAME = "#2c3e50"
WHEEL = "#111111"
AXLE = "#c0392b"
CENTRE = "#c0392b"
TRAIL = "#7f8c8d"
FRONT_TRAIL = "#2980b9"
WHEEL_RADIUS = 0.25
# How far each axle is drawn past the wheels when the turning centre lies at infinity.
REACH = 2.0
# Half the width of the view, which follows the middle of the rear axle.
VIEW = 3.0


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
    """Space in perspective around the tricycle, the turning centre a vertical line, met with the ground
    where it is drawn."""
    mv = NumpyContext(centre.algebra).multivector
    wheels, ahead, middle = ground(contacts), ground(forward), ground(middle)
    extent = following(middle)
    foot = ground(centre ^ mv.z)
    figure = plt.figure(figsize=(6, 5))
    ax = figure.add_subplot(projection="3d")
    ax.plot(*middle.T, np.zeros(len(middle)), color=TRAIL, linewidth=1.0)
    ends = axle_ends(wheels, ahead, foot)
    for start, end in zip(wheels, ends):
        ax.plot(*np.stack([start, end]).T, [WHEEL_RADIUS] * 2, color=AXLE, linewidth=0.8, linestyle="--")
    if np.isfinite(foot).all():
        ax.plot([foot[0]] * 2, [foot[1]] * 2, [0, 4 * WHEEL_RADIUS], color=CENTRE, linewidth=1.5)
    hubs = np.concatenate([wheels, np.full((3, 1), WHEEL_RADIUS)], axis=-1)
    ax.plot(*hubs[[0, 1]].T, color=FRAME, linewidth=2.5)
    ax.plot(*np.stack([(hubs[0] + hubs[1]) / 2, hubs[2]]).T, color=FRAME, linewidth=2.5)
    around = np.linspace(0, 2 * np.pi, 33)
    for start, end in zip(wheels, ahead):
        heading = (end - start) / np.linalg.norm(end - start)
        rim = start + WHEEL_RADIUS * np.sin(around)[:, None] * heading
        ax.plot(*rim.T, WHEEL_RADIUS * (1 - np.cos(around)), color=WHEEL, linewidth=1.5)
    ax.set(xlim=extent[:, 0], ylim=extent[:, 1], zlim=(0, np.ptp(extent[:, 0]) / 2))
    ax.set_box_aspect((1, np.ptp(extent[:, 1]) / np.ptp(extent[:, 0]), 0.5))
    ax.view_init(elev=35, azim=-65)
    ax.axis("off")
    figure.tight_layout()
    return figure


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
