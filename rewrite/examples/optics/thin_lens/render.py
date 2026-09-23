"""Ray diagrams of the thin-lens example: drawing only.

Coordinates leave the algebra here: a point is read in the blade layout yw, wx, xy and
divided by its weight, a line in the layout x, y, w.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.animation import capture
from examples.optics.thin_lens.core import Line, LineMap, Point, ga, mv

POINT_LAYOUT = ga.subspace("yw wx xy")
LINE_LAYOUT = ga.subspace("x y w")


def xy(points: Point) -> np.ndarray:
    """(..., 2) Euclidean coordinates of finite points."""
    k = points.cast(POINT_LAYOUT).kernel
    return k[..., :2] / k[..., 2:]


def travel(rays: Line, handedness: np.ndarray) -> np.ndarray:
    """Unit direction of travel along each ray: its normal turned a quarter turn, times the handedness."""
    normal = rays.cast(LINE_LAYOUT).kernel[..., :2]
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    return handedness[:, None] * np.stack([-normal[:, 1], normal[:, 0]], axis=-1)


def segments(ax, start: np.ndarray, stop: np.ndarray, color: str) -> None:
    """Draw segments between matching (n, 2) start and stop points."""
    ax.plot(np.stack([start[:, 0], stop[:, 0]]), np.stack([start[:, 1], stop[:, 1]]), color=color, linewidth=0.7)


def draw_rays(ax, rays: Line, start: Line, stop: Line, color: str) -> None:
    """Draw ray segments between two planes (vertical lines)."""
    segments(ax, xy(rays ^ start), xy(rays ^ stop), color)


def draw_plane(ax, plane: Line, half_height: float) -> None:
    """Draw an element's plane between the lines y = ±half_height."""
    a, b = xy(plane ^ (mv.y + mv.w * half_height)), xy(plane ^ (mv.y - mv.w * half_height))
    ax.plot([a[0], b[0]], [a[1], b[1]], color="gray")


def draw_lenses(one_lens: list, one_planes: list[Line], two_lenses: list, two_planes: list[Line]) -> plt.Figure:
    """One lens imaging a point, and two lenses focusing parallel rays."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=120, sharex=True)
    systems = ((one_lens, one_planes, ("tab:orange", "tab:blue"), "one thin lens"),
               (two_lenses, two_planes, ("tab:orange", "tab:green", "tab:blue"), "two lenses"))
    for ax, (legs, planes, colors, title) in zip(axes, systems):
        for (rays, start, stop), color in zip(legs, colors):
            draw_rays(ax, rays, start, stop, color)
        for plane in planes:
            draw_plane(ax, plane, 1.0)
        ax.set_title(title)
        ax.set_aspect("equal")
    return fig


def draw_scene(ax, subject: Point, planes: Line, legs: Line, train: LineMap, picture: Point) -> None:
    """Draw the bundle leg by leg between the element planes, then onward along its direction of travel.

    A line has no direction of travel, so the last leg follows a handedness that flips once per
    orientation-reversing element, which is what a mirror is: the sign of the train's determinant.
    """
    ax.cla()
    count = planes.shape[0]
    start = xy(legs[0] ^ (subject & mv.wx))
    for stage in range(count):
        stop = xy(legs[stage] ^ planes[stage])
        segments(ax, start, stop, f"C{stage}")
        draw_plane(ax, planes[stage], 0.8)
        start = xy(legs[stage + 1] ^ planes[stage])
    first = xy(legs[0] ^ planes[0]) - xy(legs[0] ^ (subject & mv.wx))
    heading = np.sign(np.einsum("ij,ij->i", first, travel(legs[0], np.ones(len(first)))))
    handedness = heading * np.sign(train.det().to_array())
    segments(ax, start, start + 2.5 * travel(legs[-1], handedness), f"C{count}")
    ax.scatter(*xy(subject), color="C0", zorder=3); ax.scatter(*xy(picture), color=f"C{count}", zorder=3)
    ax.set_xlim(-1.3, 4.7); ax.set_ylim(-1.5, 3.2); ax.set_aspect("equal")


def animate_train(states) -> list[np.ndarray]:
    """Frames of the optical train from its geometric states."""
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot()
    frames = []
    for subject, planes, legs, train, picture in states:
        draw_scene(ax, subject, planes, legs, train, picture)
        frames.append(capture(fig))
    plt.close(fig)
    return frames
