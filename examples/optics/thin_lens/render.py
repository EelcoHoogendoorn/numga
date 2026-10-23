"""Ray diagrams of the thin-lens example: drawing only.

Coordinates leave the algebra here: a point is read by its pairings with the lines x = 0 and y = 0 over
its weight, a line in the layout x, y, w.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from numga import stack

from examples.util.animation import capture
from examples.optics.thin_lens.core import Line, LineMap, Plane, Point, ga, mv

LINE_LAYOUT = ga.subspace("x y w")


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """(..., 2) Euclidean coordinates of finite points: their pairings with the lines x = 0 and y = 0,
    over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def travel(rays: Line, handedness: np.ndarray) -> np.ndarray:
    """Unit direction of travel along each ray: its normal turned a quarter turn, times the handedness."""
    normal = rays.cast(LINE_LAYOUT).kernel[..., :2]
    normal = normal / np.linalg.norm(normal, axis=-1, keepdims=True)
    return handedness[:, None] * np.stack([-normal[:, 1], normal[:, 0]], axis=-1)


def segments(ax, start: np.ndarray, stop: np.ndarray, color: str) -> list[Line2D]:
    """Draw segments between matching (n, 2) start and stop points, a line each."""
    return ax.plot(np.stack([start[:, 0], stop[:, 0]]), np.stack([start[:, 1], stop[:, 1]]), color=color,
                   linewidth=0.7)


def draw_rays(ax, rays: Line, start: Line, stop: Line, color: str) -> None:
    """Draw ray segments between two planes (vertical lines)."""
    segments(ax, xy(rays ^ start), xy(rays ^ stop), color)


def plane_ends(planes: Plane, half_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Where elements' planes `[...]` meet the lines y == half_height and y == -half_height, `[..., 2]` each."""
    return xy(planes ^ (mv.y + mv.w * half_height)), xy(planes ^ (mv.y - mv.w * half_height))


def draw_plane(ax, plane: Plane, half_height: float) -> Line2D:
    """Draw an element's plane between the lines y == -half_height and y == half_height."""
    a, b = plane_ends(plane, half_height)
    return ax.plot([a[0], b[0]], [a[1], b[1]], color="gray")[0]


def cross(ax) -> None:
    """A faint cross through the origin, its lines named x and y at their ends, in place of axes and ticks."""
    ax.axhline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#cbd5e1", linewidth=0.7, zorder=0)
    # Each name sits at its line's end of the panel, whatever the panel's limits.
    ax.text(1, 0, "$x$", transform=ax.get_yaxis_transform(), ha="right", va="bottom", color="#94a3b8", fontsize=11)
    ax.text(0, 1, " $y$", transform=ax.get_xaxis_transform(), ha="left", va="top", color="#94a3b8", fontsize=11)
    ax.set_axis_off()


def draw_lenses(one_lens: list, one_planes: list[Plane], two_lenses: list, two_planes: list[Plane]) -> plt.Figure:
    """One lens imaging a point, and two lenses focusing parallel rays."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), dpi=120, sharex=True)
    systems = ((one_lens, one_planes, ("tab:orange", "tab:blue")),
               (two_lenses, two_planes, ("tab:orange", "tab:green", "tab:blue")))
    for ax, (legs, planes, colors) in zip(axes, systems):
        for (rays, start, stop), color in zip(legs, colors):
            draw_rays(ax, rays, start, stop, color)
        for plane in planes:
            draw_plane(ax, plane, 1.0)
        cross(ax)
        ax.set_aspect("equal")
    return fig


def bundle_legs(subject: Point, planes: Plane, legs: Line, train: LineMap) -> list[tuple[np.ndarray, np.ndarray]]:
    """The bundle leg by leg between the element planes, then onward along its direction of travel: each leg's starts
    and stops `[rays, 2]`.

    A line has no direction of travel, so the last leg follows a handedness that flips once per
    orientation-reversing element, which is what a mirror is: the sign of the train's determinant.
    """
    count = planes.shape[0]
    start = xy(legs[0] ^ (subject & mv.wx))
    ends = []
    for stage in range(count):
        ends.append((start, xy(legs[stage] ^ planes[stage])))
        start = xy(legs[stage + 1] ^ planes[stage])
    first = xy(legs[0] ^ planes[0]) - xy(legs[0] ^ (subject & mv.wx))
    heading = np.sign(np.einsum("ij,ij->i", first, travel(legs[0], np.ones(len(first)))))
    handedness = heading * np.sign(train.det().to_array())
    return ends + [(start, start + 2.5 * travel(legs[-1], handedness))]


def draw_scene(ax, subject: Point, planes: Plane, legs: Line, train: LineMap,
               picture: Point) -> tuple[list[list[Line2D]], list[Line2D], list[PathCollection]]:
    """Draw the bundle leg by leg, each element's plane after the leg that reaches it, and the subject and its image,
    over the cross; the lines of each leg, the planes' lines and the two points, for `place_scene` to move."""
    count = planes.shape[0]
    rays, elements = [], []
    for stage, (start, stop) in enumerate(bundle_legs(subject, planes, legs, train)):
        rays.append(segments(ax, start, stop, f"C{stage}"))
        if stage < count:
            elements.append(draw_plane(ax, planes[stage], 0.8))
    marks = [ax.scatter(*xy(subject), color="C0", zorder=3), ax.scatter(*xy(picture), color=f"C{count}", zorder=3)]
    cross(ax)
    ax.set_xlim(-1.3, 4.7); ax.set_ylim(-1.5, 3.2); ax.set_aspect("equal")
    return rays, elements, marks


def place_scene(rays: list[list[Line2D]], elements: list[Line2D], marks: list[PathCollection], subject: Point,
                planes: Plane, legs: Line, train: LineMap, picture: Point) -> None:
    """Move the lines and points `draw_scene` drew to another state of the train."""
    for lines, (start, stop) in zip(rays, bundle_legs(subject, planes, legs, train)):
        for line, a, b in zip(lines, start, stop):
            line.set_data([a[0], b[0]], [a[1], b[1]])
    for line, a, b in zip(elements, *plane_ends(planes, 0.8)):
        line.set_data([a[0], b[0]], [a[1], b[1]])
    for mark, place in zip(marks, (subject, picture)):
        mark.set_offsets(xy(place)[None])


def animate_train(states) -> Iterator[np.ndarray]:
    """Frames of the optical train from its geometric states: the first state drawn, and each one after it moving
    its lines and points."""
    fig = plt.figure(figsize=(7, 5), dpi=100)
    ax = fig.add_subplot()
    states = iter(states)
    artists = draw_scene(ax, *next(states))
    yield capture(fig)
    for state in states:
        place_scene(*artists, *state)
        yield capture(fig)
    plt.close(fig)
