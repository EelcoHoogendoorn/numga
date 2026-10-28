"""Ray diagrams of the thin-lens example: drawing only.

Coordinates leave the algebra here: a point is read by its pairings with the lines x = 0 and y = 0 over its
weight, and a ray's direction of travel by where it meets the line at infinity.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D

from numga import stack

from examples.util.animation import capture
from examples.optics.thin_lens.core import Crossing, Line, LineMap, Plane, Point

# The colours of a bundle's legs before, between and after the lenses; where parallel rays start ahead of the
# first lens, and how far past its meet each bundle is drawn.
leg_colours = ("tab:orange", "tab:green", "tab:blue")
lead, past = 1.0, 0.5
# The figures' sizes in inches at their dots per inch; the widths of the rays and the elements' planes, in points;
# the area of the subject's and the image's dots, in square points.
lenses_size, train_size, dpi = (4.2, 3.3), (3.2, 2.5), 100
ray_width, plane_width = 0.6, 1.0
dot_area = 12


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """(..., 2) Euclidean coordinates of finite points: their pairings with the lines x = 0 and y = 0,
    over their weights."""
    mv = points.context.multivector
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def travel(rays: Line, handedness: np.ndarray) -> np.ndarray:
    """Unit direction of travel along each ray: where it meets the line at infinity, times the handedness."""
    mv = rays.context.multivector
    direction = (stack((mv.x, mv.y)) & (rays ^ mv.w)[..., None]).to_array()
    return handedness[:, None] * direction / np.linalg.norm(direction, axis=-1, keepdims=True)


def segments(ax, start: np.ndarray, stop: np.ndarray, color: str) -> list[Line2D]:
    """Draw segments between matching (n, 2) start and stop points, a line each."""
    return ax.plot(np.stack([start[:, 0], stop[:, 0]]), np.stack([start[:, 1], stop[:, 1]]), color=color,
                   linewidth=ray_width)


def draw_rays(ax, rays: Line, start: Line, stop: Line, color: str) -> None:
    """Draw ray segments between two planes (vertical lines)."""
    segments(ax, xy(rays ^ start), xy(rays ^ stop), color)


def plane_ends(planes: Plane, half_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Where elements' planes `[...]` meet the lines y == half_height and y == -half_height, `[..., 2]` each."""
    mv = planes.context.multivector
    return xy(planes ^ (mv.y + mv.w * half_height)), xy(planes ^ (mv.y - mv.w * half_height))


def draw_plane(ax, plane: Plane, half_height: float) -> Line2D:
    """Draw an element's plane between the lines y == -half_height and y == half_height."""
    a, b = plane_ends(plane, half_height)
    return ax.plot([a[0], b[0]], [a[1], b[1]], color="gray", linewidth=plane_width)[0]


def draw_lenses(subject: Point, imaging: Line, image: Crossing, focusing: Line, focus: Crossing,
                planes: Plane) -> plt.Figure:
    """The first lens imaging the subject, above, and both lenses focusing parallel rays, below: each bundle
    `[legs, rays]` drawn leg by leg between the lenses' planes, from a vertical through the subject or a little
    ahead of the first lens, to a little past where it meets."""
    mv = subject.context.multivector
    vertical = mv.direction([0, 1])
    ahead = (mv.wx * (lead / 2)).exp()
    beyond = (mv.wx * (-past / 2)).exp()
    systems = ((imaging, [planes[0]], subject & vertical, beyond >> (image & vertical)),
               (focusing, [planes[0], planes[1]], ahead >> planes[0], beyond >> (focus & vertical)))
    fig, axes = plt.subplots(2, 1, figsize=lenses_size, dpi=dpi, sharex=True)
    for ax, (bundles, lens_planes, start, stop) in zip(axes, systems):
        limits = [start, *lens_planes, stop]
        colours = leg_colours[:len(lens_planes)] + leg_colours[-1:]
        for leg, colour in enumerate(colours):
            draw_rays(ax, bundles[leg], limits[leg], limits[leg + 1], colour)
        for plane in lens_planes:
            draw_plane(ax, plane, 1.0)
        ax.set_axis_off()
        ax.set_aspect("equal")
    return fig


def bundle_legs(subject: Point, planes: Plane, legs: Line, train: LineMap) -> list[tuple[np.ndarray, np.ndarray]]:
    """The bundle leg by leg between the element planes, then onward along its direction of travel: each leg's starts
    and stops `[rays, 2]`.

    A line has no direction of travel, so the last leg follows a handedness that flips once per
    orientation-reversing element, which is what a mirror is: the sign of the train's determinant.
    """
    mv = subject.context.multivector
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
               picture: Crossing) -> tuple[list[list[Line2D]], list[Line2D], list[PathCollection]]:
    """Draw the bundle leg by leg, each element's plane after the leg that reaches it, and the subject and its image;
    the lines of each leg, the planes' lines and the two points are returned for `place_scene` to move."""
    count = planes.shape[0]
    rays, elements = [], []
    for stage, (start, stop) in enumerate(bundle_legs(subject, planes, legs, train)):
        rays.append(segments(ax, start, stop, f"C{stage}"))
        if stage < count:
            elements.append(draw_plane(ax, planes[stage], 0.8))
    marks = [ax.scatter(*xy(subject), s=dot_area, color="C0", zorder=3),
             ax.scatter(*xy(picture), s=dot_area, color=f"C{count}", zorder=3)]
    ax.set_axis_off()
    ax.set_xlim(-1.3, 4.7); ax.set_ylim(-1.5, 3.2); ax.set_aspect("equal")
    return rays, elements, marks


def place_scene(rays: list[list[Line2D]], elements: list[Line2D], marks: list[PathCollection], subject: Point,
                planes: Plane, legs: Line, train: LineMap, picture: Crossing) -> None:
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
    fig = plt.figure(figsize=train_size, dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    states = iter(states)
    artists = draw_scene(ax, *next(states))
    yield capture(fig)
    for state in states:
        place_scene(*artists, *state)
        yield capture(fig)
    plt.close(fig)
