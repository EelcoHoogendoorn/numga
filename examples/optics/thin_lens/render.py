"""Ray diagrams of the thin-lens example: drawing only.

Coordinates leave the algebra here: a point is read by its pairings with the planes x = 0 and y = 0 over its
weight.
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
# How far the train's last leg runs on past its last element.
onward = 2.5
# The figures' sizes in inches at their dots per inch; the widths of the rays and the elements' planes, in points;
# the area of the subject's and the image's dots, in square points.
lenses_size, train_size, dpi = (4.2, 3.3), (3.2, 2.5), 100
ray_width, plane_width = 0.6, 1.0
dot_area = 12


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """(..., 2) Euclidean coordinates of finite points: their pairings with the planes x = 0 and y = 0,
    over their weights."""
    mv = points.context.multivector
    return ((stack((mv.x, mv.y)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def segments(ax, start: np.ndarray, stop: np.ndarray, color: str) -> list[Line2D]:
    """Draw segments between matching (n, 2) start and stop points, a line each."""
    return ax.plot(np.stack([start[:, 0], stop[:, 0]]), np.stack([start[:, 1], stop[:, 1]]), color=color,
                   linewidth=ray_width)


def draw_rays(ax, rays: Line, start: Line, stop: Line, color: str) -> None:
    """Draw ray segments between two planes."""
    segments(ax, xy(rays ^ start), xy(rays ^ stop), color)


def plane_ends(planes: Plane, half_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Where elements' planes `[...]` meet the planes y == half_height and y == -half_height, `[..., 2]` each."""
    mv = planes.context.multivector
    return xy(planes ^ (mv.y + mv.w * half_height)), xy(planes ^ (mv.y - mv.w * half_height))


def draw_plane(ax, plane: Plane, half_height: float) -> Line2D:
    """Draw an element's plane between the planes y == -half_height and y == half_height."""
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


def bundle_legs(subject: Point, planes: Plane, legs: Line, train: LineMap) -> list[tuple[Crossing, Crossing]]:
    """The bundle leg by leg between the element planes, then onward along its direction of travel: each leg's starts
    and stops `[rays]`, at unit weight.

    A line has no direction of travel, so the last leg follows a handedness that flips once per
    orientation-reversing element, which is what a mirror is: the sign of the train's determinant.
    """
    mv = subject.context.multivector
    count = planes.shape[0]
    starts = [legs[0] ^ (subject & mv.wx)] + [legs[stage + 1] ^ planes[stage] for stage in range(count)]
    stops = [legs[stage] ^ planes[stage] for stage in range(count)]
    # Each crossing at unit weight, its sign divided out too, so that joins run from start to stop; `.normalized()`
    # would keep the sign.
    starts, stops = ([crossing / (mv.w & crossing) for crossing in crossings] for crossings in (starts, stops))
    # The first leg's line runs the way the ray travels, from its start to the first element, where the two lines'
    # scalar product is positive.
    heading = np.sign((starts[0] & stops[0]).scalar_product(legs[0]).to_array())   # [rays]
    handedness = heading * np.sign(train.det().to_array())                   # [rays]
    # The last leg runs on along its line's meet with the line at infinity, a unit direction for a unit line.
    travel = legs[-1].normalized() ^ mv.w                                    # [rays] Crossing-
    return list(zip(starts, stops + [starts[-1] + travel * (onward * handedness)]))


def draw_scene(ax, subject: Point, planes: Plane, legs: Line, train: LineMap,
               picture: Crossing) -> tuple[list[list[Line2D]], list[Line2D], list[PathCollection]]:
    """Draw the bundle leg by leg, each element's plane after the leg that reaches it, and the subject and its image;
    the segments of each leg and of the planes, and the two points, are returned for `place_scene` to move."""
    count = planes.shape[0]
    rays, elements = [], []
    for stage, (start, stop) in enumerate(bundle_legs(subject, planes, legs, train)):
        rays.append(segments(ax, xy(start), xy(stop), f"C{stage}"))
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
        for line, a, b in zip(lines, xy(start), xy(stop)):
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
