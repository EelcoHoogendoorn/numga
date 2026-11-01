"""Ray diagrams of the thin-lens example: drawing only, on the GPU in the plane's own algebra.

Rays and the elements' planes are drawn between the points where they cross, read by a camera of the planes x and y
over their weight; coordinates leave the algebra only to frame the lenses' figure.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack

from examples.util.animation import capture
from examples.util.wgsl import simplices
from examples.util.wgsl.camera import Camera, panel
from examples.optics.thin_lens.core import Crossing, Line, LineMap, Plane, Point

# The colours of a bundle's legs before, between and after the lenses; where parallel rays start ahead of the
# first lens, and how far past its meet each bundle is drawn.
leg_colours = ("tab:orange", "tab:green", "tab:blue")
lead, past = 1.0, 0.5
# How far the train's last leg runs on past its last element.
onward = 2.5
# The figures' sizes in inches at their dots per inch; the widths of the rays and the elements' planes, and the
# radius of the subject's and the image's dots, in pixels; the planes' colour.
lenses_size, train_size, dpi = (4.2, 3.3), (3.2, 2.5), 100
ray_width, plane_width = 1.0, 1.5
dot_radius = 2.4
plane_colour = to_rgb("gray")


# --- plumbing -------------------------------------------------------------------------
def segments(starts: Crossing, stops: Crossing) -> tuple[Crossing, simplices.Topology]:
    """Segments between matching starts and stops `[segments]`: their ends, and an edge from each start to its
    stop."""
    n_segments = len(starts)
    edges = np.stack([np.arange(n_segments), n_segments + np.arange(n_segments)], axis=-1)
    return concatenate([starts, stops]), simplices.Topology(2 * n_segments, edges)


def plane_ends(planes: Plane, half_height: float) -> tuple[Crossing, Crossing]:
    """Where elements' planes `[...]` meet the lines along x at heights half_height and -half_height, intrinsic
    crossings, points."""
    mv = planes.context.multivector
    along = mv.direction([1, 0])
    return planes ^ (mv.point([0, half_height]) & along), planes ^ (mv.point([0, -half_height]) & along)


def draw_lenses(subject: Point, imaging: Line, image: Crossing, focusing: Line, focus: Crossing,
                planes: Plane) -> plt.Figure:
    """The first lens imaging the subject, above, and both lenses focusing parallel rays, below: each bundle
    `[legs, rays]` drawn leg by leg between the lenses' planes, from a vertical through the subject or a little
    ahead of the first lens, to a little past where it meets."""
    mv = subject.context.multivector
    vertical = mv.direction([0, 1])
    ahead = (mv.wx * (lead / 2)).exp()
    beyond = (mv.wx * (-past / 2)).exp()
    # The rays start on the plane through the subject, `subject & mv.wx`, as the train's do, so that every crossing
    # they are drawn between is intrinsic, a point.
    systems = ((imaging, [planes[0]], subject & mv.wx, beyond >> (image & vertical)),
               (focusing, [planes[0], planes[1]], ahead >> planes[0], beyond >> (focus & vertical)))
    fig, axes = plt.subplots(2, 1, figsize=lenses_size, dpi=dpi, sharex=True)
    # Each panel's rays leg by leg between its limits, in their colours, and its lenses' planes.
    drawn = []
    for ax, (bundles, lens_planes, start, stop) in zip(axes, systems):
        limits = [start, *lens_planes, stop]
        colours = leg_colours[:len(lens_planes)] + leg_colours[-1:]
        legs = [(segments(bundles[leg] ^ limits[leg], bundles[leg] ^ limits[leg + 1]), colour)
                for leg, colour in enumerate(colours)]
        elements = [segments(*plane_ends(plane[None], 1.0)) for plane in lens_planes]
        # The panel framed around what it draws: the low and high corners of its extent, read by the planes x and y
        # over the weights.
        ends = concatenate([ends for (ends, _), _ in legs] + [ends for ends, _ in elements])   # [ends] Crossing-
        readings = (stack((mv.x, mv.y))[:, None] & ends) / (mv.w & ends)     # [2, ends] Scalar+
        ax.update_datalim(stack((readings.min(axis=-1), readings.max(axis=-1))).to_array())
        drawn.append((legs, elements))
        ax.set_axis_off()
        ax.set_aspect("equal")
    for ax in axes:
        ax.autoscale_view()
    for ax, (legs, elements) in zip(axes, drawn):
        camera = panel(ax, mv.x, mv.y)
        simplices.overlay(ax, [simplices.strokes(*leg, to_rgb(colour), ray_width, camera) for leg, colour in legs]
                          + [simplices.strokes(*element, plane_colour, plane_width, camera) for element in elements])
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
    agreement = (starts[0] & stops[0]).scalar_product(legs[0])
    heading = np.asarray(agreement > 0, dtype=int) - np.asarray(agreement < 0, dtype=int)
    determinant = train.det()
    handedness = heading * (np.asarray(determinant > 0, dtype=int) - np.asarray(determinant < 0, dtype=int))
    # The last leg runs on along its line's meet with the line at infinity, a unit direction for a unit line.
    travel = legs[-1].normalized() ^ mv.w                                    # [rays] Crossing-
    return list(zip(starts, stops + [starts[-1] + travel * (onward * handedness)]))


def scene(camera: Camera, subject: Point, planes: Plane, legs: Line, train: LineMap, picture: Crossing) -> np.ndarray:
    """The bundle leg by leg, each element's plane after the leg that reaches it, and the subject and its image, as
    the camera sees them."""
    count = planes.shape[0]
    tops, bottoms = plane_ends(planes, 0.8)                                  # [elements] Crossing-, Crossing-
    layers = []
    for stage, (start, stop) in enumerate(bundle_legs(subject, planes, legs, train)):
        layers.append(simplices.strokes(*segments(start, stop), to_rgb(f"C{stage}"), ray_width, camera))
        if stage < count:
            layers.append(simplices.strokes(*segments(tops[stage][None], bottoms[stage][None]), plane_colour,
                                            plane_width, camera))
    return simplices.composited(layers + [simplices.markers(subject[None], to_rgb("C0"), dot_radius, camera),
                                          simplices.markers(picture[None], to_rgb(f"C{count}"), dot_radius, camera)])


def animate_train(states) -> Iterator[np.ndarray]:
    """Frames of the optical train from its geometric states, each drawn over the same view."""
    fig = plt.figure(figsize=train_size, dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(-1.3, 4.7); ax.set_ylim(-1.5, 3.2); ax.set_aspect("equal")
    states = iter(states)
    first = next(states)
    camera = panel(ax, first[0].context.multivector.x, first[0].context.multivector.y)
    image = simplices.overlay(ax, [scene(camera, *first)])
    yield capture(fig)
    for state in states:
        image.set_data(scene(camera, *state))
        yield capture(fig)
    plt.close(fig)
