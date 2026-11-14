"""GPU ray-diagram rendering for single lenses, two-lens systems, and a moving planar optical train."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.optics.thin_lens.core import Crossing, Line, LineMap, Plane, Point
from examples.util.animation import capture
from examples.util.panels import overlay, panel
from numga import concatenate, stack
from numga.render import Camera, drawing, simplices
from numga.topology import Topology

# Colours for ray segments before, between, and after lenses, plus lead-in and run-out distances.
leg_colours = ("tab:orange", "tab:green", "tab:blue")
lead_distance, past_distance = 1.0, 0.5
onward_distance = 2.5
# Figure dimensions in inches at 100 dpi, stroke widths and marker radius in pixels, and element-plane colour.
lenses_size, train_size, dpi = (4.2, 3.3), (3.2, 2.5), 100
ray_width, plane_width = 1.0, 1.5
dot_radius = 2.4
plane_colour = to_rgb("gray")


# --- plumbing -------------------------------------------------------------------------
def segment_topology(starts: Point, stops: Point) -> tuple[Point, Topology]:
    """Pairs `starts` and `stops` (`[...] Point-`) into concatenated endpoints and a 1-simplex segment `Topology`."""
    n_segments = len(starts)
    edges = np.stack([np.arange(n_segments), n_segments + np.arange(n_segments)], axis=-1)
    return concatenate([starts, stops]), Topology(2 * n_segments, edges)


def plane_endpoints(planes: Plane, half_height: float) -> tuple[Point, Point]:
    """Intersects `planes` (`[...] Plane+`) with horizontal lines at `+half_height` and `-half_height` (`[...] Point-`)."""
    mv = planes.context.multivector
    along = mv.direction([1, 0])                                                # [] Point-
    return planes ^ (mv.point([0, half_height]) & along), planes ^ (mv.point([0, -half_height]) & along)


def lenses_figure(
    subject: Point,
    imaging: Line,
    image: Crossing,
    focusing: Line,
    focus: Crossing,
    planes: Plane,
) -> plt.Figure:
    """Renders single-lens point imaging in the upper panel and two-lens parallel focusing in the lower panel."""
    mv = subject.context.multivector
    vertical = mv.direction([0, 1])                                             # [] Point-
    ahead = (mv.wx * (lead_distance / 2)).exp()                                 # [] Rotor+
    beyond = (mv.wx * (-past_distance / 2)).exp()                               # [] Rotor+
    # Joining `image` or `focus` (`Bivector+`) with `vertical` (`Antivector-`) produces a vertical `Plane+` stop.
    systems = (
        (imaging, [planes[0]], subject & mv.wx, beyond >> (image & vertical)),
        (focusing, [planes[0], planes[1]], ahead >> planes[0], beyond >> (focus & vertical)),
    )
    fig, axes = plt.subplots(2, 1, figsize=lenses_size, dpi=dpi, sharex=True)
    drawn = []
    for ax, (bundles, lens_planes, start, stop) in zip(axes, systems):
        limits = [start, *lens_planes, stop]
        colours = leg_colours[:len(lens_planes)] + leg_colours[-1:]
        legs = [
            (segment_topology(bundles[leg] ^ limits[leg], bundles[leg] ^ limits[leg + 1]), colour)
            for leg, colour in enumerate(colours)
        ]
        elements = [segment_topology(*plane_endpoints(plane[None], 1.0)) for plane in lens_planes]
        endpoints = concatenate([pts for (pts, _), _ in legs] + [pts for pts, _ in elements])  # [...] Point-
        readings = (stack((mv.x, mv.y))[:, None] & endpoints) / (mv.w & endpoints)  # [2, ...] Scalar+
        ax.update_datalim(stack((readings.min(axis=-1), readings.max(axis=-1))).to_array())
        drawn.append((legs, elements))
        ax.set_axis_off()
        ax.set_aspect("equal")
    for ax in axes:
        ax.autoscale_view()
    for ax, (legs, elements) in zip(axes, drawn):
        camera = panel(ax, mv.x, mv.y, mv.w)
        overlay(
            ax,
            [simplices.strokes(*leg, to_rgb(colour), ray_width, camera) for leg, colour in legs]
            + [simplices.strokes(*element, plane_colour, plane_width, camera) for element in elements],
        )
    return fig


def bundle_segments(subject: Point, planes: Plane, legs: Line, train: LineMap) -> list[tuple[Point, Point]]:
    """Intersects each stage of `legs` (`[n_elements + 1, n_rays] Line-`) with `planes` (`[n_elements] Plane+`) and extends the final bundle along its travel direction."""
    mv = subject.context.multivector
    n_elements = planes.shape[0]
    starts = [legs[0] ^ (subject & mv.wx)] + [legs[stage + 1] ^ planes[stage] for stage in range(n_elements)]
    stops = [legs[stage] ^ planes[stage] for stage in range(n_elements)]
    starts, stops = ([pt / (mv.w & pt) for pt in pts] for pts in (starts, stops))
    # Orientation-reversing reflections flip `train.det()` sign, reversing ray propagation relative to line orientation.
    agreement = (starts[0] & stops[0]).scalar_product(legs[0])                  # [...] Scalar+
    heading = np.asarray(agreement > 0, dtype=int) - np.asarray(agreement < 0, dtype=int)
    determinant = train.det()                                                   # [] Scalar+
    handedness = heading * (np.asarray(determinant > 0, dtype=int) - np.asarray(determinant < 0, dtype=int))
    travel = legs[-1].normalized() ^ mv.w                                       # [...] Point-
    return list(zip(starts, stops + [starts[-1] + travel * (onward_distance * handedness)]))


def train_composite(
    camera: Camera, subject: Point, planes: Plane, legs: Line, train: LineMap, picture: Crossing,
) -> np.ndarray:
    """Composites the coloured ray stages, element planes, subject dot, and image dot for one train frame."""
    n_elements = planes.shape[0]
    tops, bottoms = plane_endpoints(planes, 0.8)                                # [n_elements] Point-, Point-
    layers = []
    for stage, (start, stop) in enumerate(bundle_segments(subject, planes, legs, train)):
        layers.append(simplices.strokes(*segment_topology(start, stop), to_rgb(f"C{stage}"), ray_width, camera))
        if stage < n_elements:
            layers.append(
                simplices.strokes(
                    *segment_topology(tops[stage][None], bottoms[stage][None]), plane_colour, plane_width, camera,
                )
            )
    return drawing.composited(
        layers
        + [
            simplices.markers(subject[None], to_rgb("C0"), dot_radius, camera),
            simplices.markers(picture[None], to_rgb(f"C{n_elements}"), dot_radius, camera),
        ]
    )


def train_animation(
    states: Iterable[tuple[Point, Plane, Line, LineMap, Crossing]],
) -> Iterator[np.ndarray]:
    """Yields one RGB frame per `(subject, planes, legs, train, picture)` state in `states`."""
    fig = plt.figure(figsize=train_size, dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.set_xlim(-1.3, 4.7)
    ax.set_ylim(-1.5, 3.2)
    ax.set_aspect("equal")
    iterator = iter(states)
    first = next(iterator)
    mv = first[0].context.multivector
    camera = panel(ax, mv.x, mv.y, mv.w)
    image = overlay(ax, [train_composite(camera, *first)])
    yield capture(fig)
    for state in iterator:
        image.set_data(train_composite(camera, *state))
        yield capture(fig)
    plt.close(fig)
