"""GPU rendering for three-point pose reconstruction draws the moving triangle as a translucent plate with solid edges and the measured markers as balls joined by a dashed outline."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np

from numga import concatenate
from examples.geometry.three_point_pose import core
from numga.topology import Topology
from numga.render import solids
from examples.util.panels import image_figure

colour = np.array(to_rgb("#218c9d"))
target_colour = np.array(to_rgb("#ce713f"))
# Opacity of each of the moving triangle's two faces.
triangle_opacity = 0.26
# Closed three-edge loop joining each marker corner to the next.
outline = Topology.loops([3])
# Camera elevation and azimuth in radians (azimuth zero looks along the positive y axis), frame margin, and pixel width.
elevation, azimuth = np.radians(55.0), np.radians(-110.0 + 90.0)
margin, n_pixels = 0.28, 600
# Extra framing factor around the initial and target markers so the intermediate screw path stays in view.
view_room = 1.5
# Widths of the triangle edges and target dashes, dash period (one dash and one gap), and marker ball radius, in pixels.
edge_width, dash_width, dash_period, marker_radius = 5.0, 4.5, 18.0, 10.0


# --- plumbing -------------------------------------------------------------------------
def framing(points: list[core.Point]) -> solids.View:
    """A shared camera view frames every marker triple (`[3] Point-`) in `points`."""
    return solids.view(points, 0.0, elevation, azimuth, margin, n_pixels)


def dashes(tails: core.Point, heads: core.Point, period: float) -> tuple[core.Point, core.Point]:
    """Dash start and end points (`[...] Point-`) alternate one dash and one gap per `period` along each edge from `tails` to `heads` (`[3] Point-`)."""
    pieces = []
    for start, end in zip(tails.normalized(), heads.normalized()):
        # The Euclidean length of a segment is the norm of the line joining its unit-weight endpoints.
        n_dashes = max(1, round(float((start & end).norm().to_array() / period)))
        fractions = np.arange(n_dashes) / n_dashes
        pieces.append((start + (end - start) * fractions, start + (end - start) * (fractions + 0.5 / n_dashes)))
    return concatenate([starts for starts, _ in pieces]), concatenate([ends for _, ends in pieces])


def target_markers(target: core.Point, view: solids.View) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Solid quadric balls and dashed outline rods represent the measured `target` markers (`[3] Point-`)."""
    tails, heads = (outline.incidence[1, 0] * target.field()).batch()        # [3] Point-, Point-
    starts, ends = dashes(tails, heads, dash_period * view.pixel)            # [...] Point-, Point-
    return [(solids.ball(target, marker_radius * view.pixel)[:, None], target_colour),
            (solids.rods(starts, ends, dash_width / 2 * view.pixel), target_colour)]


def triangle(current: core.Point, view: solids.View) -> tuple[list[tuple[solids.Quadric, np.ndarray]],
                                                              list[tuple[solids.Quadric, np.ndarray, float]]]:
    """Solid edge quadrics and a one-pixel-thick translucent plate represent the `current` marker triple (`[3] Point-`)."""
    tails, heads = (outline.incidence[1, 0] * current.field()).batch()       # [3] Point-, Point-
    rods, joints = solids.edges(tails, heads, edge_width / 2 * view.pixel)
    # Represent the interior as a translucent plate one pixel thick.
    plate = solids.plates(current[None], view.pixel)                         # [1, 5] Plane+
    return [(rods, colour), (joints, colour)], [(plate, colour, triangle_opacity)]


def scene(target: core.Point, current: core.Point, view: solids.View) -> np.ndarray:
    """Renders the measured `target` markers and the translucent `current` triangle (`[3] Point-`) into an RGB image."""
    edges, translucent = triangle(current, view)
    return solids.draw(target_markers(target, view) + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])


def pose_figure(source: core.Point, target: core.Point, current: core.Point) -> plt.Figure:
    """A figure frames `source`, `target`, and `current` (`[3] Point-`) together and draws `current` against `target`."""
    return image_figure(scene(target, current, framing([source, target, current])))


def pose_animation(source: core.Point, target: core.Point, states: Iterator[core.Point]) -> Iterator[np.ndarray]:
    """Yields RGB frames as the marker triple (`[3] Point-`) moves through `states` toward `target`."""
    framed = framing([source, target])
    view = framed._replace(half=framed.half * view_room)
    measured = target_markers(target, view)
    for state in states:
        edges, translucent = triangle(state, view)
        yield solids.draw(measured + edges, view, np.ones(3), inks=[], translucent=translucent, meshes=[])
