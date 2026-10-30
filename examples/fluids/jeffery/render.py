"""Drawing for Jeffery orbits: the paths of the long axes on the sphere of directions, and the bodies
tumbling in the shear flow. The flow runs to the right, its gradient points up, and the normal of the vortex plane
points out of the picture.

The bodies are traced solids: each the ellipsoid its shape bounds, cut by its own three planes into eight octants,
painted light and dark by the signs of the body's own directions in them, so the pattern is carried by the body
exactly.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.fluids.jeffery import core
from examples.util import charts
from examples.util.wgsl import labels, show, solids, strokes
from examples.util.wgsl.scene import Scene

# The colours of the starts, of the bodies, of the flow and of the frame's names. The view as (elevation, azimuth) in
# degrees: raised from the normal of the vortex plane towards the gradient, then turned towards the flow. Where the
# marks on the long axis end past its tips, as a share of the axis, and the margin around the farthest of them, as a
# share of it.
STARTS = cm.viridis(np.linspace(0.1, 0.9, 5))
BODY, BODY_DARK, TIP, FLOW, NAMES = "#7dd3fc", "#0369a1", "#f97316", "#94a3b8", "#475569"
VIEW = (18, 28)
MARK_END = 1.15
FRAME_MARGIN = 0.08
# A body's octants, by the signs of its own directions along its three axes, and their colours: dark where the
# signs multiply to plus. The width in pixels of the paths of the tips and of the marks past them.
OCTANTS = np.array(list(product((1.0, -1.0), repeat=3)))                     # [octants, 3]
OCTANT_COLOURS = np.where((OCTANTS.prod(axis=-1) > 0)[:, None], to_rgb(BODY_DARK), to_rgb(BODY))   # [octants, 3]
TIP_RGB = np.array(to_rgb(TIP))
PATH_WIDTH, MARK_WIDTH = 2.0, 3.0
# The sphere of directions, drawn on the GPU: its colour and how much each of its surfaces hides, and its grid's
# width in pixels and opacity; the frame's arrows' length in radii of the sphere, the flow's arrows' width and heads
# in pixels, and how far the frame's names reach; the frame's arrows' width and heads in pixels; the orbits' width in
# pixels, the margin around the names' reach, and the pixels across a sphere.
SPHERE, SPHERE_OPACITY = np.array(to_rgb("#cbd5e1")), 0.15
GRID_WIDTH, GRID_OPACITY = 1.5, 0.3
FLOW_RGB = np.array(to_rgb(FLOW))
ARROW_LENGTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH, LABEL_DISTANCE = 1.35, 3.0, 16.0, 10.0, 1.47
SPHERE_ARROW_WIDTH, SPHERE_HEAD_LENGTH, SPHERE_HEAD_WIDTH = 1.5, 8.0, 5.0
ORBIT_WIDTH, ORBIT_MARGIN, ORBIT_PIXELS = 1.5, 0.05, 260


# The core's frame turned into the scene's, so that the gradient points up and the order stays right-handed: the
# flow x to y, the gradient y to z and the normal of the vortex plane z to x, by the mirrors swapping y with z, then
# x with y.
to_scene = ((solids.mv.x - solids.mv.y) * (solids.mv.y - solids.mv.z)).normalized()   # [] Rotor+


def sphere(view: solids.View) -> tuple[list, list, list]:
    """The faint unit sphere, traced: the frame of the normal of the vortex plane, the flow and the gradient as arrows,
    the ball translucent among them, and its parallels and meridians inked on it, where their planes cut it."""
    # The frame: along the normal of the vortex plane, along the flow and along the gradient.
    shafts, heads = solids.frame(ARROW_LENGTH, SPHERE_ARROW_WIDTH / 2 * view.pixel, SPHERE_HEAD_LENGTH * view.pixel,
                                 SPHERE_HEAD_WIDTH / 2 * view.pixel)
    parts = [(shafts, FLOW_RGB), (heads, FLOW_RGB)]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE, SPHERE_OPACITY)]
    # The parallels: the planes across the vertical through the points the pole turns to at each polar angle; the
    # meridians: the plane x turned about the vertical by each azimuth.
    polar_turns = (solids.mv.zx * (-np.radians(np.arange(30, 180, 30)) / 2)).exp()   # [parallels] Rotor+
    on_parallels = polar_turns >> (solids.origin + solids.mv.z.dual())        # [parallels] Point-
    parallels = solids.mv.z - solids.mv.w * (solids.mv.z & on_parallels)      # [parallels] Plane+
    meridians = (solids.mv.xy * (-np.radians(np.arange(0, 180, 30)) / 2)).exp() >> solids.mv.x   # [meridians] Plane+
    planes = concatenate([parallels, meridians])                              # [circles] Plane+
    # The ball is the first translucent part, after the opaque ones.
    grid = solids.Ink(len(parts), solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape), SPHERE,
                      GRID_WIDTH, GRID_OPACITY)
    return parts, ball, [grid]


@charts.style()
def draw_orbits(trails: core.Point) -> plt.Figure:
    """The path of each body's long axis from each start `[steps + 1, bodies, starts]`, on the sphere of
    directions, both ends of the axis drawn, among the sphere's own surfaces sample by sample."""
    # Both ends of every long axis, the scene's points the axis reaches from its origin.
    axis_ends = trails * np.array([1.0, -1.0])[:, None, None, None]          # [ends, steps + 1, bodies, starts] Point-
    ends = solids.origin + (to_scene >> solids.context.include_from_algebra(axis_ends))   # [ends, steps + 1, bodies, starts]
    count = trails.shape[1]
    view = solids.View(solids.aim(solids.origin, np.radians(VIEW[0]), np.radians(VIEW[1] + 90)),
                       np.full(2, LABEL_DISTANCE * (1 + ORBIT_MARGIN)), ORBIT_PIXELS)
    figure = plt.figure(figsize=(count * ORBIT_PIXELS / 100, ORBIT_PIXELS / 100), dpi=100)
    parts, ball, inks = sphere(view)
    for body in range(count):
        orbits = strokes.polylines([(ends[end, :, body, start], colour[:3])
                                    for start, colour in enumerate(STARTS) for end in range(2)], ORBIT_WIDTH, view)
        ax = figure.add_axes((body / count, 0, 1 / count, 1))
        show(ax, solids.draw(parts, view, np.ones(3), inks, translucent=ball, meshes=[(orbits, 1.0)]), view)
        # Each arrow named just past its tip by the direction it runs along: the normal z, the flow x, the gradient y.
        labels(ax, view, solids.origin + solids.axes.dual() * LABEL_DISTANCE, ["$z$", "$x$", "$y$"], ha="center",
               va="center", color=NAMES)
    return figure


def bodies(stretches: core.Stretch, orientations: core.Rotor) -> solids.Quadric:
    """Each body as traced solids about the origin, `[..., octants, 4]`: its ellipsoid, the offsets its inverse shape
    takes into the unit ball, cut by its own planes into eight octants."""
    # The body's shape, its dual quadric, the solid about the origin, turned into the scene's frame.
    shapes = solids.context.include_from_algebra(core.placed(stretches, orientations))   # [...] Point- <- Plane+
    ellipsoids = solids.placed(solids.solid_of_dual_quadric(shapes), to_scene)   # [...] Plane+ <- Point-
    # Each octant on the side of each of the body's own planes its signs pick: the planes across the body's axes.
    normals = to_scene >> solids.context.include_from_algebra(orientations[..., None] >> core.axes)   # [..., 3] Plane+
    sides = solids.half_space(normals[..., None, :] * -OCTANTS)               # [..., octants, 3] Quadric
    return concatenate([ellipsoids[..., None, None].broadcast_to(sides.shape[:-1] + (1,)), sides], axis=-1)


def animate_tumbling(stretches: core.Stretch, heights: core.Point, velocities: core.Point,
                     frames: Iterable[tuple[core.Rotor, core.Point]], pixels: int) -> Iterator[np.ndarray]:
    """Each frame of the bodies' orientations `[bodies]` and the tips of their long axes `[bodies]`, two or more:
    every body tumbling in the flow, side by side in a row, each panel so many pixels across, with the paths of its
    tips so far, each frame traced in one pass among the flow, which is set up once."""
    orientations, tips = (stack(list(part)) for part in zip(*frames))        # [frames, bodies] Rotor+, Point-
    count = tips.shape[1]
    # One frame for every panel, so that the bodies' sizes compare: the farthest marks past a long axis's tips,
    # and the margin.
    radii = tips.norm().reshape(-1)
    limit = solids.context.include_from_algebra(radii.max()) * (MARK_END * (1 + FRAME_MARGIN))
    camera = solids.aim(solids.origin, np.radians(VIEW[0]), np.radians(VIEW[1] + 90))
    view = solids.View(camera, (limit * np.array([count, 1.0])).to_array(), count * pixels)
    # Each body's panel about its own centre, two half-widths apart across the image, the row about the origin.
    across = (camera >> solids.mv.x.dual()) * (2 * limit * (np.arange(count) - (count - 1) / 2))   # [bodies] Point-
    shifts = solids.translator(across)                                        # [bodies] Motor+
    # Every body in its own panel, at every frame, `[frames, bodies, octants, 4]`.
    placed_bodies = solids.placed(bodies(stretches, orientations), shifts[:, None, None])
    colours = np.tile(OCTANT_COLOURS, (count, 1))                             # [bodies * octants, 3]
    # The shear flow's velocity at each height, centred on it, about every body.
    starts, ends = (shifts[:, None] >> (solids.origin + (to_scene >> solids.context.include_from_algebra(offsets)))
                    for offsets in (heights - velocities / 2, heights + velocities / 2))   # [bodies, heights] Point-, Point-
    shafts, heads = solids.arrows(starts, ends, ARROW_WIDTH / 2 * view.pixel, HEAD_LENGTH * view.pixel,
                                  HEAD_WIDTH / 2 * view.pixel)
    flow = [(shafts.reshape(-1, 3), FLOW_RGB), (heads.reshape(-1, 3), FLOW_RGB)]
    # Both ends of every long axis, in its panel, over the frames, and the flow, which stays put.
    tip_ends = tips * np.array([1.0, -1.0])[:, None, None]                    # [ends, frames, bodies] Point-
    tip_offsets = to_scene >> solids.context.include_from_algebra(tip_ends)             # [ends, frames, bodies] Point-
    axis_ends = shifts >> (solids.origin + tip_offsets)                       # [ends, frames, bodies] Point-
    mark_ends = shifts >> (solids.origin + tip_offsets * MARK_END)            # [ends, frames, bodies] Point-
    scene = Scene(view, np.ones(3), flow, [], inks=[], translucent=[])
    for index in range(len(tips)):
        # The paths of the tips so far, and the marks from each end out past it.
        paths = strokes.segments(axis_ends[:, :index].reshape(-1), axis_ends[:, 1:index + 1].reshape(-1), TIP_RGB,
                                 PATH_WIDTH, view)
        marks = strokes.segments(axis_ends[:, index].reshape(-1), mark_ends[:, index].reshape(-1), TIP_RGB, MARK_WIDTH,
                                 view)
        yield scene.draw([(placed_bodies[index].reshape(-1, 4), colours)], inks=[], translucent=[],
                         meshes=[(paths, 1.0), (marks, 1.0)])
