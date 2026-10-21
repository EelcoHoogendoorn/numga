"""Drawing for Jeffery orbits: the paths of the long axes on the sphere of directions, and the bodies
tumbling in the shear flow. The flow runs to the right, its gradient points up, and the normal of the vortex plane
points out of the picture.

The bodies are ray cast. Each pixel's ray is taken into the body's frame by its rotor and back
through its stretch onto the unit sphere the body was stretched from, where the ray keeps its
parameter and meets the sphere where a quadratic in it vanishes. The point met is the body's own
direction there: its octant paints the body, so the pattern is carried by the body exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb

from numga import stack
from examples.animation import capture
from examples.fluids.jeffery import core

# The directions along the flow and along the gradient, and the normal of the vortex plane.
x, y, z = core.axes.dual()
# The colours of the starts, of the bodies and of the flow. The view as (elevation, azimuth) in
# degrees: raised from the normal of the vortex plane towards the gradient, then turned towards the flow. The lamp
# over the viewer's left shoulder, by its parts towards the viewer, to the right and up, and the share
# of a body's colour lit even where it faces away. The half width of the tumbling view, its pixels
# across, and the rays per pixel along each side.
STARTS = cm.viridis(np.linspace(0.1, 0.9, 5))
BODY, BODY_DARK, TIP, FLOW = "#7dd3fc", "#0369a1", "#f97316", "#94a3b8"
VIEW = (18, 28)
LAMP = (1.0, -0.4, 0.7)
AMBIENT = 0.4
LIMIT = 4.2
PIXELS = 200
SUPERSAMPLE = 2


def coordinates(points: core.Point) -> np.ndarray:
    """Points for matplotlib's three-dimensional axes: along the normal of the vortex plane, along the flow, and
    along the gradient, so that the gradient points up and the order stays right-handed."""
    return points.cast(core.ga.subspace("xy yz zx")).kernel


def draw_sphere(ax) -> None:
    """A faint unit sphere, with the flow's direction and the normal of the vortex plane."""
    polar, azimuth = np.meshgrid(np.linspace(0, np.pi, 13), np.linspace(0, 2 * np.pi, 25), indexing="ij")
    ax.plot_wireframe(np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar),
                      color="#cbd5e1", linewidth=0.4, alpha=0.6)
    ax.quiver(0, 0, 0, 0, 1.35, 0, color=FLOW, linewidth=1.4, arrow_length_ratio=0.12)
    ax.quiver(0, 0, 0, 1.35, 0, 0, color=FLOW, linewidth=1.4, arrow_length_ratio=0.12)


def frame_axes(ax, limit: float) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1, 1, 1], zoom=1.35)
    ax.set_proj_type("ortho")
    ax.view_init(*VIEW)
    ax.set_axis_off()


def draw_orbits(trails: core.Point) -> plt.Figure:
    """The path of each body's long axis from each start `[steps + 1, bodies, starts]`, on the sphere of
    directions, both ends of the axis drawn."""
    paths = coordinates(trails)                                              # [steps + 1, bodies, starts, 3]
    count = paths.shape[1]
    figure = plt.figure(figsize=(4.4 * count, 4.6), dpi=120)
    for body in range(count):
        ax = figure.add_subplot(1, count, body + 1, projection="3d")
        draw_sphere(ax)
        for start, colour in enumerate(STARTS):
            for end in (1, -1):
                ax.plot(*(end * paths[:, body, start]).T, color=colour, linewidth=1.3)
        frame_axes(ax, 1.05)
    figure.tight_layout()
    return figure


def camera() -> tuple[core.Point, core.Point, core.Point]:
    """The orthographic camera's right, up and back directions: the axes turned by the view, which
    raises the normal of the vortex plane towards the gradient, then turns it about the gradient towards the flow;
    the same view as `frame_axes` gives matplotlib."""
    elevation, azimuth = np.radians(VIEW)
    view = ((core.mv.z ^ core.mv.x) * (-azimuth / 2)).exp() * ((core.mv.z ^ core.mv.y) * (-elevation / 2)).exp()   # [] Rotor
    return view >> x, view >> y, view >> z


def screen(points: core.Point, right: core.Point, up: core.Point) -> np.ndarray:
    """Positions in the picture `[..., 2]`: along the camera's right and up."""
    return np.stack([points.dot(right).to_array(), points.dot(up).to_array()], axis=-1)


def pictures(stretches: core.Stretch, rotors: core.Rotor, right: core.Point, up: core.Point,
             back: core.Point) -> np.ndarray:
    """Each body, ray cast, as an RGBA picture `[bodies, pixels, pixels, 4]`, transparent where the
    rays miss it."""
    samples = PIXELS * SUPERSAMPLE
    offsets = (np.arange(samples) + 0.5) / samples * 2 * LIMIT - LIMIT        # [samples]
    origins = right * offsets[None, :] + up * offsets[::-1, None] + back * (2 * LIMIT)   # [samples, samples] Point
    turns, stretches = rotors[:, None, None], stretches[:, None, None]        # [bodies, 1, 1] each
    # Into each body's frame and back through its stretch: the unit sphere, and the ray in its parameter.
    start = stretches.solve(turns << origins)                                 # [bodies, samples, samples] Point
    heading = stretches.solve(turns << -back)                                 # [bodies, 1, 1] Point
    a = heading.dot(heading).to_array()
    b = heading.dot(start).to_array()
    c = start.dot(start).to_array() - 1
    discriminant = b * b - a * c
    hit = discriminant > 0                                                    # [bodies, samples, samples]
    entering = (-b - np.sqrt(np.where(hit, discriminant, 0.0))) / a
    direction = start + heading * entering                                    # the body's own direction met
    # The body's normal, the stretch undone once more and turned back out; lit by the lamp.
    normal = turns >> stretches.solve(direction)
    lamp = right * LAMP[1] + up * LAMP[2] + back * LAMP[0]
    lit = normal.dot(lamp).to_array() / (normal.norm().to_array() * lamp.norm().to_array())
    shade = AMBIENT + (1 - AMBIENT) * np.clip(lit, 0.0, 1.0)
    # The octant of the body's own direction picks the colour.
    octant = np.prod(np.sign(direction.cast(core.ga.subspace("yz zx xy")).kernel), axis=-1) > 0
    colour = np.array([to_rgb(BODY), to_rgb(BODY_DARK)])[octant.astype(int)] * shade[..., None]
    # Averaged over each pixel's rays, the colour weighted by coverage.
    coverage = hit[..., None].astype(float)
    blocks = (len(hit), PIXELS, SUPERSAMPLE, PIXELS, SUPERSAMPLE)
    covered = (colour * coverage).reshape(*blocks, 3).mean(axis=(2, 4))
    alpha = coverage.reshape(*blocks, 1).mean(axis=(2, 4))
    return np.concatenate([covered / np.maximum(alpha, 1e-12), alpha], axis=-1)


def draw_split(ax, points: core.Point, view: tuple, **style) -> None:
    """A path of points no nearer the centre than a body centred on the origin reaches: its part
    nearer the viewer than the centre in front of the body, the rest behind it."""
    right, up, back = view
    positions = screen(points, right, up)                                    # [points, 2]
    near = points.dot(back).to_array() > 0
    for side, zorder in ((near, 3), (~near, 1)):
        ax.plot(*np.where(side[:, None], positions, np.nan).T, zorder=zorder, **style)


def draw_body(ax, picture: np.ndarray, tip: core.Point, tips: core.Point, heights: core.Point,
              velocities: core.Point, view: tuple) -> None:
    """One body at one frame, its picture, the ends of its long axis, the paths of the axis's tips
    `[frames so far]` from the start, and the shear flow's velocity behind it."""
    right, up, back = view
    for end in (1, -1):
        draw_split(ax, tip * (end * np.array([1.0, 1.15])), view, color=TIP, linewidth=1.4)
        draw_split(ax, tips * end, view, color=TIP, linewidth=1.0, alpha=0.9)
    bases = screen(heights - velocities / 2, right, up)                       # [heights, 2]
    arrows = screen(velocities, right, up)                                    # [heights, 2]
    ax.quiver(*bases.T, *arrows.T, angles="xy", scale_units="xy", scale=1, color=FLOW, width=0.005, zorder=0)
    ax.imshow(picture, extent=(-LIMIT, LIMIT, -LIMIT, LIMIT), interpolation="bilinear", zorder=2)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.set_axis_off()


def animate_tumbling(stretches: core.Stretch, heights: core.Point, velocities: core.Point,
                     frames: Iterable[tuple[core.Rotor, core.Point]]) -> list[np.ndarray]:
    """Each frame of the bodies' orientations `[bodies]` and the tips of their long axes `[bodies]`: every
    body tumbling in the flow, side by side, with the paths of its tips so far."""
    view = camera()
    count = stretches.shape[0]
    images, tips = [], []
    for rotors, tip in frames:
        tips.append(tip)
        path = stack(tips)                                                    # [frames so far, bodies] Point
        shown = pictures(stretches, rotors, *view)                            # [bodies, pixels, pixels, 4]
        figure = plt.figure(figsize=(3.6 * count, 3.6), dpi=100)
        for body in range(count):
            draw_body(figure.add_subplot(1, count, body + 1), shown[body], tip[body], path[:, body], heights,
                      velocities, view)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        images.append(capture(figure))
        plt.close(figure)
    return images
