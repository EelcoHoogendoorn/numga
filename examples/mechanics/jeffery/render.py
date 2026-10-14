"""Drawing for Jeffery orbits: the paths of the long axes on the sphere of directions, and the bodies
tumbling in the shear flow. The flow runs to the right, its gradient points up, and the vorticity
axis points out of the picture.

The bodies are ray cast. Each pixel's ray is taken into the body's frame by its rotor and back
through its stretch onto the unit sphere the body was stretched from, where the ray keeps its
parameter and meets the sphere where a quadratic in it vanishes. The point met is the body's own
direction there: its octant paints the body, so the pattern is carried by the body exactly.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb

from examples.animation import capture
from examples.mechanics.jeffery import core

x, y, z = core.axes                                        # flow, gradient, vorticity
# The colours of the starts, of the bodies and of the flow. The view as (elevation, azimuth) in
# degrees: raised from the vorticity axis towards the gradient, then turned towards the flow. The lamp
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


def coordinates(vectors: core.Vector) -> np.ndarray:
    """Vectors for matplotlib's three-dimensional axes: along the vorticity axis, along the flow, and
    along the gradient, so that the gradient points up and the order stays right-handed."""
    return vectors.cast(core.ga.subspace("z x y")).kernel


def draw_sphere(ax) -> None:
    """A faint unit sphere, with the flow's direction and the vorticity axis."""
    polar, azimuth = np.meshgrid(np.linspace(0, np.pi, 13), np.linspace(0, 2 * np.pi, 25), indexing="ij")
    ax.plot_wireframe(np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar),
                      color="#cbd5e1", linewidth=0.4, alpha=0.6)
    ax.quiver(0, 0, 0, 0, 1.35, 0, color=FLOW, linewidth=1.4, arrow_length_ratio=0.12)
    ax.text(0, 1.35, 0.12, "flow", fontsize=8, color="#64748b", ha="right")
    ax.quiver(0, 0, 0, 1.35, 0, 0, color=FLOW, linewidth=1.4, arrow_length_ratio=0.12)
    ax.text(1.45, 0, 0, "vorticity", fontsize=8, color="#64748b")


def frame_axes(ax, limit: float, title: str) -> None:
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_box_aspect([1, 1, 1], zoom=1.35)
    ax.set_proj_type("ortho")
    ax.view_init(*VIEW)
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=0)


def draw_orbits(tumbling: core.Tumbling, names: list[str]) -> plt.Figure:
    """The path of each body's long axis from each start, on the sphere of directions, both ends of
    the axis drawn."""
    trails = coordinates(tumbling.trails)                                    # [steps + 1, bodies, starts, 3]
    figure = plt.figure(figsize=(4.4 * len(names), 4.6), dpi=120)
    for body, name in enumerate(names):
        ax = figure.add_subplot(1, len(names), body + 1, projection="3d")
        draw_sphere(ax)
        for start, colour in enumerate(STARTS):
            for end in (1, -1):
                ax.plot(*(end * trails[:, body, start]).T, color=colour, linewidth=1.3)
        frame_axes(ax, 1.05, name)
    figure.tight_layout()
    return figure


def camera() -> tuple[core.Vector, core.Vector, core.Vector]:
    """The orthographic camera's right, up and back directions: the axes turned by the view, which
    raises the vorticity axis towards the gradient, then turns it about the gradient towards the flow;
    the same view as `frame_axes` gives matplotlib."""
    elevation, azimuth = np.radians(VIEW)
    view = ((z ^ x) * (-azimuth / 2)).exp() * ((z ^ y) * (-elevation / 2)).exp()   # [] Rotor
    return view >> x, view >> y, view >> z


def screen(vectors: core.Vector, right: core.Vector, up: core.Vector) -> np.ndarray:
    """Positions in the picture `[..., 2]`: along the camera's right and up."""
    return np.stack([(vectors | right).to_array(), (vectors | up).to_array()], axis=-1)


def pictures(stretches: core.Stretch, rotors: core.Rotor, right: core.Vector, up: core.Vector,
             back: core.Vector) -> np.ndarray:
    """Each body, ray cast, as an RGBA picture `[bodies, pixels, pixels, 4]`, transparent where the
    rays miss it."""
    samples = PIXELS * SUPERSAMPLE
    offsets = (np.arange(samples) + 0.5) / samples * 2 * LIMIT - LIMIT        # [samples]
    origins = right * offsets[None, :] + up * offsets[::-1, None] + back * (2 * LIMIT)   # [samples, samples] Vector
    turns, stretches = rotors[:, None, None], stretches[:, None, None]        # [bodies, 1, 1] each
    # Into each body's frame and back through its stretch: the unit sphere, and the ray in its parameter.
    start = stretches.solve(turns << origins)                                 # [bodies, samples, samples] Vector
    heading = stretches.solve(turns << -back)                                 # [bodies, 1, 1] Vector
    a = (heading | heading).to_array()
    b = (heading | start).to_array()
    c = (start | start).to_array() - 1
    discriminant = b * b - a * c
    hit = discriminant > 0                                                    # [bodies, samples, samples]
    entering = (-b - np.sqrt(np.where(hit, discriminant, 0.0))) / a
    direction = start + heading * entering                                    # the body's own direction met
    # The body's normal, the stretch undone once more and turned back out; lit by the lamp.
    normal = turns >> stretches.solve(direction)
    lamp = right * LAMP[1] + up * LAMP[2] + back * LAMP[0]
    lit = (normal | lamp).to_array() / (normal.norm().to_array() * lamp.norm().to_array())
    shade = AMBIENT + (1 - AMBIENT) * np.clip(lit, 0.0, 1.0)
    # The octant of the body's own direction picks the colour.
    octant = np.prod(np.sign(direction.cast(core.ga.subspace("x y z")).kernel), axis=-1) > 0
    colour = np.array([to_rgb(BODY), to_rgb(BODY_DARK)])[octant.astype(int)] * shade[..., None]
    # Averaged over each pixel's rays, the colour weighted by coverage.
    coverage = hit[..., None].astype(float)
    blocks = (len(hit), PIXELS, SUPERSAMPLE, PIXELS, SUPERSAMPLE)
    covered = (colour * coverage).reshape(*blocks, 3).mean(axis=(2, 4))
    alpha = coverage.reshape(*blocks, 1).mean(axis=(2, 4))
    return np.concatenate([covered / np.maximum(alpha, 1e-12), alpha], axis=-1)


def draw_split(ax, vectors: core.Vector, view: tuple, **style) -> None:
    """A path of points no nearer the centre than a body centred on the origin reaches: its part
    nearer the viewer than the centre in front of the body, the rest behind it."""
    right, up, back = view
    positions = screen(vectors, right, up)                                   # [points, 2]
    near = (vectors | back).to_array() > 0
    for side, zorder in ((near, 3), (~near, 1)):
        ax.plot(*np.where(side[:, None], positions, np.nan).T, zorder=zorder, **style)


def draw_body(ax, tumbling: core.Tumbling, picture: np.ndarray, frame: int, body: int, title: str,
              view: tuple) -> None:
    """One body at one frame, its picture, the ends of its long axis, the paths of the axis's tips
    from the start, and the shear flow's velocity behind it."""
    right, up, back = view
    tip = tumbling.tips[frame, body]                                         # [] Vector
    tips = tumbling.tips[:frame + 1, body]                                   # [frame + 1] Vector
    for end in (1, -1):
        draw_split(ax, tip * (end * np.array([1.0, 1.15])), view, color=TIP, linewidth=1.4)
        draw_split(ax, tips * end, view, color=TIP, linewidth=1.0, alpha=0.9)
    bases = screen(tumbling.heights - tumbling.velocities / 2, right, up)    # [heights, 2]
    arrows = screen(tumbling.velocities, right, up)                          # [heights, 2]
    ax.quiver(*bases.T, *arrows.T, angles="xy", scale_units="xy", scale=1, color=FLOW, width=0.005, zorder=0)
    ax.imshow(picture, extent=(-LIMIT, LIMIT, -LIMIT, LIMIT), interpolation="bilinear", zorder=2)
    ax.set_xlim(-LIMIT, LIMIT)
    ax.set_ylim(-LIMIT, LIMIT)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=11, pad=0)


def animate_tumbling(tumbling: core.Tumbling, names: list[str]) -> list[np.ndarray]:
    """Each frame: every body tumbling in the flow, side by side."""
    view = camera()
    images = []
    for frame in range(tumbling.rotors.shape[0]):
        shown = pictures(tumbling.stretches, tumbling.rotors[frame], *view)  # [bodies, pixels, pixels, 4]
        figure = plt.figure(figsize=(3.6 * len(names), 3.8), dpi=100)
        for body, name in enumerate(names):
            draw_body(figure.add_subplot(1, len(names), body + 1), tumbling, shown[body], frame, body, name, view)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=0.92, wspace=0)
        images.append(capture(figure))
        plt.close(figure)
    return images
