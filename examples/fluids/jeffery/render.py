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
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import to_rgb

from numga import concatenate, stack
from examples.animation import capture
from examples.fluids.jeffery import core
from examples.wgsl import Shader, labels, meshes, show, solids, supersampled, vec3

# The directions along the flow and along the gradient, and the normal of the vortex plane.
x, y, z = core.axes.dual()
# The colours of the starts, of the bodies, of the flow and of the frame's names. The view as (elevation, azimuth) in
# degrees: raised from the normal of the vortex plane towards the gradient, then turned towards the flow. The lamp
# over the viewer's left shoulder, by its parts towards the viewer, to the right and up, and the share
# of a body's colour lit even where it faces away. How far the marks on the long axis reach past its tips, as a share
# of the axis; the margin around the farthest reach, as a share of it, and the pixels across a panel.
STARTS = cm.viridis(np.linspace(0.1, 0.9, 5))
BODY, BODY_DARK, TIP, FLOW, NAMES = "#7dd3fc", "#0369a1", "#f97316", "#94a3b8", "#475569"
VIEW = (18, 28)
LAMP = (1.0, -0.4, 0.7)
AMBIENT = 0.4
TIP_REACH = 1.15
FRAME_MARGIN = 0.08
PIXELS = 360
# The sphere of directions, drawn on the GPU: its colour and how much each of its surfaces hides, and its grid's
# width in pixels and opacity; the frame's arrows' length in radii of the sphere, and their width and heads in pixels, and
# how far their names reach; the orbits' width in pixels, the margin around the names' reach, and the pixels across a
# sphere.
SPHERE, SPHERE_OPACITY = np.array(to_rgb("#cbd5e1")), 0.15
GRID_WIDTH, GRID_OPACITY = 3.0, 0.3
FLOW_RGB = np.array(to_rgb(FLOW))
ARROW_LENGTH, ARROW_WIDTH, HEAD_LENGTH, HEAD_WIDTH, LABEL_DISTANCE = 1.35, 3.0, 16.0, 10.0, 1.47
ORBIT_WIDTH, ORBIT_MARGIN, ORBIT_PIXELS = 3.0, 0.05, 520


def coordinates(points: core.Point) -> np.ndarray:
    """Points for the sphere of directions: along the normal of the vortex plane, along the flow, and along the
    gradient, so that the gradient points up and the order stays right-handed."""
    return points.cast(core.ga.subspace("xy yz zx")).kernel


def sphere(view: solids.View) -> tuple[list, list, list]:
    """The faint unit sphere, traced: the frame of the normal of the vortex plane, the flow and the gradient as arrows,
    the ball translucent among them, and its parallels and meridians inked on it, where their planes cut it."""
    # The frame: along the normal of the vortex plane, along the flow and along the gradient.
    shafts, heads = solids.frame(ARROW_LENGTH, ARROW_WIDTH / 2 * view.pixel, HEAD_LENGTH * view.pixel,
                                 HEAD_WIDTH / 2 * view.pixel)
    parts = [(shafts, FLOW_RGB), (heads, FLOW_RGB)]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], SPHERE, SPHERE_OPACITY)]
    # The parallels: the planes across the vertical at the height of each polar angle; the meridians: the plane x
    # turned about the vertical by each azimuth.
    heights = np.cos(np.radians(np.arange(30, 180, 30)))
    meridians = (solids.mv.xy * (-np.radians(np.arange(0, 180, 30)) / 2)).exp() >> solids.mv.x   # [meridians] Plane
    planes = concatenate([solids.mv.z - solids.mv.w * heights, meridians])          # [circles] Plane
    # The ball is the first translucent part, after the opaque ones.
    grid = solids.Ink(len(parts), solids.half_space(planes), solids.everywhere[None].broadcast_to(planes.shape), SPHERE,
                      GRID_WIDTH, GRID_OPACITY)
    return parts, ball, [grid]


def draw_orbits(trails: core.Point) -> plt.Figure:
    """The path of each body's long axis from each start `[steps + 1, bodies, starts]`, on the sphere of
    directions, both ends of the axis drawn, among the sphere's own surfaces sample by sample."""
    paths = coordinates(trails)                                              # [steps + 1, bodies, starts, 3]
    count = paths.shape[1]
    view = solids.View(solids.aim(solids.origin, np.radians(VIEW[0]), np.radians(VIEW[1] + 90)),
                       np.full(2, LABEL_DISTANCE * (1 + ORBIT_MARGIN)), ORBIT_PIXELS)
    figure = plt.figure(figsize=(count * ORBIT_PIXELS / 100, ORBIT_PIXELS / 100), dpi=100)
    parts, ball, inks = sphere(view)
    for body in range(count):
        orbits = meshes.tube_samples([(end * paths[:, body, start], colour[:3]) for start, colour in enumerate(STARTS)
                                      for end in (1, -1)], ORBIT_WIDTH, view)
        ax = figure.add_axes((body / count, 0, 1 / count, 1))
        show(ax, solids.draw(parts, view, np.ones(3), inks, translucent=ball, meshes=[(*orbits, 1.0)]), view)
        # Each arrow named just past its tip by the direction it runs along: the normal z, the flow x, the gradient y.
        labels(ax, view, solids.mv.point(LABEL_DISTANCE * np.eye(3)), ["$z$", "$x$", "$y$"], ha="center",
               va="center", fontsize=13, color=NAMES)
    return figure


def camera() -> tuple[core.Point, core.Point, core.Point]:
    """The orthographic camera's right, up and back directions: the axes turned by the view, which
    raises the normal of the vortex plane towards the gradient, then turns it about the gradient towards the flow;
    the same view as `draw_orbits` sees the sphere from."""
    elevation, azimuth = np.radians(VIEW)
    view = ((core.mv.z ^ core.mv.x) * (-azimuth / 2)).exp() * ((core.mv.z ^ core.mv.y) * (-elevation / 2)).exp()   # [] Rotor
    return view >> x, view >> y, view >> z


def screen(points: core.Point, right: core.Point, up: core.Point) -> np.ndarray:
    """Positions in the picture `[..., 2]`: along the camera's right and up."""
    return np.stack([points.dot(right).to_array(), points.dot(up).to_array()], axis=-1)


def pictures(stretches: core.Stretch, rotors: core.Rotor, limit: float, right: core.Point, up: core.Point,
             back: core.Point) -> np.ndarray:
    """Each body, ray cast, as an RGBA picture `[bodies, pixels, pixels, 4]` of half-width `limit`, transparent
    where the rays miss it."""
    lamp = right * LAMP[1] + up * LAMP[2] + back * LAMP[0]
    unstretches = stretches.inverse()                                        # [bodies] Point <- Point
    shown = [shader()((PIXELS, PIXELS), turn=turn, unstretch=unstretch, right=right, up=up, back=back, lamp=lamp,
                      half_width=limit)["image"] for turn, unstretch in zip(rotors, unstretches)]
    pictures = np.stack(shown)                                               # [bodies, pixels, pixels, 4]
    alpha = pictures[..., 3:]
    return np.concatenate([pictures[..., :3] / np.maximum(alpha, 1e-12), alpha], axis=-1)


source = supersampled() + """
// A body's turn and the map from it back onto the unit sphere it was stretched from; the camera's right, up and
// back directions, the lamp, and the half width of the view.
var<uniform> turn: Rotor;
var<uniform> unstretch: Stretch;
var<uniform> right: Point;
var<uniform> up: Point;
var<uniform> back: Point;
var<uniform> lamp: Point;
var<uniform> half_width: f32;

// The colour, and its coverage.
alias Sample = vec4<f32>;

const body = BODY;
const body_dark = BODY_DARK;
const ambient = AMBIENT;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place) * half_width;
    // Into the body's frame and back through its stretch: the unit sphere, and the ray in its parameter.
    let start = unstretch(turn << (right * across.x + up * across.y + back * (2 * half_width)));
    let heading = unstretch(turn << -back);
    let a = heading.dot(heading);
    let b = heading.dot(start);
    let c = start.dot(start) - 1;
    let discriminant = b * b - a * c;
    if (discriminant <= 0) {
        return vec4<f32>(0.0);
    }
    // The body's own direction met; its normal, the stretch undone once more and turned back out.
    let direction = start + heading * ((-b - sqrt(discriminant)) / a);
    let normal = turn >> unstretch(direction);
    let lit = normal.dot(lamp) / (normal.norm() * lamp.norm());
    let shade = ambient + (1 - ambient) * clamp(lit, 0.0, 1.0);
    // The octant of the body's own direction picks the colour.
    let octant = direction.dot(mv.yz) * direction.dot(mv.zx) * direction.dot(mv.xy) > 0;
    return vec4<f32>(select(body, body_dark, octant) * shade, 1.0);
}
"""
source = (source.replace("BODY_DARK", vec3(to_rgb(BODY_DARK))).replace("BODY", vec3(to_rgb(BODY)))
          .replace("AMBIENT", repr(AMBIENT)))


@cache
def shader() -> Shader:
    return Shader(source, core.ga, dict(Point=core.Point, Rotor=core.Rotor, Stretch=core.Stretch))


def draw_split(ax, points: core.Point, view: tuple, **style) -> None:
    """A path of points no nearer the centre than a body centred on the origin reaches: its part
    nearer the viewer than the centre in front of the body, the rest behind it."""
    right, up, back = view
    positions = screen(points, right, up)                                    # [points, 2]
    near = points.dot(back).to_array() > 0
    for side, zorder in ((near, 3), (~near, 1)):
        ax.plot(*np.where(side[:, None], positions, np.nan).T, zorder=zorder, **style)


def draw_body(ax, picture: np.ndarray, limit: float, tip: core.Point, tips: core.Point, heights: core.Point,
              velocities: core.Point, view: tuple) -> None:
    """One body at one frame, its picture, the ends of its long axis, the paths of the axis's tips
    `[frames so far]` from the start, and the shear flow's velocity behind it, within `limit` of the centre."""
    right, up, back = view
    for end in (1, -1):
        draw_split(ax, tip * (end * np.array([1.0, TIP_REACH])), view, color=TIP, linewidth=1.4)
        draw_split(ax, tips * end, view, color=TIP, linewidth=1.0, alpha=0.9)
    bases = screen(heights - velocities / 2, right, up)                       # [heights, 2]
    arrows = screen(velocities, right, up)                                    # [heights, 2]
    ax.quiver(*bases.T, *arrows.T, angles="xy", scale_units="xy", scale=1, color=FLOW, width=0.005, zorder=0)
    ax.imshow(picture, extent=(-limit, limit, -limit, limit), interpolation="bilinear", zorder=2)
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
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
        # One frame for every panel, so that the bodies' sizes compare: the farthest marks past a long axis's tips,
        # and the margin.
        limit = TIP_REACH * (1 + FRAME_MARGIN) * tip.norm().to_array().max()
        shown = pictures(stretches, rotors, limit, *view)                     # [bodies, pixels, pixels, 4]
        figure = plt.figure(figsize=(count * PIXELS / 100, PIXELS / 100), dpi=100)
        for body in range(count):
            draw_body(figure.add_subplot(1, count, body + 1), shown[body], limit, tip[body], path[:, body], heights,
                      velocities, view)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
        images.append(capture(figure))
        plt.close(figure)
    return images
