"""Constructors, scene geometry and rendering for the projective camera example in PGA3D."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

# ---------------------------------------------------------------------------
# 1. PGA3D Setup
# ---------------------------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector

# Whole-extensor types (GATypes):
Point = ga.gatype.antivector()
Line = ga.gatype.bivector()
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Pseudoscalar = ga.gatype.pseudoscalar()


# ---------------------------------------------------------------------------
# 2. Euclidean Constructors and Motors
# ---------------------------------------------------------------------------
def point(coords: np.ndarray) -> Point:
    """Construct affine points (x, y, z, 1) from an (..., 3) coordinate array."""
    return mv.antivector(np.concatenate([coords, np.ones_like(coords[..., :1])], axis=-1))


def direction(coords: np.ndarray) -> Point:
    """Construct ideal points (x, y, z, 0) from an (..., 3) direction array."""
    return mv.antivector(np.concatenate([coords, np.zeros_like(coords[..., :1])], axis=-1))


def plane(normal: np.ndarray, offset: float) -> Plane:
    """Construct the plane normal · x = offset."""
    return mv.vector(np.concatenate([normal, -offset * np.ones_like(normal[..., :1])], axis=-1))


origin: Point = point(np.zeros(3))
plane_at_infinity: Plane = mv.vector(np.array([0.0, 0.0, 0.0, 1.0]))


def translator(displacement: np.ndarray) -> Motor:
    """Construct the motor translating by the given displacement."""
    generator = plane_at_infinity.wedge(direction(displacement).dual())
    return (generator * 0.5).exp()


def rotator(axis: Line, angle: float) -> Motor:
    """Construct the motor rotating by angle about a unit axis line."""
    return (axis * (angle / 2.0)).exp()


def screen_coordinates(motor: Motor, image: Point) -> np.ndarray:
    """Read (x, y) screen coordinates of image points on a rig moved by motor."""
    local = (motor << image).cast(Point.output_subspace).kernel
    return local[..., :2] / local[..., 3:]


# ---------------------------------------------------------------------------
# 3. Scene Geometry
# ---------------------------------------------------------------------------
CUBE_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def circle(n: int) -> Point:
    """Construct n points around the unit circle in the xy plane, centred on the origin."""
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return point(np.stack([np.cos(t), np.sin(t), np.zeros_like(t)], axis=-1))


def cube(size: float) -> Point:
    """Construct the eight corner points of an axis-aligned cube centred on the origin."""
    corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
    return point(corners * (size / 2.0))


# ---------------------------------------------------------------------------
# 4. Rendering
# ---------------------------------------------------------------------------
def euclidean(points: Point) -> np.ndarray:
    """Dehomogenize points to (..., 3) coordinates, cast into the layout Point declares."""
    k = points.cast(Point.output_subspace).kernel
    return k[..., :3] / k[..., 3:]


def draw_edges(ax, coords: np.ndarray, color: str, **kwargs) -> None:
    """Draw cube edges through 2D or 3D coordinates."""
    for a, b in CUBE_EDGES:
        ax.plot(*zip(coords[a], coords[b]), color=color, **kwargs)


def render_shadow_scene(
    ax,
    body: Point,
    point_light: Point,
    sun: Point,
    point_shadow: Point,
    sun_shadow: Point,
    shadow_trail: Point,
) -> None:
    """Draw the body, both lights, both shadows, and one corner's shadow trail in 3D."""
    draw_edges(ax, euclidean(body), "#38bdf8", linewidth=2.0)
    draw_edges(ax, euclidean(point_shadow), "#fbbf24", linewidth=1.5)
    draw_edges(ax, euclidean(sun_shadow), "#a855f7", linewidth=1.5, linestyle="--")
    trail = euclidean(shadow_trail)
    ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], color="#fbbf24", linewidth=1.0, linestyle=":", label="corner shadow as light moves")
    ax.scatter(*euclidean(point_light), color="#fbbf24", s=60, label="point light")
    sun_dir = sun.kernel[:3] / np.linalg.norm(sun.kernel[:3])
    ax.quiver(-2.0, 2.0, 4.0, *sun_dir, length=1.0, color="#a855f7", label="sun direction")
    ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_zlim(0, 4.5)
    ax.set_box_aspect([1, 1, 0.75])
    ax.set_title("Shadows: (light ∨ point) ∧ ground")
    ax.legend(loc="upper left", fontsize=8)


def screen_line_endpoints(rig: Motor, lines: Line, half_width: float) -> np.ndarray:
    """Clip lines on a rig's screen to x = ±half_width, returning (..., 2, 2) screen coordinates."""
    ends = []
    for x in (-half_width, half_width):
        ends.append(screen_coordinates(rig, lines.wedge(rig >> plane(np.array([1.0, 0.0, 0.0]), x))))
    return np.stack(ends, axis=-2)


def render_stereo_scene(
    ax_1, ax_2,
    rig_1: Motor, rig_2: Motor,
    image_1: Point, image_2: Point,
    epipolar_lines_2: Line,
    half_width: float = 0.5,
) -> None:
    """Draw both images; camera 2 also shows the epipolar lines of camera 1's corners."""
    px_1 = screen_coordinates(rig_1, image_1)
    px_2 = screen_coordinates(rig_2, image_2)
    draw_edges(ax_1, px_1, "#38bdf8", linewidth=2.0)
    draw_edges(ax_2, px_2, "#38bdf8", linewidth=2.0)
    for (a, b) in screen_line_endpoints(rig_2, epipolar_lines_2, half_width):
        ax_2.plot([a[0], b[0]], [a[1], b[1]], color="#f43f5e", linewidth=0.8, alpha=0.8)
    ax_2.scatter(px_2[:, 0], px_2[:, 1], color="#f43f5e", s=18, zorder=3)
    for ax, title in ((ax_1, "Camera 1"), (ax_2, "Camera 2 with epipolar lines of camera 1")):
        ax.set_xlim(-half_width, half_width); ax.set_ylim(-half_width, half_width)
        ax.set_aspect("equal"); ax.set_title(title)
        ax.grid(True, alpha=0.3)


def draw_projection(body, point_light, sun, point_shadow, sun_shadow, shadow_trail, rig_1, rig_2, image_1, image_2, epipolar_lines_2, plot_path) -> plt.Figure:
    fig = plt.figure(figsize=(16, 5), dpi=120)
    render_shadow_scene(fig.add_subplot(1, 3, 1, projection="3d"), body, point_light, sun, point_shadow, sun_shadow, shadow_trail)
    render_stereo_scene(fig.add_subplot(1, 3, 2), fig.add_subplot(1, 3, 3), rig_1, rig_2, image_1, image_2, epipolar_lines_2)
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")
    return fig
