"""Four panels: the three poses of the second ellipse, and the determinant curves of the blend."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quadrics.quadric_collision.core import (
    Line, Point, Polarity, Quadric, Scalar, ga, infinity, mv, tangent_line,
)

XLIM, YLIM = (-3.8, 4.2), (-3.0, 3.0)
# The grid the separating hyperbola is evaluated on, per side, and the tangent normals an outline is swept over.
RESOLUTION = 250
N_OUTLINE = 150
FIRST_COLOURS, SECOND_COLOURS = ("#3b82f6", "#1d4ed8"), ("#f97316", "#ea580c")
POSE_COLOURS = ("#16a34a", "#ea580c", "#dc2626")


# --- plumbing -------------------------------------------------------------------------
def xy(points: Point) -> np.ndarray:
    """Coordinates of points, `[..., 2]`: their pairings with the lines x = 0 and y = 0, over their weights."""
    return ((stack((mv.x, mv.y)) & points[..., None]) / (infinity & points)[..., None]).to_array()


def outline(dual: Quadric) -> np.ndarray:
    """Coordinates of the ellipse's contact points, swept over the tangent normals."""
    theta = np.linspace(0, 2 * np.pi, N_OUTLINE)
    return xy(dual(tangent_line(dual, mv.x * np.cos(theta) + mv.y * np.sin(theta))))


def draw_line(ax, line: Line, **style) -> None:
    """Draw the line `mv.x * a + mv.y * b + mv.w * c` through its foot from the origin, along its direction."""
    a, b, c = line.cast(ga.subspace("x y w")).kernel
    foot = -np.array([a, b]) * c / (a * a + b * b)
    ax.axline(foot, foot + np.array([-b, a]), zorder=4, **style)


def draw_ellipses(ax, fixed: Quadric, moved: Quadric) -> None:
    """Both ellipses, filled, with their centres."""
    for dual, (fill, edge) in ((fixed, FIRST_COLOURS), (moved, SECOND_COLOURS)):
        points = outline(dual)
        ax.fill(points[:, 0], points[:, 1], color=fill, alpha=0.35)
        ax.plot(points[:, 0], points[:, 1], color=edge, linewidth=2.2)
        ax.scatter(*xy(dual(infinity)), color=edge, s=40, zorder=5)
    ax.set(xlim=XLIM, ylim=YLIM, aspect="equal")
    ax.set_axis_off()


def draw_witnesses(ax, fixed: Quadric, moved: Quadric, first: Line, second: Line, midline: Line, midline_colour: str) -> None:
    """The tangent lines of both ellipses along the normal, their midline, and their contact points."""
    draw_line(ax, first, color=FIRST_COLOURS[1], linestyle="--", linewidth=1.5)
    draw_line(ax, second, color=SECOND_COLOURS[1], linestyle="--", linewidth=1.5)
    draw_line(ax, midline, color=midline_colour, linestyle="-", linewidth=2.0)
    near, far = xy(fixed(first)), xy(moved(second))
    ax.scatter(*near, color=FIRST_COLOURS[1], s=60, zorder=6)
    ax.scatter(*far, color=SECOND_COLOURS[1], s=60, zorder=6)
    ax.plot([near[0], far[0]], [near[1], far[1]], "k:", linewidth=1.8, zorder=5)


def draw_collision(
    fixed: Quadric, moved: Quadric, parameter: Scalar, maximum: Scalar, shares: Scalar, determinants: Scalar,
    first: Line, second: Line, midline: Line, separating: Polarity, deforming: Quadric, bridging: Quadric,
    contact_line: Line, contact_point: Point,
) -> plt.Figure:
    """The pair apart with the separating hyperbola, touching with a blended ellipse and the shared tangent,
    overlapping with the bridging ellipse; and the blend's determinant against its share, for each pose."""
    fig, ((ax_apart, ax_touch), (ax_over, ax_det)) = plt.subplots(2, 2, figsize=(12, 10), dpi=140)

    draw_ellipses(ax_apart, fixed, moved[0])
    xs, ys = np.meshgrid(np.linspace(*XLIM, RESOLUTION), np.linspace(*YLIM, RESOLUTION))
    grid = mv.point(np.stack([xs, ys], axis=-1))
    field = (grid & separating(grid)).to_array()
    ax_apart.contour(xs, ys, field, levels=[0.0], colors=["#059669"], linestyles=["-."], linewidths=1.6)
    ax_apart.contourf(xs, ys, field, levels=[0.0, field.max()], colors=["#10b981"], alpha=0.10)
    draw_witnesses(ax_apart, fixed, moved[0], first, second[0], midline[0], POSE_COLOURS[0])

    draw_ellipses(ax_touch, fixed, moved[1])
    middle = outline(deforming)
    ax_touch.fill(middle[:, 0], middle[:, 1], color="#a855f7", alpha=0.18)
    ax_touch.plot(middle[:, 0], middle[:, 1], color="#9333ea", linestyle=":", linewidth=1.5, alpha=0.7)
    draw_line(ax_touch, contact_line, color=POSE_COLOURS[2], linestyle="-", linewidth=2.4)
    ax_touch.scatter(*xy(contact_point), color=POSE_COLOURS[2], s=110, zorder=8)

    draw_ellipses(ax_over, fixed, moved[2])
    bridge = outline(bridging)
    ax_over.fill(bridge[:, 0], bridge[:, 1], color="#a855f7", alpha=0.22)
    ax_over.plot(bridge[:, 0], bridge[:, 1], color="#7c3aed", linestyle="-.", linewidth=1.8)
    draw_witnesses(ax_over, fixed, moved[2], first, second[2], midline[2], "#9333ea")

    share, peak = parameter.to_array(), maximum.to_array()
    for index, colour in enumerate(POSE_COLOURS):
        ax_det.plot(shares.to_array(), determinants.to_array()[:, index], color=colour, linewidth=2.4)
        ax_det.scatter([share[index]], [peak[index]], color=colour, s=70, zorder=6)
    ax_det.axhline(0.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
    # Only the sign matters: the determinant's zero across.
    ax_det.set(xlim=(0.0, 1.0), xticks=[], yticks=[], xlabel="share of the blend", ylabel="determinant")
    ax_det.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
