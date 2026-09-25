"""Four panels: the three poses of the second ellipse, and the determinant curves of the blend."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.quadric_collision.core import (
    Line, Point, Polarity, Quadric, Scalar, ga, infinity, mv, tangent_line,
)

XLIM, YLIM = (-3.8, 3.8), (-3.0, 3.0)


def xy(points: Point) -> np.ndarray:
    """Euclidean coordinates (..., 2) of points."""
    k = points.cast(ga.subspace("yw wx xy")).kernel
    return k[..., :2] / k[..., 2:]


def outline(Q: Quadric) -> np.ndarray:
    """Coordinates of the ellipse's contact points, swept over the tangent normals."""
    theta = np.linspace(0, 2 * np.pi, 150)
    return xy(Q(tangent_line(Q, mv.x * np.cos(theta) + mv.y * np.sin(theta))))


def draw_line(ax, line: Line, **style) -> None:
    """Draw the line mv.x * a + mv.y * b + mv.w * c through its foot from the origin, along its
    direction."""
    a, b, c = line.cast(ga.subspace("x y w")).kernel
    foot = -np.array([a, b]) * c / (a * a + b * b)
    ax.axline(foot, foot + np.array([-b, a]), zorder=4, **style)


def draw_ellipses(ax, Q1: Quadric, Q2: Quadric, title: str) -> None:
    for Q, fill, edge, label in ((Q1, "#3b82f6", "#1d4ed8", "Ellipse Q1"), (Q2, "#f97316", "#ea580c", "Ellipse Q2")):
        points = outline(Q)
        ax.fill(points[:, 0], points[:, 1], color=fill, alpha=0.35, label=label)
        ax.plot(points[:, 0], points[:, 1], color=edge, linewidth=2.2)
        ax.scatter(*xy(Q(infinity)), color=edge, s=40, zorder=5)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11, fontweight="bold")


def draw_witnesses(ax, Q1: Quadric, Q2: Quadric, first: Line, second: Line, midline: Line,
                   midline_color: str, midline_label: str, which: str) -> None:
    """The tangent lines of both ellipses along the normal, their midline, and their contact points."""
    draw_line(ax, first, color="#2563eb", linestyle="--", linewidth=1.5, label="Tangent Line on Q1 (L1)")
    draw_line(ax, second, color="#ea580c", linestyle="--", linewidth=1.5, label="Tangent Line on Q2 (L2)")
    draw_line(ax, midline, color=midline_color, linestyle="-", linewidth=2.0, label=midline_label)
    p1, p2 = xy(Q1(first)), xy(Q2(second))
    ax.scatter(*p1, color="#2563eb", s=60, zorder=6, label=f"{which} on Q1 ({p1[0]:.2f}, {p1[1]:.2f})")
    ax.scatter(*p2, color="#ea580c", s=60, zorder=6, label=f"{which} on Q2 ({p2[0]:.2f}, {p2[1]:.2f})")
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "k:", linewidth=1.8, zorder=5)


def draw_collision(
    Q1: Quadric, Q2: Quadric, parameter: Scalar, maximum: Scalar, sweep: Scalar, determinants: Scalar,
    first: Line, second: Line, midline: Line, separating: Polarity, deforming: Quadric, bridging: Quadric,
    contact_line: Line, contact_point: Point,
) -> plt.Figure:
    lam, peak = parameter.to_array(), maximum.to_array()
    fig, ((ax_sep, ax_touch), (ax_over, ax_det)) = plt.subplots(2, 2, figsize=(15, 12), dpi=160)

    draw_ellipses(ax_sep, Q1, Q2[0], f"State 1: Separated Ellipses\nmax det(Q(λ)) = {peak[0]:+.3f} > 0")
    # The blend at the peak is a hyperbola separating the two ellipses; its form has one positive
    # and two negative eigenvalues.
    resolution = 250
    xs, ys = np.linspace(*XLIM, resolution), np.linspace(*YLIM, resolution)
    X, Y = np.meshgrid(xs, ys)
    grid = mv.yw * X + mv.wx * Y + mv.xy
    field = (grid & separating(grid)).to_array()
    ax_sep.contour(X, Y, field, levels=[0.0], colors=["#059669"], linestyles=["-."], linewidths=1.6)
    ax_sep.contourf(X, Y, field, levels=[0.0, field.max()], colors=["#10b981"], alpha=0.10)
    draw_witnesses(ax_sep, Q1, Q2[0], first, second[0], midline[0], "#16a34a", "Separating Plane (Midplane)", "Closest")
    ax_sep.plot([], [], color="#059669", linestyle="-.", linewidth=1.6, label="Hyperbola Q(λ*) (det>0)")
    ax_sep.legend(loc="lower left", fontsize=7.5)

    draw_ellipses(ax_touch, Q1, Q2[1], f"State 2: Touching Ellipses (Exact Contact)\nmax det(Q(λ)) = {peak[1]:+.1e} ≈ 0 (at λ* = {lam[1]:.3f})")
    # An intermediate blended ellipse, deforming from one ellipse towards the other.
    middle = outline(deforming)
    ax_touch.fill(middle[:, 0], middle[:, 1], color="#a855f7", alpha=0.18, label="Blended Ellipse Q(0.45)")
    ax_touch.plot(middle[:, 0], middle[:, 1], color="#9333ea", linestyle=":", linewidth=1.5, alpha=0.7)
    draw_line(ax_touch, contact_line, color="#dc2626", linestyle="-", linewidth=2.4, label="Unique Shared Tangent Line L*")
    contact = xy(contact_point)
    ax_touch.scatter(*contact, color="#dc2626", s=110, zorder=8, label=f"Contact Point p* ({contact[0]:.2f}, {contact[1]:.2f})")
    ax_touch.annotate(
        f"Contact p*\n({contact[0]:.2f}, {contact[1]:.2f})", (contact[0] + 0.15, contact[1] + 0.25),
        fontsize=9.5, fontweight="bold", color="#b91c1c",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5),
    )
    ax_touch.legend(loc="lower left", fontsize=7.5)

    draw_ellipses(ax_over, Q1, Q2[2], f"State 3: Overlapping Ellipses\nmax det(Q(λ)) = {peak[2]:+.3f} < 0 (Penetration)")
    # The blend at the peak, an ellipse bridging the intersection.
    bridge = outline(bridging)
    ax_over.fill(bridge[:, 0], bridge[:, 1], color="#a855f7", alpha=0.22, label="Interpolated Ellipse Q(λ*)")
    ax_over.plot(bridge[:, 0], bridge[:, 1], color="#7c3aed", linestyle="-.", linewidth=1.8)
    draw_witnesses(ax_over, Q1, Q2[2], first, second[2], midline[2], "#9333ea", "Contact Plane (Midplane)", "Deepest")
    ax_over.legend(loc="lower left", fontsize=7.5)

    lams, dets = sweep.to_array(), determinants.to_array()
    labels = (f"Separated: max = {peak[0]:+.3f} > 0", f"Touching: max = {peak[1]:+.1e} ≈ 0 (at λ*={lam[1]:.3f})",
              f"Overlapping: max = {peak[2]:+.3f} < 0")
    for index, (color, label) in enumerate(zip(("#16a34a", "#ea580c", "#dc2626"), labels)):
        ax_det.plot(lams, dets[:, index], color=color, linewidth=2.4, label=label)
        ax_det.scatter([lam[index]], [peak[index]], color=color, s=70, zorder=6)
    ax_det.axhline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Collision Threshold: det(Q(λ)) = 0")
    ax_det.set_xlim(0.0, 1.0)
    ax_det.set_xlabel("Interpolation Parameter λ", fontsize=10, fontweight="bold")
    ax_det.set_ylabel("Determinant det(Q(λ))", fontsize=10, fontweight="bold")
    ax_det.grid(True, alpha=0.3)
    ax_det.set_title("Determinant of the Blend det(Q(λ))\nBinary Classification: > 0 Separated | = 0 Contact | < 0 Overlap", fontsize=11, fontweight="bold")
    ax_det.legend(loc="lower center", fontsize=9)

    fig.tight_layout()
    return fig
