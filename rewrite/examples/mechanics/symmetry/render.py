"""Drawing for the symmetry example: heat-flow ellipsoids, flywheel moments, lattice responses."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from examples.mechanics.symmetry.core import Point, Scalar, Vector


def draw_conduction(labels: list[str], surfaces: Vector, driving: Vector, fluxes: Vector) -> plt.Figure:
    """Draw the image of the unit input sphere, and one common input/output pair."""
    coordinates = surfaces.cast(Vector.output_subspace).kernel
    flow = fluxes.cast(Vector.output_subspace).kernel
    direction = driving.cast(Vector.output_subspace).kernel
    radius = np.linalg.norm(coordinates, axis=-1).max() * 1.05
    arrow = direction * radius * 0.85
    angles = np.degrees(np.arccos(np.clip(flow @ direction / np.linalg.norm(flow, axis=-1), -1, 1)))

    fig = plt.figure(figsize=(15, 5.6), dpi=140)
    fig.suptitle("What heat flow does crystal symmetry allow?", fontsize=19, y=0.97)
    fig.text(0.5, 0.895, "Each surface is K applied to every unit driving field: a heat-flow ellipsoid.",
             ha="center", fontsize=12, color="#46505a")
    for i, (label, surface, flux, angle) in enumerate(zip(labels, coordinates, flow, angles)):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d", computed_zorder=False)
        ax.plot_surface(*np.moveaxis(surface, -1, 0), color="#438ab0", alpha=0.4,
                        linewidth=0.15, edgecolor="#356781", rstride=2, cstride=2)
        ax.quiver(0, 0, 0, *arrow, color="#262b32", linewidth=2, arrow_length_ratio=0.12, zorder=3)
        ax.quiver(0, 0, 0, *flux, color="#d44e1a", linewidth=3, arrow_length_ratio=0.16, zorder=4)
        ax.set(xlim=(-radius, radius), ylim=(-radius, radius), zlim=(-radius, radius),
               xlabel="flow x", ylabel="flow y", zlabel="flow z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-120)
        ax.set_xticks([-4, 0, 4])
        ax.set_yticks([-4, 0, 4])
        ax.set_zticks([-4, 0, 4])
        ax.tick_params(labelsize=8, pad=0)
        ax.set_title(label, fontsize=11, pad=7)
        ax.text2D(0.5, -0.16, f"Flow deflection: {angle:.1f}°", transform=ax.transAxes,
                  ha="center", fontsize=11, color="#a53c14")
    fig.legend([Line2D([], [], color="#262b32", lw=2), Line2D([], [], color="#d44e1a", lw=3)],
               ["Same driving direction (arrow length arbitrary)", "Resulting heat flow"],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.025, "Symmetry constrains the response: 6 → 3 → 2 → 1 independent components.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.01, right=0.975, bottom=0.22, top=0.78, wspace=0.03)
    return fig


def draw_flywheel(arms: Point, angles: np.ndarray, moments: Scalar) -> plt.Figure:
    """Show the point masses and moment of inertia versus the in-plane axis direction."""
    homogeneous = arms.cast(Point.output_subspace).kernel
    coordinates = homogeneous[..., :3] / homogeneous[..., 3:]
    curves = moments.kernel[..., 0]
    fig = plt.figure(figsize=(11, 5.6), dpi=140)
    fig.suptitle("Threefold shape, axially symmetric inertia", fontsize=18, y=0.97)
    ax = fig.add_subplot(1, 2, 1)
    for points, color in zip(coordinates, ("#d44e1a", "#438ab0", "#438ab0")):
        points = points.reshape(-1, 3)
        ax.scatter(points[:, 0], points[:, 1], s=12, color=color)
    ax.scatter([0], [0], s=30, color="#262b32")
    ax.axhline(0, color="#adb4bb", lw=0.7, zorder=0)
    ax.axvline(0, color="#adb4bb", lw=0.7, zorder=0)
    ax.set(aspect="equal", xlabel="x", ylabel="y", title="Three unit-mass arms, 120° apart")
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(1, 2, 2, projection="polar")
    for curve, color, label in zip(curves, ("#d44e1a", "#438ab0"), ("One arm", "Whole flywheel")):
        ax.plot(angles, curve, color=color, lw=2.5, label=label)
    ax.set_title("Moment about an axis in the wheel's plane", pad=24)
    ax.set_rlabel_position(65)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    fig.text(0.5, 0.025, "Polar angle = axis direction; radius = moment of inertia. All axes pass through the hub.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.07, right=0.94, top=0.79, bottom=0.2, wspace=0.32)
    return fig


def draw_crystal(sites: Point, axial: Point, diagonal: Point,
                 conduction: Vector, stiffness: Vector) -> plt.Figure:
    """Plot the lattice bonds and radial directional responses, each in its own units."""
    fig = plt.figure(figsize=(15, 5.6), dpi=140)
    fig.suptitle("One cubic symmetry group, two different kinds of response", fontsize=18, y=0.97)
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    homogeneous = sites.cast(Point.output_subspace).kernel
    coordinates = homogeneous[..., :3] / homogeneous[..., 3:]
    ax.scatter(*coordinates.T, s=30, color="#7e8b98", alpha=0.6)
    for neighbours, color, label in ((axial, "#d44e1a", "6 axial neighbours"),
                                      (diagonal, "#438ab0", "12 face-diagonal neighbours")):
        homogeneous = neighbours.cast(Point.output_subspace).kernel
        coordinates = homogeneous[..., :3] / homogeneous[..., 3:]
        for end in coordinates:
            ax.plot([0, end[0]], [0, end[1]], [0, end[2]], color=color, lw=1.5)
        ax.scatter(*coordinates.T, s=45, color=color, label=label)
    ax.scatter([0], [0], [0], s=70, color="#262b32")
    ax.set(title="Two seed bonds generate both families", xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=23, azim=-55)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.25), frameon=False, fontsize=10)

    for panel, surface, color, title, caption in (
        (2, conduction, "#438ab0", "Conductivity: isotropic", "Radius = d · K(d) = 2/3 in every direction"),
        (3, stiffness, "#d44e1a", "Elastic stiffness: cubic anisotropy", "Radius = C(d, d, d, d)\n1/2 on axes; 1/3 on body diagonals"),
    ):
        coordinates = surface.cast(Vector.output_subspace).kernel
        limit = np.linalg.norm(coordinates, axis=-1).max() * 1.05
        ax = fig.add_subplot(1, 3, panel, projection="3d")
        ax.plot_surface(*np.moveaxis(coordinates, -1, 0), color=color,
                        linewidth=0.15, edgecolor="#48505a", rstride=1, cstride=1, alpha=0.85)
        ax.set(title=title, xlim=(-limit, limit), ylim=(-limit, limit), zlim=(-limit, limit),
               xlabel="x", ylabel="y", zlabel="z", xticks=[], yticks=[], zticks=[])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=23, azim=-55)
        ax.text2D(0.5, -0.14, caption, transform=ax.transAxes, ha="center", fontsize=10)
    fig.text(0.5, 0.035, "Averaging over the same 24 cube rotations makes conductivity isotropic while preserving cubic elastic anisotropy.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.22, wspace=0.05)
    return fig
