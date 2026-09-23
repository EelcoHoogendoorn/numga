"""Drawing and coordinate readout for the ellipsoid example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.quadrics.core import DualQuadric, Plane, Point, ga, mv, support_plane


def surface(quadric: DualQuadric) -> Point:
    """Sample the ellipsoid by sweeping its tangent-plane normal over a sphere."""
    longitude = np.linspace(0, 2 * np.pi, 65)[None, :]
    latitude = np.linspace(0, np.pi, 33)[:, None]
    normals = (mv.x * (np.sin(latitude) * np.cos(longitude))
               + mv.y * (np.sin(latitude) * np.sin(longitude))
               + mv.z * np.cos(latitude))
    return quadric(support_plane(quadric, normals, mv.w))


def euclidean(points: Point) -> np.ndarray:
    values = points.cast(ga.subspace("yzw zxw xyw zyx")).kernel
    return values[..., :3] / values[..., 3:]


def draw_panels(panels: list) -> plt.Figure:
    """Draw each ellipsoid with a tangent patch and its contact point."""
    fig = plt.figure(figsize=(7 * len(panels), 6), dpi=120, layout="constrained")
    for index, (title, quadric, tangent, contact) in enumerate(panels, start=1):
        ax = fig.add_subplot(1, len(panels), index, projection="3d")
        xyz = euclidean(surface(quadric))
        ax.plot_wireframe(*np.moveaxis(xyz, -1, 0), color="#0284c7", alpha=0.4,
                          rstride=2, cstride=2, linewidth=0.7)
        center = euclidean(quadric(mv.w))
        contact_xyz = euclidean(contact)
        ax.scatter(*center, color="#0284c7", s=40, label="Center Q(infinity)")
        ax.scatter(*contact_xyz, color="#e11d48", s=60, label="Contact Q(tangent)")

        # A basis of the plane's normal complement also handles vertical planes.
        normal = tangent.cast(ga.subspace("x y z w")).kernel[..., :3]
        _, _, frame = np.linalg.svd(normal[None, :])
        extent = np.ptp(xyz.reshape(-1, 3), axis=0).max()
        u = np.linspace(-0.22, 0.22, 2)[:, None] * extent
        v = np.linspace(-0.22, 0.22, 2)[None, :] * extent
        patch = contact_xyz + u[..., None] * frame[1] + v[..., None] * frame[2]
        ax.plot_surface(*np.moveaxis(patch, -1, 0), color="#e11d48", alpha=0.25)

        ax.set(xlabel="x", ylabel="y", zlabel="z", title=title)
        ax.set_box_aspect([1, 1, 1])
        half_width = extent * 0.6
        for coordinate, set_limit in zip(center, (ax.set_xlim, ax.set_ylim, ax.set_zlim)):
            set_limit(coordinate - half_width, coordinate + half_width)
        ax.view_init(elev=23, azim=38)
        ax.legend(loc="upper left", fontsize=9)
    return fig


def draw_polar_reciprocity(dual: DualQuadric, tangent: Plane, contact: Point) -> plt.Figure:
    return draw_panels([("Tangent → contact → tangent", dual, tangent, contact)])


def draw_motor_transport(
    local: DualQuadric, tangent: Plane, contact: Point, world: DualQuadric, world_tangent: Plane, world_contact: Point,
) -> plt.Figure:
    return draw_panels([("Body frame", local, tangent, contact),
                        ("World frame: motor >> Q(motor << Plane)", world, world_tangent, world_contact)])
