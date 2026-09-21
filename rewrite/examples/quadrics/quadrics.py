"""Ellipsoids as plane-to-point maps in PGA3D.

A dual quadric maps each tangent plane to its contact point. Its inverse maps
the contact point back to the tangent plane. Moving the whole ellipsoid means
pulling planes into its body frame and pushing the resulting points back out.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, NumpyContext
from numga.algebras import PGA3D

from examples import PLOT_DIR


# --- scenario algebra ------------------------------------------------------
ga = PGA3D
ctx = NumpyContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Plane = ga.gatype.vector()
DualQuadric = ga.gatype((Point, Plane))


# --- math ------------------------------------------------------------------
def support_plane(quadric: DualQuadric, normal: Plane, infinity: Plane) -> Plane:
    """Find an ellipsoid's tangent plane facing a given normal.

    quadric maps planes to points; infinity selects the affine chart.
    The normal's offset is discarded, retaining only its orientation.
    """
    normal = normal.normalized()
    center = quadric(infinity)                       # pole of the plane at infinity
    through_center = normal - infinity * ((normal & center) / (infinity & center))

    # Shift the plane until it contains its own pole: tangent & Q(tangent) = 0.
    # Dividing by the center's weight makes the distance independent of Q's scale.
    radius = (-(through_center & quadric(through_center)) / (infinity & center)).square_root()
    return through_center - infinity * radius


# --- plumbing: sampling and coordinate readout -----------------------------
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


# --- plotting --------------------------------------------------------------
def draw(panels: list[tuple[str, DualQuadric, Plane, Point]], path: Path) -> plt.Figure:
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
        normal = tangent.cast(ga.subspace("x y z w")).kernel[:3]
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

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"Figure saved to {path}")
    return fig


# --- scenarios -------------------------------------------------------------
def polar_reciprocity() -> plt.Figure:
    """Construct an ellipsoid, then map tangent -> contact -> tangent."""
    axes = Extensor.stack([mv.yzw * 3, mv.zxw * 2, mv.xyw])
    center = mv.zyx

    # Ideal points encode the semi-axes. Their dyads give the directional shape;
    # subtracting the center dyad makes a closed envelope of tangent planes.
    dual = (axes * (Plane & axes)).sum(axis=0) - center * (Plane & center)
    tangent = support_plane(dual, mv.x + mv.y * 2 + mv.z * 3, mv.w)
    contact = dual(tangent)                          # Point <- Plane

    primal = dual.inverse()                         # Plane <- Point
    recovered_tangent = primal(contact)
    # Incidence is reciprocal: tangent & contact = contact & primal(contact) = 0.
    return draw([("Tangent → contact → tangent", dual, recovered_tangent, contact)],
                PLOT_DIR / "quadrics_polar_reciprocity.png")


def motor_transport() -> plt.Figure:
    """Move an ellipsoid and its tangent together by transforming the map."""
    axes = Extensor.stack([mv.yzw * 3, mv.zxw * 2, mv.xyw])
    center = mv.zyx
    motor = (mv.xw * 0.5 + mv.yw + mv.zw * 1.5).exp() * (mv.xy * (-np.pi / 12)).exp()

    body = (axes * (Plane & axes)).sum(axis=0) - center * (Plane & center)
    tangent = support_plane(body, mv.x + mv.y * 2 + mv.z * 3, mv.w)
    contact = body(tangent)

    # Pull the input plane into the body frame; push the output point into world.
    world = motor >> body(motor << Plane)
    world_tangent = motor >> tangent
    world_contact = world(world_tangent)             # same point as motor >> contact
    return draw([("Body frame", body, tangent, contact),
                 ("World frame: motor >> Q(motor << Plane)", world, world_tangent, world_contact)],
                PLOT_DIR / "quadrics_motor_transport.png")


if __name__ == "__main__":
    polar_reciprocity()
    motor_transport()
    plt.show()
