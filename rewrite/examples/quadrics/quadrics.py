"""Dual Quadrics, Bounding Ellipsoids, and Motor Transformations in PGA3D.

Geometric Concept: Primal vs. Dual Quadrics
-------------------------------------------
In Euclidean geometry, an ellipsoid is usually described as a locus of points
(a "primal quadric"):
    X^T C X = 0,   where C = diag(1/a^2, 1/b^2, 1/c^2, -1) in homogeneous coordinates.

In projective geometry and 3D PGA, geometry is dual: planes are primitive
elements (vectors, grade 1) and points are dual (antivectors, grade 3).
A "dual quadric" describes the exact same ellipsoid as an envelope of its
tangent planes:
    pi^T Q pi = 0   <==>   pi v Q(pi) == 0,   where Q = C^{-1} = diag(a^2, b^2, c^2, -1).

Here Q is an arity-1 linear Extensor mapping Planes -> Points (vector -> antivector):
1. Polar Reciprocity: For any plane pi, p = Q(pi) is the pole of that plane.
   If pi is tangent to the ellipsoid, Q(pi) is the exact contact point!
2. Tangency Condition: pi is tangent iff the contact point lies on the plane:
   pi v Q(pi) == 0.
3. Center Extraction: The pole of the plane at infinity (pi_inf = w) is the
   ellipsoid center: center = -Q(w).

The Clean Geometric Formula:
----------------------------
In PGA, directions are ideal points (antivectors with zero projective weight, w=0),
while the center is an affine point (antivector with weight w=1).
An ellipsoid is fundamentally defined by its principal directions and center:
    Q = Sigma - c (x) c
where:
    Sigma = sum(r_i^2 * (v_i * V.regressive(v_i)))
is the pure directional shape tensor (built from ideal points v_i), and c (x) c
is the projective center offset.
- v_i * V.regressive(v_i) measures the squared projection of planes along axis i.
- Subtracting c (x) c introduces the homogeneous projective offset (-1).

Why Should You Care? (Robotics, Physics, & Computer Vision)
-----------------------------------------------------------
1. Instantaneous Collision Detection (GJK & Support Functions):
   Given an incoming contact plane normal n, finding the bounding support plane
   and contact point on a mesh or primal quadric requires numerical optimization.
   On a dual quadric, it is a closed-form, instantaneous evaluation:
       support distance:  d = sqrt(n v Q(n))
       contact point:     p = Q(n - d * w)
2. Eliminating T^{-T} Transpose-Inverses in Rigid Body Transformations:
   In classical matrix algebra, transforming a quadric requires keeping track of
   dual representations: points transform by T, planes by T^{-T}, and quadrics
   by T Q T^T.
   In PGA with Numga Extensors, Q transforms coordinate-free in a single sandwich:
       Q_world = motor >> Q_body(motor << V)
   The pullback on planes (motor << V) and pushforward on points (motor >> ...)
   are handled uniformly and automatically across grades.
3. 3D Bounding Volumes in SLAM & Robotics:
   Point cloud sample covariances E[(p - c)(p - c)^T] map directly into dual
   quadric bounding ellipsoids, enabling analytical tracking of 3D object envelopes.
"""

import sys
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D

ctx = NumpyContext(PGA3D)
mv = ctx.multivector
V = PGA3D.subspace.vector()
P = PGA3D.subspace.antivector()

# 1. Principal directions (ideal points in PGA) and center point:
# In PGA, directions are points at infinity (antivectors with zero projective weight).
vx = mv.antivector([1, 0, 0, 0])
vy = mv.antivector([0, 1, 0, 0])
vz = mv.antivector([0, 0, 1, 0])
origin = mv.antivector([0, 0, 0, 1])

# Shape tensor Sigma = sum(r_i^2 * (v_i (x) v_i)) and center operator:
# The dual quadric envelope of an ellipsoid centered at c with semi-axes r is:
#     Q = Sigma - c (x) c
# which yields the diagonal kernel diag(rx^2, ry^2, rz^2, -1).
rx, ry, rz = 3.0, 2.0, 1.0
Sigma = (vx * V.regressive(vx)) * rx**2 + (vy * V.regressive(vy)) * ry**2 + (vz * V.regressive(vz)) * rz**2
Q_body = Sigma - origin * V.regressive(origin)
assert Q_body.arity == 1
assert Q_body.axes == (P, V)

# 2. General rigid motor: rotation (30° in xy-plane) + translation (dx=1, dy=2, dz=3)
rotor = (mv.xy * (-np.pi / 12.0)).exp()
translator = (mv.xw * 0.5 + mv.yw * 1.0 + mv.zw * 1.5).exp()
motor = translator * rotor

# 3. Transform dual quadric via inline GA sandwich:
# motor << V pulls back planes via M^{-1}, Q_body maps planes to points, motor >> (...) pushes forward points via M
Q_world = motor >> Q_body(motor << V)

# 4. Form the two distinct 4x4 transformation matrices on planes and points:
# (Planes and points are dual grades 1 and 3; unlike 4D bivectors, they require separate matrices)
M_planes = motor >> V      # 4x4 matrix on planes (equivalent to T^{-T})
M_points = motor >> P      # 4x4 matrix on points (equivalent to T)

# Transform via two-matrix operator composition:
Q_world_matrices = M_points(Q_body(M_planes.inverse()))
np.testing.assert_allclose(Q_world.kernel, Q_world_matrices.kernel, atol=1e-10)

# Or using exact GA pullback (avoiding numerical matrix inversion):
Q_world_pullback = M_points(Q_body(motor << V))
np.testing.assert_allclose(Q_world.kernel, Q_world_pullback.kernel, atol=1e-14)

# 5. Verify exact equivalence with transforming the generating directions and center:
rvx, rvy, rvz = motor >> vx, motor >> vy, motor >> vz
rorigin = motor >> origin
Q_direct = (
    (rvx * V.regressive(rvx)) * rx**2 +
    (rvy * V.regressive(rvy)) * ry**2 +
    (rvz * V.regressive(rvz)) * rz**2 -
    (rorigin * V.regressive(rorigin))
)
np.testing.assert_allclose(Q_world.kernel, Q_direct.kernel, atol=1e-14)

# 6. Quadric center is the pole of the plane at infinity (pi_inf = w):
center_body = -Q_body(mv.w)
center_world = -Q_world(mv.w)
expected_center = motor >> center_body
np.testing.assert_allclose(center_world.kernel, expected_center.kernel, atol=1e-14)

# 7. Support planes and contact points (polar reciprocity):
# For any normal direction n, the support distance from the center is sqrt(n v Q(n)):
n_dir = mv.vector([1.0, 2.0, 3.0, 0.0]).normalized()
dist_n = np.sqrt(float(n_dir.regressive(Q_body(n_dir)).kernel.item()))
pi_body = n_dir - mv.w * dist_n
contact_body = Q_body(pi_body)

# Tangency condition in body frame: contact point lies on the tangent plane (pi v p == 0)
tangency_body = pi_body.regressive(contact_body)
np.testing.assert_allclose(tangency_body.kernel, 0.0, atol=1e-12)

# Tangency condition in world frame under transformed quadric:
pi_world = motor >> pi_body
contact_world = Q_world(pi_world)
tangency_world = pi_world.regressive(contact_world)
np.testing.assert_allclose(tangency_world.kernel, 0.0, atol=1e-12)

# Contact point transforms covariantly:
expected_contact = motor >> contact_body.normalized()
np.testing.assert_allclose(contact_world.normalized().kernel, expected_contact.kernel, atol=1e-12)


def plot_quadrics():
    """Plot 3D visualization of the ellipsoid, tangent plane, and contact point in body and world frames."""
    import matplotlib.pyplot as plt

    u = np.linspace(0, 2 * np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    U, V_grid = np.meshgrid(u, v)
    xb = rx * np.sin(V_grid) * np.cos(U)
    yb = ry * np.sin(V_grid) * np.sin(U)
    zb = rz * np.cos(V_grid)

    pts_b = mv.antivector(np.stack([xb.ravel(), yb.ravel(), zb.ravel(), np.ones(xb.size)], axis=-1))
    pts_w = (motor >> pts_b).kernel
    xw, yw, zw = pts_w[:, 0].reshape(xb.shape), pts_w[:, 1].reshape(xb.shape), pts_w[:, 2].reshape(xb.shape)

    fig = plt.figure(figsize=(14, 6), dpi=120)

    # Body frame
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_wireframe(xb, yb, zb, color="royalblue", alpha=0.35, linewidth=0.8)
    ax1.scatter([0], [0], [0], color="blue", s=60, label="Center (0,0,0)")
    cb = contact_body.normalized().kernel[:3]
    ax1.scatter([cb[0]], [cb[1]], [cb[2]], color="crimson", s=70, zorder=5, label="Contact Point Q(pi)")

    px, py = np.meshgrid(np.linspace(cb[0] - 1.5, cb[0] + 1.5, 10), np.linspace(cb[1] - 1.5, cb[1] + 1.5, 10))
    nk = n_dir.kernel[:3]
    pz = (dist_n - nk[0] * px - nk[1] * py) / nk[2]
    ax1.plot_surface(px, py, pz, color="crimson", alpha=0.25)
    ax1.set_title("Body Frame: Ellipsoid & Tangent Plane", fontsize=12, fontweight="bold")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(loc="upper left")

    # World frame
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_wireframe(xw, yw, zw, color="forestgreen", alpha=0.35, linewidth=0.8)
    cw = center_world.kernel[:3]
    ax2.scatter([cw[0]], [cw[1]], [cw[2]], color="green", s=60, label=f"Center ({cw[0]:.1f}, {cw[1]:.1f}, {cw[2]:.1f})")
    c_pt = contact_world.normalized().kernel[:3]
    ax2.scatter([c_pt[0]], [c_pt[1]], [c_pt[2]], color="crimson", s=70, zorder=5, label="Contact Point Q(pi)")

    pw_x, pw_y = np.meshgrid(np.linspace(c_pt[0] - 1.5, c_pt[0] + 1.5, 10), np.linspace(c_pt[1] - 1.5, c_pt[1] + 1.5, 10))
    pik = pi_world.kernel
    pw_z = (-pik[3] - pik[0] * pw_x - pik[1] * pw_y) / pik[2]
    ax2.plot_surface(pw_x, pw_y, pw_z, color="crimson", alpha=0.25)
    ax2.set_title("World Frame: Motor-Transformed M >> Q(M << V)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend(loc="upper left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Body-frame Dual Quadric (4x4, planes -> points):\n", np.around(Q_body.kernel, 2))
    print("\nWorld-frame Dual Quadric via Inline Sandwich (4x4):\n", np.around(Q_world.kernel, 2))
    print("\nEllipsoid Center (Body):", center_body.kernel)
    print("Ellipsoid Center (World):", np.around(center_world.kernel, 2))
    print(f"\nTangent Plane (World): normal={np.around(pi_world.kernel[:3], 3)}, d={pi_world.kernel[3]:.3f}")
    print("Contact Point (World):", np.around(contact_world.normalized().kernel, 3))
    print(f"Tangency Residual: {float(tangency_world.kernel.item()):.2e}")

    if "--plot" in sys.argv:
        plot_quadrics()
