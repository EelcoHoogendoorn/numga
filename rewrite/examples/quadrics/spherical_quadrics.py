"""Spherical Quadrics and Spherical Conics in Cl(3) (3D Euclidean Geometric Algebra).

Demonstrates:
1. Quadratic forms in Cl(3) as symmetric extensors Q: V -> V (3x3 operator).
2. Locus Q(v) . v = 0 defines a 3D quadratic cone with apex at the origin.
3. Intersection with the unit sphere S^2 = {v in V | |v| = 1}:
       - Forms closed, symmetric spherical ovals known as spherical ellipses / spherical conics.
       - Exhibits spherical focal property: dist_{S^2}(P, F1) + dist_{S^2}(P, F2) = 2 * theta_a.
4. Plane-based (dual) interpretation:
       - Each vector n in S^2 represents a great circle (plane n . x = 0).
       - Tangent great circles satisfy the dual quadric equation: n . Q^{-1}(n) = 0.
       - The spherical oval is the envelope of its tangent great circles!
       - Dual focal property: product of sines of distances from foci to tangent great circles is constant.
5. Physical connection:
       - Poinsot's polhode curves in rigid body dynamics: intersection of the inertia quadric with the angular momentum sphere.
"""

import numpy as np

from numga import NumpyContext
from numga.algebra import Algebra
from examples import PLOT_DIR

# 1. Dimension-agnostic plane-based PGA setup for Cl(3):
# Planes are Grade 1 (vectors). Points are Antivectors (Grade d-1 = 2).
# Using custom cyclic subspace ordering so Points[i] is naturally dual to Planes[i]:
#   Planes:  x,   y,   z      (Grade 1)
#   Points: yz,  zx,  xy      (Grade 2 antivectors: x*, y*, z*)
ga = Algebra("x+y+z+")
ctx = NumpyContext(ga)
mv = ctx.multivector
spaces = ga.subspace

Planes = spaces("x y z")
Points = spaces("yz zx xy")
Scalar = ga.gatype.scalar()
Polarity = ga.gatype((Planes, Points))

def plot_spherical_quadric(P, dist_F1, dist_F2, pi_tangent, x0, y0, rotor, F1, F2, theta_a, C, save_path: str):
    """Render a 3-panel visualization of the spherical quadric in Cl(3)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure(figsize=(18, 6), dpi=140)

    # Parametric unit sphere
    phi = np.linspace(0, 2 * np.pi, 60)
    theta = np.linspace(0, np.pi, 30)
    Phi, Theta = np.meshgrid(phi, theta)
    Sx = np.sin(Theta) * np.cos(Phi)
    Sy = np.sin(Theta) * np.sin(Phi)
    Sz = np.cos(Theta)

    sphere_pts = mv(Points, np.stack([Sx.ravel(), Sy.ravel(), Sz.ravel()], axis=-1))
    # Potential on sphere in PGA: V(p) = p v C(p)
    potential = sphere_pts.regressive(C(sphere_pts)).kernel.reshape(Sx.shape)

    # -------------------------------------------------------------
    # Panel 1: Primal Spherical Oval & Spherical Foci on S^2
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax1.plot_surface(Sx, Sy, Sz, facecolors=cm.coolwarm((potential - potential.min()) / (potential.max() - potential.min())),
                     alpha=0.35, rstride=2, cstride=2, shade=False)

    # Primal curve (both upper and antipodal lower loops)
    pk = P.kernel
    ax1.plot(pk[:, 0], pk[:, 1], pk[:, 2], color="gold", linewidth=3.5, label=r"Spherical Oval $p \vee C(p) = 0$")
    ax1.plot(-pk[:, 0], -pk[:, 1], -pk[:, 2], color="gold", linewidth=2.0, linestyle="--", alpha=0.7, label=r"Antipodal loop $-p$")

    # Foci
    f1k = F1.kernel.ravel()
    f2k = F2.kernel.ravel()
    ax1.scatter([f1k[0]], [f1k[1]], [f1k[2]], color="red", s=70, zorder=6, label=r"Foci $F_1, F_2$")
    ax1.scatter([f2k[0]], [f2k[1]], [f2k[2]], color="red", s=70, zorder=6)

    # Sample geodesic arcs from foci to a sample point on the oval
    idx_sample = 25
    ps = pk[idx_sample]
    # Geodesic arc from F1 to ps on S^2 (SLERP)
    slerp_t = np.linspace(0, 1, 30)[:, None]
    omega1 = dist_F1[idx_sample].kernel.item()
    arc1 = (np.sin((1 - slerp_t) * omega1) * f1k + np.sin(slerp_t * omega1) * ps) / np.sin(omega1)
    omega2 = dist_F2[idx_sample].kernel.item()
    arc2 = (np.sin((1 - slerp_t) * omega2) * f2k + np.sin(slerp_t * omega2) * ps) / np.sin(omega2)

    ax1.plot(arc1[:, 0], arc1[:, 1], arc1[:, 2], color="lime", linewidth=2.2, label=r"Geodesic $d(P, F_1)$")
    ax1.plot(arc2[:, 0], arc2[:, 1], arc2[:, 2], color="cyan", linewidth=2.2, label=r"Geodesic $d(P, F_2)$")
    ax1.scatter([ps[0]], [ps[1]], [ps[2]], color="white", edgecolor="black", s=80, zorder=7, label=r"Sample point $P$")

    ax1.set_title(f"Primal Point Locus on $S^2$\n$d(P, F_1) + d(P, F_2) = {2*theta_a:.3f}$ rad (const)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.legend(loc="lower left", fontsize=7.5)
    ax1.set_box_aspect([1, 1, 1])

    # -------------------------------------------------------------
    # Panel 2: Plane-Based Dual View: Envelope of Tangent Great Circles
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax2.plot_wireframe(Sx, Sy, Sz, color="slategray", alpha=0.15, linewidth=0.5)

    # Plot the spherical oval
    ax2.plot(pk[:, 0], pk[:, 1], pk[:, 2], color="gold", linewidth=3.0, label="Spherical Oval (Envelope)")

    # Draw a sequence of tangent great circles
    num_circles = 16
    circle_theta = np.linspace(0, 2 * np.pi, 120)
    colors = cm.plasma(np.linspace(0.1, 0.9, num_circles))

    sample_indices = np.linspace(0, len(pk) - 1, num_circles, dtype=int)
    for i, idx in enumerate(sample_indices):
        p_pt = pk[idx]
        n_pt = pi_tangent.kernel[idx]
        # Orthonormal basis for great circle plane: u1 = p_pt, u2 = n x p
        u1 = p_pt
        u2 = np.cross(n_pt, p_pt)
        u2 = u2 / np.linalg.norm(u2)

        gc = np.cos(circle_theta)[:, None] * u1 + np.sin(circle_theta)[:, None] * u2
        ax2.plot(gc[:, 0], gc[:, 1], gc[:, 2], color=colors[i], alpha=0.55, linewidth=1.2)

        # Plot normal vector (pole)
        ax2.quiver(p_pt[0], p_pt[1], p_pt[2], 0.35 * n_pt[0], 0.35 * n_pt[1], 0.35 * n_pt[2],
                   color=colors[i], arrow_length_ratio=0.3, alpha=0.8, linewidth=1.0)

    # Plot dual spherical conic (locus of tangent planes pi in S^2)
    nk = pi_tangent.kernel
    ax2.plot(nk[:, 0], nk[:, 1], nk[:, 2], color="magenta", linewidth=2.0, linestyle=":", label=r"Dual Plane Conic $\pi \in S^2$")

    ax2.set_title(r"Plane-Based Dual View" "\n" r"Envelope of Tangent Great Circles $\pi \cdot x = 0$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X"); ax2.set_ylabel("Y"); ax2.set_zlabel("Z")
    ax2.legend(loc="lower left", fontsize=7.5)
    ax2.set_box_aspect([1, 1, 1])

    # -------------------------------------------------------------
    # Panel 3: 3D Quadric Cone & Poinsot Confocal Polhodes
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    ax3.plot_wireframe(Sx, Sy, Sz, color="slategray", alpha=0.15, linewidth=0.5)

    # 3D quadratic cone intersecting the sphere
    cone_r = np.linspace(0.1, 1.25, 20)
    cone_t = np.linspace(0, 2 * np.pi, 60)
    R_cone, T_cone = np.meshgrid(cone_r, cone_t)

    # Unrotated cone
    cx_unrot = R_cone * x0 * np.cos(T_cone)
    cy_unrot = R_cone * y0 * np.sin(T_cone)
    cz_unrot = R_cone * np.sqrt(1.0 - (x0 * np.cos(T_cone))**2 - (y0 * np.sin(T_cone))**2)
    cone_pts_unrot = mv(Points, np.stack([cx_unrot.ravel(), cy_unrot.ravel(), cz_unrot.ravel()], axis=-1))
    cone_pts = (rotor >> cone_pts_unrot).kernel
    cx = cone_pts[:, 0].reshape(cx_unrot.shape)
    cy = cone_pts[:, 1].reshape(cy_unrot.shape)
    cz = cone_pts[:, 2].reshape(cz_unrot.shape)

    ax3.plot_surface(cx, cy, cz, color="khaki", alpha=0.3, rstride=2, cstride=2, shade=True)

    # Intersection curve (spherical oval)
    ax3.plot(pk[:, 0], pk[:, 1], pk[:, 2], color="gold", linewidth=3.5, label="Cone-Sphere Intersection")

    # Family of confocal polhodes (level sets p . Q(p) = c)
    levels = np.linspace(potential.min() * 0.7, potential.max() * 0.7, 7)
    for lev in levels:
        if abs(lev) < 0.05:
            continue
        # Contour on the sphere surface
        ax3.contour(Sx, Sy, Sz, potential, levels=[lev], colors=["deepskyblue" if lev > 0 else "salmon"],
                    linewidths=1.2, alpha=0.7)

    ax3.set_title(r"3D Quadratic Cone & Confocal Polhodes" "\n" r"$Q(v) \cdot v = c$ on the Momentum Sphere", fontsize=11, fontweight="bold")
    ax3.set_xlabel("X"); ax3.set_ylabel("Y"); ax3.set_zlabel("Z")
    ax3.legend(loc="lower left", fontsize=7.5)
    ax3.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    return fig


def main(save_path: str = str(PLOT_DIR / "spherical_quadrics.png")):
    # 2. Quadrics in plane-based PGA:
    # Primal quadric C maps Points -> Planes:  gatype((Planes, Points))
    # Dual quadric Q maps Planes -> Points:    gatype((Points, Planes))
    # In principal axes, both operators are purely diagonal matrices:
    l1, l2, l3 = 1.0, 0.4, -0.8
    C_diag = ctx.extensor(Polarity, np.diag([l1, l2, l3]))
    Q_diag = C_diag.inverse()

    # 3. Universal PGA motor transform via inline sandwich:
    rotor = (mv.xy * 0.25 + mv.yz * 0.15).exp()
    C = rotor >> C_diag(rotor << Points)
    Q = rotor >> Q_diag(rotor << Planes)

    # 4. Analytic spherical oval parameters in the unrotated frame:
    # Projected ellipse semi-axes: x^2 / x0^2 + y^2 / y0^2 = 1
    x0 = np.sqrt(-l3 / (l1 - l3))
    y0 = np.sqrt(-l3 / (l2 - l3))
    theta_b = np.arcsin(x0)  # semi-minor arc along x
    theta_a = np.arcsin(y0)  # semi-major arc along y
    theta_c = np.arccos(np.cos(theta_a) / np.cos(theta_b))  # focal arc along major axis y

    # Foci in unrotated frame as antivectors (points) and rotated into world frame:
    F1_unrot = mv(Points, [0.0, np.sin(theta_c), np.cos(theta_c)])
    F2_unrot = mv(Points, [0.0, -np.sin(theta_c), np.cos(theta_c)])
    F1 = rotor >> F1_unrot
    F2 = rotor >> F2_unrot

    # 5. Generate points along the spherical oval:
    t = np.linspace(0, 2 * np.pi, 200)
    x_diag = x0 * np.cos(t)
    y_diag = y0 * np.sin(t)
    z_diag = np.sqrt(1.0 - x_diag**2 - y_diag**2)
    P_diag = mv(Points, np.stack([x_diag, y_diag, z_diag], axis=-1))

    # Rotate points to world frame:
    P = rotor >> P_diag

    # Primal quadric condition: point lies on quadric iff P v C(P) == 0
    residuals_primal = P.regressive(C(P))

    # 6. Verify the Spherical Focal Property:
    # In PGA, the distance between unit antivector points on the sphere is arccos(- P . F)
    # The sum of geodesic distances from F1 and F2 to any point on the oval is constant: 2 * theta_a
    dist_F1 = (-(P | F1)).clip(-1.0, 1.0).arccos()
    dist_F2 = (-(P | F2)).clip(-1.0, 1.0).arccos()
    focal_sum = dist_F1 + dist_F2


    # 7. Plane-based (Dual) Geometry: Tangent Great Circles
    # The polar plane to point P with respect to quadric C is pi = C(P):
    pi_tangent = C(P).normalized()

    # Dual quadric condition: every tangent great circle satisfies pi v Q(pi) == 0:
    residuals_dual = pi_tangent.regressive(Q(pi_tangent))

    # Dual focal property: product of sines of distances from F1, F2 to tangent great circles is constant:
    # In PGA, sin(dist(F, plane)) = |F v pi|
    sin_d1 = F1.regressive(pi_tangent).norm()
    sin_d2 = F2.regressive(pi_tangent).norm()
    dual_prod = sin_d1 * sin_d2


    fig = plot_spherical_quadric(P, dist_F1, dist_F2, pi_tangent, x0, y0, rotor, F1, F2, theta_a, C, save_path)

    # --- checks -------------------------------------------------------------
    np.testing.assert_allclose(residuals_primal.kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(focal_sum.kernel, 2.0 * theta_a, atol=1e-11)
    np.testing.assert_allclose(residuals_dual.kernel, 0.0, atol=1e-14)
    np.testing.assert_allclose(dual_prod.kernel, dual_prod.mean().kernel.item(), atol=1e-14)

    return fig


if __name__ == "__main__":
    main()
