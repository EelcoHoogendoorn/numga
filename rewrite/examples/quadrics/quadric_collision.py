"""2D Dual Quadric (Ellipse) Collision Detection in PGA2D via the Pencil Q(lambda).

Demonstrates:
1. Dual Quadrics as First-Class Extensors in 2D PGA (R_{2,0,1}):
       Planes/Lines are Grade 1 vectors: (x, y, w) -> nx*x + ny*y - d*w = 0.
       Points are Grade 2 antivectors: (yw, wx, xy) -> (px, py, pw).
       A canonical dual ellipse at the origin is an arity-1 symmetric Extensor:
           Q_body = rx^2 * (yw (x) yw) + ry^2 * (wx (x) wx) - xy (x) xy.
       Rigid motion (SE(2) motor = translation * rotation) acts dynamically via:
           Q_world = motor >> Q_body(motor << Lines).
2. Tangency Condition:
       Line L is tangent to Q iff L v Q(L) == 0.
       The contact point on the ellipse is p = Q(L).
3. The Dual Quadric Pencil Q(lambda):
       Q(lambda) = (1 - lambda) * Q1 + lambda * Q2,  for lambda in [0, 1].
       Blends the two dual quadrics, sharing their common tangent planes.
4. The Binary Collision Criterion:
       The determinant curve det(Q(lambda)) is concave on [0, 1]:
       - Separated:   max_{lambda} det(Q(lambda)) > 0 (separating line exists).
       - Touching:    max_{lambda} det(Q(lambda)) == 0 (exact point contact at lambda*).
       - Overlapping: max_{lambda} det(Q(lambda)) < 0 (penetration, no separating line).
5. Contact Plane and Point Extraction:
       At the contact parameter lambda*, the null vector of Q(lambda*) is the
       unique common tangent line L*:
           L* v Q1(L*) == 0  and  L* v Q2(L*) == 0.
       The exact contact point is:
           p* = Q1(L*) = Q2(L*).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from numga import NumpyContext
from numga.algebras import PGA2D
from examples import PLOT_DIR


# 1. 2D Projective Geometric Algebra (PGA2D: R_{2,0,1})
context = NumpyContext(PGA2D)
spaces = PGA2D.subspace
mv = context.multivector

Lines = spaces.vector()
Points = spaces.antivector()
Line = PGA2D.gatype(Lines)
Point = PGA2D.gatype(Points)
Motor = PGA2D.gatype.rotor()
Scalar = PGA2D.gatype.scalar()
Quadric = PGA2D.gatype((Points, Lines))

# Ideal line at infinity (w = 0) and canonical origin (x ^ y)
line_at_infinity = mv.w
canonical_origin = mv.xy


def normalize_point(p: Point) -> Point:
    """Normalize an antivector point coordinate-free so its homogeneous weight is 1."""
    weight = p.regressive(line_at_infinity)
    return p / weight


# 2. Quadric & Motor Helpers
def make_quadric(rx: float, ry: float) -> Quadric:
    """Construct a canonical dual ellipse Q : Lines -> Points at the origin.
    
    In PGA2D, principal directions are the ideal points mv.yw and mv.wx.
    Q_body = rx^2 * (yw (x) yw) + ry^2 * (wx (x) wx) - xy (x) xy.
    """
    Sigma = (mv.yw * Lines.regressive(mv.yw)) * (rx**2) + (mv.wx * Lines.regressive(mv.wx)) * (ry**2)
    return Sigma - canonical_origin * Lines.regressive(canonical_origin)


def motor(tx: float = 0.0, ty: float = 0.0, angle_rad: float = 0.0) -> Motor:
    """Construct a 2D rigid motor in SE(2): translation * rotation."""
    translator = (line_at_infinity.wedge(mv.x * tx + mv.y * ty) * -0.5).exp()
    rotor = (canonical_origin * (-angle_rad / 2.0)).exp()
    return translator * rotor


def tangent_line(Q: Quadric, normal: Line) -> Line:
    """Evaluate the tangent line of dual quadric Q with given outward normal direction."""
    n = normal.normalized()
    L0 = n - line_at_infinity * n.regressive(-Q(line_at_infinity))
    return L0 - line_at_infinity * (L0.regressive(Q(L0))).square_root()


def quadric_boundary(Q: Quadric, n_pts: int = 150) -> Point:
    """Derive boundary points directly from dual quadric Q via tangent contact: p = Q(L)."""
    theta = np.linspace(0, 2 * np.pi, n_pts)
    normals = mv.x * mv.scalar(np.cos(theta)[:, None]) + mv.y * mv.scalar(np.sin(theta)[:, None])
    return normalize_point(Q(tangent_line(Q, normals)))


# 3. Dual Quadric Pencil & Contact Solvers
def dual_pencil(Q1: Quadric, Q2: Quadric, lam: Scalar) -> Quadric:
    """Evaluate the dual quadric pencil Q(lambda) = (1 - lambda) * Q1 + lambda * Q2."""
    return Q1 * (1.0 - lam) + Q2 * lam


def pencil_determinant(Q1: Quadric, Q2: Quadric, lam: Scalar) -> np.ndarray:
    """Evaluate det(Q(lambda)) for a scalar or batched Extensor lambda."""
    return dual_pencil(Q1, Q2, lam).dual().det().kernel[..., 0]


def find_contact_parameter(Q1: Quadric, Q2: Quadric) -> tuple[float, float]:
    """Find the optimal pencil parameter lambda* maximizing det(Q(lambda)) analytically in closed form.
    
    The pencil determinant P(lambda) = det((1 - lambda) * Q1 + lambda * Q2) is an exact cubic
    polynomial. Its maximum is found in closed form via the quadratic formula
    applied to P'(lambda) = 0 with zero iteration loops.
    
    Returns:
        (lam_star, max_det): The peak interpolation parameter and maximum determinant value.
    """
    samples = mv.scalar([[0.0], [1.0], [2.0], [-1.0]])
    values = (Q1 * (1 - samples) + Q2 * samples).dual().det()
    parameter, maximum = cubic_peak(values)
    return parameter.kernel.item(), maximum.kernel.item()


def cubic_peak(values: Scalar) -> tuple[Scalar, Scalar]:
    """Locate the interior maximum of a cubic sampled at 0, 1, 2 and -1."""
    y0, y1, y2, y3 = values.kernel[..., 0]
    c3 = (3.0 * y0 - 3.0 * y1 + y2 - y3) / 6.0
    c2 = -y0 + 0.5 * y1 + 0.5 * y3
    c1 = -0.5 * y0 + y1 - y2 / 6.0 - y3 / 3.0
    c0 = y0

    disc = c2 * c2 - 3.0 * c3 * c1
    if disc > 0.0 and abs(c3) > 1e-12:
        lam_star = (-c2 - np.sqrt(disc)) / (3.0 * c3)
    elif abs(c2) > 1e-12:
        lam_star = -c1 / (2.0 * c2)
    else:
        lam_star = 0.5

    lam_star = float(np.clip(lam_star, 0.001, 0.999))
    max_det = c3 * lam_star**3 + c2 * lam_star**2 + c1 * lam_star + c0
    return mv.scalar([lam_star]), mv.scalar([max_det])


def extract_contact_line_and_point(
    Q1: Quadric,
    Q2: Quadric,
    lam_star: float
) -> tuple[Line, Point]:
    """Extract the common tangent line L* and contact point p* at the collision parameter."""
    Q_star = dual_pencil(Q1, Q2, mv.scalar([lam_star]))
    eigvals, eigvecs = Q_star.dual().eigh()
    L_contact = eigvecs[eigvals.abs().argmin()].normalized()

    # Ensure normal of L points from Q1 towards Q2:
    # Under regressive product, the displacement c2 - c1 must have positive signed projection
    c1 = normalize_point(-(Q1(line_at_infinity)))
    c2 = normalize_point(-(Q2(line_at_infinity)))
    if (L_contact.regressive(c2 - c1)).kernel.item() < 0:
        L_contact = -L_contact

    p_contact = normalize_point(Q1(L_contact))
    return L_contact, p_contact


# 4. Plotting & Visualization Helpers
xy_subspace = spaces("yw wx")


def get_xy(p: Point) -> np.ndarray:
    """Extract Euclidean 2D coordinates for plotting."""
    p_norm = normalize_point(p)
    return p_norm.select_subspace(xy_subspace).kernel


def plot_line_on_ax(ax: plt.Axes, L: Line, color: str = "red", linestyle: str = "-", linewidth: float = 1.8, label: str = "") -> None:
    """Draw a PGA line nx*x + ny*y - d = 0 across the axis limits."""
    l_coords = L.select_subspace(spaces.vector()).kernel.ravel()
    nx, ny, d_neg = l_coords[0], l_coords[1], l_coords[2]
    d = -d_neg
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    kw = {"label": label} if label else {}
    if np.abs(ny) > 1e-4:
        xs = np.array([xlim[0], xlim[1]])
        ys = (d - nx * xs) / ny
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth, zorder=4, **kw)
    else:
        x_val = d / nx
        ax.axvline(x_val, color=color, linestyle=linestyle, linewidth=linewidth, zorder=4, **kw)


def draw_collision(Q1: Quadric, states, n_dir: Line, L_contact_touch: Line, p_contact_touch: Point, save_path: str) -> None:
    Q2_sep, lam_sep, max_sep = states[0]
    Q2_touch, lam_touch, max_touch = states[1]
    Q2_over, lam_over, max_over = states[2]
    lam_sep, lam_touch, lam_over = (v.kernel.item() for v in (lam_sep, lam_touch, lam_over))
    max_sep, max_touch, max_over = (v.kernel.item() for v in (max_sep, max_touch, max_over))
    pts1, c1_xy = get_xy(quadric_boundary(Q1)), get_xy(-Q1(line_at_infinity))
    pts2_sep, c2_sep_xy = get_xy(quadric_boundary(Q2_sep)), get_xy(-Q2_sep(line_at_infinity))
    pts2_touch, c2_touch_xy = get_xy(quadric_boundary(Q2_touch)), get_xy(-Q2_touch(line_at_infinity))
    pts2_over, c2_over_xy = get_xy(quadric_boundary(Q2_over)), get_xy(-Q2_over(line_at_infinity))
    p_touch_xy = get_xy(p_contact_touch)
    lams = np.linspace(.001, .999, 500)
    weights = mv.scalar(lams[:, None])
    dets_sep, dets_touch, dets_over = [(Q1*(1-weights)+other*weights).dual().det().kernel[..., 0]
                                    for other in (Q2_sep, Q2_touch, Q2_over)]
    # 3. Create 4-Panel Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    (ax_sep, ax_touch), (ax_over, ax_det) = axes

    def draw_ellipses(ax: plt.Axes, p1: np.ndarray, c1: np.ndarray, p2: np.ndarray, c2: np.ndarray, title: str, subtitle: str) -> None:
        ax.fill(p1[:, 0], p1[:, 1], color="#3b82f6", alpha=0.35, label="Ellipse Q1")
        ax.plot(p1[:, 0], p1[:, 1], color="#1d4ed8", linewidth=2.2)
        ax.scatter([c1[0]], [c1[1]], color="#1d4ed8", s=40, zorder=5)

        ax.fill(p2[:, 0], p2[:, 1], color="#f97316", alpha=0.35, label="Ellipse Q2")
        ax.plot(p2[:, 0], p2[:, 1], color="#ea580c", linewidth=2.2)
        ax.scatter([c2[0]], [c2[1]], color="#ea580c", s=40, zorder=5)

        ax.set_xlim(-3.8, 3.8)
        ax.set_ylim(-3.0, 3.0)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title}\n{subtitle}", fontsize=11, fontweight="bold")

    # Tangent lines and witness points along the contact normal for non-exact contact cases:
    L1_sep = tangent_line(Q1, n_dir)
    L2_sep = tangent_line(Q2_sep, -n_dir)
    p1_sep_xy = get_xy(Q1(L1_sep))
    p2_sep_xy = get_xy(Q2_sep(L2_sep))
    L_mid_sep = ((L1_sep - L2_sep) * 0.5).normalized()

    L1_over = tangent_line(Q1, n_dir)
    L2_over = tangent_line(Q2_over, -n_dir)
    p1_over_xy = get_xy(Q1(L1_over))
    p2_over_xy = get_xy(Q2_over(L2_over))
    L_mid_over = ((L1_over - L2_over) * 0.5).normalized()

    # --- PANEL 1: SEPARATED ---
    draw_ellipses(ax_sep, pts1, c1_xy, pts2_sep, c2_sep_xy, "State 1: Separated Ellipses", f"max det(Q(λ)) = {max_sep:+.3f} > 0")
    # Low-alpha interpolated hyperbola Q(λ*) separating the two ellipses (signature +, -, -)
    Q_star_sep = dual_pencil(Q1, Q2_sep, mv.scalar([lam_sep]))
    xs_grid = np.linspace(-3.8, 3.8, 250)
    ys_grid = np.linspace(-3.0, 3.0, 250)
    X_grid, Y_grid = np.meshgrid(xs_grid, ys_grid)
    P_grid = mv.antivector(np.stack([X_grid, Y_grid, np.ones_like(X_grid)], axis=-1))
    F_sep = P_grid.regressive(Q_star_sep.inverse()(P_grid)).kernel[..., 0]
    ax_sep.contour(X_grid, Y_grid, F_sep, levels=[0.0], colors=["#059669"], linestyles=["-."], linewidths=1.6)
    ax_sep.contourf(X_grid, Y_grid, F_sep, levels=[0.0, F_sep.max()], colors=["#10b981"], alpha=0.10)

    plot_line_on_ax(ax_sep, L1_sep, color="#2563eb", linestyle="--", linewidth=1.5, label="Tangent Line on Q1 (L1)")
    plot_line_on_ax(ax_sep, L2_sep, color="#ea580c", linestyle="--", linewidth=1.5, label="Tangent Line on Q2 (L2)")
    plot_line_on_ax(ax_sep, L_mid_sep, color="#16a34a", linestyle="-", linewidth=2.0, label="Separating Plane (Midplane)")
    ax_sep.scatter([p1_sep_xy[0]], [p1_sep_xy[1]], color="#2563eb", s=60, zorder=6, label=f"Closest on Q1 ({p1_sep_xy[0]:.2f}, {p1_sep_xy[1]:.2f})")
    ax_sep.scatter([p2_sep_xy[0]], [p2_sep_xy[1]], color="#ea580c", s=60, zorder=6, label=f"Closest on Q2 ({p2_sep_xy[0]:.2f}, {p2_sep_xy[1]:.2f})")
    ax_sep.plot([p1_sep_xy[0], p2_sep_xy[0]], [p1_sep_xy[1], p2_sep_xy[1]], "k:", linewidth=1.8, zorder=5)
    ax_sep.plot([], [], color="#059669", linestyle="-.", linewidth=1.6, label="Hyperbola Q(λ*) (det>0)")
    ax_sep.legend(loc="lower left", fontsize=7.5)

    # --- PANEL 2: TOUCHING (EXACT POINT CONTACT) ---
    draw_ellipses(ax_touch, pts1, c1_xy, pts2_touch, c2_touch_xy, "State 2: Touching Ellipses (Exact Contact)", f"max det(Q(λ)) = {max_touch:+.1e} ≈ 0 (at λ* = {lam_touch:.3f})")
    # Intermediate pencil ellipse Q(0.45) in low alpha showing deformation towards contact
    Q_mid_touch = dual_pencil(Q1, Q2_touch, mv.scalar([0.45]))
    pts_mid_touch = get_xy(quadric_boundary(Q_mid_touch))
    ax_touch.fill(pts_mid_touch[:, 0], pts_mid_touch[:, 1], color="#a855f7", alpha=0.18, label="Pencil Ellipse Q(0.45)")
    ax_touch.plot(pts_mid_touch[:, 0], pts_mid_touch[:, 1], color="#9333ea", linestyle=":", linewidth=1.5, alpha=0.7)

    plot_line_on_ax(ax_touch, L_contact_touch, color="#dc2626", linestyle="-", linewidth=2.4, label="Unique Shared Tangent Line L*")
    ax_touch.scatter([p_touch_xy[0]], [p_touch_xy[1]], color="#dc2626", s=110, zorder=8, label=f"Contact Point p* ({p_touch_xy[0]:.2f}, {p_touch_xy[1]:.2f})")
    ax_touch.annotate(
        f"Contact p*\n({p_touch_xy[0]:.2f}, {p_touch_xy[1]:.2f})",
        (p_touch_xy[0] + 0.15, p_touch_xy[1] + 0.25),
        fontsize=9.5,
        fontweight="bold",
        color="#b91c1c",
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.5),
    )
    ax_touch.legend(loc="lower left", fontsize=7.5)

    # --- PANEL 3: OVERLAPPING ---
    draw_ellipses(ax_over, pts1, c1_xy, pts2_over, c2_over_xy, "State 3: Overlapping Ellipses", f"max det(Q(λ)) = {max_over:+.3f} < 0 (Penetration)")
    # Low-alpha interpolated ellipse Q(λ*) bridging the intersection zone
    Q_star_over = dual_pencil(Q1, Q2_over, mv.scalar([lam_over]))
    pts_star_over = get_xy(quadric_boundary(Q_star_over))
    ax_over.fill(pts_star_over[:, 0], pts_star_over[:, 1], color="#a855f7", alpha=0.22, label="Interpolated Ellipse Q(λ*)")
    ax_over.plot(pts_star_over[:, 0], pts_star_over[:, 1], color="#7c3aed", linestyle="-.", linewidth=1.8)

    plot_line_on_ax(ax_over, L1_over, color="#2563eb", linestyle="--", linewidth=1.5, label="Tangent Line on Q1 (L1)")
    plot_line_on_ax(ax_over, L2_over, color="#ea580c", linestyle="--", linewidth=1.5, label="Tangent Line on Q2 (L2)")
    plot_line_on_ax(ax_over, L_mid_over, color="#9333ea", linestyle="-", linewidth=2.0, label="Contact Plane (Midplane)")
    ax_over.scatter([p1_over_xy[0]], [p1_over_xy[1]], color="#2563eb", s=60, zorder=6, label=f"Deepest on Q1 ({p1_over_xy[0]:.2f}, {p1_over_xy[1]:.2f})")
    ax_over.scatter([p2_over_xy[0]], [p2_over_xy[1]], color="#ea580c", s=60, zorder=6, label=f"Deepest on Q2 ({p2_over_xy[0]:.2f}, {p2_over_xy[1]:.2f})")
    ax_over.plot([p1_over_xy[0], p2_over_xy[0]], [p1_over_xy[1], p2_over_xy[1]], "k:", linewidth=1.8, zorder=5)
    ax_over.legend(loc="lower left", fontsize=7.5)

    # --- PANEL 4: THE PENCIL DETERMINANT CURVES ---
    ax_det.plot(lams, dets_sep, color="#16a34a", linewidth=2.4, label=f"Separated: max = {max_sep:+.3f} > 0")
    ax_det.plot(lams, dets_touch, color="#ea580c", linewidth=2.4, label=f"Touching: max = {max_touch:+.1e} ≈ 0 (at λ*={lam_touch:.3f})")
    ax_det.plot(lams, dets_over, color="#dc2626", linewidth=2.4, label=f"Overlapping: max = {max_over:+.3f} < 0")

    ax_det.axhline(0.0, color="black", linestyle="--", linewidth=1.5, alpha=0.8, label="Collision Threshold: det(Q(λ)) = 0")
    ax_det.scatter([lam_touch], [max_touch], color="#ea580c", s=70, zorder=6)
    ax_det.scatter([lam_sep], [max_sep], color="#16a34a", s=70, zorder=6)
    ax_det.scatter([lam_over], [max_over], color="#dc2626", s=70, zorder=6)

    ax_det.set_xlim(0.0, 1.0)
    ax_det.set_xlabel("Interpolation Parameter λ", fontsize=10, fontweight="bold")
    ax_det.set_ylabel("Determinant det(Q(λ))", fontsize=10, fontweight="bold")
    ax_det.grid(True, alpha=0.3)
    ax_det.set_title("The Dual Pencil Characteristic Curves det(Q(λ))\nBinary Classification: > 0 Separated | = 0 Contact | < 0 Overlap", fontsize=11, fontweight="bold")
    ax_det.legend(loc="lower center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=160)
        print(f"Saved visualization to {save_path}")
    plt.close()


def run_quadric_collision_demo(save_path: str = str(PLOT_DIR / "quadric_collision_2d.png")) -> None:
    """Demonstrate 2D dual quadric collision detection across 3 states and plot results."""
    print("=" * 72)
    print("2D Dual Quadric (Ellipse) Collision Detection in PGA2D via Pencil Q(lambda)")
    print("=" * 72)

    # 1. Base Ellipse Q1: rx=2.0, ry=1.0, posed at center=(-1.2, 0.0), angle=25 deg
    rx1, ry1 = 2.0, 1.0
    Q1_body = mv.yw * (Lines & mv.yw) * rx1**2 + mv.wx * (Lines & mv.wx) * ry1**2 - mv.xy * (Lines & mv.xy)
    m1 = motor(-1.2, 0.0, np.radians(25))
    Q1 = m1 >> Q1_body(m1 << Lines)


    # 2. Ellipse Q2 in exact tangential contact with Q1:
    rx2, ry2 = 1.6, 0.9
    Q2_body = mv.yw * (Lines & mv.yw) * rx2**2 + mv.wx * (Lines & mv.wx) * ry2**2 - mv.xy * (Lines & mv.xy)

    # Contact normal direction: 22 degrees
    phi_contact = np.radians(22)
    n_dir = (mv.x * np.cos(phi_contact) + mv.y * np.sin(phi_contact)).normalized()
    through_centre = n_dir - line_at_infinity * (n_dir & -Q1(line_at_infinity))
    L_target = through_centre - line_at_infinity * (through_centre & Q1(through_centre)).square_root()
    p_target = normalize_point(Q1(L_target))

    # Rotate Q2 at origin by angle2 to find the contact point for opposite normal:
    angle2 = np.radians(-35)
    m2_rot = motor(0.0, 0.0, angle2)
    Q2_rot = m2_rot >> Q2_body(m2_rot << Lines)

    through_centre = -n_dir - line_at_infinity * (-n_dir & -Q2_rot(line_at_infinity))
    L2_rot = through_centre - line_at_infinity * (through_centre & Q2_rot(through_centre)).square_root()
    p2_rot = normalize_point(Q2_rot(L2_rot))

    # Pure PGA translation motor: takes p2_rot to p_target
    disp = p_target - p2_rot
    T_touch = (line_at_infinity.wedge(disp.dual()) * -0.5).exp()
    m2_touch = T_touch * m2_rot
    Q2_touch = m2_touch >> Q2_body(m2_touch << Lines)

    # Separated pose: translate away along normal by +0.8
    T_sep = (line_at_infinity.wedge(n_dir * 0.8) * -0.5).exp()
    m2_sep = T_sep * m2_touch
    Q2_sep = m2_sep >> Q2_body(m2_sep << Lines)

    # Overlapping pose: translate inward along normal by -0.6
    T_over = (line_at_infinity.wedge(n_dir * -0.6) * -0.5).exp()
    m2_over = T_over * m2_touch
    Q2_over = m2_over >> Q2_body(m2_over << Lines)

    # Four determinant samples determine the cubic pencil characteristic exactly.
    # Only locating its peak leaves the algebra.
    samples = mv.scalar([[0.0], [1.0], [2.0], [-1.0]])
    states = []
    for other in (Q2_sep, Q2_touch, Q2_over):
        pencil = Q1 * (1 - samples) + other * samples
        parameter, maximum = cubic_peak(pencil.dual().det())
        states.append((other, parameter, maximum))

    # At contact the pencil has a null line. Its singular vector is the shared tangent;
    # each quadric maps that tangent to the same contact point.
    parameter = states[1][1]
    contact_pencil = Q1 * (1 - parameter) + Q2_touch * parameter
    _, _, lines = contact_pencil.dual().svd()
    contact_line = lines[-1].normalized()
    centre_delta = normalize_point(-Q2_touch(line_at_infinity)) - normalize_point(-Q1(line_at_infinity))
    contact_line = (contact_line / (contact_line & centre_delta)).normalized()
    contact_point = normalize_point(Q1(contact_line))
    draw_collision(Q1, states, n_dir, contact_line, contact_point, save_path)

    # --- checks -------------------------------------------------------------
    assert states[0][2].kernel.item() > 0
    np.testing.assert_allclose(states[1][2].kernel, 0, atol=1e-8)
    assert states[2][2].kernel.item() < 0
    np.testing.assert_allclose((contact_line & Q1(contact_line)).kernel, 0, atol=1e-6)
    np.testing.assert_allclose(normalize_point(Q2_touch(contact_line)).kernel, contact_point.kernel, atol=1e-5)


if __name__ == "__main__":
    run_quadric_collision_demo()
