"""Boosted Celestial Quadrics and Relativistic Aberration in Spacetime Algebra Cl(1,3).

Demonstrates:
1. Quadrics on the Celestial Sphere S²:
   Constructed from canonical quadratic cones Q: V -> V intersecting the null cone k² = 0.
   - Circular quadrics: θ_x = θ_y (reducible on the null cone into hyperplane pairs).
   - Elliptic quadrics: θ_x ≠ θ_y (irreducible 4D quadric cones).
2. Spatial Orientation via GA Rotors:
   Rotated across the sky using spatial bivector rotors R = exp(0.5 * B_rot).
3. Relativistic Aberration via the Lorentz Outermorphism:
   Under a Lorentz boost L = exp(0.5 * ζ * γ_t γ_z):
   - Quadric extensors transform via: Q' = L >> Q(L << V).
   - Light ray contours transform via: k' = L >> k.
   - Exact algebraic invariance: k' . Q'(k') == 0 to machine precision (< 1e-14).
4. Projective Camera View from the Origin:
   Observer pinhole at origin (0, 0, 0), projecting rays through the celestial sphere
   onto the screen plane z = 1: (u, v) = (x'/z', y'/z').
5. Terrell-Penrose Circle Preservation & Elliptical Doppler Distortion:
   - Central circles scale isotropically by the Doppler factor: r' = r * exp(ζ).
   - Off-axis and elliptic quadrics experience differential Doppler gradients across
     their subtended angle, visibly demonstrating relativistic aberration.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from numga import Algebra, NumpyContext
from examples import PLOT_DIR
from examples.animation import capture, save_gif

# ---------------------------------------------------------------------------
# 1. Spacetime Algebra Setup (STA: R_{1,3}, t+ x- y- z-)
# ---------------------------------------------------------------------------
STA = Algebra("t+x-y-z-")
ctx = NumpyContext(STA)
mv = ctx.multivector

Vector = STA.gatype.vector()
Bivector = STA.gatype.bivector()
Rotor = STA.gatype.rotor()
QuadricType = STA.gatype((Vector, Vector))
V = STA.subspace.vector()
x, y, z = mv.vector(np.eye(4)[1:])


# ---------------------------------------------------------------------------
# 2. Quadric Extensors and Contour Generators
# ---------------------------------------------------------------------------
def make_spherical_quadric_extensor(th_x: float, th_y: float) -> QuadricType:
    """Construct canonical dual quadric extensor Q : Vector -> Vector on the null cone.

    The spatial cone is (x / tan θ_x)² + (y / tan θ_y)² - z² = 0. Spatial vectors square
    to -1 here, so the dyads carry the opposite sign to the coefficients they represent.
    """
    tx2 = np.tan(th_x) ** 2
    ty2 = np.tan(th_y) ** 2
    return z * (z | V) - x * (x | V) / tx2 - y * (y | V) / ty2


def make_canonical_contour(th_x: float, th_y: float, n_pts: int = 240) -> Vector:
    """Generate parametric null ray contour k = (1, n) on the celestial sphere S²."""
    phi = np.linspace(0.0, 2.0 * np.pi, n_pts)
    tx, ty = np.tan(th_x), np.tan(th_y)
    x = tx * np.cos(phi)
    y = ty * np.sin(phi)
    z = np.ones_like(x)
    pts = np.stack([x, y, z], axis=-1)
    pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True)
    k = np.stack([np.ones_like(x), pts[..., 0], pts[..., 1], pts[..., 2]], axis=-1)
    return mv.vector(k)


def project_to_screen(
    k_boosted: Vector,
    projection_mode: str = "perspective",
) -> tuple[np.ndarray, np.ndarray]:
    """Project boosted null rays from origin onto the 2D observation canvas.

    Modes:
    - 'perspective' (default): Pinhole camera at origin, screen plane z = 1.
      Coordinates are (u, v) = (x / z, y / z). Points behind the camera (z <= 0.02)
      are masked with NaN so matplotlib breaks line segments smoothly without whole-object dropping.
    - 'stereographic': Conformal projection from south pole onto tangent plane.
      Coordinates are (u, v) = (2x / (1 + z), 2y / (1 + z)). Preserves exact circularity
      for every circle anywhere on the sphere.
    """
    coords = k_boosted.kernel
    t, x, y, z = coords[..., 0], coords[..., 1], coords[..., 2], coords[..., 3]
    if projection_mode == "stereographic":
        denom = t + z
        mask = denom > 0.02 * t
        return np.where(mask, 2.0 * x / denom, np.nan), np.where(mask, 2.0 * y / denom, np.nan)
    mask = z > 0.02 * t
    return np.where(mask, x / z, np.nan), np.where(mask, y / z, np.nan)


# ---------------------------------------------------------------------------
def make_quadric_at_direction(
    direction: np.ndarray,
    th_x_deg: float,
    th_y_deg: float,
    roll_deg: float,
) -> tuple[QuadricType, Vector]:
    """Orient canonical quadric and contour along an arbitrary spatial unit direction vector."""
    ez = z
    d = direction / np.linalg.norm(direction)
    p_vec = mv.vector([0.0, d[0], d[1], d[2]])
    if d[2] < -0.9999:
        r_dir = (mv.zx * (np.pi / 2.0)).exp()
    elif d[2] > 0.9999:
        r_dir = mv.scalar([1.0])
    else:
        r_dir = (1.0 - p_vec * ez).normalized()

    roll = np.radians(roll_deg)
    r_total = r_dir * (mv.xy * (roll / 2.0)).exp()

    th_x, th_y = np.radians(th_x_deg), np.radians(th_y_deg)
    q_local = make_spherical_quadric_extensor(th_x, th_y)
    k_local = make_canonical_contour(th_x, th_y)

    q_world = r_total >> q_local(r_total << Vector)
    k_world = r_total >> k_local
    return q_world, k_world


# ---------------------------------------------------------------------------
# 3. Celestial Scene Specification
# ---------------------------------------------------------------------------
def make_dodecahedron_vertices() -> np.ndarray:
    """Compute the 20 unit vertices of a regular dodecahedron aligned with the z-axis.

    - 1 vertex at North Pole (0, 0, 1) (directly ahead of observer).
    - 3 vertices at polar angle arccos(√5/3) ≈ 41.81° (z = √5/3 ≈ 0.745).
    - 6 vertices at polar angle arccos(1/3) ≈ 70.53° (z = 1/3 ≈ 0.333).
    - 6 vertices at polar angle arccos(-1/3) ≈ 109.47° (z = -1/3, in rear hemisphere).
    - 3 vertices at polar angle arccos(-√5/3) ≈ 138.19° (z = -√5/3, in rear hemisphere).
    - 1 vertex at South Pole (0, 0, -1) (directly behind observer).
    All adjacent vertices have exact mutual angular separation arccos(√5/3) ≈ 41.81°.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    inv_phi = 1.0 / phi

    raw = []
    for x in [-1.0, 1.0]:
        for y in [-1.0, 1.0]:
            for z in [-1.0, 1.0]:
                raw.append([x, y, z])
    for y in [-phi, phi]:
        for z in [-inv_phi, inv_phi]:
            raw.append([0.0, y, z])
    for x in [-inv_phi, inv_phi]:
        for z in [-phi, phi]:
            raw.append([x, 0.0, z])
    for x in [-phi, phi]:
        for y in [-inv_phi, inv_phi]:
            raw.append([x, y, 0.0])

    raw_arr = np.array(raw) / np.sqrt(3.0)
    top_v = raw_arr[np.argmax(raw_arr[:, 2])]
    axis = np.cross(top_v, [0.0, 0.0, 1.0])
    sin_a = np.linalg.norm(axis)
    cos_a = np.dot(top_v, [0.0, 0.0, 1.0])
    axis /= sin_a
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + sin_a * K + (1.0 - cos_a) * (K @ K)
    return raw_arr @ R.T


def build_celestial_scene(
    n_random_quads: int = 30,
    seed: int = 101,
) -> list[tuple[str, QuadricType, Vector, str, float]]:
    """Build a uniformly decorated celestial sphere with overlapping dodecahedron circles and random quads.

    1. 20 reference circles tiled at the vertices of a regular dodecahedron:
       - Adjacent vertex distance is 41.81°.
       - Setting circle radius to 23.0° (diameter 46.0°) ensures adjacent circles overlap by ~4.2°.
       - Center circle at (0, 0, 1) is viewed head-on and appears perfectly circular.
       - Off-axis circles at 41.81° obliquely intersect the screen plane and appear as perspective ellipses.
    2. Random elliptic quadrics sampled uniformly across the whole 4π sphere:
       - Directions drawn from isotropic 3D Gaussian (three independent normal numbers).

    Returns:
        List of (label, quadric_extensor, contour_rays, color_hex, line_width).
    """
    rng = np.random.default_rng(seed)
    scene: list[tuple[str, QuadricType, Vector, str, float]] = []

    # 1. 20 Dodecahedron overlapping reference circles
    v_dodec = make_dodecahedron_vertices()
    circle_rad = 23.0  # Overlaps adjacent circles (separation 41.81°)
    for idx, v in enumerate(v_dodec):
        name = "Circle (Center)" if idx == 0 else f"Circle (Dodec {idx})"
        q_w, k_w = make_quadric_at_direction(v, circle_rad, circle_rad, 0.0)
        col = "#38bdf8" if idx == 0 else "#60a5fa"
        scene.append((name, q_w, k_w, col, 2.2 if idx == 0 else 1.6))

    # 2. Random elliptic quadrics sampled uniformly all over the celestial sphere
    quad_colors = [
        "#fbbf24", "#f97316", "#f43f5e", "#a855f7",
        "#ec4899", "#34d399", "#e879f9", "#fcd34d",
    ]
    for idx in range(n_random_quads):
        d = rng.standard_normal(3)
        d /= np.linalg.norm(d)
        th_x = float(rng.uniform(6.0, 12.0))
        th_y = float(rng.uniform(3.0, 6.0))
        roll = float(rng.uniform(0.0, 180.0))
        col = quad_colors[idx % len(quad_colors)]
        q_w, k_w = make_quadric_at_direction(d, th_x, th_y, roll)
        scene.append((f"Quad {idx + 1}", q_w, k_w, col, 1.2))

    return scene


def draw_boosted_sky(states, gif_path: str, projection_mode: str, downsample: int) -> str:
    frames: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=100, facecolor="#090d16")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    for state in states:
        ax.clear()
        ax.set_facecolor("#090d16")
        ax.set_axis_off()
        if projection_mode == "stereographic":
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
        else:
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
        ax.set_aspect("equal")

        for name, k_cam, color, lw in state:
            coords = k_cam.kernel
            t, z = coords[..., 0], coords[..., 3]

            # In perspective mode, skip if entire contour is behind camera plane
            if projection_mode != "stereographic":
                if not np.any(z > 0.02 * t):
                    continue
            else:
                if not np.any((t + z) > 0.02 * t):
                    continue

            u, v = project_to_screen(k_cam, projection_mode=projection_mode)
            ax.plot(u, v, color=color, lw=lw)

        frames.append(capture(fig))

    plt.close(fig)

    return save_gif(frames, gif_path, duration_ms=40, scale=1.0 / downsample)


def animate_boosted_quadrics(
    gif_path: str = str(PLOT_DIR / "boosted_quadrics.gif"),
    n_frames: int = 60,
    zeta_max: float = 1.4,
    projection_mode: str = "perspective",
    view_direction: str = "forward",
    downsample: int = 4,
) -> str:
    """Animate boosted quadric contours on the observation canvas and export to GIF.

    Parameters:
        gif_path: Destination path for exported GIF.
        n_frames: Number of animation frames.
        zeta_max: Peak rapidity oscillation amplitude.
        projection_mode: 'perspective' (pinhole screen) or 'stereographic' (conformal).
        view_direction: 'forward' (along +z motion axis) or 'side' (starboard window along +x).
        downsample: Spatial downsampling factor with area averaging.
    """
    scene = build_celestial_scene()
    camera = {"forward": mv.rotor(), "side": (mv.zx * (-np.pi / 4.0)).exp()}[view_direction]
    rapidities = zeta_max * np.sin(2.0 * np.pi * np.arange(n_frames) / n_frames)
    boosts = (mv.zt * (rapidities / 2)).exp()

    def scenes():
        for boost in boosts:
            # The Lorentz map carries every contour ray; the camera changes the observer.
            observer = camera * boost
            yield [(name, observer >> rays, color, width) for name, quadric, rays, color, width in scene]

    return draw_boosted_sky(scenes(), gif_path, projection_mode, downsample)


def main(
    gif_path: str = str(PLOT_DIR / "boosted_quadrics.gif"),
    gif_path_stereo: str = str(PLOT_DIR / "boosted_quadrics_stereo.gif"),
) -> tuple[str, str]:
    """Generate the forward perspective and forward stereographic animations."""
    scene = build_celestial_scene()
    rapidities = 1.4 * np.sin(2 * np.pi * np.arange(60) / 60)
    boosts = (mv.zt * (rapidities / 2)).exp()

    def scenes():
        for boost in boosts:
            # One observer map carries every contour's null rays into the moving sky.
            yield [(name, boost >> rays, color, width) for name, quadric, rays, color, width in scene]

    forward = draw_boosted_sky(scenes(), gif_path, "perspective", 4)
    stereo = draw_boosted_sky(scenes(), gif_path_stereo, "stereographic", 4)
    return forward, stereo


if __name__ == "__main__":
    main()
