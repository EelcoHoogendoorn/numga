"""Spherical Quadric Vortex in 4D CGA Cl(3, 1) = R_{3,1}.

Spherical quadrics (conical donuts, peanuts, twin islands, lemniscates, crescents,
Dupin cyclides, pinched horns, spindles, teardrops, triadic clovers, hourglasses,
and parabolic bows) as extensors Q : Vector -> Plane on S², animated by the conformal
rotor exp(theta (c1 ^ c2) / 2) formed from the intersection of two off-center circles
on S², rendered in the visual style of the S² physics demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from matplotlib.colors import to_rgb
import numpy as np

from numga import Algebra, NumpyContext
from examples import PLOT_DIR
from examples.animation import save_gif

# ---------------------------------------------------------------------------
# 1. 4D CGA Cl(3, 1) Setup
# ---------------------------------------------------------------------------
ga = Algebra("x+y+z+w-")
ctx = NumpyContext(ga)
mv = ctx.multivector

Vector = ga.gatype.vector()
Plane = ga.gatype.antivector()
Quadric = ga.gatype((Plane, Vector))
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()

# Dual basis planes via pseudoscalar I = xyzw
I = mv.xyzw
px = (mv.x * I).cast(Plane.output_subspace)
py = (mv.y * I).cast(Plane.output_subspace)
pz = (mv.z * I).cast(Plane.output_subspace)
pw = (-mv.w * I).cast(Plane.output_subspace)
p_wz = pw - pz

BACKGROUND, DISK, RIM = "#0a0f1d", "#111827", "#334155"


# ---------------------------------------------------------------------------
# 2. CGA Spherical Quadric Shape Constructors
# ---------------------------------------------------------------------------
def make_spherical_donut(r_core: float, r_tube: float) -> Quadric:
    """Spherical donut (torus) extensor Q : Vector -> Plane on S²."""
    c_out = np.cos(r_core + r_tube)
    c_in = np.cos(r_core - r_tube)
    z0 = (c_out + c_in) / 2.0
    dz = (c_in - c_out) / 2.0
    return (
        pz * (pz & Vector)
        - z0 * (pz * (pw & Vector) + pw * (pz & Vector))
        + (z0**2 - dz**2) * pw * (pw & Vector)
    )


def make_conical_donut(a: float, b: float) -> Quadric:
    """Spherical donut (Limaçon) with central hole meeting in a razor conical apex on S²."""
    line = pw - pz - a * px
    return line * (line & Vector) - b**2 * (px * (px & Vector) + py * (py & Vector))


def make_bernoulli_lemniscate(scale_a: float) -> Quadric:
    """True Lemniscate of Bernoulli figure-8 on S²: (w - z)² - 2 a² (x² - y²) < 0."""
    return p_wz * (p_wz & Vector) - 2.0 * scale_a**2 * (px * (px & Vector) - py * (py & Vector))


def make_pinched_horn(r_outer: float) -> Quadric:
    """Pinched horn cyclide on S² whose central hole pinches to a single cusp."""
    c_out = np.cos(r_outer)
    c_in = 1.0
    z0 = (c_out + c_in) / 2.0
    dz = (c_in - c_out) / 2.0
    return (
        pz * (pz & Vector)
        - z0 * (pz * (pw & Vector) + pw * (pz & Vector))
        + (z0**2 - dz**2) * pw * (pw & Vector)
    )


def make_eccentric_cyclide(r_core: float, r_tube: float, boost_beta: float) -> Quadric:
    """Eccentric Dupin cyclide on S² with unequal tube width via Lorentz boost."""
    base_donut = make_spherical_donut(r_core, r_tube)
    boost = (mv.xw * (boost_beta / 2.0)).exp()
    return boost >> base_donut(boost << Vector)


def make_spherical_cassini(alpha1: float, alpha2: float, c_threshold: float) -> Quadric:
    """Spherical Cassini oval quadric (1 - n1.x)(1 - n2.x) < C * w² on S².

    Yields twin islands (C small), figure-8 lemniscates (C near pinch),
    pinched-waist peanuts (C larger), or asymmetric teardrops (alpha1 != alpha2).
    """
    p1 = pw - (np.sin(alpha1) * px + np.cos(alpha1) * pz)
    p2 = pw - (-np.sin(alpha2) * px + np.cos(alpha2) * pz)
    return 0.5 * (p1 * (p2 & Vector) + p2 * (p1 & Vector)) - c_threshold * pw * (pw & Vector)


def make_spherical_crescent(r_outer: float, r_inner: float, offset: float) -> Quadric:
    """Spherical crescent moon (sickle) bounded by two eccentric circles on S²."""
    c_outer = pz - np.cos(r_outer) * pw
    c_inner = (np.sin(offset) * px + np.cos(offset) * pz) - np.cos(r_inner) * pw
    return 0.5 * (c_outer * (c_inner & Vector) + c_inner * (c_outer & Vector))


def make_spherical_spindle(weight_z: float, weight_xy: float, bias: float) -> Quadric:
    """Spherical spindle with two opposite conical poles on S²."""
    return (
        weight_z * pz * (pz & Vector)
        - weight_xy * px * (px & Vector)
        - weight_xy * py * (py & Vector)
        - bias * pw * (pw & Vector)
    )


def make_spherical_clover(tilt_angle: float, radius: float, bias: float) -> Quadric:
    """Triadic 3-lobed rounded deltoid (cloverleaf) on S²."""
    angles = np.radians([0.0, 120.0, 240.0])
    q_clover = None
    for a in angles:
        nx = np.sin(tilt_angle) * np.cos(a)
        ny = np.sin(tilt_angle) * np.sin(a)
        nz = np.cos(tilt_angle)
        c_i = nx * px + ny * py + nz * pz - np.cos(radius) * pw
        term = c_i * (c_i & Vector)
        q_clover = term if q_clover is None else q_clover + term
    return q_clover - bias * pw * (pw & Vector)


def make_spherical_hourglass(theta: float, c_waist: float) -> Quadric:
    """True vertical hourglass on S²: two symmetric bulbs connected by a narrow waist."""
    p1 = pw - (np.sin(theta) * py + np.cos(theta) * pz)
    p2 = pw - (-np.sin(theta) * py + np.cos(theta) * pz)
    return 0.5 * (p1 * (p2 & Vector) + p2 * (p1 & Vector)) - c_waist * pw * (pw & Vector)


def make_spherical_parabola(weight_y: float, linear_x: float, bias: float) -> Quadric:
    """Parabolic bow curve on S²."""
    return (
        weight_y * py * (py & Vector)
        - 0.5 * linear_x * (px * (pw & Vector) + pw * (px & Vector))
        - bias * pw * (pw & Vector)
    )


def make_circle_intersection_vortex(
    center1: np.ndarray,
    radius1: float,
    center2: np.ndarray,
    radius2: float,
) -> Bivector:
    """Intersection 2-blade of two off-center circles on S²."""
    c1 = mv.vector([center1[0], center1[1], center1[2], np.cos(radius1)])
    c2 = mv.vector([center2[0], center2[1], center2[2], np.cos(radius2)])
    return (c1 ^ c2).normalized()


# ---------------------------------------------------------------------------
# 3. Orthographic Hemisphere Pixel Grid & Rendering
# ---------------------------------------------------------------------------
def make_hemisphere_pixels(
    resolution: int,
    supersample: int,
) -> tuple[Vector, np.ndarray, np.ndarray]:
    """Generate fixed null vector pixels on the front hemisphere of S²."""
    n = resolution * supersample
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, n), np.linspace(1.0, -1.0, n))
    r2 = u**2 + v**2
    inside = r2 <= 1.0
    z = np.sqrt(np.clip(1.0 - r2[inside], 0.0, None))
    w = np.ones_like(z)
    coords = np.stack([u[inside], v[inside], z, w], axis=-1)
    return mv.vector(coords), inside, r2


def render_frame(
    quadrics: list[Quadric],
    colors: list[str],
    pixels: Vector,
    inside: np.ndarray,
    r2: np.ndarray,
    resolution: int,
    supersample: int,
) -> np.ndarray:
    """Rasterise front hemisphere by evaluating Q(p) & p < 0 at null pixels."""
    n = resolution * supersample
    disk = np.where((r2[inside] > 0.985)[:, None], to_rgb(RIM), to_rgb(DISK))

    for Q, color in zip(quadrics, colors):
        covered = (Q(pixels) & pixels) < 0.0
        disk[covered] = to_rgb(color)

    image = np.empty((n, n, 3))
    image[:] = to_rgb(BACKGROUND)
    image[inside] = disk
    image = image.reshape(resolution, supersample, resolution, supersample, 3).mean(axis=(1, 3))
    return (image * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Shape Setup & Animation Runner
# ---------------------------------------------------------------------------
def setup_conical_donut_shape() -> tuple[list[Quadric], list[str]]:
    """Donut (solid ring + central hole) that narrows and meets in a sharp conical apex."""
    donut = make_conical_donut(0.55, 0.22)
    tilt = (mv.xz * -0.20).exp()
    return [tilt >> donut(tilt << Vector)], ["#38bdf8"]


def setup_peanut_shape() -> tuple[list[Quadric], list[str]]:
    """Pinched-waist Cassini oval (peanut) on S²."""
    peanut = make_spherical_cassini(np.radians(25.0), np.radians(25.0), 0.012)
    tilt = (mv.yz * 0.30).exp()
    return [tilt >> peanut(tilt << Vector)], ["#f59e0b"]


def setup_islands_shape() -> tuple[list[Quadric], list[str]]:
    """Two disconnected droplet beads (twin islands) on S²."""
    islands = make_spherical_cassini(np.radians(25.0), np.radians(25.0), 0.006)
    tilt = (mv.yz * 0.30).exp()
    return [tilt >> islands(tilt << Vector)], ["#10b981"]


def setup_lemniscate_shape() -> tuple[list[Quadric], list[str]]:
    """Figure-8 Bernoulli lemniscate with two symmetric lobes crossing at a node on S²."""
    lemn = make_bernoulli_lemniscate(0.38)
    return [lemn], ["#a855f7"]


def setup_crescent_shape() -> tuple[list[Quadric], list[str]]:
    """Eccentric sickle (crescent moon) with two sharp cusps on S²."""
    crescent = make_spherical_crescent(np.radians(45.0), np.radians(24.0), np.radians(16.0))
    tilt = (mv.yz * 0.25).exp()
    return [tilt >> crescent(tilt << Vector)], ["#fb7185"]


def setup_cyclide_shape() -> tuple[list[Quadric], list[str]]:
    """Single eccentric Dupin cyclide with unequal tube width."""
    cyclide = make_eccentric_cyclide(np.radians(34.0), np.radians(13.0), 0.65)
    tilt = (mv.yz * 0.30).exp()
    return [tilt >> cyclide(tilt << Vector)], ["#fbbf24"]


def setup_pinched_shape() -> tuple[list[Quadric], list[str]]:
    """Single pinched horn cyclide with singular self-touching cusp."""
    horn = make_pinched_horn(np.radians(65.0))
    tilt = (mv.yz * 0.35).exp()
    return [tilt >> horn(tilt << Vector)], ["#f43f5e"]


def setup_spindle_shape() -> tuple[list[Quadric], list[str]]:
    """Spindle / lemon with two sharp opposite conical poles on S²."""
    spindle = make_spherical_spindle(1.8, 0.8, 0.35)
    tilt = (mv.yz * 0.30).exp()
    return [tilt >> spindle(tilt << Vector)], ["#06b6d4"]


def setup_teardrop_shape() -> tuple[list[Quadric], list[str]]:
    """Asymmetric Cassini droplet tapering to a fine tail."""
    teardrop = make_spherical_cassini(np.radians(10.0), np.radians(35.0), 0.010)
    tilt = (mv.yz * 0.30).exp()
    return [tilt >> teardrop(tilt << Vector)], ["#ec4899"]


def setup_clover_shape() -> tuple[list[Quadric], list[str]]:
    """Triadic 3-lobed deltoid (cloverleaf) on S²."""
    clover = make_spherical_clover(np.radians(32.0), np.radians(28.0), 0.35)
    tilt = (mv.yz * 0.20).exp()
    return [tilt >> clover(tilt << Vector)], ["#4ade80"]


def setup_hourglass_shape() -> tuple[list[Quadric], list[str]]:
    """Vertical hourglass on S²: two symmetric bells connected by a narrow waist."""
    hour = make_spherical_hourglass(np.radians(24.0), 0.010)
    return [hour], ["#f59e0b"]


def setup_parabola_shape() -> tuple[list[Quadric], list[str]]:
    """Parabolic bow curve on S²."""
    parabola = make_spherical_parabola(1.0, 0.5, 0.20)
    tilt = (mv.yz * 0.25).exp()
    return [tilt >> parabola(tilt << Vector)], ["#f97316"]


def setup_trio_shape() -> tuple[list[Quadric], list[str]]:
    """Single scene combining conical donut, lemniscate, and hourglass by name on S²."""
    q_don, c_don = setup_conical_donut_shape()
    q_lem, c_lem = setup_lemniscate_shape()
    q_hour, c_hour = setup_hourglass_shape()
    return q_don + q_lem + q_hour, c_don + c_lem + c_hour


SHAPES = [
    ("conical_donut", setup_conical_donut_shape),
    ("peanut", setup_peanut_shape),
    ("islands", setup_islands_shape),
    ("lemniscate", setup_lemniscate_shape),
    ("crescent", setup_crescent_shape),
    ("cyclide", setup_cyclide_shape),
    ("pinched", setup_pinched_shape),
    ("spindle", setup_spindle_shape),
    ("teardrop", setup_teardrop_shape),
    ("clover", setup_clover_shape),
    ("hourglass", setup_hourglass_shape),
    ("parabola", setup_parabola_shape),
    ("trio", setup_trio_shape),
]


def render_vortex_animation(
    shape_name: str,
    frame_count: int,
    resolution: int,
    supersample: int,
    output_path: str,
) -> str:
    """Animate one or more CGA quadric shapes by the exp of two off-center circle intersections."""
    names = [s.strip() for s in shape_name.split(",")]
    shapes_dict = dict(SHAPES)
    quadrics_base = []
    colors = []
    for n in names:
        qs, cs = shapes_dict[n]()
        quadrics_base.extend(qs)
        colors.extend(cs)

    generator = make_circle_intersection_vortex(
        np.array([np.sin(0.35), 0.0, np.cos(0.35)]),
        np.radians(48.0),
        np.array([0.0, np.sin(0.40), np.cos(0.40)]),
        np.radians(52.0),
    )
    pixels, inside, r2 = make_hemisphere_pixels(resolution, supersample)

    phases = np.linspace(0.0, 2.0 * np.pi, frame_count, endpoint=False)
    frames = []
    for phase in phases:
        flow = (generator * (float(phase) / 2.0)).exp()
        quadrics_world = [flow >> Q(flow << Vector) for Q in quadrics_base]
        frame = render_frame(quadrics_world, colors, pixels, inside, r2, resolution, supersample)
        frames.append(frame)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_gif(frames, output_path, duration_ms=33, scale=1.0, colors=256)
    return output_path


def main(
    shape: str,
    frames: int,
    resolution: int,
    supersample: int,
    output_path: str,
) -> str:
    """CLI driver with explicit arguments."""
    render_vortex_animation(shape, frames, resolution, supersample, output_path)
    print(f"Exported {shape} vortex animation to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spherical Quadric Vortex in Cl(3, 1)")
    parser.add_argument("--shape", type=str, default="trio")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--resolution", type=int, default=300)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--output", type=str, default=str(PLOT_DIR / "cga_trio_vortex.gif"))
    args = parser.parse_args()

    main(
        shape=args.shape,
        frames=args.frames,
        resolution=args.resolution,
        supersample=args.supersample,
        output_path=args.output,
    )
