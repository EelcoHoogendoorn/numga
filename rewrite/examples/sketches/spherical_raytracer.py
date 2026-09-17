"""Rendering quadrics on the 3-sphere, Cl(4), by projection: a quadric projects to a conic.

A body is a dual quadric cap placed by a motor, exactly as in the spherical quadric physics one
dimension down. Seen from an eye E, its outline on the image sphere a quarter turn ahead is
the body projected from E: the dual quadric pushed through the central projection
M = (E ∨ Point) ∧ image, the same pushforward the lens camera applies to its aperture. Pulled
into the eye's frame that is a conic in the pixel chart (1, u, v), and a pixel is inside the
outline where the conic's form is negative. Together with the eye's projected polar plane,
that conic gives a depth proportional to cot t, with t the angle travelled along the great
circle. The largest depth selects the body per pixel; only that hit is reconstructed and
shaded with its polar plane. The check is that a pixel passes the 2D test exactly when its
great circle through the body has real roots.

The spherical signature shows in one thing: identical caps placed further and further away
along a geodesic do not keep shrinking. Past a quarter turn the great circles reconverge on
the antipode of the eye, and the farthest cap looms larger than the middle one.
"""

from __future__ import annotations

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from numga import Algebra, NumpyContext
from numga.gatype.traits import Versor
from numga.subspace import SubSpaceFactory

from examples import PLOT_DIR
from examples.animation import save_gif

ga = Algebra("x+y+z+w+", subspace_factory=partial(SubSpaceFactory, default="1 x y z w yz zx xy xw yw zw yzw zxw xyw zyx xyzw"))
ctx = NumpyContext(ga)
mv = ctx.multivector
P = ga.subspace.antivector()
Point = ga.gatype.antivector()
ScreenPoint = ga.gatype.from_blades("yzw zxw xyw")
ScreenConic = ga.gatype((ga.gatype.scalar(), ScreenPoint, ScreenPoint))
ScreenPolar = ga.gatype((ga.gatype.scalar(), ScreenPoint))
Plane = ga.gatype.vector()
Motor = ga.gatype.rotor()
Quadric = ga.gatype((Plane, Point))           # primal quadric: polar plane <= point
DualQuadric = ga.gatype((Point, Plane))       # dual quadric: pole <= plane


# --- plumbing -------------------------------------------------------------------------
def unit(points: Point) -> Point:
    """The representative of each point with positive unit weight; a unit point is a versor."""
    return (points / (mv.w & points)).normalized().with_traits(Versor)


origin = unit(mv.zyx)


def direction(coords: np.ndarray) -> Point:
    """A point a quarter turn from the origin: the direction (x, y, z) seen from it."""
    return mv.antivector(np.concatenate([coords, np.zeros_like(coords[..., :1])], axis=-1)).normalized()


def pixel_chart(fov: float, shape: tuple[int, int]) -> ScreenPoint:
    """Homogeneous screen points (1, u, v) of a pinhole grid looking along +x from the origin."""
    v, u = np.meshgrid(np.linspace(1.0, -1.0, shape[0]) * shape[0] / shape[1], np.linspace(-1.0, 1.0, shape[1]), indexing="ij")
    return mv.yzw + mv.zxw * (np.tan(fov / 2) * u.ravel()) + mv.xyw * (np.tan(fov / 2) * v.ravel())


def panorama_chart(shape: tuple[int, int]) -> ScreenPoint:
    """Direction points of an equirectangular panorama: every direction, longitude across, latitude down."""
    latitude, longitude = np.meshgrid(np.linspace(np.pi / 2, -np.pi / 2, shape[0]), np.linspace(-np.pi, np.pi, shape[1]), indexing="ij")
    return mv(ScreenPoint, np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=-1).reshape(-1, 3))


def outlines(eye_frame: Motor, surfaces: Quadric) -> Quadric:
    """The cone of rays from the eye tangent to each body: the eye's polar plane squared, less the
    form scaled by the eye's own value. Non-negative on the directions whose ray meets the body."""
    eye = eye_frame >> origin
    polar = surfaces(eye)
    return polar * (Point & polar) - surfaces * (eye & polar)


def inside(cone: Quadric, rays: Point) -> np.ndarray:
    """Hit test of ray directions against the bodies' outline cones."""
    return (rays & cone.reshape(-1, 1)(rays)) >= 0.0


def project(eye_frame: Motor, surfaces: Quadric) -> tuple[ScreenConic, ScreenPolar]:
    """Project each quadric to its screen conic and eye-polar linear form; the eye must be off-surface."""
    eye = eye_frame >> origin
    screen = eye_frame >> ScreenPoint
    normalized = surfaces * (eye & surfaces(eye)).inverse()
    polar = screen & normalized(eye)
    return (screen & normalized(screen)) - polar * polar, polar


def reproject(conic: ScreenConic, polar: ScreenPolar, pixels: ScreenPoint) -> np.ndarray:
    """First-hit screen depth, proportional to cot(angle): larger is nearer; -inf means a miss."""
    with np.errstate(invalid="ignore"):
        depth = -polar(pixels) + (-conic(pixels, pixels)).square_root()
    return np.where(depth.isnan(), -np.inf, depth.to_array())


def shadowed(hit: Point, surfaces: Quadric, light: Point) -> np.ndarray:
    """Test the short hit-to-light arcs for entry into any negative-inside quadric, without roots."""
    hit = hit + (light - hit) * 1e-6
    polar = surfaces(light)
    l = light & polar
    blocked = np.zeros(hit.shape, dtype=bool)
    for body in range(surfaces.shape[0]):
        h = hit & surfaces[body](hit)
        m = hit & polar[body]
        blocked |= (h < 0.0) | (l[body] < 0.0) | ((m < 0.0) & (m * m > h * l[body]))
    return blocked


def shade(hit: Point, surfaces: Quadric, body_idx: np.ndarray, colors: np.ndarray, light: Point) -> np.ndarray:
    """Lighting is done on the 3-sphere itself, not on the projective space: the light is one point of
    S³, and the hit is the point of S³ the ray reaches first, kept with its own sign rather than
    re-signed to a positive weight, since the antipode of a hit is a different point with the
    opposite facing. The one great circle out of the light through the hit reaches it along the
    arc of length t; the surface is lit where that arc arrives from outside, which the pairing of
    the polar plane with the light decides, with the flux falloff 1/sin²t of a point source.
    """
    polar = surfaces[body_idx](hit)
    with np.errstate(invalid="ignore", divide="ignore"):
        arc = ((hit | light) / (light | light)).clip(-1.0, 1.0).arccos()     # from the light to the hit, 0..π
        sine = arc.sin()
        cosine = (polar & light) / (polar.norm() * sine)                   # negative where the light is outside
        lambert = np.where(cosine < 0.0, (-cosine / (sine * sine)).to_array(), 0.0)
    lambert = np.where(shadowed(hit, surfaces, light), 0.0, lambert)
    return colors[body_idx] * (0.15 + 0.85 * np.clip(lambert, 0.0, 1.0))[:, None]   # radiance carries undimmed


def render(eye_frame: Motor, bodies: DualQuadric, surfaces: Quadric, colors: np.ndarray, light: Point, chart: ScreenPoint, shape: tuple[int, int], supersample: int) -> np.ndarray:
    """Accumulate the nearest body from screen conics, then gather surfaces and shade every pixel."""
    conics, polars = project(eye_frame, surfaces)
    depth = np.full(chart.shape, -np.inf)
    body_idx = np.zeros(chart.shape, dtype=int)
    for body in range(surfaces.shape[0]):
        candidate = reproject(conics[body], polars[body], chart)
        nearer = candidate > depth
        depth = np.where(nearer, candidate, depth)
        body_idx = np.where(nearer, body, body_idx)
    visible = np.isfinite(depth)
    depth = np.where(visible, depth, 0.0)
    hit = ((eye_frame >> origin) * depth + (eye_frame >> chart)).normalized()   # the point of S³ hit first, sign and all
    image = np.where(visible[:, None], shade(hit, surfaces, body_idx, colors, light), 0.02)
    rows, cols = shape
    return np.clip(image.reshape(rows, supersample, cols, supersample, 3).mean(axis=(1, 3)), 0.0, 1.0)


# --- math -----------------------------------------------------------------------------
def draw_walk(eye_frames: Motor, bodies: DualQuadric, surfaces: Quadric, colors, light: Point, chart: ScreenPoint, shape, supersample, plot_path: str, animation_path: str) -> plt.Figure:
    image = render(eye_frames[0], bodies, surfaces, colors, light, chart, shape, supersample)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(image, interpolation="nearest"); ax.axis("off")
    ax.set_title("four identical caps at angles 0.7, 1.4, 2.1, 2.6 along the line of sight")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")

    if animation_path:
        frames_out = [(render(eye, bodies, surfaces, colors, light, chart, shape, supersample) * 255).astype(np.uint8)
                      for eye in eye_frames]
        save_gif(frames_out, animation_path, duration_ms=80)
    return fig


def main(plot_path: str = str(PLOT_DIR / "sketch_spherical_raytracer.png"), animation_path: str = str(PLOT_DIR / "sketch_spherical_raytracer.gif")) -> plt.Figure:
    # A cap: the dual quadric with the origin as centre, principal half-widths tan(θ) along the
    # basis directions, and the -1 on the centre that closes it. Four copies of one shape,
    # carried to increasing angular distances along +x and offset so they don't overlap.
    dx, dy, dz = direction(np.eye(3))
    cap = (dx * (dx & Plane)) * np.tan(0.2)**2 + (dy * (dy & Plane)) * np.tan(0.28)**2 + (dz * (dz & Plane)) * np.tan(0.15)**2 - origin * (origin & Plane)
    distances = np.array([0.7, 1.4, 2.1, 2.6])
    sideways, upward = np.array([-0.4, 0.4, -0.4, 0.4]), np.array([-0.3, -0.3, 0.3, 0.3])
    colors = np.array([[0.9, 0.3, 0.3], [0.3, 0.8, 0.4], [0.3, 0.5, 0.95], [0.95, 0.8, 0.3]])
    placed = (mv.xw * (distances / 2)).exp() * (mv.yw * (sideways / 2)).exp() * (mv.zw * (upward / 2)).exp() * (mv.xy * 0.4).exp()
    bodies = placed >> cap(placed << Plane)                       # dual quadrics in the world
    surfaces = bodies.inverse()                                   # their primal forms: polar plane <= point
    light = direction(np.array([-0.4, 0.6, 0.7]))

    shape, supersample = (180, 240), 2
    chart = pixel_chart(np.radians(80.0), (shape[0] * supersample, shape[1] * supersample))

    # The eye walks along the x geodesic; the scene remains fixed in the world.
    eye_frames = (mv.xw * (np.linspace(0.0, 1.2, 48, endpoint=False) / 2)).exp()
    fig = draw_walk(eye_frames, bodies, surfaces, colors, light, chart, shape, supersample, plot_path, animation_path)

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    # A ray is inside a body's outline cone exactly when its great circle origin + λ·dir meets the
    # body: the quadratic a λ² + 2 b λ + c has real roots. Compared away from the outline itself.
    covered = inside(outlines(mv.rotor(), surfaces), chart)
    k_eye, k_dir = surfaces.reshape(-1, 1)(origin), surfaces.reshape(-1, 1)(chart)
    a, b, c = (chart & k_dir).kernel[..., 0], (chart & k_eye).kernel[..., 0], (origin & k_eye).kernel[..., 0]
    disc = b * b - a * c
    clear = np.abs(disc) > 1e-3 * np.abs(disc).max(axis=1, keepdims=True)
    assert np.array_equal(covered[clear], (disc >= 0.0)[clear])
    # Reprojected hits lie on their surfaces.
    conics, polars = project(mv.rotor(), surfaces)
    np.testing.assert_allclose(conics.reshape(-1, 1)(chart, chart).kernel[..., 0], -disc / c**2, atol=1e-8)
    for body in range(4):
        pixels = np.nonzero(covered[body])[0][::50]
        depth = reproject(conics[body], polars[body], chart[pixels])
        hit = (origin * depth + chart[pixels]).normalized()
        np.testing.assert_allclose((hit & surfaces[body](hit)).kernel, 0.0, atol=1e-8)
    return fig


if __name__ == "__main__":
    main()
