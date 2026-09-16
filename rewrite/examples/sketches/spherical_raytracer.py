"""Rendering quadrics on the 3-sphere, Cl(4), by projection: a quadric projects to a conic.

A body is a dual quadric cap placed by a motor, exactly as in the spherical quadric physics one
dimension down. Seen from an eye E, its outline on the image sphere a quarter turn ahead is
the body projected from E: the dual quadric pushed through the central projection
M = (E ∨ Point) ∧ image, the same pushforward the lens camera applies to its aperture. Pulled
into the eye's frame that is a conic in the pixel chart (1, u, v), and a pixel is inside the
outline where the conic's form is negative. That 2D test culls; the covered pixels are then
reprojected into the scene along their great circles, a quadratic in the affine parameter
λ = tan t with t the angle travelled, to order the bodies per pixel and to shade with the
polar plane at the hit. The check is that a pixel passes the 2D test exactly when its great
circle through the body has real roots.

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


def pixel_chart(fov: float, shape: tuple[int, int]) -> Point:
    """Direction points of a pinhole grid on the image sphere, looking along +x from the origin."""
    v, u = np.meshgrid(np.linspace(1.0, -1.0, shape[0]) * shape[0] / shape[1], np.linspace(-1.0, 1.0, shape[1]), indexing="ij")
    return direction(np.stack([np.ones_like(u), np.tan(fov / 2) * u, np.tan(fov / 2) * v], axis=-1).reshape(-1, 3))


def panorama_chart(shape: tuple[int, int]) -> Point:
    """Direction points of an equirectangular panorama: every direction, longitude across, latitude down."""
    latitude, longitude = np.meshgrid(np.linspace(np.pi / 2, -np.pi / 2, shape[0]), np.linspace(-np.pi, np.pi, shape[1]), indexing="ij")
    return direction(np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=-1).reshape(-1, 3))


def outlines(eye_frame: Motor, surfaces: Quadric) -> Quadric:
    """The cone of rays from the eye tangent to each body: the eye's polar plane squared, less the
    form scaled by the eye's own value. Non-negative on the directions whose ray meets the body."""
    eye = eye_frame >> origin
    polar = surfaces(eye)
    return polar * (Point & polar) - surfaces * (eye & polar)


def inside(cone: Quadric, rays: Point) -> np.ndarray:
    """Hit test of ray directions against the bodies' outline cones."""
    return (rays & cone.reshape(-1, 1)(rays)).kernel[..., 0] >= 0.0


def reproject(eye_frame: Motor, surface: Quadric, rays: Point, light: Point) -> tuple[np.ndarray, np.ndarray]:
    """For ray directions, the angle travelled along the great circle from the eye to the first hit on
    the surface, and the shading there; inf and 0 where the circle misses.

    Lighting is done on the 3-sphere itself, not on the projective space: the light is one point of
    S³, and the hit is the point of S³ the ray reaches first, kept with its own sign rather than
    re-signed to a positive weight, since the antipode of a hit is a different point with the
    opposite facing. The one great circle out of the light through the hit reaches it along the
    arc of length t; the surface is lit where that arc arrives from outside, which the pairing of
    the polar plane with the light decides, with the flux falloff 1/sin²t of a point source.
    """
    eye = eye_frame >> origin
    k_eye, k_dir = surface(eye), surface(rays)
    a, b, c = (rays & k_dir).kernel[..., 0], (rays & k_eye).kernel[..., 0], (eye & k_eye).kernel[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        root = np.sqrt(b * b - a * c)
        angle = (np.arctan((-b + np.stack([root, -root])) / a) % np.pi).min(axis=0)   # the nearer of the two roots
    hit = eye * np.cos(angle) + rays * np.sin(angle)                  # the point of S³ hit first, sign and all
    polar = surface(hit)
    with np.errstate(invalid="ignore", divide="ignore"):
        arc = np.arccos(np.clip(((hit | light) / (light | light)).kernel[..., 0], -1.0, 1.0))     # from the light to the hit, 0..π
        cosine = (polar & light).kernel[..., 0] / (polar.norm().kernel[..., 0] * np.sin(arc))    # negative where the light is outside
        lambert = np.where(cosine < 0.0, -cosine / np.sin(arc) ** 2, 0.0)
    missed = np.isnan(angle)
    return np.where(missed, np.inf, angle), np.where(missed, 0.0, np.clip(lambert, 0.0, 1.0))


def render(eye_frame: Motor, bodies: DualQuadric, surfaces: Quadric, colors: np.ndarray, light: Point, chart: Point, shape: tuple[int, int], supersample: int) -> np.ndarray:
    """Cull by the outline cones, then order and shade: reproject only the covered rays into the
    scene for depth and the polar-plane normal."""
    rays = eye_frame >> chart
    covered = inside(outlines(eye_frame, surfaces), rays)
    depth, lambert = np.full(covered.shape, np.inf), np.zeros(covered.shape)
    for body in range(len(colors)):
        pixels = np.nonzero(covered[body])[0]
        depth[body, pixels], lambert[body, pixels] = reproject(eye_frame, surfaces[body], rays[pixels], light)
    nearest = depth.argmin(axis=0)
    image = colors[nearest] * (0.15 + 0.85 * lambert[nearest, np.arange(len(nearest))])[:, None]   # radiance carries undimmed
    image[~np.isfinite(depth.min(axis=0))] = 0.02
    rows, cols = shape
    return np.clip(image.reshape(rows, supersample, cols, supersample, 3).mean(axis=(1, 3)), 0.0, 1.0)


# --- math -----------------------------------------------------------------------------
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

    image = render(mv.rotor(), bodies, surfaces, colors, light, chart, shape, supersample)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(image, interpolation="nearest"); ax.axis("off")
    ax.set_title("four identical caps at angles 0.7, 1.4, 2.1, 2.6 along the line of sight")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")

    # The eye walks forward along the x geodesic; caps slide past and the far ones loom.
    frames_out = [(render((mv.xw * (step / 2)).exp(), bodies, surfaces, colors, light, chart, shape, supersample) * 255).astype(np.uint8) for step in np.linspace(0.0, 1.2, 48, endpoint=False)]
    if animation_path:
        save_gif(frames_out, animation_path, duration_ms=80)

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
    for body in range(4):
        pixels = np.nonzero(covered[body])[0][::50]
        angle, _ = reproject(mv.rotor(), surfaces[body], chart[pixels], light)
        hit = origin * np.cos(angle) + chart[pixels] * np.sin(angle)
        np.testing.assert_allclose((hit & surfaces[body](hit)).kernel, 0.0, atol=1e-8)
    return fig


if __name__ == "__main__":
    main()
