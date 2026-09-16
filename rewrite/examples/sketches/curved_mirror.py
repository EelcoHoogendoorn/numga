"""Curved mirrors in PGA2D: a mirror is a conic, and a conic is a polarity.

A conic is the map from a point to its polar line; at a point of the conic that line is the
tangent there. So a ray reflects off the mirror by the sandwich with the tangent at its hit.
A parabola sends every ray parallel to its axis through its focus. A sphere with the same
paraxial focal length does not: its axis crossings walk inward with ray height, which is
spherical aberration, straight from the reflection law with no paraxial expansion anywhere.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.sketches.thin_lens import Line, Point, draw_rays, ga, mv, xy

Polarity = ga.gatype((Line.output_subspace, Point.output_subspace))     # polar line <= point


# --- plumbing -------------------------------------------------------------------------
def meet_conic(rays: Line, conic: Polarity, start: Line) -> Point:
    """Where each ray first meets the conic, travelling from its point on the start plane.

    Along the ray a + t·b, with a on the start plane and b the ray's ideal point, the conic
    restricted to the ray is the quadratic Q(a,a) + 2t Q(a,b) + t² Q(b,b) with Q(u, v) = u ∨ C(v).
    """
    a, b = rays ^ start, rays ^ mv.w
    Qaa, Qab, Qbb = (a & conic(a)).kernel[:, 0], (a & conic(b)).kernel[:, 0], (b & conic(b)).kernel[:, 0]
    linear = np.abs(Qbb) < 1e-12
    t = np.where(linear, -Qaa / (2 * Qab), (-Qab + np.sqrt(np.maximum(Qab**2 - Qaa * Qbb, 0.0))) / np.where(linear, 1.0, Qbb))
    return a + b * t


def draw_conic(ax, conic: Polarity, box: tuple[float, float, float, float]) -> None:
    """Draw the zero level of p ∨ C(p) over a grid."""
    x, y = np.meshgrid(np.linspace(box[0], box[1], 300), np.linspace(box[2], box[3], 300))
    grid = mv.antivector(np.stack([x, y, np.ones_like(x)], axis=-1))
    ax.contour(x, y, (grid & conic(grid)).kernel[..., 0], levels=[0.0], colors="gray")


# --- math -----------------------------------------------------------------------------
def main(plot_path: str = str(PLOT_DIR / "sketch_curved_mirror.png")) -> plt.Figure:
    # Two mirrors with vertex at the origin opening toward +x and the same paraxial focal
    # length f: the parabola x = y²/4f and the sphere of radius 2f, each as the polarity whose
    # quadratic form p ∨ C(p) vanishes on the curve.
    f = 1.0
    parabola = mv.y * (mv.y & Point) - (mv.x * (mv.w & Point) + mv.w * (mv.x & Point)) * (2 * f)
    sphere = mv.x * (mv.x & Point) + mv.y * (mv.y & Point) - (mv.x * (mv.w & Point) + mv.w * (mv.x & Point)) * (2 * f)
    focus = (mv.x - mv.w * f) ^ mv.y

    # A beam parallel to the axis, coming in from the far plane. The polarity applied to the
    # hit point is the tangent line there; reflecting the beam is the sandwich by that tangent,
    # normalised so the sandwich is a certified reflection.
    far = mv.x - mv.w * 3.0
    beam = mv.y - mv.w * np.linspace(-1.4, 1.4, 8)                 # even count: no axial ray
    hit_p = meet_conic(beam, parabola, far)
    hit_s = meet_conic(beam, sphere, far)
    mirrored_p = parabola(hit_p).normalized() >> beam
    mirrored_s = sphere(hit_s).normalized() >> beam
    print("spherical mirror, axis crossings by ray height:", np.round(xy(mirrored_s ^ mv.y)[:, 0], 3))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)
    for ax, conic, hit, mirrored, title in (
        (axes[0], parabola, hit_p, mirrored_p, "parabolic mirror: one focus"),
        (axes[1], sphere, hit_s, mirrored_s, "spherical mirror: aberration"),
    ):
        draw_rays(ax, beam, far, hit & mv.wx, "tab:orange")
        draw_rays(ax, mirrored, hit & mv.wx, (mv.wx * -0.4).exp() >> (focus & mv.wx), "tab:blue")
        draw_conic(ax, conic, (-0.2, 3.0, -1.6, 1.6))
        ax.scatter(*xy(focus), color="tab:red", zorder=3); ax.set_title(title)
        ax.set_xlim(-0.3, 3.1); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal")
    if plot_path:
        fig.savefig(plot_path, bbox_inches="tight")
        print(f"Figure saved to {plot_path}")

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    # Every parabola reflection passes through the focus; the sphere's crossings spread.
    np.testing.assert_allclose((mirrored_p ^ focus).kernel, 0.0, atol=1e-12)
    assert np.ptp(xy(mirrored_s ^ mv.y)[:, 0]) > 0.1
    return fig


if __name__ == "__main__":
    main()
