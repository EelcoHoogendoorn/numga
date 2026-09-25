"""Raster rendering on the 3-sphere: nearest body per pixel from the screen conics, then shading."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.quadrics.s3_raytracer.core import Motor, Point, Quadric, ScreenPoint, origin, pixel_chart, project, reproject


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
    opposite facing. The one great circle out of the light through the hit reaches it along an
    arc; the surface is lit where that arc arrives from outside, which the pairing of the polar
    plane with the light decides, with the flux falloff 1 / arc.sin()**2 of a point source.
    """
    polar = surfaces[body_idx](hit)
    with np.errstate(invalid="ignore", divide="ignore"):
        # The arc from the light to the hit, between 0 and np.pi.
        arc = ((hit | light) / (light | light)).clip(-1.0, 1.0).arccos()
        sine = arc.sin()
        # Negative where the light is outside.
        cosine = (polar & light) / (polar.norm() * sine)
        lambert = np.where(cosine < 0.0, (-cosine / (sine * sine)).to_array(), 0.0)
    lambert = np.where(shadowed(hit, surfaces, light), 0.0, lambert)
    # Radiance carries undimmed.
    return colors[body_idx] * (0.15 + 0.85 * np.clip(lambert, 0.0, 1.0))[:, None]


def render(eye_frame: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, chart: ScreenPoint, shape: tuple[int, int], supersample: int) -> np.ndarray:
    """Accumulate the nearest body from screen conics, then gather surfaces and shade every pixel."""
    conics, polars = project(eye_frame, surfaces)
    depth = np.full(chart.shape, -np.inf)
    body_idx = np.zeros(chart.shape, dtype=int)
    for body in range(surfaces.shape[0]):
        candidate = reproject(conics[body], polars[body], chart)
        # A miss is infinitely far.
        candidate = np.where(candidate.isnan(), -np.inf, candidate.to_array())
        nearer = candidate > depth
        depth = np.where(nearer, candidate, depth)
        body_idx = np.where(nearer, body, body_idx)
    visible = np.isfinite(depth)
    depth = np.where(visible, depth, 0.0)
    # The point of S³ hit first, sign and all.
    hit = ((eye_frame >> origin) * depth + (eye_frame >> chart)).normalized()
    image = np.where(visible[:, None], shade(hit, surfaces, body_idx, colors, light), 0.02)
    rows, cols = shape
    return np.clip(image.reshape(rows, supersample, cols, supersample, 3).mean(axis=(1, 3)), 0.0, 1.0)


def frames(eye_frames: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
           shape: tuple[int, int], supersample: int) -> list[np.ndarray]:
    """One RGB frame per eye frame, through a pinhole of the given field of view."""
    chart = pixel_chart(fov, (shape[0] * supersample, shape[1] * supersample))
    return [(render(eye, surfaces, colors, light, chart, shape, supersample) * 255).astype(np.uint8) for eye in eye_frames]


def draw_walk(eye_frames: Motor, surfaces: Quadric, colors: np.ndarray, light: Point, fov: float,
              shape: tuple[int, int], supersample: int) -> plt.Figure:
    image = frames(eye_frames[:1], surfaces, colors, light, fov, shape, supersample)[0]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    ax.imshow(image, interpolation="nearest")
    ax.axis("off")
    ax.set_title("four identical ellipsoids at angles 0.7, 1.4, 2.1, 2.6 along the line of sight")
    return fig
