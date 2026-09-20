"""Rendering utilities for rigid body physics simulations."""

from __future__ import annotations

from typing import Callable, Sequence
import numpy as np

try:
    import imageio.v3 as iio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from numga import Context, Extensor


def quantize_image(data: np.ndarray, n_colors: int = 32) -> np.ndarray:
    """Quantize floating point image to uint8 palette."""
    norm = data - data.min()
    max_val = norm.max()
    if max_val > 0:
        norm = norm / max_val
    palette = np.linspace(0, 255, n_colors, endpoint=True, dtype=np.uint8)
    indices = np.clip((norm * (n_colors - 1)).round().astype(int), 0, n_colors - 1)
    return palette[indices]


def render_chain_2d(bodies: object, resolution: int = 200, radius: float = 0.08) -> np.ndarray:
    """Render a 2D projection of the rigid body positions."""
    motor = bodies if isinstance(bodies, Extensor) else bodies.motor
    context = motor.context
    ndim = context.algebra.dimension - 1
    origin_coords = np.zeros(len(context.algebra.subspace.antivector()), dtype=float)
    origin_coords[-1] = 1.0
    origin_pt = context.multivector.antivector(origin_coords)

    # World positions of each body link: shape (N_bodies, n_coords)
    world_pts = motor >> origin_pt
    # Extract x and y coordinates
    pts = world_pts.kernel[..., :2]

    # Render on a 2D pixel grid
    xs = np.linspace(-1.5, 1.5, resolution)
    ys = np.linspace(-1.5, 1.5, resolution)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    grid = np.stack([gx, gy], axis=-1)  # (H, W, 2)

    # Distance to closest link
    # grid: (H, W, 1, 2), pts: (1, 1, N, 2)
    diff = grid[:, :, None, :] - pts[None, None, :, :]
    dist = np.linalg.norm(diff, axis=-1).min(axis=-1)  # (H, W)

    # Pixel intensity: inside radius -> bright, outside -> dark
    img = np.clip(1.0 - dist / radius, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)


def write_simulation_gif(
    frames: Sequence[np.ndarray],
    output_filename: str = "simulation.gif",
    fps: int = 20,
) -> None:
    """Write an array of 2D image frames to an animated GIF."""
    if not HAS_IMAGEIO:
        raise RuntimeError("imageio is required to write simulation GIF animations")
    iio.imwrite(output_filename, np.asarray(frames), fps=fps, loop=0)


def render_simulation(
    states: Sequence[Body],
    output_filename: str = "simulation.gif",
    resolution: int = 200,
    fps: int = 20,
) -> np.ndarray:
    """Render a sequence of Body simulation states to frames and optionally save a GIF."""
    frames = [render_chain_2d(b, resolution=resolution) for b in states]
    frames_arr = np.array(frames)
    if output_filename and HAS_IMAGEIO:
        write_simulation_gif(frames_arr, output_filename=output_filename, fps=fps)
    return frames_arr
