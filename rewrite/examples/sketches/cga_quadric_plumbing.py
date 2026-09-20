"""CGA primitives, quartic roots and image output."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import mkstemp

import numpy as np
from PIL import Image

from numga import Algebra, NumpyContext
from examples import PLOT_DIR
from examples.animation import save_gif

ga = Algebra("x+y+z+p+n-")
ctx = NumpyContext(ga)
mv = ctx.multivector
Sphere = ga.gatype.vector()
Point = ga.gatype.antivector()
Direction = ga.gatype.from_blades("x y z")
Scalar = ga.gatype.scalar()
Quadric = ga.gatype((Sphere, Point))

infinity = mv.p + mv.n
zero_sphere = (mv.n - mv.p) * 0.5
origin = (mv.x ^ mv.y ^ mv.z ^ zero_sphere).cast(Point.output_subspace)


def nearest_depth(*coefficients: Scalar) -> np.ndarray:
    """Largest positive reciprocal root, or zero on a miss.

    Arguments are ascending distance-polynomial coefficients. Reversing them
    solves inverse distance: lower degrees contribute zero roots at infinity.
    The constant coefficient must be nonzero (the camera is off the surface).
    """
    c = np.stack(np.broadcast_arrays(*(c.to_array() for c in coefficients)), axis=-1)
    companion = np.zeros(c.shape[:-1] + (4, 4))
    companion[..., 1:, :-1] = np.eye(3)
    companion[..., :, -1] = -c[..., 4:0:-1] / c[..., :1]
    roots = np.linalg.eigvals(companion)
    return np.where((roots.imag == 0) & (roots.real > 0), roots.real, 0).max(axis=-1)


def shade(states, up: Direction, shape: tuple[int, int], supersample: int) -> np.ndarray:
    """Shade the nearest bodies and average subpixels in linear light."""
    rgb = []
    for normal, direction, visible, base, main_light_direction, fill_light_direction, main_strength, fill_strength in states:
        diffuse = (normal | main_light_direction).clip(0, 1) * main_strength
        soft = (normal | fill_light_direction).clip(0, 1) * fill_strength
        facing = (normal | -direction).clip(0, 1)
        half = (main_light_direction - direction).normalized()
        gloss = (normal | half).clip(0, 1)
        gloss = gloss.squared().squared().squared().squared().squared()
        rim = (1 - facing).squared().squared()

        highlight = np.array([1.0, 0.86, 0.63])
        color = base * (0.14 + 0.8 * diffuse + 0.3 * soft).to_array()[:, None]
        color += highlight * (0.75 * gloss * main_strength).to_array()[:, None]
        color += np.array([0.2, 0.38, 0.5]) * (0.35 * rim).to_array()[:, None]
        sky = (direction | up).to_array()
        background = np.array([0.013, 0.02, 0.036]) + (sky[:, None] + 1) * np.array([0.008, 0.011, 0.014])
        rgb.append(np.where(visible[:, None], color, background))

    height, width = shape
    linear = np.concatenate(rgb).reshape(height, supersample, width, supersample, 3).mean(axis=(1, 3))
    image = np.clip(linear, 0, 1) ** (1 / 2.2)
    return image


def export(images, name: str, output_dir: Path, scale: float, duration_ms: int) -> str:
    """One frame becomes a PNG; a sequence becomes a shared-palette GIF."""
    frames = []
    for frame, image in enumerate(images):
        frames.append(np.round(image * 255).astype(np.uint8))
        print(f"Rendered frame {frame + 1}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if len(frames) == 1 else ".gif"
    # Reserve a unique filename atomically; only this newly created file is written.
    descriptor, path = mkstemp(prefix=f"cga_{name}_", suffix=suffix, dir=output_dir)
    os.close(descriptor)
    if len(frames) == 1:
        image = Image.fromarray(frames[0])
        image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BOX).save(path)
        print(f"Figure saved to {path}")
    else:
        save_gif(frames, path, duration_ms=duration_ms, scale=scale, colors=256)
    return path


def arguments() -> dict:
    from examples.sketches.cga_quadric_scenarios import SCENES

    parser = argparse.ArgumentParser(description="Trace a CGA cyclide or animate its circle vortex.")
    parser.add_argument("--scene", choices=SCENES, default="cyclide")
    parser.add_argument("--frames", dest="frame_count", type=int, default=1)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--supersample", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--duration-ms", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=PLOT_DIR)
    options = vars(parser.parse_args())
    options["shape"] = (options.pop("height"), options.pop("width"))
    options["name"] = options.pop("scene")
    return options
