"""Drawing for quadric physics on S² and S³."""

from __future__ import annotations

from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga.algebras import Spherical3D
from examples import instantiate
from examples.quadrics.s3_raytracer import core as raytracer
from examples.quadrics.s3_raytracer.render import picture as trace
from examples.wgsl import Shader, image_figure, supersampled

S2 = instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)
S3 = instantiate("examples.quadrics.elliptic_physics.core", raytracer.ga)

BACKGROUND, DISK, RIM = "#0a0f1d", "#111827", "#334155"


# --- the front hemisphere of S² -------------------------------------------------------
def hemisphere(size: int):
    """The front hemisphere seen along z on a size by size grid: coordinates (m, 3) of the pixels
    inside the disk, the disk mask, and which of the disk's pixels form its rim."""
    u, v = np.meshgrid(np.linspace(-1.0, 1.0, size), np.linspace(1.0, -1.0, size))
    r2 = u**2 + v**2
    inside = r2 <= 1.0
    coordinates = np.stack([u, v, np.sqrt(np.clip(1.0 - r2, 0.0, None))], axis=-1)[inside]
    return coordinates, inside, r2[inside] > 0.985


def paint(coverage: np.ndarray, colors: np.ndarray, inside: np.ndarray, rim: np.ndarray, supersample: int) -> np.ndarray:
    """An RGB image of the disk: each pixel takes the colour of the last body covering it.
    coverage is (bodies, pixels) boolean; the image is box-filtered by the supersampling."""
    covered = coverage.any(axis=0)
    last = coverage.shape[0] - 1 - coverage[::-1].argmax(axis=0)
    disk = np.where(covered[:, None], colors[last], np.where(rim[:, None], to_rgb(RIM), to_rgb(DISK)))
    image = np.empty(inside.shape + (3,))
    image[:] = to_rgb(BACKGROUND)
    image[inside] = disk
    size = inside.shape[0] // supersample
    image = image.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
    return (image * 255.0).round().astype(np.uint8)


def picture(image: np.ndarray) -> plt.Figure:
    """A painted image of the hemisphere as a figure, at its own size."""
    height, width = image.shape[:2]
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = figure.add_axes((0, 0, 1, 1))
    ax.imshow(image)
    ax.set_axis_off()
    return figure


def rgb(colors: list[str]) -> np.ndarray:
    """RGB rows (bodies, 3) of matplotlib colour names."""
    return np.array([to_rgb(color) for color in colors])


def hues(fractions: np.ndarray) -> np.ndarray:
    """RGB rows (bodies, 3) around the hue circle."""
    return plt.get_cmap("hsv")(fractions)[:, :3]


# --- S² -------------------------------------------------------------------------------
source = """
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;
var<uniform> disk: vec3<f32>;
var<uniform> rim: vec3<f32>;
var<uniform> background: vec3<f32>;

alias Sample = vec3<f32>;

// The share of the disk's radius inside its rim.
const rim_start = 0.985;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let squared = dot(across, across);
    if (squared > 1) {
        return background;
    }
    if (squared > rim_start) {
        return rim;
    }
    // The point of S² under the sample, on the front hemisphere.
    let point = (mv.x * across.x + mv.y * across.y + mv.z * sqrt(1 - squared)).dual();
    var colour = disk;
    for (var body = 0u; body < arrayLength(&quadrics); body++) {
        if ((quadrics[body](point) & point) < 0) {
            colour = colours[body];
        }
    }
    return colour;
}
"""


@cache
def shader() -> Shader:
    return Shader(supersampled() + source, S2.ga, dict(Point=S2.Point, Plane=S2.Plane, Quadric=S2.Quadric))


def hemisphere_frames(surfaces: S2.Quadric, colors: np.ndarray, resolution: int) -> list[np.ndarray]:
    """One frame per row of world forms, painted on the GPU, `resolution` pixels across: a sample of the front
    hemisphere is a point of S², and it belongs to the last body whose primal form is negative there, as `paint`
    paints on the CPU. The spherical quadric is a cone through the origin, so the front hemisphere already shows both
    halves."""
    painter = shader()
    return [np.round(np.clip(painter((resolution, resolution), quadrics=bodies, colours=colors, disk=to_rgb(DISK),
                                     rim=to_rgb(RIM), background=to_rgb(BACKGROUND))["image"], 0, 1) * 255).astype(np.uint8)
            for bodies in surfaces]


def draw_tumbling(trajectory: S2.Trajectory, dt: float) -> plt.Figure:
    """The tumbling oval's rate in each of its body's planes against time: the spin in the intermediate plane flips
    over and back, periodically, while the others swing through zero."""
    rate = trajectory.rate[:, 0].cast(S2.ga.subspace("yz zx xy")).kernel     # [steps, 3]
    time = np.arange(rate.shape[0]) * dt
    figure, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    for spin, colour, name in zip(rate.T, ("#0284c7", "#e11d48", "#d97706"),
                                  ("in yz, the intermediate plane", "in zx", "in xy")):
        ax.plot(time, spin, color=colour, linewidth=1.8, label=name)
    # The rates against their zero, without numbers; the legend above, clear of the curves.
    ax.axhline(0, color="0.3", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlim=(time[0], time[-1]), xticks=[], yticks=[], xlabel="time", ylabel="body rate")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    return figure


# --- S³ -------------------------------------------------------------------------------
# A wide pinhole view from the eye.
FOV = np.radians(120.0)                            # rad


def s3_frames(trajectory: S3.Trajectory, colors: np.ndarray, eye: S3.Motor, light: S3.Point,
              shape: tuple[int, int]) -> list[np.ndarray]:
    """Every frame traced from the eye along great circles, each ellipsoid lit by the light."""
    return [trace(eye, surfaces, colors, light, FOV, shape) for surfaces in trajectory.surfaces]


def draw_last_frame(frames: list[np.ndarray]) -> plt.Figure:
    """The last frame, as it was traced."""
    return image_figure(frames[-1])
