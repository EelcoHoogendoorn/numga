"""Implicit front-hemisphere painting, tumbling-rate charts, and 3-sphere ray-traced animations for spherical quadric bodies."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.algebras import Spherical3D
from numga.wgsl import Shader, supersampled
from examples import instantiate
from examples.quadrics.s3_raytracer import core as raytracer
from examples.quadrics.s3_raytracer.render import picture as trace_s3
from examples.util import charts
from examples.util.panels import image_figure, show_pixels

S2 = instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)
S3 = instantiate("examples.quadrics.elliptic_physics.core", raytracer.ga)

background_colour, disk_colour, rim_colour = "#0a0f1d", "#111827", "#334155"
# Squared screen radius beyond which pixels inside the unit disk belong to the sphere's rim.
rim_start = 0.985
axes = stack((S2.mv.x, S2.mv.y, S2.mv.z))                                       # [3] Plane+
toward_viewer = S2.mv.z.dual()                                                  # [] Point-
inside_colour = "#38bdf8"
painted_resolution, samples_per_side = 240, 2


# --- front hemisphere of the 2-sphere -------------------------------------------------
def hemisphere(n_grid: int) -> tuple[S2.Point, np.ndarray, np.ndarray]:
    """Orthographic screen samples on an `n_grid` by `n_grid` square lift inside the unit disk to front-hemisphere points (`[...] Point-`), the disk mask (`[n_grid, n_grid]`), and the rim mask."""
    steps = np.linspace(-1.0, 1.0, n_grid)
    across = S2.mv.x.dual() * steps[None, :] + S2.mv.y.dual() * steps[::-1, None]  # [n_grid, n_grid] Point-
    inside = across.scalar_norm_squared() <= 1.0                                # [n_grid, n_grid]
    disk = across[inside]                                                       # [...] Point-
    squared_distance = disk.scalar_norm_squared()                               # [...] Scalar+
    points = disk + toward_viewer * (1 - squared_distance).square_root()        # [...] Point-
    return points, inside, squared_distance > rim_start


def hemisphere_points() -> S2.Point:
    """Supersampled orthographic points (`[...] Point-`) cover the front hemisphere inside the painted disk."""
    points, _, _ = hemisphere(painted_resolution * samples_per_side)
    return points                                                               # [...] Point-


def painted_disk(
    coverage: np.ndarray,
    colors: np.ndarray,
    inside: np.ndarray,
    rim: np.ndarray,
    supersample: int,
) -> np.ndarray:
    """Boolean `coverage` (`[n_bodies, n_samples]`) and RGB `colors` (`[n_bodies, 3]`) box-average by `supersample` into a `[n_pixels, n_pixels, 3]` `uint8` disk image."""
    covered = coverage.any(axis=0)
    last = coverage.shape[0] - 1 - coverage[::-1].argmax(axis=0)
    disk = np.where(
        covered[:, None],
        colors[last],
        np.where(rim[:, None], to_rgb(rim_colour), to_rgb(disk_colour)),
    )
    image = np.empty(inside.shape + (3,))
    image[:] = to_rgb(background_colour)
    image[inside] = disk
    n_pixels = inside.shape[0] // supersample
    averaged = image.reshape(n_pixels, supersample, n_pixels, supersample, 3).mean(axis=(1, 3))
    return (averaged * 255.0).round().astype(np.uint8)


def hemisphere_figure(image: np.ndarray) -> plt.Figure:
    """Borderless `100`-dpi figure displaying `image` pixel for pixel without resampling."""
    n_rows, n_columns = image.shape[:2]
    figure = plt.figure(figsize=(n_columns / 100, n_rows / 100), dpi=100)
    show_pixels(figure.add_axes((0, 0, 1, 1)), image)
    return figure


def inside_figure(inside: np.ndarray) -> plt.Figure:
    """Boolean `inside` samples (`[n_samples]`) on `hemisphere_points()` render on the CPU as a front-hemisphere figure."""
    _, disk, rim = hemisphere(painted_resolution * samples_per_side)
    return hemisphere_figure(painted_disk(inside[None], rgb_colors([inside_colour]), disk, rim, samples_per_side))


def rgb_colors(colors: list[str]) -> np.ndarray:
    """Matplotlib colour strings convert to `[n_bodies, 3]` float RGB rows."""
    return np.array([to_rgb(color) for color in colors])


def hue_colors(fractions: np.ndarray) -> np.ndarray:
    """Unit hue `fractions` (`[n_bodies]`) map around the HSV colour wheel into `[n_bodies, 3]` float RGB rows."""
    return plt.get_cmap("hsv")(fractions)[:, :3]


# --- WebGPU painting and tumbling chart on the 2-sphere -------------------------------
source = """
var<storage, read> quadrics: array<Quadric>;
var<storage, read> colours: array<vec3<f32>>;

alias Sample = vec3<f32>;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let foot = (axes.dual() * vec3<f32>(across, 0.0)).sum();
    let squared = foot.scalar_norm_squared();
    if (squared > 1) {
        return background;
    }
    if (squared > rim_start) {
        return rim;
    }
    // Lift the screen sample to its unit antivector on the front hemisphere.
    let point = foot + toward * sqrt(1 - squared);
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
def hemisphere_shader() -> Shader:
    """Compiles and caches the supersampled WebGPU shader that paints spherical quadrics on the front hemisphere."""
    return Shader(
        supersampled() + source,
        S2.ga,
        dict(Point=S2.Point, Plane=S2.Plane, Quadric=S2.Quadric),
        dict(
            axes=axes,
            toward=axes[2].dual(),
            rim_start=rim_start,
            disk=to_rgb(disk_colour),
            rim=to_rgb(rim_colour),
            background=to_rgb(background_colour),
        ),
    )


def hemisphere_image(surfaces: S2.Quadric, colors: np.ndarray, resolution: int) -> np.ndarray:
    """World-frame primal `surfaces` (`[n_bodies] Plane+ <- Point-`) paint on the GPU into a `[resolution, resolution, 3]` `uint8` RGB image."""
    image = hemisphere_shader()((resolution, resolution), quadrics=surfaces, colours=colors)["image"]
    return np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def contact_figure(surfaces: S2.Quadric, colors: np.ndarray) -> plt.Figure:
    """World-frame primal `surfaces` (`[n_bodies] Plane+ <- Point-`) render at `painted_resolution` inside a borderless hemisphere figure."""
    return hemisphere_figure(hemisphere_image(surfaces, colors, painted_resolution))


def s2_animation(
    forms: S2.Quadric,
    colors: np.ndarray,
    resolution: int,
    states: Iterable[tuple],
) -> Iterator[np.ndarray]:
    """Yields one `[resolution, resolution, 3]` `uint8` RGB frame per `(motors, ...)` state by placing `forms` in the world frame."""
    for state in states:
        surfaces = S2.world_quadrics(state[0], forms)                           # [n_bodies] Plane+ <- Point-
        yield hemisphere_image(surfaces, colors, resolution)


@charts.style()
def tumbling_figure(rates: S2.Rate, dt: float) -> plt.Figure:
    """Body-frame `rates` (`[n_frames] Rate+`) paired with the three coordinate turning patches plot the intermediate-axis reversals against time."""
    n_frames = rates.shape[0]
    patches = stack((S2.mv.yz, S2.mv.zx, S2.mv.xy)).dual()                      # [3] Point-
    spin_components = (rates[:, None] & patches).to_array()                     # [n_frames, 3]
    times = np.arange(n_frames) * dt
    figure, ax = charts.subplots(520, 220, layout="constrained")
    for spin, colour, label in zip(
        spin_components.T,
        ("#0284c7", "#e11d48", "#d97706"),
        ("yz (intermediate)", "zx", "xy"),
    ):
        ax.plot(times, spin, color=colour)
        ax.text(times[-1], spin[-1], f" {label}", color=colour, va="center")
    ax.axhline(0, color="0.3", linewidth=charts.FRAME_WIDTH)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlim=(times[0], times[-1] * 1.28), xticks=[0, round(float(times[-1]), 1)], yticks=[0], xlabel="time", ylabel="body rate")
    return figure


# --- ray tracing on the 3-sphere ------------------------------------------------------
field_of_view = np.radians(120.0)


def s3_animation(
    forms: S3.Quadric,
    colors: np.ndarray,
    eye_frame: S3.Motor,
    light: S3.Point,
    shape: tuple[int, int],
    states: Iterable[tuple],
) -> Iterator[np.ndarray]:
    """Yields one `[n_rows, n_columns, 3]` `uint8` RGB frame per `(motors, ...)` state traced along great circles of the 3-sphere."""
    for state in states:
        surfaces = S3.world_quadrics(state[0], forms)                           # [n_bodies] Plane+ <- Point-
        yield trace_s3(eye_frame, surfaces, colors, light, field_of_view, shape)


def frame_figure(image: np.ndarray) -> plt.Figure:
    """Wraps a ray-traced RGB `image` (`[n_rows, n_columns, 3]`) in a borderless figure."""
    return image_figure(image)
