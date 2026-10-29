"""Drawing for quadric physics on S² and S³."""

from __future__ import annotations

from collections.abc import Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.algebras import Spherical3D
from examples import instantiate
from examples.quadrics.s3_raytracer import core as raytracer
from examples.quadrics.s3_raytracer.render import picture as trace
from examples.util import charts
from examples.util.wgsl import Shader, image_figure, show_pixels, supersampled

S2 = instantiate("examples.quadrics.elliptic_physics.core", Spherical3D)
S3 = instantiate("examples.quadrics.elliptic_physics.core", raytracer.ga)

background_colour, disk_colour, rim_colour = "#0a0f1d", "#111827", "#334155"
# Where the disk's rim begins: the squared distance from its centre, in its radii, past which a pixel is rim.
rim_start = 0.985
# The planes x, y and z of S²'s space: a point of S² is the dual of their sum weighted by its unit coordinates.
axes = stack((S2.mv.x, S2.mv.y, S2.mv.z))                                     # [3] Plane+
# The direction toward the viewer, who looks along -z.
toward_viewer = S2.mv.z.dual()                                                # [] Point-
# The colour of the pixels inside a body painted on the CPU, the pixels across a painting, on the CPU or the GPU, and
# the samples along each side of a pixel on the CPU.
inside_colour = "#38bdf8"
painted, samples_per_side = 240, 2


# --- the front hemisphere of S² -------------------------------------------------------
def hemisphere(size: int) -> tuple[S2.Point, np.ndarray, np.ndarray]:
    """The points of S² on the front hemisphere seen along z under the samples of a size by size grid inside the unit
    disk, each its place across the disk lifted toward the viewer onto the sphere, as the shader builds it; the disk
    mask, and which of the disk's samples form its rim."""
    steps = np.linspace(-1.0, 1.0, size)
    # Across to the right along the columns, up along the rows from the top.
    across = S2.mv.x.dual() * steps[None, :] + S2.mv.y.dual() * steps[::-1, None]   # [rows, columns] Point-
    inside = across.scalar_norm_squared() <= 1.0                                  # [rows, columns]
    disk = across[inside]                                                         # [pixels] Point-
    squared_distance = disk.scalar_norm_squared()                                 # [pixels] Scalar+
    return disk + toward_viewer * (1 - squared_distance).square_root(), inside, squared_distance > rim_start


def hemisphere_points() -> S2.Point:
    """The points of S² on the front hemisphere under the samples of `draw_inside`'s painting, inside its disk."""
    points, _, _ = hemisphere(painted * samples_per_side)
    return points                                                                 # [pixels] Point-


def paint(coverage: np.ndarray, colors: np.ndarray, inside: np.ndarray, rim: np.ndarray, supersample: int) -> np.ndarray:
    """An RGB image of the disk: each pixel takes the colour of the last body covering it.
    coverage is (bodies, pixels) boolean; the image is box-filtered by the supersampling."""
    covered = coverage.any(axis=0)
    last = coverage.shape[0] - 1 - coverage[::-1].argmax(axis=0)
    disk = np.where(covered[:, None], colors[last], np.where(rim[:, None], to_rgb(rim_colour), to_rgb(disk_colour)))
    image = np.empty(inside.shape + (3,))
    image[:] = to_rgb(background_colour)
    image[inside] = disk
    size = inside.shape[0] // supersample
    image = image.reshape(size, supersample, size, supersample, 3).mean(axis=(1, 3))
    return (image * 255.0).round().astype(np.uint8)


def picture(image: np.ndarray) -> plt.Figure:
    """A painted image of the hemisphere as a figure, at its own size."""
    height, width = image.shape[:2]
    figure = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    show_pixels(figure.add_axes((0, 0, 1, 1)), image)
    return figure


def draw_inside(inside: np.ndarray) -> plt.Figure:
    """The front hemisphere painted on the CPU, the samples of `hemisphere_points` inside a body in `inside_colour`."""
    _, disk, rim = hemisphere(painted * samples_per_side)
    return picture(paint(inside[None], rgb([inside_colour]), disk, rim, samples_per_side))


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

alias Sample = vec3<f32>;

fn sample(place: vec2<f32>) -> Sample {
    let across = screen(place);
    let squared = dot(across, across);
    if (squared > 1) {
        return background;
    }
    if (squared > rim_start) {
        return rim;
    }
    // The point of S² under the sample, on the front hemisphere: the dual of its unit vector.
    let point = (axes * vec3<f32>(across, sqrt(1 - squared))).sum().dual();
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
    return Shader(supersampled() + source, S2.ga, dict(Point=S2.Point, Plane=S2.Plane, Quadric=S2.Quadric),
                  dict(axes=axes, rim_start=rim_start, disk=to_rgb(disk_colour), rim=to_rgb(rim_colour), background=to_rgb(background_colour)))


def hemisphere_frames(surfaces: S2.Quadric, colors: np.ndarray, resolution: int) -> Iterator[np.ndarray]:
    """One frame per row of world forms, painted on the GPU, `resolution` pixels across: a sample of the front
    hemisphere is a point of S², and it belongs to the last body whose primal form is negative there, as `paint`
    paints on the CPU. The spherical quadric is a cone through the origin, so the front hemisphere already shows both
    halves."""
    painter = shader()
    for bodies in surfaces:
        image = painter((resolution, resolution), quadrics=bodies, colours=colors)["image"]
        yield np.round(np.clip(image, 0, 1) * 255).astype(np.uint8)


def hemisphere_image(bodies: S2.Quadric, colors: np.ndarray) -> np.ndarray:
    """The bodies' world forms painted as `hemisphere_frames` paints them, as many pixels across as the CPU's
    painting."""
    return next(hemisphere_frames(bodies[None], colors, painted))


@charts.style()
def draw_tumbling(trajectory: S2.Trajectory, dt: float) -> plt.Figure:
    """The tumbling oval's rate in each of its body's planes against time: the spin in the intermediate plane flips
    over and back, periodically, while the others swing through zero."""
    # The rate's turning in each of the body's planes, read by pairing it with the plane's patch.
    patches = stack((S2.mv.yz, S2.mv.zx, S2.mv.xy)).dual()                   # [3] Vector-
    rate = (trajectory.rate[:, 0, None] & patches).to_array()                 # [steps, 3]
    time = np.arange(rate.shape[0]) * dt
    # Over half the column: three curves along the run, for the flips to stand apart, under a legend of three.
    figure, ax = charts.subplots(520, 220, layout="constrained")
    for spin, colour, name in zip(rate.T, ("#0284c7", "#e11d48", "#d97706"),
                                  ("in yz, the intermediate plane", "in zx", "in xy")):
        ax.plot(time, spin, color=colour, label=name)
    # The rates against their zero, without numbers; the legend above, clear of the curves.
    ax.axhline(0, color="0.3", linewidth=charts.FRAME_WIDTH)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlim=(time[0], time[-1]), xticks=[], yticks=[], xlabel="time", ylabel="body rate")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    return figure


# --- S³ -------------------------------------------------------------------------------
# A wide pinhole view from the eye.
field_of_view = np.radians(120.0)


def s3_frames(trajectory: S3.Trajectory, colors: np.ndarray, eye: S3.Motor, light: S3.Point,
              shape: tuple[int, int]) -> Iterator[np.ndarray]:
    """Every frame traced from the eye along great circles, each ellipsoid lit by the light."""
    for surfaces in trajectory.surfaces:
        yield trace(eye, surfaces, colors, light, field_of_view, shape)


def draw_last_frame(frames: list[np.ndarray]) -> plt.Figure:
    """The last frame, as it was traced."""
    return image_figure(frames[-1])
