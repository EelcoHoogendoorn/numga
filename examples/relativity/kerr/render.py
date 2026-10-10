"""Equatorial light rays around a rotating black hole."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from examples.animation import capture
from examples.relativity.kerr import core

# The spin sweep's resolution: its thin rays need no more.
SWEEP_DPI = 64
# Wavelengths across the visible band, nm, and the second radiation constant, nm K.
WAVELENGTHS = np.linspace(380.0, 780.0, 81)
RADIATION_CONSTANT = 1.4388e7
# Linear sRGB from CIE XYZ.
XYZ_TO_RGB = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
# The luminance of linear sRGB, and the exposure of the reference temperature.
LUMINANCE = np.array([0.2126, 0.7152, 0.0722])
MID_GREY = 0.5
# A milder lift than a display's gamma of 2.2, so brightness differences stay visible.
GAMMA = 1.6
# Blackbody colours look pale on a display; their saturation is exaggerated to show the shift.
SATURATION = 2.5
# The white balance: a blackbody at this temperature shows neutral, hotter ones blue, cooler ones red.
WHITE_TEMPERATURE = 6500.0                                                     # K
# The disk's clumpy texture: how many random waves make it, their seed, and its contrast.
CLUMP_MODES = 24
CLUMP_SEED = 5
CLUMP_CONTRAST = 0.35


# --- plumbing -------------------------------------------------------------------------
def animate_rays(paths: core.Vector, horizons: core.Vector,
                 extent: float) -> list[np.ndarray]:
    """Sweep stationary spins with fixed ray colours and spatial scales."""
    tracks = paths.cast(core.ga.subspace("x y")).kernel.transpose(1, 2, 0, 3)
    outlines = horizons.cast(core.ga.subspace("x y")).kernel
    colours = plt.colormaps["coolwarm"](np.linspace(0.05, 0.95, tracks.shape[1]))
    ticks = [-extent / 2, 0, extent / 2]

    figure, ax = plt.subplots(figsize=(5, 5), dpi=SWEEP_DPI, layout="constrained")
    rays = LineCollection(tracks[0], colors=colours, linewidths=1.0, alpha=0.9)
    horizon = Polygon(outlines[0], facecolor="#161b22", edgecolor="none", zorder=3)
    ax.add_collection(rays)
    ax.add_patch(horizon)
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal",
           xticks=ticks, yticks=ticks, xlabel="$x/M$", ylabel="$y/M$")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#cbd5e1")
    ax.tick_params(color="#cbd5e1")
    figure.canvas.draw()
    figure.set_layout_engine("none")

    frames = []
    for paths, outline in zip(tracks, outlines):
        rays.set_segments(paths)
        horizon.set_xy(outline)
        frames.append(capture(figure))
    plt.close(figure)
    return frames


def star_texture(width: int, height: int, seed: int) -> np.ndarray:
    """A seamless sky image with unresolved stars and a faint dusty stellar band."""
    random = np.random.default_rng(seed)
    star_count = width * height // 140
    longitude = (np.arange(width) + 0.5) * (2 * np.pi / width) - np.pi
    latitude = (0.5 - (np.arange(height) + 0.5) / height) * np.pi
    horizontal = np.cos(latitude[:, None])
    sky_x = horizontal * np.cos(longitude)
    sky_y = horizontal * np.sin(longitude)
    sky_z = np.sin(latitude[:, None])
    band_height = 0.2 * sky_x - 0.6 * sky_y - np.sqrt(0.6) * sky_z
    cloud = (np.sin(13 * sky_x + 9 * sky_y - 7 * sky_z)
             + 0.45 * np.sin(31 * sky_x - 23 * sky_y + 19 * sky_z)
             + 0.18 * np.sin(79 * sky_x + 47 * sky_y + 53 * sky_z))
    band = np.exp(-(band_height / 0.16) ** 2) * (0.75 + 0.2 * cloud)
    dust = np.exp(-((band_height + 0.025 * cloud) / 0.035) ** 2)
    background = np.array([0.006, 0.009, 0.019])
    band_colour = np.array([0.065, 0.058, 0.09])
    texture = background + band[..., None] * band_colour * (1 - 0.7 * dust[..., None])

    # Distribute a catalogue on the sphere, then stamp small Gaussian stellar images.
    star_x = random.uniform(0, width, star_count)
    star_latitude = np.arcsin(random.uniform(-1, 1, star_count))
    star_y = (0.5 - star_latitude / np.pi) * height
    brightness = 0.35 + 2.5 * random.random(star_count) ** 6
    warmth = random.uniform(0, 1, (star_count, 1))
    colours = np.array([0.57, 0.72, 1.0]) * (1 - warmth) + np.array([1.0, 0.79, 0.51]) * warmth
    offsets = np.arange(-2, 3)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    columns = np.floor(star_x).astype(int)[:, None] + offset_x.ravel()
    rows = np.floor(star_y).astype(int)[:, None] + offset_y.ravel()
    distance_squared = (columns + 0.5 - star_x[:, None]) ** 2 + (rows + 0.5 - star_y[:, None]) ** 2
    weights = np.exp(-distance_squared / (2 * 0.6 ** 2)) * brightness[:, None]
    valid = (rows >= 0) & (rows < height)
    contribution = weights[..., None] * colours[:, None]
    np.add.at(texture, (rows[valid], columns[valid] % width), contribution[valid])
    return np.clip(texture, 0, 1)


def sky_colour(direction: core.Vector, texture: np.ndarray) -> np.ndarray:
    """Look up the celestial image using the escaped ray's spatial direction."""
    horizontal_x, horizontal_y, vertical = np.moveaxis(
        direction.cast(core.ga.subspace("x y z")).kernel, -1, 0)
    length = np.sqrt(horizontal_x ** 2 + horizontal_y ** 2 + vertical ** 2)
    height, width = texture.shape[:2]
    longitude = np.arctan2(horizontal_y, horizontal_x)
    latitude = np.arcsin(np.clip(vertical / length, -1, 1))
    column = (longitude / (2 * np.pi) + 0.5) * width - 0.5
    row = np.clip((0.5 - latitude / np.pi) * height - 0.5, 0, height - 1)
    left = np.floor(column).astype(int)
    top = np.floor(row).astype(int)
    right = (left + 1) % width
    bottom = np.minimum(top + 1, height - 1)
    fraction_x = (column - left)[..., None]
    fraction_y = (row - top)[..., None]
    upper = texture[top, left % width] * (1 - fraction_x) + texture[top, right] * fraction_x
    lower = texture[bottom, left % width] * (1 - fraction_x) + texture[bottom, right] * fraction_x
    return upper * (1 - fraction_y) + lower * fraction_y


def clump_waves(points: core.Vector, radius: core.Scalar) -> np.ndarray:
    """Random waves around and across the disk at each point, summed to unit variance, `[...]`; whole
    turns keep each continuous in angle."""
    horizontal_x, horizontal_y = np.moveaxis(points.cast(core.ga.subspace("x y")).kernel, -1, 0)
    angle = np.arctan2(horizontal_y, horizontal_x)
    modes = np.random.default_rng(CLUMP_SEED)
    turns = modes.integers(3, 18, CLUMP_MODES)
    wavenumbers = modes.uniform(0.8, 4.0, CLUMP_MODES)
    phases = modes.uniform(0, 2 * np.pi, CLUMP_MODES)
    waves = np.sin(turns * angle[..., None] + wavenumbers * radius.kernel + phases)                # [..., modes]
    return waves.sum(axis=-1) / np.sqrt(CLUMP_MODES / 2)


def disk_emissivity(radius: core.Scalar, waves: np.ndarray) -> np.ndarray:
    """An illustrative texture on the disk's emission, `[crossings]`: fine rings, which the orbital
    flow leaves in place, and clumps from `waves`."""
    radius = radius.kernel[..., 0]
    rings = 0.90 + 0.045 * np.sin(11 * radius) + 0.025 * np.sin(19 * radius)
    return rings * np.exp(CLUMP_CONTRAST * waves)


def colour_matching(wavelengths: np.ndarray) -> np.ndarray:
    """The CIE 1931 colour-matching functions, by the multi-lobe fits of Wyman, Sloan and Shirley
    (2013), `[wavelengths, 3]`."""
    def lobe(centre: float, below: float, above: float) -> np.ndarray:
        width = np.where(wavelengths < centre, below, above)
        return np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)
    x = 1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2)
    y = 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
    z = 1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def blackbody(temperature: np.ndarray) -> np.ndarray:
    """The visible light of a blackbody at each temperature, K, in linear sRGB, relative units,
    `[..., 3]`; a temperature of zero is dark."""
    with np.errstate(divide="ignore", over="ignore"):
        spectrum = 1 / (WAVELENGTHS ** 5 * np.expm1(RADIATION_CONSTANT / (WAVELENGTHS * temperature[..., None])))
    return np.clip(spectrum @ colour_matching(WAVELENGTHS) @ XYZ_TO_RGB.T, 0, None)


def exposed(light: np.ndarray, reference: float) -> np.ndarray:
    """Linear light shown on the display: the luminance scaled so a blackbody at the reference
    temperature sits at mid-grey, kept linear below it and rolled off smoothly above, the colour kept,
    then gamma-encoded and its saturation exaggerated, any channel above one scaled back."""
    luminance = light @ LUMINANCE                                                   # [...]
    exposure = luminance / (blackbody(np.array(reference)) @ LUMINANCE)
    mapped = np.where(exposure < 1, MID_GREY * exposure, 1 - (1 - MID_GREY) * np.exp(-(exposure - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(luminance > 0, mapped / luminance, 0)
    shown = (light * scale[..., None]) ** (1 / GAMMA)
    grey = shown @ LUMINANCE
    vivid = np.clip(grey[..., None] + SATURATION * (shown - grey[..., None]), 0, None)
    return vivid / np.maximum(vivid.max(axis=-1, keepdims=True), 1)


def disk_glow(image: core.Image, temperature: core.Scalar) -> np.ndarray:
    """Each disk crossing's linear light before its texture, `[crossings, 3]`: a blackbody at its
    observed temperature, white-balanced, and weighted by the share of its pixel the crossing gives."""
    white = blackbody(np.array(WHITE_TEMPERATURE))
    balanced = blackbody(temperature.kernel[..., 0]) * (white @ LUMINANCE) / white
    return balanced * image.disk_weights.kernel


def disk_light(image: core.Image, temperature: core.Scalar, waves: np.ndarray) -> np.ndarray:
    """Each disk crossing's linear light, `[crossings, 3]`: its glow, textured by `waves`."""
    return disk_glow(image, temperature) * disk_emissivity(image.disk_radii, waves)[:, None]


def scattered(light: np.ndarray, pixels: np.ndarray, pixel_count: int) -> np.ndarray:
    """Light `[entries, 3]` summed into its pixels, `[pixel_count, 3]`; a pixel may take several."""
    summed = np.zeros((pixel_count, 3))
    np.add.at(summed, pixels, light)
    return summed


def camera_colours(image: core.Image, temperature: core.Scalar, image_shape: tuple[int, int],
                   sky_texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The disk's linear light and the sky's colours, each summed into its pixels, `[pixels, 3]` each."""
    height, width = image_shape
    light = disk_light(image, temperature, clump_waves(image.disk[0], image.disk_radii))
    sky = sky_colour(image.sky_directions, sky_texture) * image.sky_weights.kernel
    return (scattered(light, image.disk_pixels, height * width),
            scattered(sky, image.sky_pixels, height * width))


def draw_camera(image: core.Image, temperature: core.Scalar, reference: float, image_shape: tuple[int, int],
                sky_texture: np.ndarray) -> plt.Figure:
    """The camera's view: the glowing disk at its observed temperatures, clearing at its edges, exposed
    so a blackbody at the `reference` temperature shows mid-grey, the horizon's shadow and the lensed
    sky."""
    height, width = image_shape
    disk, sky = camera_colours(image, temperature, image_shape, sky_texture)
    figure = plt.figure(figsize=(9, 9 * height / width), facecolor="black")
    ax = figure.add_axes((0, 0, 1, 1))
    ax.imshow(np.clip(exposed(disk, reference) + sky, 0, 1).reshape(height, width, 3), interpolation="lanczos")
    ax.set_axis_off()
    return figure


def animate_polarizer(image: core.Image, temperature: core.Scalar, reference: float, transmission: core.Scalar,
                      image_shape: tuple[int, int], sky_texture: np.ndarray) -> list[np.ndarray]:
    """The camera's view through each analyzer: each disk crossing by the power it transmits,
    `transmission [analyzers, crossings]`. The exposure is doubled to make up for the half of
    unpolarized light a polarizer stops, so the sky and unpolarized light look as in the still."""
    height, width = image_shape
    light = disk_light(image, temperature, clump_waves(image.disk[0], image.disk_radii))
    _, sky = camera_colours(image, temperature, image_shape, sky_texture)
    return [np.round(np.clip(exposed(scattered(2 * light * passed[:, None], image.disk_pixels, height * width), reference)
                             + sky, 0, 1) * 255).astype(np.uint8).reshape(height, width, 3)
            for passed in transmission.kernel[..., 0]]


def animate_disk(image: core.Image, temperature: core.Scalar, reference: float, flows: Iterable[core.Vector],
                 loop_frames: int, image_shape: tuple[int, int], sky_texture: np.ndarray) -> list[np.ndarray]:
    """The camera's view as the disk's gas orbits: its light steady, its texture read where each
    crossing's gas was, `flows [2, crossings]` per frame, in this loop and the one before. Turning the
    clump waves from the first to the second over the loop closes it without a jump, and without the
    loss of contrast an average would bring."""
    height, width = image_shape
    _, sky = camera_colours(image, temperature, image_shape, sky_texture)
    glow = disk_glow(image, temperature)                                            # [crossings, 3]
    frames = []
    for index, (now, before) in enumerate(flows):
        fade = np.pi / 2 * index / loop_frames
        waves = (np.cos(fade) * clump_waves(now, image.disk_radii)
                 + np.sin(fade) * clump_waves(before, image.disk_radii))
        light = glow * disk_emissivity(image.disk_radii, waves)[:, None]           # [crossings, 3]
        disk = scattered(light, image.disk_pixels, height * width)
        frames.append(np.round(np.clip(exposed(disk, reference) + sky, 0, 1) * 255).astype(np.uint8).reshape(height, width, 3))
    return frames
