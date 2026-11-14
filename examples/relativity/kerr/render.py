"""Equatorial light-ray sweeps and camera views around a rotating Kerr black hole."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack

from examples.util.animation import capture
from numga.topology import Topology
from numga.render import Camera, contours, drawing, simplices
from examples.util.panels import offset_panel, overlay
from examples.relativity.kerr import core
from examples.relativity.kerr.plumbing import Image

# The spin sweep's resolution: its thin rays need no more. The rays' width in pixels and how much they cover, and the
# horizon's shade.
sweep_dpi = 64
ray_width, ray_opacity = 1.0, 0.9
horizon_shade = np.array(to_rgb("#161b22"))
# Wavelengths across the visible band, nm, and the second radiation constant, nm K.
visible_wavelengths = np.linspace(380.0, 780.0, 81)
radiation_constant = 1.4388e7
# Linear sRGB from CIE XYZ.
xyz_to_rgb = np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
# The luminance of linear sRGB, and the exposure of the reference temperature.
luminance_weights = np.array([0.2126, 0.7152, 0.0722])
mid_grey = 0.5
# A milder lift than a display's gamma of 2.2, so brightness differences stay visible.
display_gamma = 1.6
# Blackbody colours look pale on a display; their saturation is exaggerated to show the shift.
colour_saturation = 2.5
# The white balance: a blackbody at this temperature shows neutral, hotter ones blue, cooler ones red.
white_temperature = 6500.0                                                     # K
# The disk's clumpy texture: how many random waves make it, their seed, and its contrast.
n_clump_modes = 24
clump_seed = 5
clump_contrast = 0.35


# --- plumbing -------------------------------------------------------------------------
def rays_and_horizon(paths: core.Antivector, outline: core.Antivector, camera: Camera) -> np.ndarray:
    """Composites equatorial light rays (`[n_steps, n_rays]`) in cool-to-warm colours underneath the filled horizon outline (`[n_sides]`)."""
    n_steps, n_rays = paths.shape
    # The events ray by ray, each ray's in its order along it.
    by_ray = paths[np.arange(n_steps)[None, :], np.arange(n_rays)[:, None]].reshape(-1)     # [n_rays * n_steps] Antivector-
    colours = plt.colormaps["coolwarm"](np.linspace(0.05, 0.95, n_rays))[:, :3]
    # The horizon is a fan of triangles from the hole's centre to each side of its outline: the outline's cone.
    n_sides = outline.shape[0]
    fan = Topology.loops([n_sides]).cone()
    disc = concatenate((outline, 0 * outline[:1]))                                          # [n_sides + 1] Antivector-
    return drawing.composited([
        simplices.strokes(by_ray, Topology.chains([n_steps] * n_rays), np.repeat(colours, n_steps - 1, axis=0),
                          ray_width, camera, opacity=ray_opacity),
        contours.picture(disc, fan, np.zeros(n_sides + 1), horizon_shade, camera)])


def ray_animation(paths: core.Antivector, horizons: core.Antivector,
                  extent: float) -> Iterator[np.ndarray]:
    """Yields frames sweeping stationary black-hole spins with fixed ray colours and spatial scales."""
    figure, ax = plt.subplots(figsize=(5, 5), dpi=sweep_dpi, layout="constrained")
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
    ax.set_axis_off()
    figure.canvas.draw()
    figure.set_layout_engine("none")
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    picture = overlay(ax, [np.zeros(camera.shape + (4,))])

    for spin_index in range(paths.shape[1]):
        picture.set_data(rays_and_horizon(paths[:, spin_index], horizons[spin_index], camera))
        yield capture(figure)
    plt.close(figure)


def ray_comparison_figure(paths: tuple[core.Antivector, core.Antivector], horizon: core.Antivector,
                          extent: float) -> plt.Figure:
    """Draws gauge and metric rays side by side with identical colours, horizon, and spatial scale."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=sweep_dpi, layout="constrained")
    for ax in axes:
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect="equal")
        ax.set_axis_off()
    for ax, path in zip(axes, paths):
        overlay(ax, [rays_and_horizon(path, horizon, offset_panel(ax, core.mv.x, core.mv.y))])
    return figure


def star_texture(n_width: int, n_height: int, seed: int) -> np.ndarray:
    """Generates a seamless equirectangular sky image (`[n_height, n_width, 3]`) with unresolved stars and a dusty stellar band."""
    mv = core.mv
    random = np.random.default_rng(seed)
    n_stars = n_width * n_height // 140
    longitude = (np.arange(n_width) + 0.5) * (2 * np.pi / n_width) - np.pi
    latitude = (0.5 - (np.arange(n_height) + 0.5) / n_height) * np.pi
    # The direction at the middle of each pixel: x turned up toward z by its latitude, then about z by its longitude.
    lat_rotors = (mv.zx * (latitude / 2)).exp()                                             # [n_height] Rotor+
    meridian = lat_rotors >> mv.x.dual()                                                    # [n_height] Antivector-
    lon_rotors = (mv.xy * (-longitude / 2)).exp()                                           # [n_width] Rotor+
    sky = lon_rotors >> meridian[:, None]                                                   # [n_height, n_width] Antivector-
    # The band's height above its plane, and the cloud's waves across the planes they run along.
    band_plane = 0.2 * mv.x - 0.6 * mv.y - np.sqrt(0.6) * mv.z                              # [] Vector+
    wave_planes = stack((13 * mv.x + 9 * mv.y - 7 * mv.z, 31 * mv.x - 23 * mv.y + 19 * mv.z,
                         79 * mv.x + 47 * mv.y + 53 * mv.z))                                # [3] Vector+
    band_height = (band_plane & sky).to_array()
    cloud = (np.array([1.0, 0.45, 0.18]) * np.sin((wave_planes & sky[..., None]).to_array())).sum(axis=-1)
    band = np.exp(-(band_height / 0.16) ** 2) * (0.75 + 0.2 * cloud)
    dust = np.exp(-((band_height + 0.025 * cloud) / 0.035) ** 2)
    background = np.array([0.006, 0.009, 0.019])
    band_colour = np.array([0.065, 0.058, 0.09])
    texture = background + band[..., None] * band_colour * (1 - 0.7 * dust[..., None])

    # Distribute a catalogue on the sphere, then stamp small Gaussian stellar images.
    star_x = random.uniform(0, n_width, n_stars)
    star_latitude = np.arcsin(random.uniform(-1, 1, n_stars))
    star_y = (0.5 - star_latitude / np.pi) * n_height
    brightness = 0.35 + 2.5 * random.random(n_stars) ** 6
    warmth = random.uniform(0, 1, (n_stars, 1))
    colours = np.array([0.57, 0.72, 1.0]) * (1 - warmth) + np.array([1.0, 0.79, 0.51]) * warmth
    offsets = np.arange(-2, 3)
    offset_x, offset_y = np.meshgrid(offsets, offsets)
    columns = np.floor(star_x).astype(int)[:, None] + offset_x.ravel()
    rows = np.floor(star_y).astype(int)[:, None] + offset_y.ravel()
    distance_squared = (columns + 0.5 - star_x[:, None]) ** 2 + (rows + 0.5 - star_y[:, None]) ** 2
    weights = np.exp(-distance_squared / (2 * 0.6 ** 2)) * brightness[:, None]
    valid = (rows >= 0) & (rows < n_height)
    contribution = weights[..., None] * colours[:, None]
    np.add.at(texture, (rows[valid], columns[valid] % n_width), contribution[valid])
    return np.clip(texture, 0, 1)


def sky_colour(direction: core.Antivector, texture: np.ndarray) -> np.ndarray:
    """Samples `texture` bilinearly at the equirectangular longitude and latitude of each escaped ray `direction`."""
    mv = core.mv
    spatial = core.space(direction)                                                         # [...] Antivector-
    unit = spatial / spatial.antinorm()                                                     # [...] Antivector-
    n_height, n_width = texture.shape[:2]
    # The image's texture coordinates, longitude and latitude, from the direction's pairings with the planes x, y and z.
    longitude = np.arctan2((mv.y & unit).to_array(), (mv.x & unit).to_array())
    latitude = np.arcsin(np.clip((mv.z & unit).to_array(), -1, 1))
    column = (longitude / (2 * np.pi) + 0.5) * n_width - 0.5
    row = np.clip((0.5 - latitude / np.pi) * n_height - 0.5, 0, n_height - 1)
    left = np.floor(column).astype(int)
    top = np.floor(row).astype(int)
    right = (left + 1) % n_width
    bottom = np.minimum(top + 1, n_height - 1)
    fraction_x = (column - left)[..., None]
    fraction_y = (row - top)[..., None]
    upper = texture[top, left % n_width] * (1 - fraction_x) + texture[top, right] * fraction_x
    lower = texture[bottom, left % n_width] * (1 - fraction_x) + texture[bottom, right] * fraction_x
    return upper * (1 - fraction_y) + lower * fraction_y


def clump_waves(points: core.Antivector, radius: core.Scalar) -> np.ndarray:
    """Sums `n_clump_modes` unit-variance azimuthal and radial waves (`[...]`) across disk `points`."""
    mv = core.mv
    angle = np.arctan2((mv.y & points).to_array(), (mv.x & points).to_array())
    modes = wave_modes()
    waves = np.sin(modes["turns"] * angle[..., None] + modes["wavenumber"] * radius.to_array()[..., None]
                   + modes["phase"])                                                        # [..., 24]
    return waves.sum(axis=-1) / np.sqrt(n_clump_modes / 2)


def wave_modes() -> dict[str, np.ndarray]:
    """Generates random integer azimuthal turn counts, radial wavenumbers, and phases (`[24]`) for disk clumps."""
    generator = np.random.default_rng(clump_seed)
    turns = generator.integers(3, 18, n_clump_modes)
    wavenumbers = generator.uniform(0.8, 4.0, n_clump_modes)
    phases = generator.uniform(0, 2 * np.pi, n_clump_modes)
    return dict(turns=turns, wavenumber=wavenumbers, phase=phases)


def blackbody_table(hottest: float, n_entries: int) -> np.ndarray:
    """Tabulates white-balanced linear sRGB blackbody colours (`[1, n_entries, 3]`) from zero up to `hottest` K."""
    white = blackbody(np.array(white_temperature))
    temperatures = np.linspace(0, hottest, n_entries)
    return (blackbody(temperatures) * (white @ luminance_weights) / white)[None]


def frame_figure(frame: np.ndarray) -> plt.Figure:
    """Draws a `[n_rows, n_columns, 3]` display image edge-to-edge on a black figure."""
    figure = plt.figure(figsize=(frame.shape[1] / 100, frame.shape[0] / 100), dpi=100, facecolor="black")
    ax = figure.add_axes((0, 0, 1, 1))
    ax.imshow(frame, interpolation="none")
    ax.set_axis_off()
    return figure


def tone_mapped_frame(disk: np.ndarray, sky: np.ndarray, reference: float) -> np.ndarray:
    """Combines tone-mapped linear `disk` light with `sky` colour into a `uint8` RGB image."""
    shown = tone_mapped_light(disk, reference) + sky
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def disk_emissivity(radius: core.Scalar, waves: np.ndarray) -> np.ndarray:
    """Modulates disk emission with fine radial rings and log-normal clump contrast from `waves`."""
    radii = radius.to_array()
    rings = 0.90 + 0.045 * np.sin(11 * radii) + 0.025 * np.sin(19 * radii)
    return rings * np.exp(clump_contrast * waves)


def colour_matching(wavelengths: np.ndarray) -> np.ndarray:
    """Evaluates the Wyman, Sloan and Shirley (2013) multi-lobe CIE 1931 XYZ colour-matching functions (`[..., 3]`)."""
    def lobe(centre: float, below: float, above: float) -> np.ndarray:
        width = np.where(wavelengths < centre, below, above)
        return np.exp(-0.5 * ((wavelengths - centre) / width) ** 2)
    x = 1.056 * lobe(599.8, 37.9, 31.0) + 0.362 * lobe(442.0, 16.0, 26.7) - 0.065 * lobe(501.1, 20.4, 26.2)
    y = 0.821 * lobe(568.8, 46.9, 40.5) + 0.286 * lobe(530.9, 16.3, 31.1)
    z = 1.217 * lobe(437.0, 11.8, 36.0) + 0.681 * lobe(459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def blackbody(temperature: np.ndarray) -> np.ndarray:
    """Integrates Planck spectra across `visible_wavelengths` into linear sRGB (`[..., 3]`)."""
    with np.errstate(divide="ignore", over="ignore"):
        spectrum = 1 / (visible_wavelengths ** 5 * np.expm1(radiation_constant / (visible_wavelengths * temperature[..., None])))
    return np.clip(spectrum @ colour_matching(visible_wavelengths) @ xyz_to_rgb.T, 0, None)


def tone_mapped_light(light: np.ndarray, reference: float) -> np.ndarray:
    """Maps linear sRGB `light` so a blackbody at `reference` K sits at mid-grey, with gamma lift and saturation boost."""
    luminance = light @ luminance_weights                                                   # [...]
    exposure = luminance / (blackbody(np.array(reference)) @ luminance_weights)
    mapped = np.where(exposure < 1, mid_grey * exposure, 1 - (1 - mid_grey) * np.exp(-(exposure - 1)))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(luminance > 0, mapped / luminance, 0)
    shown = (light * scale[..., None]) ** (1 / display_gamma)
    grey = shown @ luminance_weights
    vivid = np.clip(grey[..., None] + colour_saturation * (shown - grey[..., None]), 0, None)
    return vivid / np.maximum(vivid.max(axis=-1, keepdims=True), 1)


def disk_glow(image: Image, temperature: core.Scalar) -> np.ndarray:
    """Evaluates white-balanced linear blackbody light (`[..., 3]`) weighted by each crossing's pixel share."""
    white = blackbody(np.array(white_temperature))
    balanced = blackbody(temperature.to_array()) * (white @ luminance_weights) / white
    return balanced * image.disk_weights.to_array()[..., None]


def disk_light(image: Image, temperature: core.Scalar, waves: np.ndarray) -> np.ndarray:
    """Modulates each disk crossing's blackbody glow (`[..., 3]`) by its textured emissivity."""
    return disk_glow(image, temperature) * disk_emissivity(image.disk_radii, waves)[:, None]


def pixel_light(light: np.ndarray, pixels: np.ndarray, n_pixels: int) -> np.ndarray:
    """Accumulates crossing or sky `light` contributions into `[n_pixels, 3]`."""
    summed = np.zeros((n_pixels, 3))
    np.add.at(summed, pixels, light)
    return summed


def camera_colours(image: Image, temperature: core.Scalar, image_shape: tuple[int, int],
                   sky_texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Accumulates linear disk light and escaped sky colour into `[n_pixels, 3]` arrays."""
    n_height, n_width = image_shape
    light = disk_light(image, temperature, clump_waves(image.disk[0], image.disk_radii))
    sky = sky_colour(image.sky_directions, sky_texture) * image.sky_weights.to_array()[..., None]
    return (pixel_light(light, image.disk_pixels, n_height * n_width),
            pixel_light(sky, image.sky_pixels, n_height * n_width))


def camera_figure(image: Image, temperature: core.Scalar, reference: float, image_shape: tuple[int, int],
                  sky_texture: np.ndarray) -> plt.Figure:
    """Draws the static camera view of the lensed glowing disk, horizon shadow, and background starlight."""
    n_height, n_width = image_shape
    disk, sky = camera_colours(image, temperature, image_shape, sky_texture)
    return frame_figure(np.clip(tone_mapped_light(disk, reference) + sky, 0, 1).reshape(n_height, n_width, 3))


def camera_comparison_figure(images: tuple[Image, Image], temperatures: tuple[core.Scalar, core.Scalar],
                             reference: float, image_shape: tuple[int, int], sky_texture: np.ndarray) -> plt.Figure:
    """Draws gauge and metric camera images side by side at identical resolution and exposure."""
    n_height, n_width = image_shape
    frames = []
    for image, temperature in zip(images, temperatures):
        disk, sky = camera_colours(image, temperature, image_shape, sky_texture)
        frames.append(tone_mapped_frame(disk, sky, reference).reshape(n_height, n_width, 3))
    return frame_figure(np.concatenate(frames, axis=1))


def polarizer_animation(image: Image, temperature: core.Scalar, reference: float, transmission: core.Scalar,
                        image_shape: tuple[int, int], sky_texture: np.ndarray) -> Iterator[np.ndarray]:
    """Yields camera frames viewed through a rotating linear analyzer at doubled disk exposure."""
    n_height, n_width = image_shape
    light = disk_light(image, temperature, clump_waves(image.disk[0], image.disk_radii))
    _, sky = camera_colours(image, temperature, image_shape, sky_texture)
    for passed in transmission.to_array():
        disk = pixel_light(2 * light * passed[:, None], image.disk_pixels, n_height * n_width)
        yield np.round(np.clip(tone_mapped_light(disk, reference) + sky, 0, 1) * 255).astype(np.uint8).reshape(n_height, n_width, 3)


def polarizer_comparison_animation(images: tuple[Image, Image], temperatures: tuple[core.Scalar, core.Scalar],
                                   reference: float, transmissions: tuple[core.Scalar, core.Scalar],
                                   image_shape: tuple[int, int], sky_texture: np.ndarray) -> Iterator[np.ndarray]:
    """Yields side-by-side gauge and metric camera frames through a rotating linear analyzer."""
    frames = (polarizer_animation(image, temperature, reference, transmission, image_shape, sky_texture)
              for image, temperature, transmission in zip(images, temperatures, transmissions))
    for pair in zip(*frames):
        yield np.concatenate(pair, axis=1)


def comparison_readout(position: core.Antivector, momentum: core.Vector, screen: core.ScreenFrame,
                       metric_residual: core.Vector) -> dict[str, float]:
    """Reads maximum absolute coordinate residuals between gauge and metric ray states by basis pairing."""
    mv = core.mv
    planes = stack((mv.x, mv.y, mv.z, mv.t))                                                # [4] Vector+
    directions = planes.dual()                                                              # [4] Antivector-
    screen_directions = stack((core.camera_right, core.camera_up))                          # [2] Antivector-
    readings = (("position", planes & position[..., None]),
                ("momentum", momentum[..., None] & directions),
                ("screen", planes & screen[..., None](screen_directions)[..., None]),
                ("metric inverse", metric_residual[..., None] & directions))
    return {name: float(np.abs(value.to_array()).max()) for name, value in readings}


def disk_animation(image: Image, temperature: core.Scalar, reference: float, flows: Iterable[core.Antivector],
                   n_loop_frames: int, image_shape: tuple[int, int], sky_texture: np.ndarray) -> Iterator[np.ndarray]:
    """Yields seamless looping frames of the orbiting disk texture cross-faded across two orbital periods."""
    n_height, n_width = image_shape
    _, sky = camera_colours(image, temperature, image_shape, sky_texture)
    glow = disk_glow(image, temperature)                                                    # [..., 3]
    for index, (current_points, previous_points) in enumerate(flows):
        fade = np.pi / 2 * index / n_loop_frames
        waves = (np.cos(fade) * clump_waves(current_points, image.disk_radii)
                 + np.sin(fade) * clump_waves(previous_points, image.disk_radii))
        light = glow * disk_emissivity(image.disk_radii, waves)[:, None]                    # [..., 3]
        disk = pixel_light(light, image.disk_pixels, n_height * n_width)
        yield np.round(np.clip(tone_mapped_light(disk, reference) + sky, 0, 1) * 255).astype(np.uint8).reshape(n_height, n_width, 3)
