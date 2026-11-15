"""Planar panels and GPU shaders render electric field strength, out-of-plane magnetic field,
charge density, and circling-charge radiation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize, TwoSlopeNorm, to_rgb, to_rgba
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

from numga import stack
from numga.render import Camera, contours, drawing, simplices
from numga.topology import Topology
from numga.wgsl import Shader, supersampled

from examples.electromagnetism.moving_charge import core
from examples.util.animation import capture
from examples.util.panels import offset_panel, pinned

# Colour scales for electric field magnitude, out-of-plane magnetic field, and core density contour.
electric_norm = LogNorm(0.05, 50.0)
magnetic_norm = TwoSlopeNorm(0.0, -10.0, 10.0)
density_level = 5.0
density_colour = "#ea580c"
strength_colours = plt.get_cmap("Purples")
signed_colours = plt.get_cmap("RdBu_r")
wave_zone = 4.0
wave_percentile = 99.0
wave_power = 0.4
n_palette_colours = 256
charge_colour, charge_rim_colour = "#facc15", "#0f172a"
figure_dpi = 100
panel_gap = 4
outline_ink = np.array(to_rgba(density_colour))
outline_width = 0.9
charge_size, charge_rim = 0.0105, 0.002


# --- plumbing -------------------------------------------------------------------------
def coordinates(events: core.Antivector) -> np.ndarray:
    """Pairing `events` with the `x` and `y` spatial planes reads out their planar coordinates (`[..., 2]`)."""
    mv = events.context.multivector
    return (stack((mv.x, mv.y)) & events[..., None]).to_array()                 # [..., 2]


def grid_extent(events: core.Antivector) -> tuple[float, float, float, float]:
    """Extending half a grid spacing past the corner cell centres gives the image bounding box."""
    half_step = (events[..., 1, 1] - events[..., 0, 0]) / 2                     # [...] Antivector-
    low = coordinates(events[..., 0, 0] - half_step)                            # [..., 2]
    high = coordinates(events[..., -1, -1] + half_step)                         # [..., 2]
    return low[..., 0], high[..., 0], low[..., 1], high[..., 1]


def electric_strength(field: core.Bivector) -> np.ndarray:
    """Pairing `field` with the lab's future time step isolates the electric part and takes its norm (`[...]`)."""
    mv = field.context.multivector
    time = -mv.t.dual()                                                         # [] Antivector-
    return (field & time).norm().to_array()                                     # [...]


def magnetic_component(field: core.Bivector) -> np.ndarray:
    """Pairing `field` with the intrinsic `xy` plane patch reads the out-of-plane magnetic field (`[...]`)."""
    mv = field.context.multivector
    return (field & mv.xy.dual()).to_array()                                    # [...]


def charge_density(four_current: core.Antivector) -> np.ndarray:
    """Pairing the lab's time slice with `four_current` reads the lab charge density (`[...]`)."""
    mv = four_current.context.multivector
    return (mv.t & four_current).to_array()                                     # [...]


def rgb_image(values: np.ndarray, colours: Colormap, norm: Normalize) -> np.ndarray:
    """Mapping `values` (`[..., n_rows, n_columns]`) through `norm` and `colours` produces an upright
    RGB image (`[..., n_rows, n_columns, 3]`)."""
    return colours(np.ma.filled(norm(values), 0.0))[..., ::-1, :, :3]


def side_by_side_panels(n_panels: int, pixels: tuple[int, int]) -> tuple[Figure, list[Axes]]:
    """`n_panels` pixel-exact axes sit side by side with `panel_gap` pixels between adjacent panels."""
    n_rows, n_columns = pixels
    width = n_panels * n_columns + (n_panels - 1) * panel_gap
    figure = plt.figure(figsize=(width / figure_dpi, n_rows / figure_dpi), dpi=figure_dpi)
    axes = figure.subplots(1, n_panels, squeeze=False)[0]
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=panel_gap / n_columns)
    return figure, list(axes)


def image_panel(
    ax: Axes,
    events: core.Antivector,
    values: np.ndarray,
    colours: Colormap,
    norm: Normalize,
) -> AxesImage:
    """Pinning `values` over `events` fills `ax` pixel for pixel and returns the image artist."""
    image = pinned(ax, rgb_image(values, colours, norm), grid_extent(events))
    ax.set_autoscale_on(False)
    return image


def outlined_strength_layer(
    places: core.Antivector,
    topology: Topology,
    field: core.Bivector,
    four_current: core.Antivector,
    view: Camera,
) -> np.ndarray:
    """Overlaying the `density_level` charge-density contour onto the electric field strength produces
    an RGBA layer (`[n_rows, n_columns, 4]`)."""
    strengths = strength_colours(np.ma.filled(electric_norm(electric_strength(field)), 0.0))[..., :3]
    charge_outline = contours.Level(density_level, outline_ink, outline_width)
    return contours.picture(
        places,
        topology,
        np.abs(charge_density(four_current)).reshape(-1),
        strengths.reshape(-1, 3),
        view,
        levels=(charge_outline,),
    )


def fields_figure(events: core.Antivector, fields: core.Bivector) -> Figure:
    """Each batch entry in `fields` draws an electric-strength panel beside an out-of-plane magnetic panel."""
    strengths, magnetic_planes = electric_strength(fields), magnetic_component(fields)
    figure, axes = side_by_side_panels(2 * len(strengths), events.shape[-2:])
    for left_ax, right_ax, strength, magnetic_plane in zip(axes[::2], axes[1::2], strengths, magnetic_planes):
        image_panel(left_ax, events, strength, strength_colours, electric_norm)
        image_panel(right_ax, events, magnetic_plane, signed_colours, magnetic_norm)
    return figure


def densities_figure(events: core.Antivector, four_currents: core.Antivector) -> Figure:
    """Each batch entry in `four_currents` draws its signed lab charge density on one shared colour scale."""
    values = charge_density(four_currents)                                      # [...]
    limit = np.abs(values).max()
    figure, axes = side_by_side_panels(len(values), events.shape[-2:])
    for ax, value in zip(axes, values):
        image_panel(ax, events, value, signed_colours, TwoSlopeNorm(0.0, -limit, limit))
    return figure


def boost_frames(frames: Iterable[tuple[core.Antivector, core.Bivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Yields side-by-side electric-strength (with charge-density contour) and magnetic-field RGB frames."""
    states = iter(frames)
    events, field, four_current = next(states)
    n_rows, n_columns = events.shape[-2:]
    figure, (left_ax, right_ax) = side_by_side_panels(2, (n_rows, n_columns))
    corners = coordinates(events[[0, -1], [0, -1]])                             # [2, 2]
    left_ax.set(xlim=corners[:, 0], ylim=corners[:, 1])
    view = offset_panel(left_ax, core.mv.x, core.mv.y)
    strength_image = pinned(left_ax, np.zeros(view.shape + (4,)), (*corners[:, 0], *corners[:, 1]))
    left_ax.set_autoscale_on(False)
    places = events.reshape(-1)                                                 # [n_rows * n_columns] Antivector-
    topology = Topology.grid(n_rows, n_columns)
    magnetic_image = image_panel(right_ax, events, magnetic_component(field), signed_colours, magnetic_norm)
    for _, field, four_current in chain([(events, field, four_current)], states):
        strength_image.set_data(drawing.composited([outlined_strength_layer(places, topology, field, four_current, view)]))
        magnetic_image.set_data(rgb_image(magnetic_component(field), signed_colours, magnetic_norm))
        yield capture(figure)
    plt.close(figure)


# --- the waves on the GPU ---------------------------------------------------------------
# Each supersampled pixel evaluates `orbit_field(event)` in the orbit plane and multiplies the out-of-plane magnetic
# field by radial distance from the orbit centre so outgoing spherical waves keep uniform visual amplitude across the
# window, compressed with `wave_power < 1` so both narrow relativistic pulses and the smooth spiral between them remain
# visible.
waves_source = '''
struct Sample {
    colour: vec3<f32>,
    far_amplitude: f32,
}
var<uniform> lab_time: f32;
var<uniform> half_height: f32;
var<uniform> limit: f32;
var<storage, read> palette: array<vec3<f32>>;

fn palette_colour(level: f32) -> vec3<f32> {
    let last = arrayLength(&palette) - 1u;
    let entry = (clamp(level, -1.0, 1.0) + 1) / 2 * f32(last);
    let below = u32(floor(entry));
    return mix(palette[below], palette[min(below + 1u, last)], entry - floor(entry));
}

fn sample(place: vec2<f32>) -> Sample {
    let event = (directions * screen(place)).sum() * half_height + time * lab_time;
    let spatial_event = event - time * (lab_slice & event);
    let reach = spatial_event.squared().square_root();
    let waves = (orbit_field(event) & orbit_plane) * reach;
    let scaled_wave = waves / limit;
    return Sample(palette_colour(sign(scaled_wave) * pow(abs(scaled_wave), wave_power)), select(0.0, abs(waves), reach > wave_zone));
}
'''


@cache
def waves_shader(field_source: str) -> Shader:
    """Compiling `field_source` with `waves_source` binds the planar screen directions and `xy` magnetic plane patch."""
    mv = core.mv
    return Shader(
        supersampled() + field_source + waves_source,
        core.ga,
        dict(Antivector=core.Antivector, Vector=core.Vector, Bivector=core.Bivector),
        dict(
            directions=stack((mv.x, mv.y)).dual(),
            lab_slice=mv.t,
            orbit_plane=mv.xy.dual(),
            wave_power=wave_power,
            wave_zone=wave_zone,
        ),
    )


def orbit_view(half_width: float, n_pixels: int) -> Camera:
    """A square planar camera spans `[-half_width, half_width]` across `n_pixels` pixels."""
    mv = core.mv
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.array([half_width, half_width]), n_pixels)


def wave_samples(
    field_source: str,
    orbit_charge: float,
    orbit_radius: float,
    orbit_speed: float,
    lab_time: float,
    view: Camera,
    limit: float,
) -> dict[str, np.ndarray]:
    """Evaluating `waves_shader` across `view` returns RGB `colour` (`[n_pixels, n_pixels, 3]`) and
    far-zone amplitudes `far_amplitude` (`[n_pixels, n_pixels]`)."""
    return waves_shader(field_source)(
        view.shape,
        orbit_charge=orbit_charge,
        orbit_radius=orbit_radius,
        orbit_speed=orbit_speed,
        lab_time=lab_time,
        half_height=view.half[1],
        limit=limit,
        palette=signed_colours(np.linspace(0, 1, n_palette_colours))[:, :3],
    )["image"]


def wave_limit(
    field_source: str,
    orbit_charge: float,
    orbit_radius: float,
    orbit_speed: float,
    lab_time: float,
    view: Camera,
) -> float:
    """The far-zone amplitude percentile (`wave_percentile`) fixes one shared colour limit across a full orbit."""
    far_amplitudes = wave_samples(field_source, orbit_charge, orbit_radius, orbit_speed, lab_time, view, 1.0)["far_amplitude"]
    return float(np.percentile(far_amplitudes[far_amplitudes > 0], wave_percentile))


def charge_marked_frame(colours: np.ndarray, charge_position: core.Antivector, view: Camera) -> np.ndarray:
    """Compositing a dark-rimmed yellow dot at `charge_position` onto `colours` returns a `uint8` RGB frame."""
    places = charge_position.reshape(-1)                                        # [1] Antivector-
    rim_radius = (charge_size + charge_rim) / 2 * view.pixels
    dot_radius = (charge_size - charge_rim) / 2 * view.pixels
    composite = drawing.composited([
        np.concatenate((colours, np.ones(colours.shape[:-1] + (1,))), axis=-1),
        simplices.markers(places, np.array(to_rgb(charge_rim_colour)), rim_radius, view),
        simplices.markers(places, np.array(to_rgb(charge_colour)), dot_radius, view),
    ])
    return np.round(np.clip(composite[..., :3], 0, 1) * 255).astype(np.uint8)


def wave_frames(
    field_source: str,
    orbit_charge: float,
    orbit_radius: float,
    orbit_speed: float,
    half_width: float,
    n_pixels: int,
    frames: Iterable[tuple[float, core.Antivector]],
) -> Iterator[np.ndarray]:
    """Yields square `uint8` RGB frames of the circling charge's radiation pattern over one orbit."""
    view = orbit_view(half_width, n_pixels)
    states = iter(frames)
    first_state = next(states)
    limit = wave_limit(field_source, orbit_charge, orbit_radius, orbit_speed, first_state[0], view)
    for lab_time, charge_position in chain([first_state], states):
        colours = wave_samples(field_source, orbit_charge, orbit_radius, orbit_speed, lab_time, view, limit)["colour"]
        yield charge_marked_frame(colours, charge_position, view)


def radiation_figure(
    field_source: str,
    orbit_charge: float,
    orbit_radius: float,
    orbit_speeds: np.ndarray,
    charge_position: core.Antivector,
    half_width: float,
    n_pixels: int,
) -> Figure:
    """Side-by-side panels compare the radiation pattern at lab time zero across `orbit_speeds`."""
    view = orbit_view(half_width, n_pixels)
    figure, axes = side_by_side_panels(len(orbit_speeds), view.shape)
    for ax, orbit_speed in zip(axes, orbit_speeds):
        limit = wave_limit(field_source, orbit_charge, orbit_radius, float(orbit_speed), 0.0, view)
        colours = wave_samples(field_source, orbit_charge, orbit_radius, float(orbit_speed), 0.0, view, limit)["colour"]
        pinned(ax, charge_marked_frame(colours, charge_position, view), view.extent)
    return figure
