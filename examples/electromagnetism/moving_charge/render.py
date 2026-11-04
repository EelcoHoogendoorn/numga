"""The electric and magnetic fields of a moving charge, as the lab sees them."""

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

from examples.util.animation import capture
from examples.util.wgsl import Shader, contours, pinned, simplices, supersampled
from examples.util.wgsl.camera import Camera, offset_panel
from examples.electromagnetism.moving_charge import core

# The lab's slice of space, which carries the events into the renderer's space.
# The scales of the electric field's strength and of the magnetic field.
electric_norm = LogNorm(0.05, 50.0)
magnetic_norm = TwoSlopeNorm(0.0, -10.0, 10.0)
density_level = 5.0
density_colour = "#ea580c"
# Magnitudes start at white, as the signed fields do at zero.
strength_colours = plt.get_cmap("Purples")
signed_colours = plt.get_cmap("RdBu_r")
wave_zone = 4.0
wave_percentile = 99.0
wave_power = 0.4
# The entries of the signed colours a shader looks a level up in.
n_palette_colours = 256
charge_colour, charge_rim_colour = "#facc15", "#0f172a"
# The figures' dots per inch, and the pixels between panels side by side.
figure_dpi = 100
panel_gap = 4
# The ink of the charge's outline, and its width in pixels; the charge's dot and its rim, as shares of the
# picture's width.
outline_ink = np.array(to_rgba(density_colour))
outline_width = 0.9
charge_size, charge_rim = 0.0105, 0.002


# --- plumbing -------------------------------------------------------------------------
def coordinates(events: core.Antivector) -> np.ndarray:
    """x and y of each event, on the last axis: its pairings with the slices x and y."""
    mv = events.context.multivector
    return (stack((mv.x, mv.y)) & events[..., None]).to_array()               # [..., 2]


def extent(events: core.Antivector) -> tuple[float, float, float, float]:
    """The box the grid's cells cover: its samples sit at the cell centres, half a cell inside."""
    half_cell = (events[..., 1, 1] - events[..., 0, 0]) / 2                     # [...] Antivector-
    low, high = coordinates(events[..., 0, 0] - half_cell), coordinates(events[..., -1, -1] + half_cell)
    return low[..., 0], high[..., 0], low[..., 1], high[..., 1]


def electric_strength(field: core.Bivector) -> np.ndarray:
    """The strength of the electric field the lab reads: the size of the field met with the lab's time, the part of
    the field that the planes swept along the lab's time read."""
    mv = field.context.multivector
    time = -mv.t.dual()                                                         # [] Antivector-
    return (field & time).norm().to_array()


def magnetic_across(field: core.Bivector) -> np.ndarray:
    """The lab's magnetic field across the plane: the field read on the patch of the plane of x and y."""
    mv = field.context.multivector
    return (field & mv.xy.dual()).to_array()


def density(current: core.Antivector) -> np.ndarray:
    """The charge density: the current crossing the lab's time slice."""
    mv = current.context.multivector
    return (mv.t & current).to_array()


def coloured(values: np.ndarray, colours: Colormap, norm: Normalize) -> np.ndarray:
    """Values `[..., rows, columns]` in colours, `[..., rows, columns, 3]`, the rows flipped so that the grid's first
    row, lowest in y, is at the bottom of the picture."""
    return colours(np.ma.filled(norm(values), 0.0))[..., ::-1, :, :3]


def panels(n_panels: int, pixels: tuple[int, int]) -> tuple[Figure, list[Axes]]:
    """A figure of panels side by side, each with room for exactly its pixels, rows and columns, a gap between them."""
    rows, columns = pixels
    width = n_panels * columns + (n_panels - 1) * panel_gap
    figure = plt.figure(figsize=(width / figure_dpi, rows / figure_dpi), dpi=figure_dpi)
    axes = figure.subplots(1, n_panels, squeeze=False)[0]
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=panel_gap / columns)
    return figure, list(axes)


def panel(ax: Axes, events: core.Antivector, values: np.ndarray, colours: Colormap, norm: Normalize) -> AxesImage:
    """Values on the grid, one pixel per event, held pixel for pixel over the box the grid covers; the image, for an
    animation to set its data."""
    image = pinned(ax, coloured(values, colours, norm), extent(events))
    ax.set_autoscale_on(False)
    return image


def outlined(places: core.Antivector, topology: contours.Topology, field: core.Bivector, current: core.Antivector,
             view: Camera) -> np.ndarray:
    """The electric field's strength over the grid of events at their places in the lab's space, `[rows, columns]`
    numbered row by row, with the charge's outline, where its density is high, seen by the view of its panel:
    `[rows, columns, 4]`."""
    strengths = strength_colours(np.ma.filled(electric_norm(electric_strength(field)), 0.0))[..., :3]
    charge_outline = contours.Level(density_level, outline_ink, outline_width)
    return contours.picture(places, topology, np.abs(density(current)).reshape(-1), strengths.reshape(-1, 3), view,
                            levels=(charge_outline,))


def fields(events: core.Antivector, field: core.Bivector) -> Figure:
    """Electric field strength and magnetic field side by side for each case, the cases side by side."""
    strengths, magnetic = electric_strength(field), magnetic_across(field)       # [cases, rows, columns] each
    figure, axes = panels(2 * len(strengths), events.shape[-2:])
    for left, right, strength, across in zip(axes[::2], axes[1::2], strengths, magnetic):
        panel(left, events, strength, strength_colours, electric_norm)
        panel(right, events, across, signed_colours, magnetic_norm)
    return figure


def densities(events: core.Antivector, current: core.Antivector) -> Figure:
    """The charge density, signed, one panel per case, on one colour scale."""
    values = density(current)                                                   # [cases, rows, columns]
    limit = np.abs(values).max()
    figure, axes = panels(len(values), events.shape[-2:])
    for ax, value in zip(axes, values):
        panel(ax, events, value, signed_colours, TwoSlopeNorm(0.0, -limit, limit))
    return figure


def animate(frames: Iterable[tuple[core.Antivector, core.Bivector, core.Antivector]]) -> Iterator[np.ndarray]:
    """Electric field strength with the charge's outline, and magnetic field, side by side, a frame for each of the
    lab's views over one grid of events: one figure set up for the first, its images' data set for each."""
    states = iter(frames)
    events, field, current = next(states)
    figure, (left, right) = panels(2, events.shape[-2:])
    # The strength and the charge's outline over the grid of events, from its first to its last.
    corners = coordinates(events[[0, -1], [0, -1]])                            # [2, 2]
    left.set(xlim=corners[:, 0], ylim=corners[:, 1])
    view = offset_panel(left, core.mv.x, core.mv.y)
    strength = pinned(left, np.zeros(view.shape + (4,)), (*corners[:, 0], *corners[:, 1]))
    left.set_autoscale_on(False)
    places = events.reshape(-1)                                              # [rows * columns] Antivector-
    topology = contours.Topology.grid(*events.shape[-2:])
    magnetic = panel(right, events, magnetic_across(field), signed_colours, magnetic_norm)
    for _, field, current in chain([(events, field, current)], states):
        strength.set_data(outlined(places, topology, field, current, view))
        magnetic.set_data(coloured(magnetic_across(field), signed_colours, magnetic_norm))
        yield capture(figure)
    plt.close(figure)


# --- the waves on the GPU ---------------------------------------------------------------
# Each sample places an event in the orbit's plane at the frame's lab time and asks a field's source, which declares
# the lab's `time` and `orbit_field(event)`, for the field there: the magnetic field across the plane times the distance
# from the orbit's centre, so that the outgoing waves keep their strength across the picture, coloured on a scale with
# a power below one on its magnitude, which keeps a fast charge's narrow pulses and the field between them. The sample
# also gives that product's size beyond the wave zone, which sets the scale.
waves_source = """
struct Sample {
    colour: vec3<f32>,
    far: f32,
}
var<uniform> lab_time: f32;
var<uniform> half_height: f32;
var<uniform> limit: f32;
var<storage, read> palette: array<vec3<f32>>;

// A level from minus one to one in the palette's colours, between its two nearest entries.
fn coloured(level: f32) -> vec3<f32> {
    let last = arrayLength(&palette) - 1u;
    let entry = (clamp(level, -1.0, 1.0) + 1) / 2 * f32(last);
    let below = u32(floor(entry));
    return mix(palette[below], palette[min(below + 1u, last)], entry - floor(entry));
}

fn sample(place: vec2<f32>) -> Sample {
    let event = (directions * screen(place)).sum() * half_height + time * lab_time;
    // The distance from the orbit's centre: the size of the event's place, less the lab's time.
    let centred = event - time * (mv.t & event);
    let reach = centred.squared().square_root();
    let waves = (orbit_field(event) & orbit_plane) * reach;
    let scaled = waves / limit;
    return Sample(coloured(sign(scaled) * pow(abs(scaled), wave_power)), select(0.0, abs(waves), reach > wave_zone));
}
"""


@cache
def waves_shader(field_source: str) -> Shader:
    """The shader of the waves of a field's source: the directions along x and y, which place a sample, the patch of
    the orbit's plane, which reads the magnetic field across it, and the scale's power and zone as its constants."""
    mv = core.mv
    return Shader(supersampled() + field_source + waves_source, core.ga,
                  dict(Antivector=core.Antivector, Vector=core.Vector, Bivector=core.Bivector),
                  dict(directions=stack((mv.x, mv.y)).dual(), orbit_plane=mv.xy.dual(), wave_power=wave_power,
                       wave_zone=wave_zone))


def orbit_view(half_width: float, n_pixels: int) -> Camera:
    """The camera of a square picture of the orbit's plane about its centre, `half_width` to either side, reading
    places, offsets of weight one, by the planes x and y."""
    mv = core.mv
    return Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0, np.zeros(2), np.array([half_width, half_width]), n_pixels)


def waves(field_source: str, orbit: core.Orbit, lab_time: float, view: Camera, limit: float) -> dict[str, np.ndarray]:
    """The waves the view sees at a lab time, each pixel the average of its samples: their colours on the scale that
    `limit` sets, `[rows, columns, 3]`, and the sizes beyond the wave zone, `[rows, columns]`."""
    return waves_shader(field_source)(
        view.shape, orbit_charge=orbit.charge, orbit_radius=orbit.radius, orbit_speed=orbit.speed, lab_time=lab_time,
        half_height=view.half[1], limit=limit, palette=signed_colours(np.linspace(0, 1, n_palette_colours))[:, :3]
    )["image"]


def wave_limit(field_source: str, orbit: core.Orbit, lab_time: float, view: Camera) -> float:
    """One colour scale for a whole turn: the magnetic field times the distance from the orbit's centre, far from the
    orbit, at a lab time; the sharpest crests of a fast charge's beam saturate."""
    far = waves(field_source, orbit, lab_time, view, 1.0)["far"]
    return float(np.percentile(far[far > 0], wave_percentile))


def marked(colours: np.ndarray, charge: core.Antivector, view: Camera) -> np.ndarray:
    """A picture's colours `[rows, columns, 3]` with the charge marked over them, a dot with a dark rim, sized with
    the picture: its bytes `[rows, columns, 3]`."""
    places = charge.reshape(-1)                                                 # [1] Antivector-
    rim_radius, dot_radius = (charge_size + charge_rim) / 2 * view.pixels, (charge_size - charge_rim) / 2 * view.pixels
    shown = simplices.composited([np.concatenate((colours, np.ones(colours.shape[:-1] + (1,))), axis=-1),
                                  simplices.markers(places, np.array(to_rgb(charge_rim_colour)), rim_radius, view),
                                  simplices.markers(places, np.array(to_rgb(charge_colour)), dot_radius, view)])
    return np.round(np.clip(shown[..., :3], 0, 1) * 255).astype(np.uint8)


def animate_waves(field_source: str, orbit: core.Orbit, frames: Iterable[tuple[float, core.Antivector]],
                  half_width: float, n_pixels: int) -> Iterator[np.ndarray]:
    """The waves of a field's source over one turn, a lab time and the charge then for each frame, on the first
    frame's colour scale, the charge marked: a square picture `n_pixels` across, `half_width` to either side of the
    orbit's centre."""
    view = orbit_view(half_width, n_pixels)
    frames = iter(frames)
    first = next(frames)
    limit = wave_limit(field_source, orbit, first[0], view)
    for lab_time, charge in chain([first], frames):
        yield marked(waves(field_source, orbit, lab_time, view, limit)["colour"], charge, view)


def radiation(field_source: str, orbits: list[core.Orbit], charge: core.Antivector, half_width: float,
              n_pixels: int) -> Figure:
    """The waves of circling charges at lab time zero side by side, one panel per orbit, each on its own colour scale,
    with the charge marked where each turn starts."""
    view = orbit_view(half_width, n_pixels)
    figure, axes = panels(len(orbits), view.shape)
    for ax, orbit in zip(axes, orbits):
        limit = wave_limit(field_source, orbit, 0.0, view)
        pinned(ax, marked(waves(field_source, orbit, 0.0, view, limit)["colour"], charge, view), view.extent)
    return figure
