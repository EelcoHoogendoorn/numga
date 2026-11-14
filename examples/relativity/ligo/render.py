"""Mollweide sky maps and overhead free-mass ring diagrams render gravitational-wave antenna patterns and spin-two polarization sweeps."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.colors import Normalize, to_rgb
from matplotlib.projections.geo import MollweideAxes

from numga import stack
from numga.render import Camera, contours, planar, simplices
from numga.topology import Topology
from examples.relativity.ligo import core
from examples.util import charts
from examples.util.animation import capture
from examples.util.panels import offset_panel, overlay, panel

# Colour maps for signed responses and sensitivities, plus detector-site, ring, and arm inks.
signed_cmap, sensitivity_cmap = "RdBu_r", "magma"
site_colour, label_colour, ring_colour, rest_colour, arm_colour = (
    "#22d3ee",
    "#0f172a",
    "#0369a1",
    "#cbd5e1",
    "#f97316",
)
arm_reach = 1.25
n_rest_samples = 199
ring_limit = 1.5
map_px, row_px, network_px = 760, 140, (600, 160)
site_radius, arm_width = charts.MARKER_PX, 2 * charts.LINE_PX

# Mollweide projection geometry and graticule sampling counts.
mollweide = MollweideAxes.MollweideTransform(1)
map_half, map_margin = np.array([2 * np.sqrt(2), np.sqrt(2)]), 1.02
meridian_angles = np.radians(np.arange(-150, 151, 30))
parallel_angles = np.radians(np.arange(-75, 76, 15))
n_meridians, n_parallels = len(meridian_angles), len(parallel_angles)
graticule_opacity = 0.25
n_meridian_samples, n_outline_samples = 31, 121
# Odd band count centres zero response on a white band so frame diffs move only along band edges.
n_response_bands = 9


def longitude_latitude(directions: core.Point) -> np.ndarray:
    """Longitude and latitude coordinates (`[..., 2]`, radians) read out unit spatial directions by pairing with `x`, `y`, and `z`."""
    mv = core.mv
    longitude = np.arctan2((mv.y & directions).to_array(), (mv.x & directions).to_array())
    latitude = np.arcsin(np.clip((mv.z & directions).to_array(), -1, 1))
    return np.stack([longitude, latitude], axis=-1)


def mollweide_places(longitudes_latitudes: np.ndarray) -> np.ndarray:
    """Planar coordinates (`[..., 2]`) map radian `(longitude, latitude)` pairs onto the Mollweide ellipse."""
    shape = longitudes_latitudes.shape
    return mollweide.transform(longitudes_latitudes.reshape(-1, 2)).reshape(shape)


def map_graticule() -> tuple[planar.Point, Topology]:
    """Meridian and parallel sample points on the Mollweide plane pair with their chain topology."""
    cap = parallel_angles[-1]
    latitudes = np.linspace(-cap, cap, n_meridian_samples)
    meridians = np.stack(
        np.broadcast_arrays(meridian_angles[:, None], latitudes[None, :]),
        axis=-1,
    )                                                                                   # [n_meridians, n_meridian_samples, 2]
    parallels = np.stack(
        np.broadcast_arrays(np.array([-np.pi, np.pi])[None, :], parallel_angles[:, None]),
        axis=-1,
    )                                                                                   # [n_parallels, 2, 2]
    places = np.concatenate([meridians.reshape(-1, 2), parallels.reshape(-1, 2)])
    chains = Topology.chains([n_meridian_samples] * n_meridians + [2] * n_parallels)
    return planar.mv.point(mollweide_places(places)), chains


def map_outline() -> planar.Point:
    """Antimeridian rim points (`[2 * n_outline_samples] Point-`) trace the closed outer boundary of the Mollweide ellipse."""
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, n_outline_samples)
    left = np.stack(np.broadcast_arrays(-np.pi, latitudes), axis=-1)
    right = np.stack(np.broadcast_arrays(np.pi, latitudes[::-1]), axis=-1)
    return planar.mv.point(mollweide_places(np.concatenate([left, right])))


def map_camera(ax: plt.Axes) -> Camera:
    """Planar camera frames `ax` tightly around the Mollweide ellipse with axes hidden."""
    half = map_margin * map_half
    ax.set(xlim=(-half[0], half[0]), ylim=(-half[1], half[1]), aspect="equal")
    ax.set_axis_off()
    return panel(ax, planar.mv.x, planar.mv.y, planar.mv.w)


def rgb_ink(colour: str) -> np.ndarray:
    """RGB float triplet (`[3]`) converts a hex colour string for `simplices` drawing."""
    return np.array(to_rgb(colour))


def sky_map(
    ax: plt.Axes,
    directions: core.Point,
    values: core.Scalar,
    cmap: str,
    low: float,
    high: float,
    zeniths: core.Point,
    names: list[str],
    bands: tuple[contours.Bands, ...] = (),
) -> None:
    """Renders one Mollweide sky field (`[n_latitudes, n_longitudes]`) on `ax` with graticule, rim, and labelled site markers."""
    n_latitudes, n_longitudes = directions.shape
    n_sites = zeniths.shape[0]
    sky = planar.mv.point(mollweide_places(longitude_latitude(directions))).reshape(-1)   # [n_latitudes * n_longitudes] Point-
    flat_values = values.to_array().reshape(-1)
    colours = colormaps[cmap](Normalize(low, high)(flat_values))[:, :3]
    lines, chains = map_graticule()
    sites = mollweide_places(longitude_latitude(zeniths))                                 # [n_sites, 2]
    camera = map_camera(ax)
    overlay(
        ax,
        [
            contours.picture(sky, Topology.grid(n_latitudes, n_longitudes), flat_values, colours, camera, bands=bands),
            simplices.strokes(lines, chains, np.ones(3), charts.FRAME_PX, camera, opacity=graticule_opacity),
            simplices.strokes(
                map_outline(),
                Topology.loops([2 * n_outline_samples]),
                np.zeros(3),
                charts.FRAME_PX,
                camera,
            ),
            simplices.markers(planar.mv.point(sites), rgb_ink(label_colour), site_radius + charts.FRAME_PX, camera),
            simplices.markers(planar.mv.point(sites), rgb_ink(site_colour), site_radius, camera),
        ],
    )
    for site, name in zip(sites, names):
        ax.annotate(
            name,
            site,
            xytext=(6, 4),
            textcoords="offset points",
            color=label_colour,
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
        )


@charts.style()
def patterns_figure(
    directions: core.Point,
    plus_response: core.Scalar,
    cross_response: core.Scalar,
    sensitivities: core.Scalar,
    zeniths: core.Point,
    names: list[str],
) -> plt.Figure:
    """Renders a three-column figure per detector showing plus response, cross response, and directional sensitivity."""
    figure = plt.figure(figsize=(map_px / charts.DPI, (row_px * len(names) + 10) / charts.DPI))
    for detector, _ in enumerate(names):
        for column in range(3):
            figure.add_subplot(len(names), 3, 3 * detector + column + 1)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.08, wspace=0.06)
    for detector, name in enumerate(names):
        maps = [plus_response[detector], cross_response[detector], sensitivities[detector]]
        for column, (values, cmap, low) in enumerate(zip(maps, [signed_cmap, signed_cmap, sensitivity_cmap], [-1, -1, 0])):
            sky_map(
                figure.axes[3 * detector + column],
                directions,
                values,
                cmap,
                low,
                1,
                zeniths[detector : detector + 1],
                [name],
            )
    return figure


@charts.style()
def network_figure(
    directions: core.Point,
    network_sensitivity: core.Scalar,
    alignment: core.Scalar,
    zeniths: core.Point,
    names: list[str],
) -> plt.Figure:
    """Renders side-by-side Mollweide sky maps of combined network sensitivity and detector response alignment."""
    figure = plt.figure(figsize=(network_px[0] / charts.DPI, network_px[1] / charts.DPI))
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.06)
    panels = [
        (network_sensitivity, sensitivity_cmap, 0, np.sqrt(len(names))),
        (alignment, signed_cmap, -1, 1),
    ]
    for ax, (values, cmap, low, high) in zip(axes, panels):
        sky_map(ax, directions, values, cmap, low, high, zeniths, names)
    return figure


def overhead_ring_panel(ax: plt.Axes, site: core.Rotor, arms: core.Point, ring: core.Point) -> None:
    """Renders the horizontal test-mass ring at rest and under tidal strain alongside the two interferometer arms."""
    mv = core.mv
    ax.set(xlim=(-ring_limit, ring_limit), ylim=(-ring_limit, ring_limit), aspect="equal")
    ax.set_axis_off()
    camera = offset_panel(ax, site >> mv.y, site >> mv.z)
    angles = np.linspace(0, 2 * np.pi, n_rest_samples, endpoint=False)
    turns = (mv.yz * (-angles / 2)).exp()                                               # [n_rest_samples] Rotor+
    local_rest = turns >> mv.y.dual()                                                   # [n_rest_samples] Point-
    rest = site >> local_rest                                                           # [n_rest_samples] Point-
    n_arms = arms.shape[0]
    reaches = stack((0 * arms, arm_reach * arms), axis=-1).reshape(-1)                  # [2 * n_arms] Point-
    n_masses = ring.shape[0]
    overlay(
        ax,
        [
            simplices.strokes(rest, Topology.loops([n_rest_samples]), rgb_ink(rest_colour), charts.LINE_PX, camera),
            simplices.strokes(reaches, Topology.chains([2] * n_arms), rgb_ink(arm_colour), arm_width, camera),
            simplices.strokes(ring, Topology.loops([n_masses]), rgb_ink(ring_colour), charts.LINE_PX, camera, opacity=0.6),
            simplices.markers(ring, rgb_ink(ring_colour), charts.MARKER_PX / 2, camera),
        ],
    )


def polarization_animation(
    directions: core.Point,
    site: core.Rotor,
    arms: core.Point,
    name: str,
    size: tuple[int, int],
    states: Iterable[tuple[core.Scalar, core.Point]],
) -> Iterator[np.ndarray]:
    """Yields `[rows, columns, 3]` RGB frames pairing the deformed overhead mass ring with the banded sky response."""
    spacing = 2 / (n_response_bands - 1)
    middles = np.linspace(-1, 1, n_response_bands)
    bands = contours.Bands(-1 - spacing / 2, spacing, colormaps[signed_cmap](Normalize(-1, 1)(middles)))
    zenith = site[None] >> core.mv.x.dual()                                             # [1] Point-

    @charts.style()
    def rendered_frame(sky_response: core.Scalar, deformed_ring: core.Point) -> np.ndarray:
        figure = plt.figure(figsize=(size[0] / charts.DPI, size[1] / charts.DPI))
        ring_ax = figure.add_subplot(1, 2, 1)
        map_ax = figure.add_subplot(1, 2, 2)
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97, wspace=0.05)
        overhead_ring_panel(ring_ax, site, arms, deformed_ring)
        sky_map(map_ax, directions, sky_response, signed_cmap, -1, 1, zenith, [name], (bands,))
        pixels = capture(figure)
        plt.close(figure)
        return pixels

    for sky_response, deformed_ring in states:
        yield rendered_frame(sky_response, deformed_ring)
