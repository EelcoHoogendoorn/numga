"""Drawing for gravitational-wave antenna patterns: maps of the sky in the Mollweide projection, each
direction placed over the point on Earth it is straight above, and a ring of free masses across a
wave from overhead.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps, patheffects
from matplotlib.colors import Normalize, to_rgb
from matplotlib.projections.geo import MollweideAxes

from numga import concatenate, stack
from examples.util import charts
from examples.util.animation import capture
from examples.util.wgsl import contours, planar, simplices
from examples.util.wgsl.camera import Camera, flat_panel, panel
from examples.relativity.ligo import core

# Colour maps for signed responses, for sensitivities and for likeness; the colours of the sites, the
# ring and the arms; the drawn reach of the arms on the ring, the samples of the ring at rest, and the
# ring view's half width; the sizes in pixels of the detectors' maps, three to a row and a row per detector, and of
# the network's two maps; the sites' discs, twice a dot's size, and the arms, twice a curve's width, in pixels.
SIGNED, SENSITIVE = "RdBu_r", "magma"
SITE, LABEL, RING, REST, ARM = "#22d3ee", "#0f172a", "#0369a1", "#cbd5e1", "#f97316"
ARM_REACH = 1.25
REST_SAMPLES = 200
RING_LIMIT = 1.5
MAP_PX, ROW_PX, NETWORK_PX = 760, 140, (600, 160)
SITE_RADIUS, ARM_WIDTH = charts.MARKER_PX, 2 * charts.LINE_PX
# The Mollweide projection of longitudes and latitudes, in radians, onto the plane of a map, an ellipse twice as wide as
# it is high, its half width and height these; the map's view a little wider, for its outline. The graticule: meridians
# every 30 degrees, between the parallels at 75 degrees, and parallels every 15 degrees, white and faint; the samples
# along a meridian, and along the outline.
mollweide = MollweideAxes.MollweideTransform(1)
MAP_HALF, MAP_MARGIN = np.array([2 * np.sqrt(2), np.sqrt(2)]), 1.02
MERIDIANS, PARALLELS = np.radians(np.arange(-150, 151, 30)), np.radians(np.arange(-75, 76, 15))
GRATICULE_OPACITY = 0.25
N_MERIDIAN_SAMPLES, N_OUTLINE_SAMPLES = 31, 121


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis: its pairings with the planes x, y and z."""
    mv = core.mv
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def lonlat(directions: core.Point) -> np.ndarray:
    """The longitude and latitude in radians of directions, on the last axis."""
    xyz = coordinates(directions)
    return np.stack([np.arctan2(xyz[..., 1], xyz[..., 0]), np.arcsin(np.clip(xyz[..., 2], -1, 1))], axis=-1)


def projected(longitudes_latitudes: np.ndarray) -> np.ndarray:
    """Longitudes and latitudes in radians, on the last axis, as places on the map, across and up, on the last axis:
    the Mollweide projection."""
    shape = longitudes_latitudes.shape
    return mollweide.transform(longitudes_latitudes.reshape(-1, 2)).reshape(shape)


def graticule() -> tuple[planar.Point, simplices.Mesh]:
    """The map's meridians, between the parallels at 75 degrees, and its parallels, straight across the map: their
    points on the map, one line after another, and the chains of edges along them."""
    cap = PARALLELS[-1]
    latitudes = np.linspace(-cap, cap, N_MERIDIAN_SAMPLES)
    meridians = np.stack(np.broadcast_arrays(MERIDIANS[:, None], latitudes[None, :]), axis=-1)   # [meridians, samples, 2]
    parallels = np.stack(np.broadcast_arrays(np.array([-np.pi, np.pi])[None, :], PARALLELS[:, None]), axis=-1)   # [parallels, 2, 2]
    places = np.concatenate([meridians.reshape(-1, 2), parallels.reshape(-1, 2)])
    return (planar.mv.point(projected(places)),
            simplices.Mesh.chains([N_MERIDIAN_SAMPLES] * len(MERIDIANS) + [2] * len(PARALLELS)))


def outline() -> planar.Point:
    """The map's rim, the meridian at the antimeridian on both sides, up the left and down the right, closed: its points
    on the map."""
    latitudes = np.linspace(-np.pi / 2, np.pi / 2, N_OUTLINE_SAMPLES)
    left = np.stack(np.broadcast_arrays(-np.pi, latitudes), axis=-1)
    right = np.stack(np.broadcast_arrays(np.pi, latitudes[::-1]), axis=-1)
    return planar.mv.point(projected(np.concatenate([left, right, left[:1]])))


def map_camera(ax: plt.Axes) -> Camera:
    """A map's axes framed about the projection's ellipse, without axes; the camera looking down on the map's plane."""
    half = MAP_MARGIN * MAP_HALF
    ax.set(xlim=(-half[0], half[0]), ylim=(-half[1], half[1]), aspect="equal")
    ax.set_axis_off()
    return flat_panel(ax, planar.ga)


def sky_map(ax: plt.Axes, directions: core.Point, values: core.Scalar, cmap: str, low: float, high: float,
            zeniths: core.Point, names: list[str]) -> None:
    """One map over the sky `[latitudes, longitudes]`, each direction at the place the Mollweide projection gives its
    longitude and latitude, coloured between its corners, with a faint graticule and its outline, and a disc over each
    site, with its name."""
    n_latitudes, n_longitudes = directions.shape
    sky = planar.mv.point(projected(lonlat(directions))).reshape(-1)          # [latitudes * longitudes] Point-
    values = values.to_array().reshape(-1)
    colours = colormaps[cmap](Normalize(low, high)(values))[:, :3]
    lines, chains = graticule()
    sites = projected(lonlat(zeniths))                                       # [sites, 2]
    camera = map_camera(ax)
    simplices.overlay(ax, [
        contours.picture(sky, simplices.Mesh.grid(n_latitudes, n_longitudes), values, colours, camera),
        simplices.strokes(lines, chains, np.ones(3), charts.FRAME_PX, camera, opacity=GRATICULE_OPACITY),
        simplices.strokes(outline(), simplices.Mesh.chains([2 * N_OUTLINE_SAMPLES + 1]), np.zeros(3),
                          charts.FRAME_PX, camera),
        simplices.markers(planar.mv.point(sites), ink(LABEL), SITE_RADIUS + charts.FRAME_PX, camera),
        simplices.markers(planar.mv.point(sites), ink(SITE), SITE_RADIUS, camera)])
    for site, name in zip(sites, names):
        ax.annotate(name, site, xytext=(6, 4), textcoords="offset points", color=LABEL,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])


@charts.style()
def draw_patterns(directions: core.Point, plus: core.Scalar, cross: core.Scalar, sensitivities: core.Scalar,
                  zeniths: core.Point, names: list[str]) -> plt.Figure:
    """A row per detector: its response to the plus strain, to the cross strain, and its sensitivity."""
    figure = plt.figure(figsize=(MAP_PX / charts.DPI, (ROW_PX * len(names) + 10) / charts.DPI))
    for detector, name in enumerate(names):
        maps = [plus[detector], cross[detector], sensitivities[detector]]
        for column, (values, cmap, low) in enumerate(zip(maps, [SIGNED, SIGNED, SENSITIVE], [-1, -1, 0])):
            figure.add_subplot(len(names), 3, 3 * detector + column + 1)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.08, wspace=0.06)
    # The maps drawn once the layout is set, each at the pixels its panel gets.
    for detector, name in enumerate(names):
        maps = [plus[detector], cross[detector], sensitivities[detector]]
        for column, (values, cmap, low) in enumerate(zip(maps, [SIGNED, SIGNED, SENSITIVE], [-1, -1, 0])):
            sky_map(figure.axes[3 * detector + column], directions, values, cmap, low, 1,
                    zeniths[detector:detector + 1], [name])
    return figure


@charts.style()
def draw_network(directions: core.Point, network: core.Scalar, alike: core.Scalar, zeniths: core.Point,
                 names: list[str]) -> plt.Figure:
    """The network's sensitivity, and how alike the detectors' readings of the same wave are."""
    figure = plt.figure(figsize=(NETWORK_PX[0] / charts.DPI, NETWORK_PX[1] / charts.DPI))
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.06)
    for ax, (values, cmap, low, high) in zip(axes, [(network, SENSITIVE, 0, np.sqrt(len(names))),
                                                     (alike, SIGNED, -1, 1)]):
        sky_map(ax, directions, values, cmap, low, high, zeniths, names)
    return figure


def draw_ring(ax: plt.Axes, site: core.Rotor, arms: core.Point, ring: core.Point) -> None:
    """The ring of free masses across a wave from overhead, at rest and stretched, with the detector's
    arms, seen from above: east to the right, north up."""
    mv = core.mv
    ax.set(xlim=(-RING_LIMIT, RING_LIMIT), ylim=(-RING_LIMIT, RING_LIMIT), aspect="equal")
    ax.set_axis_off()
    # East and north at the site: the site's frame carries the planes y and z there, which the camera reads by.
    camera = panel(ax, site >> mv.y, site >> mv.z)
    # The ring at rest: the unit direction east turned all the way round through north.
    turns = (mv.yz * (-np.linspace(0, 2 * np.pi, REST_SAMPLES) / 2)).exp()   # [samples] Rotor+
    rest = site >> (turns >> mv.y.dual())                                    # [samples] Point-
    # Each arm from the site out to its drawn reach.
    reaches = stack((0 * arms, ARM_REACH * arms), axis=-1).reshape(-1)       # [arms * 2] Point-
    n_masses = ring.shape[0]
    simplices.overlay(ax, [
        simplices.strokes(rest, simplices.Mesh.chains([REST_SAMPLES]), ink(REST), charts.LINE_PX, camera),
        simplices.strokes(reaches, simplices.Mesh.chains([2] * arms.shape[0]), ink(ARM), ARM_WIDTH, camera),
        simplices.strokes(concatenate((ring, ring[:1])), simplices.Mesh.chains([n_masses + 1]), ink(RING),
                          charts.LINE_PX, camera, opacity=0.6),
        simplices.markers(ring, ink(RING), charts.MARKER_PX / 2, camera)])


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def animate_polarization(directions: core.Point, responses: core.Scalar, site: core.Rotor, arms: core.Point,
                         rings: core.Point, name: str, size: tuple[int, int]) -> list[np.ndarray]:
    """Each frame, in a figure `size` pixels wide and high: the ring stretched by the plus strain at the
    polarization angle, beside the detector's response to it over the sky."""

    @charts.style()
    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(size[0] / charts.DPI, size[1] / charts.DPI))
        ring_ax = figure.add_subplot(1, 2, 1)
        ax = figure.add_subplot(1, 2, 2)
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97, wspace=0.05)
        draw_ring(ring_ax, site, arms, rings[index])
        sky_map(ax, directions, responses[index], SIGNED, -1, 1, site[None] >> core.mv.x.dual(), [name])
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(rings.shape[0])]
