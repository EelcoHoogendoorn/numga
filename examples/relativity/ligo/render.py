"""Drawing for gravitational-wave antenna patterns: maps of the sky in the Mollweide projection, each
direction placed over the point on Earth it is straight above, and a ring of free masses across a
wave from overhead.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

from examples.animation import capture
from examples.relativity.ligo import core

ga = core.ga
# Colour maps for signed responses, for sensitivities and for likeness; the colours of the sites, the
# ring and the arms; the drawn reach of the arms on the ring, the samples of the ring at rest, and the
# ring view's half width.
SIGNED, SENSITIVE = "RdBu_r", "magma"
SITE, LABEL, RING, REST, ARM = "#22d3ee", "#0f172a", "#0369a1", "#cbd5e1", "#f97316"
ARM_REACH = 1.25
REST_SAMPLES = 200
RING_LIMIT = 1.5


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for drawing and checks."""
    return values.cast(ga.subspace.scalar()).kernel[..., 0]


def coordinates(points: core.Point) -> np.ndarray:
    """x, y and z of each point, on the last axis."""
    return points.cast(ga.subspace("yz zx xy")).kernel


def lonlat(directions: core.Point) -> np.ndarray:
    """The longitude and latitude in radians of directions, on the last axis."""
    xyz = coordinates(directions)
    return np.stack([np.arctan2(xyz[..., 1], xyz[..., 0]), np.arcsin(np.clip(xyz[..., 2], -1, 1))], axis=-1)


def sky_map(ax: plt.Axes, directions: core.Point, values: core.Scalar, cmap: str, low: float, high: float) -> None:
    """One map over the sky, with a faint graticule and no tick labels."""
    longitudes, latitudes = np.moveaxis(lonlat(directions), -1, 0)
    ax.pcolormesh(longitudes, latitudes, scalars(values), shading="gouraud", cmap=cmap, vmin=low, vmax=high)
    ax.grid(color="white", alpha=0.25, linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def mark_sites(ax: plt.Axes, zeniths: core.Point, names: list[str]) -> None:
    """A star over each site, with its name."""
    for (longitude, latitude), name in zip(lonlat(zeniths), names):
        ax.plot(longitude, latitude, "*", color=SITE, markersize=9, markeredgecolor=LABEL, markeredgewidth=0.5)
        ax.annotate(name, (longitude, latitude), xytext=(6, 4), textcoords="offset points", fontsize=8, color=LABEL,
                    path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")])


def draw_patterns(directions: core.Point, plus: core.Scalar, cross: core.Scalar, sensitivities: core.Scalar,
                  zeniths: core.Point, names: list[str]) -> plt.Figure:
    """A row per detector: its response to the plus strain, to the cross strain, and its sensitivity."""
    figure = plt.figure(figsize=(14, 2.5 * len(names) + 0.2), dpi=110)
    for detector, name in enumerate(names):
        maps = [plus[detector], cross[detector], sensitivities[detector]]
        for column, (values, cmap, low) in enumerate(zip(maps, [SIGNED, SIGNED, SENSITIVE], [-1, -1, 0])):
            ax = figure.add_subplot(len(names), 3, 3 * detector + column + 1, projection="mollweide")
            sky_map(ax, directions, values, cmap, low, 1)
            mark_sites(ax, zeniths[detector:detector + 1], [name])
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, hspace=0.08, wspace=0.06)
    return figure


def draw_network(directions: core.Point, network: core.Scalar, alike: core.Scalar, zeniths: core.Point,
                 names: list[str]) -> plt.Figure:
    """The network's sensitivity, and how alike the detectors' readings of the same wave are."""
    figure = plt.figure(figsize=(11, 2.9), dpi=110)
    for column, (values, cmap, low, high) in enumerate([(network, SENSITIVE, 0, np.sqrt(len(names))),
                                                        (alike, SIGNED, -1, 1)]):
        ax = figure.add_subplot(1, 2, column + 1, projection="mollweide")
        sky_map(ax, directions, values, cmap, low, high)
        mark_sites(ax, zeniths, names)
    figure.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.02, wspace=0.06)
    return figure


def draw_ring(ax: plt.Axes, site: core.Rotor, arms: core.Point, ring: core.Point) -> None:
    """The ring of free masses across a wave from overhead, at rest and stretched, with the detector's
    arms, seen from above: east to the right, north up."""
    def level(vectors: core.Point) -> np.ndarray:
        """East and north of vectors at the site, on the last axis: their coordinates in the site's frame."""
        return coordinates(site.reverse() >> vectors)[..., 1:]

    rest = np.linspace(0, 2 * np.pi, REST_SAMPLES)
    ax.plot(np.cos(rest), np.sin(rest), color=REST, linewidth=1.0)
    for arm in level(arms):
        ax.plot([0, ARM_REACH * arm[0]], [0, ARM_REACH * arm[1]], color=ARM, linewidth=3, solid_capstyle="round")
    masses = level(ring)                                                     # [masses, 2]
    ax.plot(*np.concatenate([masses, masses[:1]]).T, color=RING, linewidth=1.0, alpha=0.6)
    ax.plot(*masses.T, "o", color=RING, markersize=5)
    ax.set(xlim=(-RING_LIMIT, RING_LIMIT), ylim=(-RING_LIMIT, RING_LIMIT), aspect="equal")
    ax.set_axis_off()


def animate_polarization(directions: core.Point, responses: core.Scalar, site: core.Rotor, arms: core.Point,
                         rings: core.Point, name: str) -> list[np.ndarray]:
    """Each frame: the ring stretched by the plus strain at the polarization angle, beside the detector's
    response to it over the sky."""

    def frame(index: int) -> np.ndarray:
        figure = plt.figure(figsize=(12, 4.0), dpi=100)
        draw_ring(figure.add_subplot(1, 2, 1), site, arms, rings[index])
        ax = figure.add_subplot(1, 2, 2, projection="mollweide")
        sky_map(ax, directions, responses[index], SIGNED, -1, 1)
        mark_sites(ax, site[None] >> core.mv.x.dual(), [name])
        figure.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97, wspace=0.05)
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(rings.shape[0])]
