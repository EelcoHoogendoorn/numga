"""The rays of a quadratic curve and its complement in the three dimensions of x, y and weight, the cone they
sweep, and where both meet the unit sphere."""

from __future__ import annotations

import contourpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from numga import stack
from examples.animation import capture
from examples.geometry.bezier import core

CURVE_COLOUR = "#287fa0"
COMPLEMENT_COLOUR = "#d06c36"
CONTROL_COLOUR = "0.55"
# The samples of a circle, of the sphere's mesh, and of the grid the cone is found on.
CIRCLE = 200
SPHERE_MESH = 60
SPHERE_GRID = 200
# The direction homogeneous space is viewed from, in degrees.
ELEVATION, AZIMUTH = 18.0, -62.0


# --- plumbing -------------------------------------------------------------------------
def scalars(values: core.Scalar) -> np.ndarray:
    """The values of a batch of scalars, for printing and checks."""
    return values.cast(values.algebra.subspace.scalar()).kernel[..., 0]


def homogeneous(points: core.Vector) -> np.ndarray:
    """x, y and weight of each point, on the last axis."""
    return points.cast(core.ga.subspace("x y z")).kernel


def cones(shapes: core.Conic) -> list[list[np.ndarray]]:
    """Where each conic's form vanishes on the unit sphere, for a batch of conics: the loops its cone cuts,
    each an array of unit vectors in x, y and weight."""
    longitude, latitude = np.meshgrid(np.linspace(-np.pi, np.pi, SPHERE_GRID), np.linspace(-np.pi / 2, np.pi / 2, SPHERE_GRID))
    xyw = np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=-1)
    grid = core.mv.vector(xyw)[..., None]                                      # [grid, grid, 1] Vector
    values = np.moveaxis(scalars(grid | shapes(grid)), -1, 0)                  # [conics, grid, grid]
    return [[np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
             for lon, lat in (loop.T for loop in contourpy.contour_generator(longitude, latitude, value).lines(0.0))]
            for value in values]


def draw_rays(controls: core.Vector, curve: core.Vector, complement: core.Vector, loops: list[np.ndarray], reach: float,
              azimuth: float) -> plt.Figure:
    """A curve and its complement as their sums are, in the three dimensions of x, y and weight, with their
    weighted control points, and with the rays from the origin to the curves filled in: the part of the
    conic's cone each sweeps. Where the rays pierce
    the unit sphere, the curves normalised, and where the cone does, its two antipodal loops. The equator
    is the line at infinity."""
    figure = plt.figure(figsize=(4.8, 4.4), layout="constrained")
    ax = figure.add_subplot(projection="3d")
    longitude, latitude = np.meshgrid(np.linspace(0, 2 * np.pi, SPHERE_MESH), np.linspace(-np.pi / 2, np.pi / 2, SPHERE_MESH))
    ax.plot_surface(np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude),
                    color="0.85", alpha=0.18, linewidth=0, shade=False)
    angles = np.linspace(0, 2 * np.pi, CIRCLE + 1)
    ax.plot(np.cos(angles), np.sin(angles), 0 * angles, color="0.45", linewidth=1)
    for loop in loops:
        ax.plot(*loop.T, color="0.55", linewidth=1)
    # The weighted control points of the curve and of its complement, each on its ray out to the sphere.
    for polygon in homogeneous(controls):                                      # [controls, 3]
        ax.plot(*polygon.T, color=CONTROL_COLOUR, linestyle=":", linewidth=1)
        ax.scatter(*polygon.T, color=CONTROL_COLOUR, s=14, depthshade=False)
        for corner in polygon:
            ax.plot(*np.stack([0 * corner, corner * max(1.0, 1 / np.linalg.norm(corner))]).T, color=CONTROL_COLOUR, linewidth=0.6)
    for points, colour in ((curve, CURVE_COLOUR), (complement, COMPLEMENT_COLOUR)):
        xyw = homogeneous(points)                                              # [samples, 3]
        ax.plot(*xyw.T, color=colour, linewidth=1.2)
        fan = np.stack([np.zeros_like(xyw[:-1]), xyw[:-1], xyw[1:]], axis=1)   # [samples, 3, 3]
        ax.add_collection3d(Poly3DCollection(fan, facecolor=colour, alpha=0.12, linewidth=0))
        ax.plot(*(xyw / np.linalg.norm(xyw, axis=-1, keepdims=True)).T, color=colour, linewidth=2.4)
    ax.scatter(0, 0, 0, color="0.3", s=10)
    ax.view_init(elev=ELEVATION, azim=azimuth)
    ax.set(xlim=(-reach, reach), ylim=(-reach, reach), zlim=(-reach, reach))
    ax.set_box_aspect((1, 1, 1), zoom=1.35)
    ax.set_axis_off()
    return figure


def framing(points: core.Vector) -> float:
    """The half width that frames every point, as its sum is, and the unit sphere."""
    return max(1.0, np.abs(homogeneous(points)).max())


def animate_rays(controls: core.Vector, curves: core.Vector, complements: core.Vector, shapes: core.Conic) -> list[np.ndarray]:
    """The ray figure for each conic of a batch, viewed from turning once around, at one framing."""
    loops = cones(shapes)
    half_width = framing(stack([curves, complements]))
    azimuths = AZIMUTH + np.linspace(0, 360, curves.shape[0], endpoint=False)

    def frame(index: int) -> np.ndarray:
        figure = draw_rays(controls[index], curves[index], complements[index], loops[index], half_width, azimuths[index])
        pixels = capture(figure)
        plt.close(figure)
        return pixels
    return [frame(index) for index in range(curves.shape[0])]
