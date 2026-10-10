"""Figures for rational Bézier curves: curves through infinity, conics with their tangent planes, and
triangles carried by curves of motors."""

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
CONIC_COLOUR = "0.8"
TANGENT_COLOUR = "#626c76"
BLEND_COLOUR = "#d06c36"
GEODESIC_COLOUR = "#287fa0"
GRID = 400
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


def homogeneous(points: core.Point) -> np.ndarray:
    """x, y and weight of each point, on the last axis."""
    return points.cast(core.ga.subspace("yw wx xy")).kernel


def path(points: core.Point) -> np.ndarray:
    """Coordinates along a curve of points, broken where the curve passes through infinity: there the
    weight changes sign."""
    xyw = homogeneous(points)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = xyw[..., :2] / xyw[..., 2:]
    crossing = np.concatenate([np.zeros(xyw.shape[:-2] + (1,), bool), np.diff(np.sign(xyw[..., 2]), axis=-1) != 0], axis=-1)
    return np.where(crossing[..., None], np.nan, xy)


def draw_controls(ax: plt.Axes, controls: core.Point, reach: float) -> None:
    """The control polygon: segments between finite control points, and from a finite control point a ray
    towards a neighbouring point at infinity."""
    xyw = homogeneous(controls)
    for start, end in zip(xyw[:-1], xyw[1:]):
        finite, other = (start, end) if start[2] != 0 else (end, start)
        tip = other[:2] / other[2] if other[2] != 0 else finite[:2] / finite[2] + reach * other[:2] / np.linalg.norm(other[:2])
        ax.plot(*np.stack([finite[:2] / finite[2], tip]).T, color=CONTROL_COLOUR, linestyle=":", linewidth=1)
    finite = xyw[xyw[:, 2] != 0]
    ax.scatter(*(finite[:, :2] / finite[:, 2:]).T, color=CONTROL_COLOUR, s=18, zorder=3)


def panels(count: int, box: tuple[float, float, float, float]) -> tuple[plt.Figure, np.ndarray]:
    figure, axes = plt.subplots(1, count, figsize=(3.6 * count, 3.4), layout="constrained", squeeze=False)
    for ax in axes[0]:
        ax.set(xlim=box[:2], ylim=box[2:], aspect="equal")
        ax.set_axis_off()
    return figure, axes[0]


def draw_curves(controls: core.Point, curves: core.Point, complements: core.Point, box: tuple[float, float, float, float]) -> plt.Figure:
    """Each case's control polygon, its curve, and the complementary curve dashed."""
    figure, axes = panels(curves.shape[0], box)
    for ax, control, curve, complement in zip(axes, controls, path(curves), path(complements)):
        draw_controls(ax, control, box[1] - box[0])
        ax.plot(*complement.T, color=COMPLEMENT_COLOUR, linestyle="--", linewidth=1.3)
        ax.plot(*curve.T, color=CURVE_COLOUR, linewidth=2)
    return figure


def draw_conics(controls: core.Point, curves: core.Point, complements: core.Point, shapes: core.Conic,
                box: tuple[float, float, float, float]) -> plt.Figure:
    """Each case's conic, evaluated on a grid and contoured at zero, with the curve and its complement."""
    figure = draw_curves(controls.broadcast_to(curves.shape[:1] + controls.shape), curves, complements, box)
    x, y = np.meshgrid(np.linspace(*box[:2], GRID), np.linspace(*box[2:], GRID))
    grid = core.point(np.stack([x, y], axis=-1))                               # [grid, grid] Point
    values = (grid[..., None] & shapes(grid[..., None])).cast(core.ga.subspace.scalar()).kernel[..., 0]
    for ax, value in zip(figure.axes, np.moveaxis(values, -1, 0)):
        ax.contour(x, y, value, levels=[0.0], colors=CONIC_COLOUR, linewidths=5, zorder=0)
    return figure


def draw_tangents(planes: core.Plane, figure: plt.Figure) -> plt.Figure:
    """Each case's tangent planes over its conic."""
    abc = planes.cast(core.ga.subspace("x y w")).kernel
    for ax, case in zip(figure.axes, abc):
        for a, b, c in case:
            foot = -np.array([a, b]) * c / (a * a + b * b)
            ax.axline(foot, foot + np.array([-b, a]), color=TANGENT_COLOUR, linewidth=0.6, alpha=0.7)
    return figure


def draw_motors(centres: core.Point, placements: core.Point, paths: core.Point, frames: core.Point) -> plt.Figure:
    """One panel per curve of motors: the control placements filled and joined by the control polygon,
    the path the curve carries the origin along, and the carried triangle every so often."""
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    for ax, curve, triangles, colour in zip(axes, path(paths), path(frames), (BLEND_COLOUR, GEODESIC_COLOUR)):
        ax.plot(*path(centres).T, color=CONTROL_COLOUR, linestyle=":", linewidth=1)
        for corners in path(placements):
            ax.fill(*corners.T, color=CONTROL_COLOUR, alpha=0.5)
        ax.plot(*curve.T, color=colour, linewidth=1.5)
        for corners in triangles:
            ax.fill(*corners.T, facecolor="none", edgecolor=colour, linewidth=0.9)
        ax.set_aspect("equal")
        ax.set_axis_off()
    return figure


# --- homogeneous space ---------------------------------------------------------------
def cones(shapes: core.Conic) -> list[list[np.ndarray]]:
    """Where each conic's form vanishes on the unit sphere, for a batch of conics: the loops its cone cuts,
    each an array of unit vectors in x, y and weight."""
    longitude, latitude = np.meshgrid(np.linspace(-np.pi, np.pi, SPHERE_GRID), np.linspace(-np.pi / 2, np.pi / 2, SPHERE_GRID))
    xyw = np.stack([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)], axis=-1)
    grid = core.mv("x y w", xyw).dual()[..., None]                             # [grid, grid, 1] Point
    values = np.moveaxis(scalars(grid & shapes(grid)), -1, 0)                  # [conics, grid, grid]
    return [[np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=-1)
             for lon, lat in (loop.T for loop in contourpy.contour_generator(longitude, latitude, value).lines(0.0))]
            for value in values]


def draw_rays(controls: core.Point, curve: core.Point, complement: core.Point, loops: list[np.ndarray], reach: float,
              azimuth: float) -> plt.Figure:
    """A curve and its complement as their sums are, in the three dimensions of x, y and weight, with their
    weighted control points, and with the rays from the origin to the curves filled in: the part of the
    conic's cone each sweeps. Where the rays pierce
    the unit sphere, the curves normalised, and where the cone does, its two antipodal loops. The equator
    is the plane at infinity."""
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


def framing(points: core.Point) -> float:
    """The half width that frames every point, as its sum is, and the unit sphere."""
    return max(1.0, np.abs(homogeneous(points)).max())


def animate_rays(controls: core.Point, curves: core.Point, complements: core.Point, shapes: core.Conic) -> list[np.ndarray]:
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
