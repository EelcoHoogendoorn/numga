"""Sensor images and side views of the lens camera: drawing only.

The sensor is rasterised from the image cones as implicit functions. Coordinates leave the
algebra here: a point is read by its pairings with the planes x, y and z, over its weight.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.animation import capture
from examples.optics.lens_camera.core import Motor, Plane, Point, Quadric, ga, mv, point

# Sensor pixels, rows by columns, and their subdivision for antialiasing; the edges are smooth already.
RESOLUTION = (210, 270)
SUPERSAMPLE = 1
# How bright a disc's light shows, per unit of its energy over the sensor's pixels, and the width of a
# disc's edge, a crude diffraction limit, in pixels.
BRIGHTNESS = 1.36e-3
EDGE_PIXELS = 1.0
# Half extents of the sensor window in its own frame.
SENSOR = (0.225, 0.175)
# The aperture whose exposure is the reference brightness, and the half-heights the front and rear lens
# are drawn with in the side view.
REFERENCE_APERTURE = 0.45
LENS_HALF_HEIGHTS = [0.6, 0.35]
# The colour each depth layer of the scene lights on the sensor, red, green and blue, in the order of the layers.
LAYER_COLOURS = np.eye(3)


# --- plumbing -------------------------------------------------------------------------
def xyz(points: Point) -> np.ndarray:
    """(..., 3) Euclidean coordinates of finite points: their pairings with the planes x, y and z, over
    their weights."""
    return ((stack((mv.x, mv.y, mv.z)) & points[..., None]) / (mv.w & points)[..., None]).to_array()


def rasterise(frame: Motor, cones: Quadric, energy: float) -> np.ndarray:
    """Rasterise the sensor from the image cones as implicit functions.

    A pixel's coverage by a point's blur disc comes from a first-order signed distance to the
    disc boundary: the cone's form at the pixel over the length of its gradient, which is twice
    the Euclidean normal of the polar plane. The edge is a logistic a pixel wide, a crude
    diffraction limit that spreads a focused point over a few pixels. Each disc deposits the
    same total energy, scaled by the aperture area, so a point in focus is a bright dot and a
    defocused one a dim wide disc. Scene points of each depth layer light one colour channel.
    """
    rows, cols = (RESOLUTION[0] * SUPERSAMPLE, RESOLUTION[1] * SUPERSAMPLE)
    y, z = np.meshgrid(np.linspace(-SENSOR[0], SENSOR[0], cols), np.linspace(SENSOR[1], -SENSOR[1], rows))
    pixels = frame >> point(np.stack([np.zeros_like(y), y, z], axis=-1).reshape(-1, 3))
    polar = cones.reshape(-1, 1)(pixels)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = (pixels & polar).to_array() / (2 * polar.norm().to_array())
    # A logistic edge, its width in output pixels.
    coverage = 1.0 / (1.0 + np.exp(-distance / (2 * SENSOR[0] / RESOLUTION[1] * EDGE_PIXELS)))
    share = coverage / np.maximum(coverage.sum(axis=-1, keepdims=True), 1e-12) * energy
    layer = np.broadcast_to(np.arange(cones.shape[0])[:, None, None], cones.shape).ravel()
    image = np.zeros((rows * cols, 3))
    for channel in range(3):
        image[:, channel] = share[layer == channel].sum(axis=0)
    image = image.reshape(RESOLUTION[0], SUPERSAMPLE, RESOLUTION[1], SUPERSAMPLE, 3).mean(axis=(1, 3))
    # Each disc's share of a pixel, over the pixel's share of the sensor, is the light's density there.
    return np.clip(image * BRIGHTNESS * rows * cols, 0.0, 1.0)


def draw_side(ax, planes: Plane, heights: list[float], legs: Point, scene: Point) -> None:
    """Side view in the x-y plane: element planes as segments, the scene layers, and a ray fan drawn leg by leg."""
    ax.cla()
    for plane, height in zip(planes, heights):
        top, bottom = xyz(plane ^ mv.z ^ (mv.y - mv.w * height)), xyz(plane ^ mv.z ^ (mv.y + mv.w * height))
        ax.plot([top[0], bottom[0]], [top[1], bottom[1]], color="gray", linewidth=2)
    xy = xyz(scene)[..., :2]                                              # [depth, height, width, 2]
    colours = np.broadcast_to(LAYER_COLOURS.reshape((3,) + (1,) * (xy.ndim - 2) + (3,)), xy.shape[:-1] + (3,))
    ax.scatter(xy[..., 0].ravel(), xy[..., 1].ravel(), s=6, color=colours.reshape(-1, 3))
    fan = xyz(legs)                                                       # [plane + 1, ray, 3]
    ax.plot(fan[..., 0], fan[..., 1], color="tab:orange", linewidth=0.7)
    ax.set_xlim(-3.6, 2.6); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal"); ax.axis("off")


def draw_stills(exposures) -> plt.Figure:
    """The sensor image of each still setting."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=120)
    for ax, (collineation, camera, frame, cones, planes, legs) in zip(axes, exposures):
        ax.imshow(rasterise(frame, cones, 1.0), interpolation="nearest")
        ax.axis("off")
    return fig


def animate_camera(states, scene: Point) -> list[np.ndarray]:
    """Frames of the sensor image over a side view of the camera, one per state."""
    fig = plt.figure(figsize=(6, 6.6), dpi=100)
    top, side = fig.subplots(2, 1, height_ratios=(3, 1))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02, hspace=0.08)
    frames = []
    for rear_at, focus_at, radius, (collineation, camera, frame, cones, planes, legs) in states:
        top.cla()
        top.imshow(rasterise(frame, cones, (radius / REFERENCE_APERTURE)**2), interpolation="nearest")
        top.axis("off")
        draw_side(side, planes, [radius] + LENS_HALF_HEIGHTS, legs, scene)
        frames.append(capture(fig))
    plt.close(fig)
    return frames
