"""Smooth, depth-tested views of a deforming triangle surface, rasterized on the GPU."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.freeform.geometry import Antivector
from examples.wgsl import meshes, solids

n_pixels = 460
elevation, azimuth = np.radians(24.0), np.radians(54.0)
# Room around the first surface for the later ones to flex into, as a share of its larger side.
margin = 0.12
surface = np.array([0.25, 0.61, 0.68])
background = np.ones(3)


def coordinates(vertices: Antivector) -> np.ndarray:
    """Cartesian positions at the drawing boundary, read by pairing with the planes across the axes."""
    mv = vertices.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & vertices[..., None]).to_array()


def viewport(points: solids.Point) -> tuple[solids.Motor, np.ndarray]:
    """A fixed three-quarter view of a surface, square, with room for it to flex."""
    camera, half = solids.view([points], 0.0, elevation, azimuth, margin)
    return camera, np.full(2, half.max())


def picture(points: solids.Point, faces: np.ndarray, camera: solids.Motor, half: np.ndarray) -> np.ndarray:
    """The surface, smooth shaded by normals blended across each face from its vertices."""
    return meshes.draw(points, faces, surface, meshes.smooth(points, faces), camera, half, elevation, azimuth,
                       n_pixels, background)


def frames(vertices: Iterable[Antivector], faces: np.ndarray) -> list[np.ndarray]:
    """Draw a sequence of surfaces with a camera fixed by the first frame."""
    states = [solids.mv.point(coordinates(state)) for state in vertices]
    view = viewport(states[0])
    return [picture(state, faces, *view) for state in states]


def draw(vertices: Antivector, faces: np.ndarray) -> plt.Figure:
    """One surface without axes or margins."""
    points = solids.mv.point(coordinates(vertices))
    image = picture(points, faces, *viewport(points))
    figure, ax = plt.subplots(figsize=(n_pixels / 100, n_pixels / 100), dpi=100)
    ax.imshow(image)
    ax.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return figure
