"""Smooth, depth-tested views of a deforming triangle surface, rasterized on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.freeform.geometry import Antivector
from examples.util.wgsl import image_figure, meshes, solids

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


def viewport(points: solids.Point) -> solids.View:
    """A fixed three-quarter view of a surface, square, with room for it to flex."""
    framed = solids.view([points], 0.0, elevation, azimuth, margin, n_pixels)
    return solids.View(framed.camera, np.full(2, framed.half.max()), n_pixels)


def picture(points: solids.Point, faces: np.ndarray, view: solids.View) -> np.ndarray:
    """The surface, smooth shaded by normals blended across each face from its vertices."""
    return meshes.draw(points, faces, surface, meshes.smooth(points, faces), view, background)


def frames(vertices: Iterable[Antivector], faces: np.ndarray) -> Iterator[np.ndarray]:
    """Draw a sequence of surfaces with a camera fixed by the first frame."""
    states = [solids.mv.point(coordinates(state)) for state in vertices]
    view = viewport(states[0])
    for state in states:
        yield picture(state, faces, view)


def draw(vertices: Antivector, faces: np.ndarray) -> plt.Figure:
    """One surface without axes or margins."""
    points = solids.mv.point(coordinates(vertices))
    return image_figure(picture(points, faces, viewport(points)))
