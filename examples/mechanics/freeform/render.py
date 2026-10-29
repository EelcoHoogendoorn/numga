"""Smooth, depth-tested views of a deforming triangle surface, rasterized on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.freeform.geometry import Antivector, Mesh
from examples.util.wgsl import image_figure, meshes, solids

n_pixels = 230
elevation, azimuth = np.radians(24.0), np.radians(54.0)
# Room around the first surface for the later ones to flex into, as a share of its larger side.
margin = 0.12
surface = np.array([0.25, 0.61, 0.68])
background = np.ones(3)


def gpu_points(vertices: Antivector) -> solids.Point:
    """The GPU library's own points at the places of the solid's: their coordinates, read by pairing with the planes
    across the axes, are the one hand-off from this algebra to the renderer's."""
    mv = vertices.context.multivector
    return solids.mv.point((stack((mv.x, mv.y, mv.z)) & vertices[..., None]).to_array())


def viewport(points: solids.Point) -> solids.View:
    """A fixed three-quarter view of a surface, square, with room for it to flex."""
    return solids.view([points], 0.0, elevation, azimuth, margin, n_pixels).squared()


def picture(points: solids.Point, faces: np.ndarray, view: solids.View) -> np.ndarray:
    """The surface, smooth shaded by normals blended across each face from its vertices."""
    return meshes.draw(points, faces, surface, meshes.smooth(points, faces), view, background, face_values=np.zeros(4),
                       colour=meshes.plain, gloss=solids.gloss)


def frames(vertices: Iterable[Antivector], mesh: Mesh) -> Iterator[np.ndarray]:
    """Draw a sequence of surfaces of the mesh's triangles with a camera fixed by the first frame."""
    states = [gpu_points(state) for state in vertices]
    view = viewport(states[0])
    for state in states:
        yield picture(state, mesh.faces, view)


def draw(vertices: Antivector, mesh: Mesh) -> plt.Figure:
    """One surface of the mesh's triangles without axes or margins."""
    points = gpu_points(vertices)
    return image_figure(picture(points, mesh.faces, viewport(points)))
