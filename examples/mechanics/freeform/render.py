"""Smooth, depth-tested views of a deforming triangle surface, rasterized on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.freeform.geometry import Antivector, Topology
from examples.util.wgsl import image_figure, meshes, solids

n_pixels = 230
elevation, azimuth = np.radians(24.0), np.radians(54.0)
# Room around the first surface for the later ones to flex into, as a share of its larger side.
margin = 0.12
surface = np.array([0.25, 0.61, 0.68])
background = np.ones(3)


def viewport(points: solids.Point) -> solids.View:
    """A fixed three-quarter view of a surface, square, with room for it to flex."""
    return solids.view([points], 0.0, elevation, azimuth, margin, n_pixels).squared()


def picture(points: solids.Point, topology: Topology, view: solids.View) -> np.ndarray:
    """The surface, smooth shaded by normals blended across each face from its vertices."""
    return meshes.draw(points, topology.elements[2], surface, meshes.smooth(points, topology), view, background, face_values=np.zeros(4),
                       colour=meshes.plain, gloss=solids.gloss)


def frames(vertices: Iterable[Antivector], topology: Topology) -> Iterator[np.ndarray]:
    """Draw a sequence of surfaces of the mesh's triangles with a camera fixed by the first frame."""
    # The projective embedding: places of space, offsets from its origin, as the renderer's points about its own.
    states = [solids.origin + solids.context.include_from_algebra(state) for state in vertices]
    view = viewport(states[0])
    for state in states:
        yield picture(state, topology, view)


def draw(vertices: Antivector, topology: Topology) -> plt.Figure:
    """One surface of the mesh's triangles without axes or margins."""
    # The projective embedding: places of space, offsets from its origin, as the renderer's points about its own.
    points = solids.origin + solids.context.include_from_algebra(vertices)               # [vertices] Point-
    return image_figure(picture(points, topology, viewport(points)))
