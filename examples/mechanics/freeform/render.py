"""Smooth, depth-tested views of a deforming triangulated surface, rasterized on the GPU."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.freeform.geometry import Antivector, Topology
from numga.render import meshes, solids
from examples.util.panels import image_figure

n_pixels = 230
elevation, azimuth = np.radians(24.0), np.radians(54.0)
# Margin around the rest surface so deforming frames stay within the camera box.
margin = 0.12
surface_colour = np.array([0.25, 0.61, 0.68])
background = np.ones(3)


def viewport(points: solids.Point) -> solids.View:
    """Three-quarter square camera view framed around `points` with room to flex."""
    return solids.view([points], 0.0, elevation, azimuth, margin, n_pixels).squared()


def picture(points: solids.Point, topology: Topology, view: solids.View) -> np.ndarray:
    """Smooth-shaded RGB rasterization of `topology` at `points` from `view`."""
    return meshes.draw(points, topology.elements[2], surface_colour, meshes.smooth(points, topology),
                       view, background, face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def surface_frames(topology: Topology, states: Iterable[Antivector]) -> Iterator[np.ndarray]:
    """Yields RGB frames of the triangulated surface `topology` across `states` with the camera fixed by the initial state."""
    iterator = iter(states)
    first_state = next(iterator, None)
    if first_state is None:
        return
    # Projective embedding maps spatial direction offsets onto points around the renderer's origin.
    first_points = solids.origin + solids.context.include_from_algebra(first_state)  # [...] Point-
    view = viewport(first_points)
    yield picture(first_points, topology, view)
    for state in iterator:
        points = solids.origin + solids.context.include_from_algebra(state)          # [...] Point-
        yield picture(points, topology, view)


def surface_figure(vertices: Antivector, topology: Topology) -> plt.Figure:
    """Renders a single borderless view of the triangulated surface `topology` at `vertices`."""
    # Projective embedding maps spatial direction offsets onto points around the renderer's origin.
    points = solids.origin + solids.context.include_from_algebra(vertices)           # [...] Point-
    return image_figure(picture(points, topology, viewport(points)))
