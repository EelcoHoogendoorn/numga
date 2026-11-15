"""Drawing for spin transformations on the GPU: triangulated surfaces coloured by a field on their
faces, and surfaces lit on both sides and textured with a checkerboard carried over from their
reference surface."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from numga.render import meshes, solids

from examples.surfaces.spin_transformations.core import Point, Scalar, Topology
from examples.util import charts
from examples.util.panels import fitted, show

# The camera's elevation and azimuth, its margin as a share of the larger side, and the image's side in pixels, until
# a figure's view is fitted to its axes.
elevation, azimuth = np.radians(20.0), np.radians(35.0)
margin = 0.06
pixels = 300
background = np.ones(3)
# The checkerboard's cubes across the reference surface, and its light and dark squares on the outer and inner
# side of the surface.
squares = 8
outside_light, outside_dark = np.array([0.98, 0.84, 0.62]), np.array([0.90, 0.45, 0.18])
inside_light, inside_dark = np.array([0.70, 0.82, 0.96]), np.array([0.22, 0.42, 0.72])

# A solid checkerboard of cubes, in the place each pixel had on the reference surface, scaled so the cubes
# are unit; light and dark squares, warm outside and cool inside.
checkerboard = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let grid = round(values.xyz);
    let dark = abs(grid.x + grid.y + grid.z) % 2.0 > 0.5;
    let outside = select(outside_light, outside_dark, dark);
    let inside = select(inside_light, inside_dark, dark);
    return vec4<f32>(select(inside, outside, front), 1.0);
}
""", dict(outside_light=outside_light, outside_dark=outside_dark, inside_light=inside_light, inside_dark=inside_dark))


# --- plumbing -------------------------------------------------------------------------
def square_view(vertices: Point, side: int) -> solids.View:
    """A square view framing the surface's `vertices` (`Point-[V]`), `side` pixels across."""
    view = solids.view([solids.origin + solids.context.include_from_algebra(vertices.batch())], 0.0, elevation, azimuth, margin, side)
    return view.squared()


def bare_layout(figure: plt.Figure, axes: np.ndarray) -> None:
    """Strips axis frames and tightens layout around `axes` before views are fitted to their panels."""
    for axis in axes.ravel():
        axis.set_axis_off()
    figure.tight_layout()


def face_colours(values: Scalar, limit: float) -> np.ndarray:
    """Maps face `values` (`Scalar+[F]`) in `[-limit, limit]` to diverging RGB colours (`[F, 3]`)."""
    return plt.get_cmap("coolwarm")((values.to_array().ravel() / limit + 1) / 2)[:, :3]


def coloured_image(topology: Topology, vertices: Point, values: Scalar, limit: float, view: solids.View) -> np.ndarray:
    """Draws the triangulated surface at `vertices` (`Point-[V]`) with faces flat-shaded by `values` (`Scalar+[F]`)."""
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())  # [V] Point-
    triangles = topology.elements[2]
    return meshes.draw(points, triangles, np.zeros(3), meshes.flat(points, triangles), view, background,
                       face_values=face_colours(values, limit), colour=meshes.by_face, gloss=solids.gloss)


def surface_image(topology: Topology, vertices: Point, reference: Point, view: solids.View) -> np.ndarray:
    """Draws the surface at `vertices` (`Point-[V]`) smoothly lit on both sides and textured with a 3D
    checkerboard sampled on `reference` (`Point-[V]`)."""
    mv = reference.context.multivector
    frame_planes = stack((mv.x, mv.y, mv.z))                                  # [3] Plane+
    # Pairing with the coordinate planes reads the reference surface's bounding reach so the cubes have unit width.
    reach = (frame_planes & reference.batch()[..., None]).abs().max()        # [] Scalar+
    corner_points = topology.incidence[2, 0] * (reference * (squares / 2) / reach)  # [3] Point-[F]
    # Read texture coordinates at each triangle corner by pairing, transposed to faces-first for the shader.
    texture_coordinates = (frame_planes & corner_points.batch()[..., None]).to_array().transpose(1, 0, 2)  # [F, 3, 3]
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())  # [V] Point-
    return meshes.draw(points, topology.elements[2], texture_coordinates, meshes.smooth(points, topology), view, background,
                       face_values=np.zeros(4), colour=checkerboard, gloss=solids.gloss)


def surface_figure(topology: Topology, vertices: Point, values: Scalar) -> plt.Figure:
    """Draws the triangulated surface at `vertices` (`Point-[V]`) coloured symmetrically by `values` (`Scalar+[F]`)."""
    limit = float(values.batch().abs().max().to_array())
    figure, axis = plt.subplots(figsize=(5, 5))
    bare_layout(figure, np.array(axis))
    view = fitted(square_view(vertices, pixels), axis)
    show(axis, coloured_image(topology, vertices, values, limit, view), view)
    return figure


def dipole_figure(topology: Topology, vertices: Point, curvature: Scalar, transformed_vertices: Point) -> plt.Figure:
    """Draws the reference sphere coloured by `curvature` (`Scalar+[F]`) beside the deformed surface
    at `transformed_vertices` (`Point-[V]`) carrying the sphere's checkerboard."""
    limit = float(curvature.batch().abs().max().to_array())
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    bare_layout(figure, axes)
    sphere_view = fitted(square_view(vertices, pixels), axes[0])
    transformed_view = fitted(square_view(transformed_vertices, pixels), axes[1])
    show(axes[0], coloured_image(topology, vertices, curvature, limit, sphere_view), sphere_view)
    show(axes[1], surface_image(topology, transformed_vertices, vertices, transformed_view), transformed_view)
    return figure


def textured_figure(topology: Topology, vertices: Point, reference: Point) -> plt.Figure:
    """Draws the surface at `vertices` (`Point-[V]`) textured with `reference`'s (`Point-[V]`) checkerboard."""
    figure, axis = plt.subplots(figsize=(3, 3))
    bare_layout(figure, np.array(axis))
    view = fitted(square_view(vertices, pixels), axis)
    show(axis, surface_image(topology, vertices, reference, view), view)
    return figure


@charts.style()
def dirac_figure(topology: Topology, gallery: dict[int, Point], reference: Point) -> plt.Figure:
    """Draws a gallery row of Dirac spheres (`[...] Point-[V]`) for each eigenvalue, textured with the
    checkerboard of `reference` (`Point-[V]`)."""
    rows, columns = len(gallery), max(len(spheres) for spheres in gallery.values())
    # A panel of 200 pixels a side for each sphere, under its row's eigenvalue label.
    figure, axes = charts.subplots(200 * columns, 200 * rows, nrows=rows, ncols=columns, squeeze=False)
    for row, eigenvalue in enumerate(gallery):
        axes[row, 0].set_title(f"n = {eigenvalue}", loc="left")
    bare_layout(figure, axes)
    for row, spheres in enumerate(gallery.values()):
        for column, sphere in enumerate(spheres):
            view = fitted(square_view(sphere, pixels), axes[row, column])
            show(axes[row, column], surface_image(topology, sphere, reference, view), view)
    return figure


def rounding_animation(topology: Topology, frames: Iterable[Point]) -> Iterator[np.ndarray]:
    """Yields RGB frames along `frames` (`Point-[V]`), textured with the initial surface's checkerboard."""
    iterator = iter(frames)
    first = next(iterator)
    for frame in chain([first], iterator):
        yield surface_image(topology, frame, first, square_view(frame, pixels))
