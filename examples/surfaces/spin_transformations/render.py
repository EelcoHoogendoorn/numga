"""Drawing for spin transformations, on the GPU: meshes coloured by a field on their faces, and surfaces lit
on both sides and textured with a checkerboard carried over from the sphere they came from."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.spin_transformations.core import Mesh, Point, Scalar, ga
from examples.util.wgsl import fitted, meshes, show, solids

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
    let cells = round(values.xyz);
    let dark = abs(cells.x + cells.y + cells.z) % 2.0 > 0.5;
    let outside = select(outside_light, outside_dark, dark);
    let inside = select(inside_light, inside_dark, dark);
    return vec4<f32>(select(inside, outside, front), 1.0);
}
""", dict(outside_light=outside_light, outside_dark=outside_dark, inside_light=inside_light, inside_dark=inside_dark))


def euclidean(points) -> np.ndarray:
    """x, y and z of points, antivectors, on the last axis."""
    return points.cast(ga.subspace("yz zx xy")).kernel


def projective(vertices: Point) -> solids.Point:
    """A mesh's vertices as points of projective space, for drawing."""
    return solids.mv.point(euclidean(vertices))


def square_view(vertices: Point, side: int) -> solids.View:
    """A square view framing a mesh's vertices, `side` pixels across."""
    view = solids.view([projective(vertices)], 0.0, elevation, azimuth, margin, side)
    return view.squared()


def bare(figure: plt.Figure, axes: np.ndarray) -> None:
    """The axes without their own axes, and the figure laid out tightly around them, before the pictures in them
    are fitted to the room they get."""
    for axis in axes.ravel():
        axis.set_axis_off()
    figure.tight_layout()


def face_colours(values: Scalar, limit: float) -> np.ndarray:
    """A diverging colour per face for values in [-limit, limit]."""
    return plt.get_cmap("coolwarm")((values.to_array().ravel() / limit + 1) / 2)[:, :3]


def coloured(mesh: Mesh, vertices: Point, values: Scalar, limit: float, view: solids.View) -> np.ndarray:
    """The mesh at its vertices, each face flat in its colour by a field in [-limit, limit], as the view sees it."""
    points = projective(vertices)
    return meshes.draw(points, mesh.faces, np.zeros(3), meshes.flat(points, mesh.faces), view, background,
                       face_values=face_colours(values, limit), colour=meshes.by_face, gloss=solids.gloss)


def render_surface(mesh: Mesh, vertices: Point, reference: Point, view: solids.View) -> np.ndarray:
    """The surface at its vertices as an RGB image, as the view sees it, smoothly lit on both sides, its texture a
    solid checkerboard of cubes sampled on the reference vertices of the same mesh: straight squares on its flat
    faces, circles on a sphere."""
    points = projective(vertices)
    original = euclidean(reference)
    cells = original / np.abs(original).max() * squares / 2
    return meshes.draw(points, mesh.faces, cells[mesh.faces], meshes.smooth(points, mesh.faces), view, background,
                       face_values=np.zeros(4), colour=checkerboard, gloss=solids.gloss)


def draw_surface(mesh: Mesh, vertices: Point, values: Scalar) -> plt.Figure:
    """The mesh at its vertices, its faces coloured by a field, over a range that shows all of it."""
    limit = np.abs(values.to_array()).max()
    figure, axis = plt.subplots(figsize=(5, 5))
    bare(figure, np.array(axis))
    view = fitted(square_view(vertices, pixels), axis)
    show(axis, coloured(mesh, vertices, values, limit, view), view)
    return figure


def draw_dipole(mesh: Mesh, vertices: Point, curvature: Scalar, deformed: Point) -> plt.Figure:
    """The sphere coloured by the curvature change it is given, and the surface that has it, at the deformed
    vertices, textured with the sphere's checkerboard."""
    limit = np.abs(curvature.to_array()).max()
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    bare(figure, axes)
    sphere_view = fitted(square_view(vertices, pixels), axes[0])
    deformed_view = fitted(square_view(deformed, pixels), axes[1])
    show(axes[0], coloured(mesh, vertices, curvature, limit, sphere_view), sphere_view)
    show(axes[1], render_surface(mesh, deformed, vertices, deformed_view), deformed_view)
    return figure


def draw_textured(mesh: Mesh, vertices: Point, reference: Point) -> plt.Figure:
    """The surface at its vertices rendered with the reference vertices' checkerboard."""
    figure, axis = plt.subplots(figsize=(3, 3))
    bare(figure, np.array(axis))
    view = fitted(square_view(vertices, pixels), axis)
    show(axis, render_surface(mesh, vertices, reference, view), view)
    return figure


def draw_dirac(mesh: Mesh, gallery: dict[int, Point], reference: Point) -> plt.Figure:
    """A row of Dirac spheres for each eigenvalue, the vertices of each `[spheres] Point[V]` on the sphere's mesh,
    textured with the sphere's checkerboard."""
    rows, columns = len(gallery), max(len(spheres) for spheres in gallery.values())
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for row, eigenvalue in enumerate(gallery):
        axes[row, 0].set_title(f"n = {eigenvalue}", loc="left")
    bare(figure, axes)
    for row, spheres in enumerate(gallery.values()):
        for column, sphere in enumerate(spheres):
            view = fitted(square_view(sphere, pixels), axes[row, column])
            show(axes[row, column], render_surface(mesh, sphere, reference, view), view)
    return figure


def animate_rounding(mesh: Mesh, frames: Iterable[Point]) -> Iterator[np.ndarray]:
    """Each step of the flow, the mesh's vertices at it, textured with the checkerboard of the first."""
    frames = iter(frames)
    first = next(frames)
    for frame in chain([first], frames):
        yield render_surface(mesh, frame, first, square_view(frame, pixels))
