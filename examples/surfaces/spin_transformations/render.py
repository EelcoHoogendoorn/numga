"""Drawing for spin transformations, on the GPU: meshes coloured by a field on their faces, and surfaces lit
on both sides and textured with a checkerboard carried over from the sphere they came from."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.spin_transformations.core import Mesh, Scalar, ga
from examples.wgsl import meshes, solids

# The camera's elevation and azimuth, its margin as a share of the larger side, and the image's side in pixels,
# alone and as a tile of a gallery.
elevation, azimuth = np.radians(20.0), np.radians(35.0)
margin = 0.06
pixels = 300
tile = 200
background = np.ones(3)
# The checkerboard's cubes across the reference surface, and its light and dark squares on the outer and inner
# side of the surface.
squares = 8
outside = np.array([[0.98, 0.84, 0.62], [0.90, 0.45, 0.18]])
inside = np.array([[0.70, 0.82, 0.96], [0.22, 0.42, 0.72]])

# A solid checkerboard of cubes, in the place each pixel had on the reference surface, scaled so the cubes
# are unit; light and dark squares, warm outside and cool inside.
def wgsl_colour(colour: np.ndarray) -> str:
    return f"vec3<f32>({', '.join(f'{value:.6f}' for value in colour)})"


checkerboard = f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32> {{
    let cells = round(values.xyz);
    let dark = abs(cells.x + cells.y + cells.z) % 2.0 > 0.5;
    let outside = select({wgsl_colour(outside[0])}, {wgsl_colour(outside[1])}, dark);
    let inside = select({wgsl_colour(inside[0])}, {wgsl_colour(inside[1])}, dark);
    return select(inside, outside, front);
}}
"""


def euclidean(points) -> np.ndarray:
    """x, y and z of points, antivectors, on the last axis."""
    return points.cast(ga.subspace("yz zx xy")).kernel


def projective(mesh: Mesh) -> solids.Point:
    """The mesh's vertices as points of projective space, for drawing."""
    return solids.mv.point(euclidean(mesh.vertices))


def square_view(points: solids.Point) -> tuple[solids.Motor, np.ndarray]:
    camera, half = solids.view([points], 0.0, elevation, azimuth, margin)
    return camera, np.full(2, half.max())


def face_colours(values: Scalar, limit: float) -> np.ndarray:
    """A diverging colour per face for values in [-limit, limit]."""
    return plt.get_cmap("coolwarm")((values.to_array().ravel() / limit + 1) / 2)[:, :3]


def coloured(mesh: Mesh, values: Scalar, limit: float, side: int) -> np.ndarray:
    """The mesh, each face flat in its colour by a field in [-limit, limit], `[side, side, 3]`."""
    points = projective(mesh)
    return meshes.draw(points, mesh.faces, np.zeros(3), meshes.flat(points, mesh.faces), *square_view(points),
                       elevation, azimuth, side, background, face_values=face_colours(values, limit), colour=meshes.by_face)


def render_surface(mesh: Mesh, reference: Mesh, side: int) -> np.ndarray:
    """The surface as an RGB image, `[side, side, 3]`, smoothly lit on both sides, its texture a solid
    checkerboard of cubes sampled on the reference mesh with the same faces: straight squares on its flat
    faces, circles on a sphere."""
    points = projective(mesh)
    original = euclidean(reference.vertices)
    cells = original / np.abs(original).max() * squares / 2
    return meshes.draw(points, mesh.faces, cells[mesh.faces], meshes.smooth(points, mesh.faces), *square_view(points),
                       elevation, azimuth, side, background, colour=checkerboard)


def shown(axis: plt.Axes, image: np.ndarray) -> None:
    axis.imshow(image)
    axis.set_axis_off()


def draw_surface(mesh: Mesh, values: Scalar, limit: float) -> plt.Figure:
    """The mesh, its faces coloured by a field in [-limit, limit]."""
    figure, axis = plt.subplots(figsize=(5, 5))
    shown(axis, coloured(mesh, values, limit, pixels))
    figure.tight_layout()
    return figure


def draw_dipole(mesh: Mesh, curvature: Scalar, deformed: Mesh) -> plt.Figure:
    """The sphere coloured by the curvature change it is given, and the surface that has it,
    textured with the sphere's checkerboard."""
    limit = np.abs(curvature.to_array()).max()
    figure, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    shown(axes[0], coloured(mesh, curvature, limit, pixels))
    shown(axes[1], render_surface(deformed, mesh, pixels))
    figure.tight_layout()
    return figure


def draw_textured(mesh: Mesh, reference: Mesh) -> plt.Figure:
    """The surface rendered with the reference's checkerboard."""
    figure, axis = plt.subplots(figsize=(3, 3))
    shown(axis, render_surface(mesh, reference, pixels))
    figure.tight_layout()
    return figure


def draw_dirac(gallery: dict[int, list[Mesh]], reference: Mesh) -> plt.Figure:
    """A row of Dirac spheres for each eigenvalue, textured with the sphere's checkerboard."""
    rows, columns = len(gallery), max(len(spheres) for spheres in gallery.values())
    figure, axes = plt.subplots(rows, columns, figsize=(2 * columns, 2 * rows), squeeze=False)
    for row, (eigenvalue, spheres) in enumerate(gallery.items()):
        for column, sphere in enumerate(spheres):
            axes[row, column].imshow(render_surface(sphere, reference, tile))
        axes[row, 0].set_title(f"n = {eigenvalue}", loc="left")
        for axis in axes[row]:
            axis.set_axis_off()
    figure.tight_layout()
    return figure


def animate_rounding(frames: Iterable[Mesh]) -> list[np.ndarray]:
    """Each step of the flow, textured with the checkerboard of the first."""
    frames = list(frames)
    return [render_surface(frame, frames[0], pixels) for frame in frames]
