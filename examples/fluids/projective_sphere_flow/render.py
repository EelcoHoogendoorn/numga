"""Readout of weighted representatives, their dual, and diffusion on a fixed sphere or affine plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, stack
from examples import instantiate
from numga.algebras import VGA3D
from examples.util.wgsl import image_figure, meshes, solids, strokes

core = instantiate("examples.fluids.projective_sphere_flow.core", VGA3D)

n_pixels = 300
n_arc_segments = 16
background = np.ones(3)
blue = np.array([0.16, 0.43, 0.62])
orange = np.array([0.80, 0.34, 0.15])
cream = np.array([0.94, 0.93, 0.89])
elevation, azimuth = np.radians(22), np.radians(55)

field_shader = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let value = clamp(values.x, -1.0, 1.0);
    let tone = select(blue, orange, value > 0.0);
    let base = mix(cream, tone, 0.85 * abs(value));
    let contour = smoothstep(0.04, 0.0, abs(fract(values.x * 4.0 + 0.5) - 0.5));
    return vec4<f32>(mix(base, vec3<f32>(0.2), 0.4 * contour), 1.0);
}
""", dict(blue=blue, orange=orange, cream=cream))


# --- plumbing -------------------------------------------------------------------------
def read_points(points: Extensor, planes: Extensor) -> np.ndarray:
    """The coordinates of any algebra's points, read against the chosen display planes."""
    return (planes & points.batch()[..., None]).to_array()


def draw_mesh(mesh: core.Mesh, vertices: core.Point) -> plt.Figure:
    """The physical spherical mesh before any choice of lifted representatives."""
    mv = vertices.context.multivector
    points = solids.mv.point(read_points(vertices, stack((mv.x, mv.y, mv.z))))
    frame = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1.15), n_pixels)
    colours = np.broadcast_to(cream, mesh.faces.shape + (3,))
    surface = meshes.samples(points, mesh.faces, colours, meshes.smooth(points, mesh.faces), frame,
                             face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    lines = strokes.segments(points[mesh.edges[:, 0]], points[mesh.edges[:, 1]], blue, 1.0, frame)
    return image_figure(solids.draw([], frame, background, inks=[], translucent=[],
                                    meshes=[(surface, 1.0), (lines, 1.0)]))


def animate_sphere(mesh: core.Mesh, display_faces: np.ndarray,
                   frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Scalar]]) -> Iterator[np.ndarray]:
    """The moving representatives and dual beside the physical solution on its fixed spherical domain."""
    space_view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1.85), n_pixels)
    sphere_view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1.15), n_pixels)
    for lifted, duals, samples, values in frames:
        mv = lifted.context.multivector
        planes = stack((mv.x, mv.y, mv.z))
        xyz = read_points(lifted, planes)
        poles = read_points(duals, planes)
        physical = read_points(samples, planes).transpose(1, 0, 2)
        n_samples = physical.shape[1]
        faces = (display_faces[None] + n_samples * np.arange(len(mesh.faces))[:, None, None]).reshape(-1, 3)
        points = solids.mv.point(physical.reshape(-1, 3))
        field = values.to_array().T.reshape(-1)
        normals = meshes.smooth(points, faces)
        surface = meshes.samples(points, faces, field[faces, None], normals, sphere_view,
                                 face_values=np.zeros(4), colour=field_shader, gloss=solids.gloss)
        sphere = solids.draw([], sphere_view, background, inks=[], translucent=[], meshes=[(surface, 1.0)])

        vertices = solids.mv.point(xyz)
        mesh_lines = strokes.segments(vertices[mesh.edges[:, 0]], vertices[mesh.edges[:, 1]], blue, 1.25, space_view)
        # Great-circle arcs draw the projected dual on the same unit sphere.
        fractions = np.linspace(0, 1, n_arc_segments + 1)[None, :, None]
        ends = poles[mesh.edge_faces]
        chords = (1 - fractions) * ends[:, 0, None] + fractions * ends[:, 1, None]
        arcs = chords / np.linalg.norm(chords, axis=-1, keepdims=True)
        dual_lines = strokes.segments(solids.mv.point(arcs[:, :-1].reshape(-1, 3)),
                                     solids.mv.point(arcs[:, 1:].reshape(-1, 3)), orange, 1.1, space_view)
        directions = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
        rays = strokes.segments(solids.mv.point(directions), vertices, np.full(3, 0.45), 1.0, space_view)
        shell = solids.ball(solids.origin, 1.0)[None, None]
        markers = solids.ball(vertices, 2.0 * space_view.pixel)[:, None]
        space = solids.draw([(markers, blue)], space_view, background, inks=[],
                            translucent=[(shell, cream, 0.35)],
                            meshes=[(mesh_lines, 1.0), (dual_lines, 1.0), (rays, 1.0)])
        yield np.concatenate((space, sphere), axis=1)


def draw_planar(points: Extensor, faces: np.ndarray, potential: Extensor, current: Extensor) -> plt.Figure:
    """Finite points and spatial current in x,y,w, with no readout of the null coordinate as distance."""
    mv = points.context.multivector
    planes = stack((mv.x, mv.y))
    xy = read_points(points.normalized(), planes)
    flow = read_points(current, planes)
    figure, axis = plt.subplots(figsize=(3, 3), dpi=100)
    axis.tripcolor(xy[:, 0], xy[:, 1], faces, potential.to_array(), shading="gouraud", cmap="viridis", vmin=0, vmax=1)
    chosen = np.arange(0, len(xy), 4)
    axis.quiver(xy[chosen, 0], xy[chosen, 1], flow[chosen, 0], flow[chosen, 1], color="white", width=0.005)
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout(pad=0.2)
    return figure
