"""Readout of weighted representatives, their dual, and diffusion on a fixed sphere or affine plane."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import Extensor, stack
from examples import instantiate
from examples.util import charts
from numga.algebras import VGA3D
from examples.util.wgsl import contours, flow, image_figure, meshes, solids, strokes

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
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())             # [vertices] Point-
    frame = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1.15), n_pixels)
    colours = np.broadcast_to(cream, mesh.faces.shape + (3,))
    surface = meshes.samples(points, mesh.faces, colours, meshes.smooth(points, mesh), frame,
                             face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)
    tails, heads = mesh.incidence[1, 0] * vertices                           # Point-[edges], Point-[edges]
    lines = strokes.segments(solids.origin + solids.context.include_from_algebra(tails.batch()),
                             solids.origin + solids.context.include_from_algebra(heads.batch()), blue, 1.0, frame)
    return image_figure(solids.draw([], frame, background, inks=[], translucent=[],
                                    meshes=[(surface, 1.0), (lines, 1.0)]))


def animate_sphere(mesh: core.Mesh, display: core.Mesh,
                   frames: Iterable[tuple[core.Point, core.Point, core.Point, core.Scalar]]) -> Iterator[np.ndarray]:
    """The moving representatives and dual beside the physical solution on its fixed spherical domain."""
    space_view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1.85), n_pixels)
    sphere_view = solids.View(solids.aim(solids.origin, elevation, azimuth), np.full(2, 1.15), n_pixels)
    # Points along each great-circle arc, as shares of its chord.
    fractions = np.linspace(0, 1, n_arc_segments + 1)
    for lifted, duals, samples, values in frames:
        # The display samples of every face, sample after sample, and the display triangles of each face among them.
        n_faces = len(mesh.faces)
        faces = (n_faces * display.faces[None] + np.arange(n_faces)[:, None, None]).reshape(-1, 3)
        points = solids.origin + solids.context.include_from_algebra(samples.batch().reshape(-1))   # [samples] Point-
        field = values.to_array().reshape(-1)
        normals = meshes.smooth(points, meshes.Mesh(len(points), faces))
        surface = meshes.samples(points, faces, field[faces, None], normals, sphere_view,
                                 face_values=np.zeros(4), colour=field_shader, gloss=solids.gloss)
        sphere = solids.draw([], sphere_view, background, inks=[], translucent=[], meshes=[(surface, 1.0)])

        vertices = solids.origin + solids.context.include_from_algebra(lifted.batch())         # [vertices] Point-
        tails, heads = mesh.incidence[1, 0] * lifted                         # Point-[edges], Point-[edges]
        mesh_lines = strokes.segments(solids.origin + solids.context.include_from_algebra(tails.batch()),
                                      solids.origin + solids.context.include_from_algebra(heads.batch()), blue, 1.25, space_view)
        # Great-circle arcs draw the projected dual on the same unit sphere: the chord between the poles of the faces
        # either side of each edge, pushed out onto it.
        left, right = (mesh.reading(mesh.edge_faces.T, 2) * duals).batch()   # [edges] Point-, [edges] Point-
        arcs = (left[:, None] * (1 - fractions) + right[:, None] * fractions).normalized()   # [edges, arc points] Point-
        arc_places = solids.origin + solids.context.include_from_algebra(arcs)                # [edges, arc points] Point-
        dual_lines = strokes.segments(arc_places[:, :-1].reshape(-1), arc_places[:, 1:].reshape(-1), orange, 1.1,
                                      space_view)
        on_sphere = solids.origin + solids.context.include_from_algebra(lifted.normalized().batch())   # [vertices] Point-
        rays = strokes.segments(on_sphere, vertices, np.full(3, 0.45), 1.0, space_view)
        shell = solids.ball(solids.origin, 1.0)[None, None]
        markers = solids.ball(vertices, 2.0 * space_view.pixel)[:, None]
        space = solids.draw([(markers, blue)], space_view, background, inks=[],
                            translucent=[(shell, cream, 0.35)],
                            meshes=[(mesh_lines, 1.0), (dual_lines, 1.0), (rays, 1.0)])
        yield np.concatenate((space, sphere), axis=1)


def draw_planar(points: Extensor, display: core.Mesh, potential: Extensor, current: Extensor) -> plt.Figure:
    """Finite points and spatial current in x,y,w, with no readout of the null coordinate as distance: the potential
    in its colours, the current as white streaks running downstream."""
    mv = points.context.multivector
    xy = read_points(points.normalized(), stack((mv.x, mv.y)))
    figure, axis = charts.subplots(300, 300)
    # The panel framed about the samples as matplotlib frames its data.
    axis.update_datalim(xy)
    axis.autoscale_view()
    axis.set_aspect("equal")
    axis.set_axis_off()
    figure.tight_layout(pad=0.2)
    # Finite points of the affine plane are the renderer's points, as weighted as they come, and the current's
    # directions its directions.
    places = solids.context.include_from_algebra(points)                     # [samples] Point-
    directions = solids.context.include_from_algebra(current)                # [samples] Point-
    values = potential.to_array()
    contours.overlay(axis, places, display, values, plt.get_cmap("viridis")(values)[:, :3]).set_zorder(1)
    # The streaks' lengths measured against the strongest current.
    fastest = float(current.dual().norm().max().to_array())
    flow.overlay(axis, places, display, directions, fastest, np.ones(3))
    return figure
