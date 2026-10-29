"""A planetary cross-section shown twice, its rate of dilation beside its vorticity, each on a fixed scale."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.whitney_elasticity import core
from examples.util.wgsl import image_figure, meshes, solids, strokes

n_pixels = 300
n_markers = 12
n_trail_frames = 8
panel_gap = 10
display_thickness = 0.12
surface_height = display_thickness / 2
view_elevation, view_azimuth = np.radians(68), np.radians(10)
background = np.ones(3)
outline_colour = np.array([0.22, 0.29, 0.35])
wall_colour = np.array([0.64, 0.70, 0.74])
moon_colour = np.array([0.58, 0.58, 0.62])
# Compression and dilation on the left; counter-clockwise and clockwise turning on the right.
compression_colour, dilation_colour = [0.80, 0.10, 0.08], [0.06, 0.30, 0.80]
counter_clockwise_colour, clockwise_colour = [0.48, 0.12, 0.62], [0.05, 0.50, 0.22]

material_surface = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let paper = select(vec3<f32>(0.94, 0.90, 0.81), vec3<f32>(0.72, 0.85, 0.91), face_values.x < 0.5);
    let edge = min(corner.x, min(corner.y, corner.z));
    let triangulation = 1.0 - smoothstep(0.0, fwidth(edge) * 0.45, edge);
    return vec4<f32>(paper * (1.0 - 0.16 * triangulation), 1.0);
}
""", {})


def wave_surface(positive: list[float], negative: list[float]) -> meshes.Colour:
    """A signed reading on the paper of its layer: white where it vanishes, deepening to one hue or the other."""
    return meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let reading = clamp(values.x, -1.0, 1.0);
    let paper = select(vec3<f32>(0.97, 0.96, 0.93), vec3<f32>(0.88, 0.90, 0.92), face_values.x < 0.5);
    let hue = select(negative, positive, reading > 0.0);
    return vec4<f32>(mix(paper, hue, abs(reading)), 1.0);
}
""", dict(positive=np.array(positive), negative=np.array(negative)))


dilation_surface = wave_surface(dilation_colour, compression_colour)
rotation_surface = wave_surface(counter_clockwise_colour, clockwise_colour)

# --- plumbing -------------------------------------------------------------------------
# Up from the section's plane, in space.
up = solids.mv.z.dual()                                                      # [] Point-


def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the section's, in the plane z = 0 of space: their coordinates,
    read by pairing with the planes x and y, are the one hand-off from this algebra to the renderer's."""
    across = (stack((core.mv.x, core.mv.y)) & points.batch()[..., None]).to_array()   # [..., 2]
    return solids.mv.point(np.concatenate((across, np.zeros(across.shape[:-1] + (1,))), axis=-1))


def panel_image(images: list[np.ndarray]) -> np.ndarray:
    """Views share equally sized viewports and narrow white gutters."""
    gap = np.full((n_pixels, panel_gap, 3), 255, dtype=np.uint8)
    return np.concatenate([piece for i, image in enumerate(images)
                           for piece in ([gap, image] if i else [image])], axis=1)


class DiskDrawing:
    """Fixed topology and framing for an extruded drawing of a two-dimensional solid."""

    def __init__(self, mesh: core.Mesh, vertices: core.Point, regions: np.ndarray):
        self.mesh = mesh
        self.regions = regions
        self.reference = vertices
        radius = vertices.batch().norm().to_array().max()
        self.view = solids.View(solids.aim(solids.origin, view_elevation, view_azimuth),
                                np.full(2, 1.12 * radius), n_pixels)
        # The outline: the free surface, and the edges between a liquid and a solid triangle around the core.
        sides = regions[mesh.edge_faces]                                          # [interior edges, 2]
        interface = mesh.interior_edges[sides[:, 0] != sides[:, 1]]
        self.rim = mesh.edges[np.concatenate((mesh.boundary_edges, interface))]
        # The markers: the vertices of the free surface nearest points turned evenly around it, from the top.
        surface_vertices = mesh.exterior(0)
        angles = np.linspace(0, 2 * np.pi, n_markers, endpoint=False) + np.pi / 2
        targets = (core.mv.xy * (-angles / 2)).exp() >> (core.mv.x.dual() * radius)   # [markers] Point-
        surface_points = vertices.batch()[surface_vertices]                       # [surface] Point-
        nearest = (surface_points[:, None] - targets[None]).norm_squared().argmin(axis=0)
        self.markers = surface_vertices[nearest]
        self.corner_mean = mesh.incidence[2, 0].mean(axis=0)                      # Scalar+[faces <- vertices]
        self.vertex_shares = ~self.corner_mean * core.as_scalar(np.ones(len(mesh.faces))).field()   # Scalar+[vertices]
        # The motion stays in the plane, so the top keeps the normals of its reference.
        self.top_normals = meshes.flat(gpu_points(vertices) + up * surface_height, mesh.faces)

    def draw(self, places: core.Point, panels: list[tuple[np.ndarray, meshes.Colour]],
             motion: list[meshes.Samples], markers: list[tuple[solids.Quadric, np.ndarray]]) -> list[np.ndarray]:
        """One image per panel of values and colouring on the same deformed section: the top carries the field
        readings; the shallow side wall only gives the section a visible edge."""
        points = gpu_points(places) + up * surface_height                         # [vertices] Point-
        top_start, top_end = points[self.rim[:, 0]], points[self.rim[:, 1]]       # [rim] Point-, Point-
        bottom_start, bottom_end = top_start - up * display_thickness, top_end - up * display_thickness
        wall_points = stack((top_start, top_end, bottom_start, bottom_end), axis=1).reshape(-1)   # [rim * 4] Point-
        wall_faces = (np.array([[0, 2, 1], [1, 2, 3]])[None]
                      + 4 * np.arange(len(self.rim))[:, None, None]).reshape(-1, 3)
        wall = meshes.samples(wall_points, wall_faces, wall_colour,
                              meshes.smooth(wall_points, wall_faces), self.view,
                              face_values=np.zeros(1), colour=meshes.plain, gloss=0.06)
        rim = strokes.segments(top_start, top_end, outline_colour, 1.0, self.view)
        tops = [meshes.samples(points, self.mesh.faces, values[self.mesh.faces], self.top_normals, self.view,
                               face_values=self.regions[:, None], colour=surface, gloss=0.0)
                for values, surface in panels]
        return [solids.draw(markers, self.view, background, inks=[], translucent=[],
                            meshes=[(wall, 1.0), (top, 1.0), (rim, 1.0)] + [(trail, 1.0) for trail in motion])
                for top in tops]

    def vertex_values(self, readings: core.Scalar) -> np.ndarray:
        """Face readings `Scalar+[faces]` averaged at each vertex, only for smooth colour interpolation in the drawing."""
        return ((~self.corner_mean * readings) / self.vertex_shares).batch().to_array()[:, None]


def draw_materials(mesh: core.Mesh, vertices: core.Point, regions: np.ndarray) -> plt.Figure:
    """The mesh, the liquid core and the solid shell around it, before anything moves."""
    drawing = DiskDrawing(mesh, vertices, regions)
    image, = drawing.draw(vertices, [(np.zeros((mesh.n_vertices, 1)), material_surface)], [], [])
    return image_figure(image)


def animate_waves(mesh: core.Mesh, vertices: core.Point, regions: np.ndarray,
                  frames: Iterable[tuple[core.Point, core.Scalar, core.Scalar]], moon: core.Point, moon_radius: float,
                  displacement_scale: float, dilation_scale: float, vorticity_scale: float) -> Iterator[np.ndarray]:
    """The rate of dilation on the left and the vorticity on the right, signed, each at its own fixed scale, with the
    moon `[frames]` that strikes the planet."""
    drawing = DiskDrawing(mesh, vertices, regions)
    history = deque(maxlen=n_trail_frames)
    for (displacement, dilation_rate, vorticity), moon_place in zip(frames, moon):
        places = vertices + displacement_scale * displacement                      # Point-[vertices]
        history.append(places.batch()[drawing.markers])
        # The beads follow material points of the free surface; their short tails trace its motion.
        trajectory = gpu_points(stack(list(history))) + up * (surface_height + 0.009)   # [trail, markers] Point-
        age = np.linspace(0.72, 0.12, len(history) - 1)
        trail_colours = np.broadcast_to(age[:, None, None], (len(age), n_markers, 3))
        tails = strokes.segments(trajectory[:-1].reshape(-1), trajectory[1:].reshape(-1),
                                 trail_colours.reshape(-1, 3), 1.0, drawing.view)
        beads = solids.ball(trajectory[-1], 0.012)[:, None]
        moon_ball = solids.ball(gpu_points(moon_place[None]), moon_radius)[:, None]
        panels = [(drawing.vertex_values(dilation_rate / dilation_scale), dilation_surface),
                  (drawing.vertex_values(vorticity / vorticity_scale), rotation_surface)]
        yield panel_image(drawing.draw(places, panels, [tails], [(beads, outline_colour), (moon_ball, moon_colour)]))



def animate_modes(mesh: core.Mesh, vertices: core.Point, regions: np.ndarray,
                  frames: Iterable[tuple[core.Point, core.Scalar, core.Scalar]],
                  displacement_scale: float) -> Iterator[np.ndarray]:
    """The ball moved by each wave in turn, its dilation on the left and its rotation on the right, both on the scale
    of the larger of the two in the frame, so that a wave of either kind shows at full strength and in its proportion."""
    drawing = DiskDrawing(mesh, vertices, regions)
    for displacement, dilation, rotation in frames:
        places = vertices + displacement_scale * displacement                      # Point-[vertices]
        dilations, rotations = dilation.batch().to_array(), rotation.batch().to_array()   # [faces], [faces]
        scale = max(np.abs(dilations).max(), np.abs(rotations).max())
        panels = [(drawing.vertex_values(dilation / scale), dilation_surface),
                  (drawing.vertex_values(rotation / scale), rotation_surface)]
        yield panel_image(drawing.draw(places, panels, [], []))
