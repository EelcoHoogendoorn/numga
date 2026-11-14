"""Dual-panel extruded cross-section rendering dilation rate beside vorticity on fixed or per-frame scales."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.mechanics.seismic_waves import core
from numga.topology import Topology
from numga.render.drawing import Draw
from numga.render import meshes, simplices, solids
from examples.util.panels import image_figure

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
compression_colour, dilation_colour = [0.80, 0.10, 0.08], [0.06, 0.30, 0.80]
counter_clockwise_colour, clockwise_colour = [0.48, 0.12, 0.62], [0.05, 0.50, 0.22]

material_surface = meshes.Colour('''
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let paper = select(vec3<f32>(0.94, 0.90, 0.81), vec3<f32>(0.72, 0.85, 0.91), face_values.x < 0.5);
    let edge = min(corner.x, min(corner.y, corner.z));
    let triangulation = 1.0 - smoothstep(0.0, fwidth(edge) * 0.45, edge);
    return vec4<f32>(paper * (1.0 - 0.16 * triangulation), 1.0);
}
''', {})


def wave_surface(positive: list[float], negative: list[float]) -> meshes.Colour:
    """Signed face readings blend from each layer's neutral paper tint toward `positive` or `negative`."""
    return meshes.Colour('''
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let reading = clamp(values.x, -1.0, 1.0);
    let paper = select(vec3<f32>(0.97, 0.96, 0.93), vec3<f32>(0.88, 0.90, 0.92), face_values.x < 0.5);
    let hue = select(negative, positive, reading > 0.0);
    return vec4<f32>(mix(paper, hue, abs(reading)), 1.0);
}
''', dict(positive=np.array(positive), negative=np.array(negative)))


dilation_surface = wave_surface(dilation_colour, compression_colour)
rotation_surface = wave_surface(counter_clockwise_colour, clockwise_colour)

# --- plumbing -------------------------------------------------------------------------
vertical = solids.mv.z.dual()                                                 # [] Point-
top_middle = solids.origin + vertical * surface_height                        # [] Point-


def panel_image(images: list[np.ndarray]) -> np.ndarray:
    """Side-by-side panels share equal viewports separated by a narrow white gutter."""
    gap = np.full((n_pixels, panel_gap, 3), 255, dtype=np.uint8)
    return np.concatenate([piece for i, image in enumerate(images)
                           for piece in ([gap, image] if i else [image])], axis=1)


class DiskDrawing:
    """Fixed topology, boundary outline, and camera framing for an extruded planetary cross-section."""

    def __init__(self, topology: Topology, vertices: core.Point, regions: np.ndarray):
        V, E, F = topology.n_elements
        self.topology = topology
        self.regions = regions
        radius = float(vertices.antinorm().batch().max().to_array())
        self.view = solids.View(solids.aim(solids.origin, view_elevation, view_azimuth),
                                np.full(2, 1.12 * radius), n_pixels)
        # Region jumps along dual edges pick out the interior core-mantle boundary alongside the free rim.
        region_field = core.mv.scalar(regions[:, None]).field()               # Scalar+[F]
        jumps = (topology.dual.d[0] * region_field).batch().to_array()        # [E]
        interface = np.intersect1d(topology.interior[1], np.flatnonzero(jumps))
        self.outline = topology.subset(np.concatenate((topology.exterior[1], interface)), 1)
        surface_vertices = topology.exterior[0]
        angles = np.linspace(0, 2 * np.pi, n_markers, endpoint=False) + np.pi / 2
        marker_rotors = (core.mv.xy * (-angles / 2)).exp()                    # [n_markers] Rotor+
        targets = marker_rotors >> (core.mv.x.dual() * radius)                # [n_markers] Point-
        surface_points = (topology.reading(surface_vertices, 0) * vertices).batch()  # [...] Point-
        nearest = (surface_points[:, None] - targets[None]).scalar_norm_squared().argmin(axis=0)
        self.markers = surface_vertices[nearest]
        self.corner_mean = topology.incidence[2, 0].mean(axis=0)              # Scalar+[F <- V]
        unit_faces = core.mv.scalar(np.ones((F, 1))).field()                  # Scalar+[F]
        self.vertex_shares = self.corner_mean.adjugate() * unit_faces         # Scalar+[V]
        top_vertices = top_middle + solids.context.include_from_algebra(vertices.batch())  # [V] Point-
        self.top_normals = meshes.flat(top_vertices, topology.elements[2])

    def draw(self, places: core.Point, panels: list[tuple[np.ndarray, meshes.Colour]],
             motion: list[Draw], markers: list[tuple[solids.Quadric, np.ndarray]]) -> list[np.ndarray]:
        """Renders one image per `(values, surface)` panel on the deformed cross-section `places`."""
        V_outline, E_outline = self.outline.n_elements
        points_field = top_middle + solids.context.include_from_algebra(places)  # Point-[V]
        points = points_field.batch()                                         # [V] Point-
        top_start, top_end = (self.outline.incidence[1, 0] * points_field).batch()  # [E_outline] Point-, Point-
        bottom_start = top_start - vertical * display_thickness               # [E_outline] Point-
        bottom_end = top_end - vertical * display_thickness                   # [E_outline] Point-
        wall_points = stack((top_start, top_end, bottom_start, bottom_end), axis=1).reshape(-1)  # [4 * E_outline] Point-
        wall_faces = (np.array([[0, 2, 1], [1, 2, 3]])[None]
                      + 4 * np.arange(E_outline)[:, None, None]).reshape(-1, 3)
        wall = meshes.samples(wall_points, wall_faces, wall_colour,
                              meshes.smooth(wall_points, Topology(len(wall_points), wall_faces)), self.view,
                              face_values=np.zeros(1), colour=meshes.plain, gloss=0.06)
        rim = simplices.tubes(points, self.outline, outline_colour, 1.0, self.view, gloss=solids.gloss)
        tops = [meshes.samples(points, self.topology.elements[2], values[self.topology.elements[2]], self.top_normals,
                               self.view, face_values=self.regions[:, None], colour=surface, gloss=0.0)
                for values, surface in panels]
        return [solids.draw(markers, self.view, background, inks=[], translucent=[],
                            meshes=[(wall, 1.0), (top, 1.0), (rim, 1.0)] + [(trail, 1.0) for trail in motion])
                for top in tops]

    def vertex_values(self, readings: core.Scalar) -> np.ndarray:
        """Averages face `readings` (`Scalar+[F]`) onto vertices for smooth colour interpolation."""
        return ((self.corner_mean.adjugate() * readings) / self.vertex_shares).batch().to_array()[:, None]


def materials_figure(topology: Topology, vertices: core.Point, regions: np.ndarray) -> plt.Figure:
    """Renders the triangulated planetary cross-section with its liquid core and solid shell at rest."""
    V, E, F = topology.n_elements
    drawing = DiskDrawing(topology, vertices, regions)
    image, = drawing.draw(vertices, [(np.zeros((V, 1)), material_surface)], [], [])
    return image_figure(image)


def waves_animation(topology: Topology, vertices: core.Point, regions: np.ndarray,
                    moon: core.Point, moon_radius: float, displacement_scale: float,
                    dilation_scale: float, vorticity_scale: float,
                    frames: Iterable[tuple[core.Point, core.Scalar, core.Scalar]]) -> Iterator[np.ndarray]:
    """Yields dual-panel RGB frames pairing dilation rate and vorticity as `moon` strikes the planet."""
    drawing = DiskDrawing(topology, vertices, regions)
    history = deque(maxlen=n_trail_frames)
    for (displacement, dilation_rate, vorticity), moon_place in zip(frames, moon):
        places = vertices + displacement_scale * displacement                 # Point-[V]
        history.append((topology.reading(drawing.markers, 0) * places).batch())
        n_history = len(history)
        trail_places = solids.context.include_from_algebra(stack(list(history), axis=1))  # [n_markers, n_history] Point-
        trajectory = top_middle + vertical * 0.009 + trail_places             # [n_markers, n_history] Point-
        motions = []
        if n_history > 1:
            tail_chains = Topology.chains([n_history] * n_markers)
            age = np.tile(np.linspace(0.72, 0.12, n_history - 1), n_markers)  # [n_markers * (n_history - 1)]
            tails = simplices.tubes(trajectory.reshape(-1), tail_chains, np.repeat(age[:, None], 3, axis=-1), 1.0,
                                    drawing.view, gloss=solids.gloss)
            motions.append(tails)
        beads = solids.ball(trajectory[:, -1], 0.012)[:, None]
        moon_ball = solids.ball(solids.origin + solids.context.include_from_algebra(moon_place[None]), moon_radius)[:, None]
        panels = [(drawing.vertex_values(dilation_rate / dilation_scale), dilation_surface),
                  (drawing.vertex_values(vorticity / vorticity_scale), rotation_surface)]
        yield panel_image(drawing.draw(places, panels, motions, [(beads, outline_colour), (moon_ball, moon_colour)]))


def modes_animation(topology: Topology, vertices: core.Point, regions: np.ndarray,
                    displacement_scale: float,
                    frames: Iterable[tuple[core.Point, core.Scalar, core.Scalar]]) -> Iterator[np.ndarray]:
    """Yields dual-panel RGB frames of travelling normal modes normalized to each frame's peak reading."""
    drawing = DiskDrawing(topology, vertices, regions)
    for displacement, dilation, rotation in frames:
        places = vertices + displacement_scale * displacement                 # Point-[V]
        dilations, rotations = dilation.batch().to_array(), rotation.batch().to_array()  # [F], [F]
        scale = max(np.abs(dilations).max(), np.abs(rotations).max())
        panels = [(drawing.vertex_values(dilation / scale), dilation_surface),
                  (drawing.vertex_values(rotation / scale), rotation_surface)]
        yield panel_image(drawing.draw(places, panels, [], []))
