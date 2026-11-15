"""Flat-shaded GPU rendering lights each triangle by its own normal patch so developable panels read as smooth sweeps of shade and creases read as sharp transitions."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.developability import core
from examples.util.panels import image_figure
from numga.render.drawing import Draw, Target, drawn
from numga.topology import Topology
from numga.wgsl import Shader, samples_per_side

Rotor = core.ga.gatype.rotor()                      # Rotor+

surface_colour = np.array([0.90, 0.89, 0.86])
light_direction = core.mv.z.dual() * 0.6 - core.mv.x.dual() * 0.4 - core.mv.y.dual() * 0.7      # [] Point-
ambient = 0.25
elevation, azimuth = np.radians(24.0), np.radians(54.0)
distance, half_view = 8.0, 0.17
frame_shape = (480, 480)
background = np.array([1.0, 1.0, 1.0])

shader_source = "\n".join([
    "var<storage, read> points: array<Point>;",
    "var<storage, read> faces: array<vec3<u32>>;",
    "var<uniform> turn: Rotor;",
    "var<uniform> eye: Point;",
    "var<uniform> lamp: Point;",
    "var<uniform> half_view: f32;",
    "var<uniform> pixels: vec2<u32>;",
    "var<uniform> colour: vec3<f32>;",
    "var<uniform> ambient: f32;",
    "",
    "const near = 0.1;",
    "const far = 100.0;",
    "",
    "struct Corner {",
    "    @builtin(position) position: vec4<f32>,",
    "    @interpolate(flat) colour: vec3<f32>,",
    "}",
    "",
    "@vertex",
    "fn place(@builtin(vertex_index) index: u32) -> Corner {",
    "    let face = faces[index / 3];",
    "    let a = points[face.x];",
    "    let b = points[face.y];",
    "    let c = points[face.z];",
    "    let normal = ((b - a) & (c - a)).normalized();",
    "    let lit = ambient + (1 - ambient) * max(normal & lamp, 0.0);",
    "    let seen = turn << (points[face[index % 3]] - eye);",
    "    let depth = -(mv.z & seen);",
    "    let aspect = f32(pixels.x) / f32(pixels.y);",
    "    let position = vec4<f32>((mv.x & seen) / (half_view * aspect), (mv.y & seen) / half_view,",
    "                             (depth - near) * far / (far - near), depth);",
    "    return Corner(position, colour * lit);",
    "}",
    "",
    "@fragment",
    "fn shade(corner: Corner) -> @location(0) vec4<f32> {",
    "    return vec4<f32>(corner.colour, 1.0);",
    "}",
])


# --- plumbing -------------------------------------------------------------------------
@cache
def surface_shader() -> Shader:
    """Compiles and caches the flat-shaded WGSL triangle shader for `core.ga`."""
    return Shader(shader_source, core.ga, dict(Point=core.Point, Rotor=Rotor), {})


def camera_pose() -> tuple[Rotor, core.Point]:
    """Camera orientation (`Rotor+`) and eye position (`Point-`) view the origin from `distance` at `elevation` and `azimuth`."""
    pitch = (core.mv.yz * (-(np.pi / 2 - elevation) / 2)).exp()                 # [] Rotor+
    turn = (core.mv.xy * (-azimuth / 2)).exp() * pitch                          # [] Rotor+
    eye_direction = turn >> core.mv.z.dual()                                    # [] Point-
    return turn, eye_direction * distance


def surface_image(topology: Topology, vertices: core.Point) -> np.ndarray:
    """Renders `vertices` (`Point-[V]`) on `topology` with flat per-triangle shading into an RGB image."""
    V, E, F = topology.n_elements
    turn, eye = camera_pose()
    # Supersample each pixel on the GPU and composite the premultiplied colour over the white background.
    surface, = drawn([Draw(surface_shader(), frame_shape, 3 * F, dict(
        points=vertices.batch(), faces=topology.elements[2], turn=turn, eye=eye, lamp=light_direction.normalized(),
        half_view=half_view, colour=surface_colour, ambient=ambient))], Target(("rgba32float",), False), samples_per_side())
    shown = surface[..., :3] + (1 - surface[..., 3:]) * background
    return np.round(np.clip(shown, 0, 1) * 255).astype(np.uint8)


def surface_figure(topology: Topology, vertices: core.Point) -> plt.Figure:
    """Wraps the flat-shaded render of `vertices` (`Point-[V]`) in a matplotlib figure."""
    return image_figure(surface_image(topology, vertices))


def flow_animation(topology: Topology, states: Iterable[tuple[core.Point, core.Scalar]]) -> Iterator[np.ndarray]:
    """Yields flat-shaded RGB frames as `vertices` (`Point-[V]`) evolve through `states`."""
    for vertices, _ in states:
        yield surface_image(topology, vertices)
