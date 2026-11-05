"""The nematic film as symmetric grain along its molecular axes, shortening where order vanishes."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from functools import cache
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from examples.fluids.liquid_crystals import core
from examples.util import charts
from examples.util.topology import Topology
from examples.util.wgsl import Shader, contours, flow, image_figure, simplices
from examples.util.wgsl.camera import Camera, placing
from examples.util.wgsl.library import samples_per_side

gap, defect_radius = 8, 4.0
# A fixed palette reserves both defect colours while limiting the texture's shades.
film_colours = np.array([
    [26, 51, 71], [66, 87, 103], [108, 128, 143], [152, 169, 181],
    [192, 204, 213], [224, 231, 237], [255, 255, 255],
    [153, 194, 240], [183, 213, 243], [213, 230, 248],
    [204, 77, 31], [31, 92, 184],
], dtype=np.uint8)
n_colours = len(film_colours)
film_palette = Image.new("P", (1, 1))
film_palette.putpalette(film_colours.ravel().tolist())
grain_pixels, speck_pixels, streak_pixels = 11.0, 1.1, 64.0
ink = np.array([0.10, 0.20, 0.28])
colours = ("#274d70", "#c46a30", "#40836e")
mv = core.mv
setup_camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0,
                      np.array([0.03, 0.07]), np.array([2.7, 1.4]), 280)
turn_camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0,
                     np.array([-1.15, 0.6]), np.array([1.65, 1.65]), 280)
# The same fixed window in every case keeps translation visible, with room for the driven pair's whole path.
defect_camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0,
                       np.array([-0.5, 0.05]), np.array([2.6, 1.35]), 300)
dense_camera = Camera(mv.x, mv.y, 0 * mv.x, 0 * mv.x, 1.0,
                      np.zeros(2), np.full(2, 7.8), 300)


alignment_source = """
var<storage, read> points: array<Point>;
var<storage, read> faces: array<vec3<u32>>;
var<storage, read> alignment: array<Order>;
var<uniform> across_direction: Point;

struct Corner {
    @builtin(position) position: vec4<f32>,
    across: Point,
    place: Point,
}
struct Shade {
    @location(0) colour: vec4<f32>,
    @location(1) flow: vec4<f32>,
    @location(2) grain: vec4<f32>,
}

@vertex
fn place(@builtin(vertex_index) index: u32) -> Corner {
    let face = faces[index / 3];
    let vertex = face[index % 3];
    return Corner(clip(points[vertex]), alignment[vertex](across_direction), points[vertex]);
}

@fragment
fn shade(corner: Corner) -> Shade {
    // Reading this traceless map gives a doubled-angle image: opposite molecular ends have the same value.
    // Only here, where the image is written, does the map become pixel components, with rows counted down.
    let doubled = 2.0 * vec2<f32>(screen_across & corner.across, -(screen_up & corner.across));
    return Shade(vec4<f32>(1.0), vec4<f32>(doubled, 0, 0), vec4<f32>(seed(corner.place)));
}
"""


@cache
def alignment_shader() -> Shader:
    return Shader(placing(core.Plane) + flow.planar_grain + alignment_source, core.ga,
                  dict(Point=core.Point, Plane=core.Plane, Order=core.Order), {},
                  targets=("rgba8unorm", "rgba32float", "rgba32float"))


def texture(topology: Topology, points: core.Point, camera: Camera, order: core.Order) -> np.ndarray:
    """Fixed grain feathered symmetrically along the alignment, without a signed director field."""
    surface, projected, grains = alignment_shader().draw_averaged(
        camera.shape, samples_per_side(), 3 * topology.n_elements[2], points=points,
        faces=topology.elements[2], alignment=order, across_direction=camera.across.dual(),
        noise_scale=camera.pixel * grain_pixels, speck=speck_pixels / grain_pixels, **camera.uniforms(1.0))
    result = flow.trace_shader(True)(
        camera.shape, surface=surface, flow=projected, grains=grains, background=np.ones(3),
        step_pixels=streak_pixels / flow.n_trace_steps, reference_speed=1.0,
        downstream_fraction=0.5, depth_limit=6 * camera.pixel, ink_opacity=0.92)
    rgba = np.empty(camera.shape + (4,))
    rgba[..., :3] = ink
    rgba[..., 3] = result["coverage"][..., 0]
    return rgba


def frame(topology: Topology, vertices: core.Point, camera: Camera, order: core.Order) -> np.ndarray:
    """The molecular axes and degree of order, with one tightly cropped row per case."""
    mv = vertices.context.multivector
    points = vertices.batch()
    middles = (topology.incidence[2, 0].mean(axis=0) * vertices).batch()
    panels = []
    for state in order:
        state = state.batch()
        strength = (2 * state(state).trace()).square_root()
        amount = np.clip(strength.to_array(), 0, 1)
        tint = 1 - (1 - amount[:, None]) * np.array([0.40, 0.24, 0.06])
        winding = core.windings(topology, state.field(), mv.x.dual()).scalar_product(~mv.xy).to_array()
        defect_colours = np.where((winding > 0)[:, None], np.array([0.80, 0.30, 0.12]),
                                 np.array([0.12, 0.36, 0.72]))
        radii = defect_radius * (np.abs(winding) > 0.25)
        layers = [contours.picture(points, topology, amount, tint, camera),
                  texture(topology, points, camera, state),
                  simplices.markers(middles, defect_colours, radii, camera)]
        picture = simplices.composited(layers)
        panels.append(np.round(255 * np.clip(picture[..., :3] + 1 - picture[..., 3:], 0, 1)).astype(np.uint8))
    spacer = np.full((gap, camera.shape[1], 3), 255, dtype=np.uint8)
    picture = np.concatenate([part for index, panel in enumerate(panels)
                              for part in ((spacer, panel) if index else (panel,))], axis=0)
    return np.asarray(Image.fromarray(picture).quantize(palette=film_palette, dither=Image.Dither.NONE).convert("RGB"))


def animate(topology: Topology, vertices: core.Point, camera: Camera,
            states: Iterable[core.Order]) -> Iterator[np.ndarray]:
    """The film's molecular axes at each state, streamed in the order computed."""
    for order in states:
        yield frame(topology, vertices, camera, order)


def draw_setup(topology: Topology, vertices: core.Point, order: core.Order) -> Figure:
    """The film's starting alignment."""
    return image_figure(frame(topology, vertices, setup_camera, order[None]))


def draw_dense_setup(topology: Topology, vertices: core.Point, order: core.Order) -> Figure:
    """The crowded initial alignment, with both signs of half-charge defect."""
    return image_figure(frame(topology, vertices, dense_camera, order[None]))


@charts.style()
def draw_energy(times: np.ndarray, energies: core.Scalar) -> Figure:
    """The free energy of each film over time."""
    figure, axis = charts.subplots(360, 190, layout="constrained")
    for values, colour in zip(energies.to_array().T, colours):
        axis.plot(times, values, color=colour)
    axis.axhline(0, color="0.55", linewidth=charts.FRAME_WIDTH)
    axis.set(xlim=(times[0], times[-1]), ylim=(0, None), xticks=[], yticks=[], xlabel="time", ylabel="free energy")
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    return figure


@charts.style()
def draw_derivative(angles: np.ndarray, energies: core.Scalar, baseline: core.Scalar, slope: core.Scalar) -> Figure:
    """The energy change under a local turn and the tangent predicted by its bivector derivative."""
    figure, axis = charts.subplots(300, 190, layout="constrained")
    axis.plot(angles, (energies - baseline).to_array(), color=colours[0])
    axis.plot(angles, (angles * slope).to_array(), color=colours[1], linestyle="--")
    axis.axhline(0, color="0.65", linewidth=charts.FRAME_WIDTH)
    axis.axvline(0, color="0.85", linewidth=charts.FRAME_WIDTH)
    axis.set(xlim=(angles[0], angles[-1]), xticks=[0], yticks=[0], xlabel="local turn", ylabel="energy change")
    axis.spines[["top", "right", "bottom", "left"]].set_visible(False)
    return figure


@charts.style()
def draw_release(times: np.ndarray, energies: core.Scalar) -> Figure:
    """The free energy released by orientational relaxation, measured from its initial state."""
    figure, axis = charts.subplots(300, 180, layout="constrained")
    axis.plot(times, (energies[0] - energies).to_array(), color=colours[0])
    axis.axhline(0, color="0.55", linewidth=charts.FRAME_WIDTH)
    axis.set(xlim=(times[0], times[-1]), ylim=(0, None), xticks=[], yticks=[], xlabel="time", ylabel="energy released")
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    return figure


def inspect_animation(path: Path, destination: Path) -> None:
    """Write sampled frames to a private inspection strip without resampling them."""
    with Image.open(path) as animation:
        indices = np.linspace(0, animation.n_frames - 1, 5, dtype=int)
        strip = Image.new("RGB", (animation.width, animation.height * len(indices)))
        for row, index in enumerate(indices):
            animation.seek(int(index))
            strip.paste(animation.convert("RGB"), (0, row * animation.height))
        destination.parent.mkdir(parents=True, exist_ok=True)
        strip.save(destination)
