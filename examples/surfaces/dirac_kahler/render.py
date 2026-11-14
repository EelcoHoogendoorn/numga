"""Continuous flowlines and lit wave reliefs display the separate grades of a Dirac–Kähler field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.extensor import Extensor
from examples.surfaces.dirac_kahler import core
from examples.util import charts
from numga.render import Camera, contours, flow, meshes, simplices, solids
from examples.util.panels import offset_panel, overlay

sm = solids.mv

n_panel_pixels = 300
n_contours = 6
panel_gap = 12
background = np.ones(3)
wall_colour = "#68747b"
# Domain rim width in pixels.
wall_width = 1.5
contour_colour = "#bac3c7"
domain_colour = "#f7f8f6"
flow_colours = ("#214b56", "#dc402c", "#386be0", "#8e4d9a")
scalar_colour = np.array([0.96, 0.24, 0.12])
bivector_colour = np.array([0.12, 0.36, 0.94])
relief_height = 0.85
relief_elevation, relief_azimuth = np.radians(39), np.radians(22)

# Thin level curves follow the rendered surface so wave crests show both elevation and contour lines.
wave_shader = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let value = values.x;
    let paper = vec3<f32>(0.93, 0.94, 0.91);
    let signed_colour = component_colour * select(0.64, 1.0, value >= 0.0);
    let tint = mix(paper, signed_colour, min(abs(value) * 2.6, 0.94));
    let level = value / 0.10;
    let distance = abs(fract(level + 0.5) - 0.5);
    let width = fwidth(level) * 0.7;
    let ink = (1.0 - smoothstep(0.0, width, distance)) * smoothstep(0.025, 0.06, abs(value));
    return vec4<f32>(mix(tint, tint * 0.58, ink * 0.58), 1.0);
}
"""
scalar_surface = meshes.Colour(wave_shader, dict(component_colour=scalar_colour))
bivector_surface = meshes.Colour(wave_shader, dict(component_colour=bivector_colour))
mixed_surface = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let weights = abs(values.rgb);
    let amplitude = values.w;
    let largest = max(max(weights.r, weights.g), max(weights.b, 0.001));
    let components = vec3<f32>(0.06) + 0.88 * weights / largest;
    let paper = vec3<f32>(0.93, 0.94, 0.91);
    return vec4<f32>(mix(paper, components, min(amplitude * 3.2, 0.97)), 1.0);
}
""", {})


# --- plumbing -------------------------------------------------------------------------
def vertex_average(topology: core.Topology, field: Extensor) -> Extensor:
    """Incident face values averaged at each vertex lift a face field (`...[F]`) onto the vertices (`...[V]`)."""
    V, E, F = topology.n_elements
    corners = topology.incidence[2, 0].sum(axis=0).adjugate()                # Scalar+[V <- F]
    ones = field.context.multivector.scalar(np.ones((F, 1))).field()         # Scalar+[F]
    return (corners * field) / (corners * ones)


def frame_panel(ax: plt.Axes, vertices: core.Point) -> None:
    """Sets equal square axis limits slightly wider than `vertices` and hides axis ticks."""
    mv = vertices.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & vertices.batch().reshape(-1)   # [2, ...] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                 # [2] Scalar+, [2] Scalar+
    centre = (low + high) / 2
    half_span = 0.54 * (high - low).max()
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()


def level_curves(topology: core.Topology, vertices: core.Point, potential: core.Scalar, camera: Camera) -> np.ndarray:
    """Draws evenly spaced level contours of `potential` across the triangulated domain."""
    values = np.asarray(potential.batch().to_array())
    bound = np.abs(values).max()
    spacing = 1.6 * bound / (n_contours - 1)
    levels = contours.Lines(spacing, spacing / 2, np.append(to_rgb(contour_colour), 0.7), 1.0)
    return contours.picture(vertices.batch(), topology, values, np.zeros(3), camera, lines=(levels,), opacity=0.0)


def panel_figure() -> tuple[plt.Figure, np.ndarray]:
    """Creates a two-by-two grid of borderless square viewports."""
    figure, axes = charts.subplots(2 * n_panel_pixels, 2 * n_panel_pixels, nrows=2, ncols=2)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return figure, axes.reshape(-1)


def grades_figure(
    topology: core.Topology, vertices: core.Point,
    vertex_multivector: core.Full, scalar_gradient: core.Plane, face_dirac: core.Full,
    whitney_multivector: core.Full,
) -> plt.Figure:
    """Renders the scalar slope, stream slope, full grade-1 Dirac–Kähler derivative, and Whitney 1-form on one streak scale."""
    fields = stack((
        scalar_gradient.dual(),
        (face_dirac.select[1] - scalar_gradient).dual(),
        face_dirac.select[1].dual(),
        whitney_multivector.select[1].dual(),
    ))
    return field_panels(
        topology, vertices, fields,
        (flow_colours[1], flow_colours[2], flow_colours[0], flow_colours[3]),
        0, vertex_multivector.select[0],
    )


def field_panels(
    topology: core.Topology, vertices: core.Point, fields: core.Point, colours: tuple[str, ...],
    contoured: int, potential: core.Scalar,
) -> plt.Figure:
    """Draws four interpolated face fields on matching panels sharing one streak speed scale."""
    directions = vertex_average(topology, fields).batch()
    points = vertices.batch()
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    figure, axes = panel_figure()
    rim = topology.border
    for index, (ax, field, colour) in enumerate(zip(axes, directions, colours)):
        frame_panel(ax, vertices)
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        levels = [level_curves(topology, vertices, potential, camera)] if index == contoured else []
        overlay(
            ax,
            [contours.picture(points, topology, np.zeros(len(points)), np.array(to_rgb(domain_colour)), camera)]
            + levels
            + [
                flow.layer(points, topology, field, camera, reference_speed, np.array(to_rgb(colour))),
                simplices.strokes(points, rim, np.array(to_rgb(wall_colour)), wall_width, camera),
            ],
        )
    return figure


def hodge_figure(
    topology: core.Topology, vertices: core.Point,
    total_flow: core.Point, potential: core.Scalar, potential_flow: core.Point,
    stream_flow: core.Point, harmonic_flow: core.Point,
) -> plt.Figure:
    """Renders the total velocity alongside its potential, stream, and harmonic Hodge components."""
    fields = stack((total_flow, potential_flow, stream_flow, harmonic_flow))
    return field_panels(topology, vertices, fields, flow_colours, 1, potential)


def wave_relief(
    topology: core.Topology, vertices: core.Point, heights: core.Scalar, channels: list[core.Scalar],
    colour: meshes.Colour, view: solids.View,
) -> np.ndarray:
    """Renders the triangulated annulus lifted along `z` by `heights` (`Scalar+[V]`) and shaded by corner `channels`."""
    V, E, F = topology.n_elements
    n_channels = len(channels)
    rest = solids.origin + solids.context.include_from_algebra(vertices.batch())  # [V] Point-
    elevated = rest + sm.z.dual() * solids.context.include_from_algebra((relief_height * heights).batch())  # [V] Point-
    rim = topology.border
    corner_values = np.stack(
        [(topology.incidence[2, 0] * channel).batch().to_array() for channel in channels],
        axis=-1,
    ).transpose(1, 0, 2)                                                     # [F, 3, n_channels]
    surface = meshes.samples(
        elevated, topology.elements[2], corner_values,
        meshes.smooth(elevated, topology), view,
        face_values=np.zeros(1), colour=colour, gloss=0.14,
    )
    edge = simplices.tubes(elevated, rim, np.array([0.24, 0.32, 0.35]), 1.25, view, solids.gloss)
    baseline = simplices.tubes(rest, rim, np.array([0.66, 0.71, 0.72]), 0.9, view, solids.gloss)
    return solids.draw(
        [], view, background, inks=[], translucent=[],
        meshes=[(surface, 1.0), (edge, 1.0), (baseline, 0.35)],
    )


def waves_animation(
    topology: core.Topology, vertices: core.Point,
    frames: Iterable[tuple[core.Even, core.Plane]],
) -> Iterator[np.ndarray]:
    """Yields side-by-side RGB frames of the scalar, combined RGB, and stream-bivector wave reliefs."""
    mv = vertices.context.multivector
    frames = iter(frames)
    first_even, first_face_field = next(frames)
    amplitudes = first_even.scalar_norm_squared().batch().square_root()
    amplitude_scale = amplitudes.max()
    view = solids.View(
        solids.aim(solids.origin, relief_elevation, relief_azimuth),
        np.full(2, 1.12), n_panel_pixels,
    )
    separator = np.full((n_panel_pixels, panel_gap, 3), 255, dtype=np.uint8)
    for even_state, face_field in chain(((first_even, first_face_field),), frames):
        scalar = even_state.select[0] / amplitude_scale                      # Scalar+[V]
        bivector = (mv.xy.dual() & even_state.select[2]) / amplitude_scale   # Scalar+[V]
        vertex_speed = vertex_average(topology, face_field.norm()) / amplitude_scale  # Scalar+[V]
        amplitude = (scalar.squared() + vertex_speed.squared() + bivector.squared()).square_root()  # Scalar+[V]
        scalar_image = wave_relief(topology, vertices, scalar, [scalar], scalar_surface, view)
        mixed_image = wave_relief(
            topology, vertices, amplitude,
            [scalar, vertex_speed, bivector, amplitude],
            mixed_surface, view,
        )
        bivector_image = wave_relief(topology, vertices, bivector, [bivector], bivector_surface, view)
        yield np.concatenate((scalar_image, separator, mixed_image, separator, bivector_image), axis=1)
