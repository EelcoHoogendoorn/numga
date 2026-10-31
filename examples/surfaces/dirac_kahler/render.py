"""Continuous flowlines and lit wave reliefs expose the different grades of a Dirac–Kähler field."""

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
from examples.util.wgsl import contours, flow, meshes, simplices, solids
from examples.util.wgsl.camera import Camera, flat_panel

sm = solids.mv

n_panel_pixels = 300
n_contours = 6
panel_gap = 12
background = np.ones(3)
wall_colour = "#68747b"
# The domain's boundary's width in pixels.
wall_width = 1.5
contour_colour = "#bac3c7"
domain_colour = "#f7f8f6"
flow_colours = ("#214b56", "#dc402c", "#386be0", "#8e4d9a")
scalar_colour = np.array([0.96, 0.24, 0.12])
bivector_colour = np.array([0.12, 0.36, 0.94])
relief_height = 0.85
relief_elevation, relief_azimuth = np.radians(39), np.radians(22)

# Thin level curves follow the rendered surface, so wave crests have shape as well as colour.
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
def plane_places(points: core.Point) -> np.ndarray:
    """x and y of points `[...]` on the last axis, for matplotlib's artists: their pairings with the planes x and y."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def at_vertices(mesh: core.Mesh, field: Extensor) -> Extensor:
    """Face fields averaged at each vertex over the faces at it, for drawing `X[faces]` as `X[vertices]`."""
    corners = ~mesh.incidence[2, 0].sum(axis=0)                                # Scalar+[vertices <- faces]
    ones = field.context.multivector.scalar(np.ones((len(mesh.faces), 1))).field()   # Scalar+[faces]
    return (corners * field) / (corners * ones)


def frame_panel(ax: plt.Axes, vertices: core.Point) -> None:
    """A square view about the domain, a little wider than it, without axes."""
    places = plane_places(vertices)                                            # [vertices, 2]
    low, high = places.min(axis=0), places.max(axis=0)
    centre = (low + high) / 2
    half_span = 0.54 * (high - low).max()
    ax.set(xlim=(centre[0] - half_span, centre[0] + half_span),
           ylim=(centre[1] - half_span, centre[1] + half_span), aspect="equal")
    ax.set_axis_off()


def level_curves(mesh: core.Mesh, vertices: core.Point, potential: core.Scalar, camera: Camera) -> np.ndarray:
    """Sparse level curves distinguish crossing a potential from following a stream potential: their picture."""
    values = np.asarray(potential.batch().to_array())
    bound = np.abs(values).max()
    # An even count of levels at even steps across four fifths of the bound either side of zero, none at zero.
    spacing = 1.6 * bound / (n_contours - 1)
    levels = contours.Lines(spacing, spacing / 2, np.append(to_rgb(contour_colour), 0.7), 1.0)
    return contours.picture(vertices.batch(), mesh, values, np.zeros(3), camera, lines=(levels,), opacity=0.0)


def panel_figure() -> tuple[plt.Figure, np.ndarray]:
    """Four distinct fields are compared in a compact square of equally sized viewports."""
    figure, axes = charts.subplots(2 * n_panel_pixels, 2 * n_panel_pixels, nrows=2, ncols=2)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return figure, axes.reshape(-1)


def draw_dirac_grades(mesh: core.Mesh, vertices: core.Point,
                      vertex_multivector: core.Full, scalar_gradient: core.Plane, face_dirac: core.Full,
                      whitney_multivector: core.Full) -> plt.Figure:
    """The scalar and circulating parts, their sum, and a Whitney reconstruction share a streak-length scale."""
    fields = stack((scalar_gradient.dual(), (face_dirac.select[1] - scalar_gradient).dual(),
                    face_dirac.select[1].dual(), whitney_multivector.select[1].dual()))
    return field_panels(mesh, vertices, fields, (flow_colours[1], flow_colours[2], flow_colours[0], flow_colours[3]),
                        0, vertex_multivector.select[0])


def field_panels(mesh: core.Mesh, vertices: core.Point, fields: core.Point, colours: tuple[str, ...],
                 contoured: int, potential: core.Scalar) -> plt.Figure:
    """Face fields interpolated for drawing on identical panels with one magnitude scale, each over a quiet domain
    under its boundary; the level curves of a potential under the flow of one panel."""
    directions = at_vertices(mesh, fields).batch()
    points = vertices.batch()
    reference_speed = float(np.percentile(directions.dual().norm().to_array(), 95))
    figure, axes = panel_figure()
    # A quiet domain, and its boundary, the mesh of its boundary's edges, leave the flow paths as the main drawing.
    rim = simplices.Mesh(len(points), mesh.edges[mesh.boundary_edges])
    for index, (ax, field, colour) in enumerate(zip(axes, directions, colours)):
        frame_panel(ax, vertices)
        camera = flat_panel(ax, points.algebra)
        levels = [level_curves(mesh, vertices, potential, camera)] if index == contoured else []
        simplices.overlay(ax, [contours.picture(points, mesh, np.zeros(len(points)), np.array(to_rgb(domain_colour)),
                                                camera)]
                          + levels
                          + [flow.layer(points, mesh, field, camera, reference_speed, np.array(to_rgb(colour))),
                             simplices.strokes(points, rim, np.array(to_rgb(wall_colour)), wall_width, camera)])
    return figure


def draw_hodge_decomposition(mesh: core.Mesh, vertices: core.Point,
                             total_flow: core.Point, potential: core.Scalar, potential_flow: core.Point,
                             stream_flow: core.Point, harmonic_flow: core.Point) -> plt.Figure:
    """The total velocity separates into source-to-sink paths, local eddies, and circulation around the hole."""
    fields = stack((total_flow, potential_flow, stream_flow, harmonic_flow))
    return field_panels(mesh, vertices, fields, flow_colours, 1, potential)


def wave_relief(mesh: core.Mesh, vertices: core.Point, heights: core.Scalar, channels: list[core.Scalar],
                colour: meshes.Colour, view: solids.View) -> np.ndarray:
    """Signed wave amplitude `Scalar+[vertices]` lifts a neutral annulus out of the drawing plane, coloured by the
    channels `Scalar+[vertices]` at the corners of its faces; both grade panels share its height and contour
    spacing."""
    # The annulus raised along z by its heights, and its rim, the mesh of its boundary's edges, raised and at rest.
    rest = solids.origin + solids.context.include_from_algebra(vertices.batch())                # [vertices] Point-
    elevated = rest + sm.z.dual() * solids.context.include_from_algebra((relief_height * heights).batch())   # [vertices] Point-
    rim = simplices.Mesh(len(rest), mesh.edges[mesh.boundary_edges])
    # The channels at each corner of each face, faces first, for the shader.
    corner_values = np.stack([(mesh.incidence[2, 0] * channel).batch().to_array() for channel in channels],
                             axis=-1).transpose(1, 0, 2)                       # [faces, corners, channels]
    surface = meshes.samples(elevated, mesh.faces, corner_values,
                              meshes.smooth(elevated, mesh), view,
                              face_values=np.zeros(1), colour=colour, gloss=0.14)
    edge = simplices.tubes(elevated, rim, np.array([0.24, 0.32, 0.35]), 1.25, view, solids.gloss)
    # A faint undisturbed rim gives height a fixed reference through crests and troughs.
    baseline = simplices.tubes(rest, rim, np.array([0.66, 0.71, 0.72]), 0.9, view, solids.gloss)
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(surface, 1.0), (edge, 1.0), (baseline, 0.35)])


def animate_dirac_waves(mesh: core.Mesh, vertices: core.Point,
                        frames: Iterable[tuple[core.Even, core.Plane]]) -> Iterator[np.ndarray]:
    """Signed scalar and bivector reliefs flank a combined RGB field: red scalar, green 1-form, blue bivector."""
    mv = vertices.context.multivector
    frames = iter(frames)
    first_even, first_face_field = next(frames)
    amplitudes = first_even.scalar_norm_squared().batch().square_root()
    amplitude_scale = amplitudes.max()
    view = solids.View(solids.aim(solids.origin, relief_elevation, relief_azimuth),
                       np.full(2, 1.12), n_panel_pixels)
    separator = np.full((n_panel_pixels, panel_gap, 3), 255, dtype=np.uint8)
    for even_state, face_field in chain(((first_even, first_face_field),), frames):
        scalar = even_state.select[0] / amplitude_scale                       # Scalar+[vertices]
        bivector = (even_state.select[2] & mv.xy.dual()) / amplitude_scale     # Scalar+[vertices]
        vertex_speed = at_vertices(mesh, face_field.norm()) / amplitude_scale  # Scalar+[vertices]
        # The size of the three together sets the combined panel's height and brightness.
        amplitude = (scalar.squared() + vertex_speed.squared() + bivector.squared()).square_root()   # Scalar+[vertices]
        scalar_image = wave_relief(mesh, vertices, scalar, [scalar], scalar_surface, view)
        mixed_image = wave_relief(mesh, vertices, amplitude, [scalar, vertex_speed, bivector, amplitude],
                                  mixed_surface, view)
        bivector_image = wave_relief(mesh, vertices, bivector, [bivector], bivector_surface, view)
        yield np.concatenate((scalar_image, separator, mixed_image, separator, bivector_image), axis=1)
