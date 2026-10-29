"""Continuous flowlines and lit wave reliefs expose the different grades of a Dirac–Kähler field."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import LinearTriInterpolator, Triangulation

from numga import stack
from examples.surfaces.dirac_kahler import core
from examples.util import charts
from examples.util.wgsl import meshes, solids, strokes

n_panel_pixels = 300
n_grid_samples = 120
n_contours = 6
panel_gap = 12
background = np.ones(3)
wall_colour = "#68747b"
contour_colour = "#bac3c7"
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
def coordinates(points: core.Point) -> np.ndarray:
    """Drawing coordinates of intrinsic positions and directions."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def domain_panel(ax: plt.Axes, mesh: core.Mesh, places: np.ndarray) -> Triangulation:
    """A quiet domain boundary leaves the flow paths as the main drawing."""
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    ax.add_collection(PolyCollection(places[mesh.faces], facecolors="#f7f8f6", edgecolors="none"))
    walls = places[mesh.edges[mesh.boundary_edges]]
    ax.add_collection(LineCollection(walls, colors=wall_colour, linewidths=1.1, zorder=4))
    low, high = places.min(axis=0), places.max(axis=0)
    centre = (low + high) / 2
    half_span = 0.54 * (high - low).max()
    ax.set(xlim=(centre[0] - half_span, centre[0] + half_span),
           ylim=(centre[1] - half_span, centre[1] + half_span), aspect="equal")
    ax.set_axis_off()
    return triangulation


def flow_panel(ax: plt.Axes, mesh: core.Mesh, places: np.ndarray,
               flow: core.Point, colour: str) -> None:
    """Streamlines show the connected geometry of a face velocity without direction glyphs."""
    triangulation = domain_panel(ax, mesh, places)
    field = coordinates(flow)
    # Face readouts are averaged only for the continuous drawing; the solver keeps its face field.
    vertex_field = np.zeros_like(places)
    np.add.at(vertex_field, mesh.faces.reshape(-1), np.repeat(field, 3, axis=0))
    vertex_field /= np.bincount(mesh.faces.reshape(-1))[:, None]
    low, high = places.min(axis=0), places.max(axis=0)
    sample_x = np.linspace(low[0], high[0], n_grid_samples)
    sample_y = np.linspace(low[1], high[1], n_grid_samples)
    grid_x, grid_y = np.meshgrid(sample_x, sample_y)
    horizontal = LinearTriInterpolator(triangulation, vertex_field[:, 0])(grid_x, grid_y)
    vertical = LinearTriInterpolator(triangulation, vertex_field[:, 1])(grid_x, grid_y)
    ax.streamplot(sample_x, sample_y, horizontal, vertical, color=colour, linewidth=1.3,
                  density=0.65, arrowstyle="-", maxlength=4, broken_streamlines=False, zorder=3)


def stream_panel(ax: plt.Axes, mesh: core.Mesh, places: np.ndarray,
                 potential: core.Scalar, colour: str) -> None:
    """Stream-potential contours close into eddies or wind around the missing region."""
    triangulation = domain_panel(ax, mesh, places)
    values = np.asarray(potential.batch().to_array())
    n_levels = 11
    levels = np.linspace(values.min(), values.max(), n_levels + 2)[1:-1]
    ax.tricontour(triangulation, values, levels=levels, colors=colour,
                  linewidths=1.3, linestyles="solid", zorder=3)


def contours(ax: plt.Axes, mesh: core.Mesh, places: np.ndarray,
             potential: core.Scalar, linestyle: str) -> None:
    """Sparse level curves distinguish crossing a potential from following a stream potential."""
    values = np.asarray(potential.batch().to_array())
    bound = np.abs(values).max()
    levels = np.linspace(-0.8 * bound, 0.8 * bound, n_contours)
    active = levels[(values.min() < levels) & (levels < values.max())]
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    ax.tricontour(triangulation, values, levels=active, colors=contour_colour,
                  linewidths=0.75, linestyles=linestyle, zorder=2)


def panel_figure() -> tuple[plt.Figure, np.ndarray]:
    """Four distinct fields are compared in a compact square of equally sized viewports."""
    figure, axes = charts.subplots(2 * n_panel_pixels, 2 * n_panel_pixels, nrows=2, ncols=2)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    return figure, axes.reshape(-1)


def draw_dirac_grades(mesh: core.Mesh, vertices: core.Point,
                      vertex_multivector: core.Full, scalar_gradient: core.Plane, face_dirac: core.Full,
                      whitney_multivector: core.Full) -> plt.Figure:
    """Scalar and stream gradients, their sum, and a separate Whitney edge reconstruction appear as flowlines."""
    mv = vertices.context.multivector
    places = coordinates(vertices)
    figure, axes = panel_figure()
    # The metric dual draws each face 1-form along the direction it grows fastest.
    flow_panel(axes[0], mesh, places, scalar_gradient.dual(), flow_colours[1])
    stream_panel(axes[1], mesh, places, vertex_multivector.select[2] & mv.xy.dual(), flow_colours[2])
    flow_panel(axes[2], mesh, places, face_dirac.select[1].dual(), flow_colours[0])
    flow_panel(axes[3], mesh, places, whitney_multivector.select[1].dual(), flow_colours[3])
    contours(axes[0], mesh, places, vertex_multivector.select[0], "dashed")
    return figure


def draw_hodge_decomposition(mesh: core.Mesh, vertices: core.Point,
                             total_flow: core.Point, potential: core.Scalar, stream: core.Scalar,
                             potential_flow: core.Point, harmonic_stream: core.Scalar) -> plt.Figure:
    """The total velocity separates into source-to-sink paths, local eddies, and loops winding around the hole."""
    places = coordinates(vertices)
    figure, axes = panel_figure()
    flow_panel(axes[0], mesh, places, total_flow, flow_colours[0])
    flow_panel(axes[1], mesh, places, potential_flow, flow_colours[1])
    stream_panel(axes[2], mesh, places, stream, flow_colours[2])
    stream_panel(axes[3], mesh, places, harmonic_stream, flow_colours[3])
    contours(axes[1], mesh, places, potential, "dashed")
    return figure


def wave_relief(mesh: core.Mesh, places: np.ndarray, amplitude: np.ndarray,
                values: np.ndarray, colour: meshes.Colour, view: solids.View) -> np.ndarray:
    """Signed wave amplitude lifts a neutral annulus; both grade panels share its height and contour spacing."""
    elevated = solids.mv.point(np.column_stack((places, relief_height * amplitude)))
    surface = meshes.samples(elevated, mesh.faces, values[mesh.faces],
                              meshes.smooth(elevated, mesh.faces), view,
                              face_values=np.zeros(1), colour=colour, gloss=0.14)
    rim = mesh.edges[mesh.boundary_edges]
    edge = strokes.segments(elevated[rim[:, 0]], elevated[rim[:, 1]],
                            np.array([0.24, 0.32, 0.35]), 1.25, view)
    # A faint undisturbed rim gives height a fixed reference through crests and troughs.
    reference = solids.mv.point(np.column_stack((places, np.zeros(len(places)))))
    baseline = strokes.segments(reference[rim[:, 0]], reference[rim[:, 1]],
                                np.array([0.66, 0.71, 0.72]), 0.9, view)
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(surface, 1.0), (edge, 1.0), (baseline, 0.35)])


def animate_dirac_waves(mesh: core.Mesh, vertices: core.Point,
                        frames: Iterable[tuple[core.Even, core.Plane]]) -> Iterator[np.ndarray]:
    """Signed scalar and bivector reliefs flank a combined RGB field: red scalar, green 1-form, blue bivector."""
    mv = vertices.context.multivector
    frames = iter(frames)
    first_even, first_face_field = next(frames)
    places = coordinates(vertices)
    amplitude_scale = float(np.asarray(first_even.scalar_norm_squared().batch().to_array()).max() ** 0.5)
    view = solids.View(solids.aim(solids.origin, relief_elevation, relief_azimuth),
                       np.full(2, 1.12), n_panel_pixels)
    separator = np.full((n_panel_pixels, panel_gap, 3), 255, dtype=np.uint8)
    for even_state, face_field in chain(((first_even, first_face_field),), frames):
        scalar = np.asarray(even_state.select[0].batch().to_array()) / amplitude_scale
        bivector = np.asarray((even_state.select[2].batch() & mv.xy.dual()).to_array()) / amplitude_scale
        face_speed = np.sqrt(np.asarray(face_field.scalar_norm_squared().batch().to_array())) / amplitude_scale
        vertex_speed = np.zeros(len(places))
        np.add.at(vertex_speed, mesh.faces.reshape(-1), np.repeat(face_speed, 3))
        vertex_speed /= np.bincount(mesh.faces.reshape(-1))
        components = np.column_stack((scalar, vertex_speed, bivector))
        amplitude = np.linalg.norm(components, axis=-1)
        mixed_values = np.column_stack((components, amplitude))
        scalar_image = wave_relief(mesh, places, scalar, scalar[:, None], scalar_surface, view)
        mixed_image = wave_relief(mesh, places, amplitude, mixed_values, mixed_surface, view)
        bivector_image = wave_relief(mesh, places, bivector, bivector[:, None], bivector_surface, view)
        yield np.concatenate((scalar_image, separator, mixed_image, separator, bivector_image), axis=1)
