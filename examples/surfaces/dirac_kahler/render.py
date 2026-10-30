"""Continuous flowlines and lit wave reliefs expose the different grades of a Dirac–Kähler field."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.tri import Triangulation

from numga import stack
from numga.extensor import Extensor
from numga.sparse import SparseExtensor, spdiag
from examples.surfaces.dirac_kahler import core
from examples.util import charts
from examples.util.wgsl import meshes, solids, strokes

sm = solids.mv

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
def plane_places(points: core.Point) -> np.ndarray:
    """x and y of points `[...]` on the last axis, for matplotlib's artists: their pairings with the planes x and y."""
    mv = points.context.multivector
    return np.asarray((stack((mv.x, mv.y)) & points.batch()[..., None]).to_array())


def at_vertices(mesh: core.Mesh, field: Extensor) -> Extensor:
    """Face fields averaged at each vertex over the faces at it, for drawing `X[faces]` as `X[vertices]`."""
    corners = ~mesh.incidence[2, 0].sum(axis=0)                                # Scalar+[vertices <- faces]
    ones = field.context.multivector.scalar(np.ones((len(mesh.faces), 1))).field()   # Scalar+[faces]
    return (corners * field) / (corners * ones)


def lifting(mesh: core.Mesh, vertices: core.Point, samples: core.Point) -> tuple[SparseExtensor, np.ndarray]:
    """The map carrying a vertex field over the mesh onto samples `[rows, columns]`, linear across each triangle: a
    sample's hats in the triangle that holds it weight that triangle's corners; and the samples no triangle holds,
    `[rows, columns]`."""
    corners = mesh.incidence[2, 0]                                             # [corners] Scalar+[faces <- vertices]
    domain_corners = corners * vertices                                        # [corners] Point-[faces]
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                       # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                        # Patch-[faces]
    # Each corner's hat rises from zero along the facing side to one at the corner.
    corner_gradients = sides / (2 * patches)                                   # [corners] Plane+[faces]
    gradients, corner_points = corner_gradients.batch()[:, None], domain_corners.batch()[:, None]   # [corners, 1, faces] Plane+, Point-
    # A sample's three hats in any triangle sum to one, and their sizes do too only in a triangle that holds it;
    # found a row of samples at a time, so that the hats of every sample in every triangle are never held at once.
    nearest = np.stack([(1 + (gradients & (row[:, None] - corner_points))).abs().sum(axis=0).argmin(axis=-1)
                        for row in samples])                                   # [rows, columns]
    holding = mesh.reading(nearest, 2)                                         # [rows] Scalar+[columns <- faces]
    hats = 1 + ((holding[None] * corner_gradients[:, None])
                & (samples.field() - holding[None] * domain_corners[:, None]))   # [corners, rows] Scalar+[columns]
    outside = hats.abs().sum(axis=0).batch() > 1 + 1e-9                         # [rows, columns]
    return (spdiag(hats) * (holding[None] * corners[:, None])).sum(axis=0), outside   # [rows] Scalar+[columns <- vertices]


def streamlines(mesh: core.Mesh, vertices: core.Point) -> Callable[[plt.Axes, core.Point, str], None]:
    """What draws the streamlines of a face velocity `Point-[faces]` on an axes: the velocity averaged to the
    vertices for the continuous drawing, the solver keeping its face field, and carried across the triangles onto a
    grid of samples over the mesh, where no triangle holding a sample leaves it without flow."""
    mv = vertices.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & vertices.batch()
    low = readings.min(axis=-1)
    high = readings.max(axis=-1)
    grid = low[:, None] + (high - low)[:, None] * np.linspace(0, 1, n_grid_samples)
    sample_x, sample_y = grid
    samples = mv.x.dual() * sample_x[None, :] + mv.y.dual() * sample_y[:, None]   # [rows, columns] Point-
    lift, outside = lifting(mesh, vertices, samples)                           # [rows] Scalar+[columns <- vertices], [rows, columns]

    def drawn(ax: plt.Axes, flow: core.Point, colour: str) -> None:
        velocities = np.ma.masked_array(plane_places(lift * at_vertices(mesh, flow)),
                                        np.broadcast_to(outside[..., None], outside.shape + (2,)))   # [rows, columns, 2]
        ax.streamplot(sample_x.to_array(), sample_y.to_array(), velocities[..., 0], velocities[..., 1], color=colour, linewidth=1.3,
                      density=0.65, arrowstyle="-", maxlength=4, broken_streamlines=False, zorder=3)

    return drawn


def domain_panel(ax: plt.Axes, mesh: core.Mesh, vertices: core.Point) -> Triangulation:
    """A quiet domain boundary leaves the flow paths as the main drawing."""
    places = plane_places(vertices)                                            # [vertices, 2]
    triangulation = Triangulation(*places.T, triangles=mesh.faces)
    faces = plane_places(mesh.incidence[2, 0] * vertices).transpose(1, 0, 2)  # [faces, corners, 2]
    ax.add_collection(PolyCollection(faces, facecolors="#f7f8f6", edgecolors="none"))
    walls = plane_places(mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices))   # [ends, rim, 2]
    ax.add_collection(LineCollection(walls.transpose(1, 0, 2), colors=wall_colour, linewidths=1.1, zorder=4))
    low, high = places.min(axis=0), places.max(axis=0)
    centre = (low + high) / 2
    half_span = 0.54 * (high - low).max()
    ax.set(xlim=(centre[0] - half_span, centre[0] + half_span),
           ylim=(centre[1] - half_span, centre[1] + half_span), aspect="equal")
    ax.set_axis_off()
    return triangulation


def flow_panel(ax: plt.Axes, mesh: core.Mesh, vertices: core.Point, flow: core.Point, colour: str,
               drawn: Callable[[plt.Axes, core.Point, str], None]) -> None:
    """Streamlines show the connected geometry of a face velocity without direction glyphs."""
    domain_panel(ax, mesh, vertices)
    drawn(ax, flow, colour)


def stream_panel(ax: plt.Axes, mesh: core.Mesh, vertices: core.Point,
                 potential: core.Scalar, colour: str) -> None:
    """Stream-potential contours close into eddies or wind around the missing region."""
    triangulation = domain_panel(ax, mesh, vertices)
    values = np.asarray(potential.batch().to_array())
    n_levels = 11
    levels = np.linspace(values.min(), values.max(), n_levels + 2)[1:-1]
    ax.tricontour(triangulation, values, levels=levels, colors=colour,
                  linewidths=1.3, linestyles="solid", zorder=3)


def contours(ax: plt.Axes, mesh: core.Mesh, vertices: core.Point,
             potential: core.Scalar, linestyle: str) -> None:
    """Sparse level curves distinguish crossing a potential from following a stream potential."""
    values = np.asarray(potential.batch().to_array())
    bound = np.abs(values).max()
    levels = np.linspace(-0.8 * bound, 0.8 * bound, n_contours)
    active = levels[(values.min() < levels) & (levels < values.max())]
    triangulation = Triangulation(*plane_places(vertices).T, triangles=mesh.faces)
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
    figure, axes = panel_figure()
    drawn = streamlines(mesh, vertices)
    # The metric dual draws each face 1-form along the direction it grows fastest.
    flow_panel(axes[0], mesh, vertices, scalar_gradient.dual(), flow_colours[1], drawn)
    stream_panel(axes[1], mesh, vertices, vertex_multivector.select[2] & mv.xy.dual(), flow_colours[2])
    flow_panel(axes[2], mesh, vertices, face_dirac.select[1].dual(), flow_colours[0], drawn)
    flow_panel(axes[3], mesh, vertices, whitney_multivector.select[1].dual(), flow_colours[3], drawn)
    contours(axes[0], mesh, vertices, vertex_multivector.select[0], "dashed")
    return figure


def draw_hodge_decomposition(mesh: core.Mesh, vertices: core.Point,
                             total_flow: core.Point, potential: core.Scalar, stream: core.Scalar,
                             potential_flow: core.Point, harmonic_stream: core.Scalar) -> plt.Figure:
    """The total velocity separates into source-to-sink paths, local eddies, and loops winding around the hole."""
    figure, axes = panel_figure()
    drawn = streamlines(mesh, vertices)
    flow_panel(axes[0], mesh, vertices, total_flow, flow_colours[0], drawn)
    flow_panel(axes[1], mesh, vertices, potential_flow, flow_colours[1], drawn)
    stream_panel(axes[2], mesh, vertices, stream, flow_colours[2])
    stream_panel(axes[3], mesh, vertices, harmonic_stream, flow_colours[3])
    contours(axes[1], mesh, vertices, potential, "dashed")
    return figure


def wave_relief(mesh: core.Mesh, vertices: core.Point, heights: core.Scalar, channels: list[core.Scalar],
                colour: meshes.Colour, view: solids.View) -> np.ndarray:
    """Signed wave amplitude `Scalar+[vertices]` lifts a neutral annulus out of the drawing plane, coloured by the
    channels `Scalar+[vertices]` at the corners of its faces; both grade panels share its height and contour
    spacing."""
    # The annulus raised along z by its heights, and its rim's edges, raised and at rest.
    rest = solids.origin + solids.context.include_from_algebra(vertices.batch())                # [vertices] Point-
    elevated = rest + sm.z.dual() * solids.context.include_from_algebra((relief_height * heights).batch())   # [vertices] Point-
    rim = mesh.reading(mesh.boundary_edges, 1) * mesh.incidence[1, 0]          # [ends] Scalar+[rim <- vertices]
    rim_tails, rim_heads = rim * elevated.field()                              # Point-[rim], Point-[rim]
    rest_tails, rest_heads = rim * rest.field()                                # Point-[rim], Point-[rim]
    # The channels at each corner of each face, faces first, for the shader.
    corner_values = np.stack([(mesh.incidence[2, 0] * channel).batch().to_array() for channel in channels],
                             axis=-1).transpose(1, 0, 2)                       # [faces, corners, channels]
    surface = meshes.samples(elevated, mesh.faces, corner_values,
                              meshes.smooth(elevated, mesh), view,
                              face_values=np.zeros(1), colour=colour, gloss=0.14)
    edge = strokes.segments(rim_tails.batch(), rim_heads.batch(), np.array([0.24, 0.32, 0.35]), 1.25, view)
    # A faint undisturbed rim gives height a fixed reference through crests and troughs.
    baseline = strokes.segments(rest_tails.batch(), rest_heads.batch(), np.array([0.66, 0.71, 0.72]), 0.9, view)
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
