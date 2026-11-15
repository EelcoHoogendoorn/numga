"""Depth-tested GPU rendering draws K-surface sheets and their Bäcklund pairs.

Each triangulated surface carries inked grid lines evaluated per fragment from corner edge distances, while a Bäcklund pair combines two translucent sheets with tangent connector rods and endpoint spheres composited by depth sample by sample so occluded connector segments remain faintly visible through the sheets.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate
from examples.surfaces.k_surfaces import core
from numga.topology import Topology
from numga.render.drawing import Draw, compose
from numga.render import meshes, solids
from examples.util.panels import image_figure

# Canvas pixel dimensions, scaled saddle viewport fraction, and grid-line stride on the single saddle.
n_width = 640
n_height = 480
saddle_scale = 0.5
saddle_line_spacing = 2
elevation, azimuth = np.radians(15.0), np.radians(75.0)
margin = 0.1
front_colour = np.array([0.29, 0.61, 0.67])
back_colour = np.array([0.79, 0.48, 0.32])
background = np.ones(3)
# Grid-line pixel width, interior and boundary ink strengths, and sheet colours/opacity on Bäcklund pairs.
edge_width = 0.8
edge_darkness = 0.28
boundary_darkness = 0.6
pair_grid_ink = 0.6
source_colour = np.array([0.28, 0.58, 0.64])
backlund_colour = np.array([0.78, 0.48, 0.28])
surface_opacity = 0.78
# Tangent connector colour, pixel width, endpoint radius, occluded ghost opacity, and sampled grid counts.
connector_colour = np.array([0.17, 0.17, 0.20])
connector_width = 1.8
endpoint_radius = 2.5
hidden_opacity = 0.35
n_connector_rows = 3
n_connector_columns = 3
n_connectors = n_connector_rows * n_connector_columns


# WGSL fragment shader colouring front and back faces while darkening pixels near marked triangle edges.
gridded_shader = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let edges = clamp(1 - corner / (edge_width * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    let ink = max(edges.x, max(edges.y, edges.z));
    return vec4<f32>(select(back_colour, front_colour, front) * (1 - ink), 1.0);
}
"""


def grid_colour(front: np.ndarray, back: np.ndarray, width: float) -> meshes.Colour:
    """Two-sided `Colour` shader shades `front` and `back` faces while darkening marked triangle edges of pixel `width`."""
    return meshes.Colour(gridded_shader, dict(edge_width=width, front_colour=front, back_colour=back))


# --- plumbing -------------------------------------------------------------------------
def sheet_topology(n_rows: int, n_columns: int, spacing: int) -> tuple[Topology, np.ndarray]:
    """Triangulating an `[n_rows, n_columns]` sheet produces its `Topology` and per-face edge ink (`[F, 3]`) without quad diagonals."""
    indices = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    quads = np.stack([indices[:-1, :-1], indices[1:, :-1],
                     indices[1:, 1:], indices[:-1, 1:]], axis=-1).reshape(-1, 4)
    topology = Topology(n_rows * n_columns, quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))
    # Mark edges opposite each triangle corner that lie on the sheet's outer border.
    boundary = np.isin(topology.boundary_indices[1], topology.exterior[1])      # [F, 3]
    # Mark quad edges along every `spacing`-th row and column while leaving every quad diagonal unmarked.
    rows, columns = np.meshgrid(np.arange(n_rows - 1), np.arange(n_columns - 1), indexing="ij")
    drawn_rows, drawn_columns = np.arange(n_rows) % spacing == 0, np.arange(n_columns) % spacing == 0
    zero = np.zeros(rows.shape, dtype=bool)
    lines = np.stack([np.stack([drawn_rows[rows + 1], zero, drawn_columns[columns]], axis=-1),
                      np.stack([drawn_columns[columns + 1], drawn_rows[rows], zero], axis=-1)], axis=-2).reshape(-1, 3)
    grid_edges = np.tile([[1, 0, 1], [1, 1, 0]], (len(quads), 1))
    ink = np.where(boundary, boundary_darkness * grid_edges, edge_darkness * lines)
    return topology, ink


def landscape_extents(half: np.ndarray) -> np.ndarray:
    """Canvas aspect ratio `n_width / n_height` expands `half` (`[2]`) so neither axis clips."""
    aspect = n_width / n_height
    return np.array([max(half[0], half[1] * aspect), max(half[1], half[0] / aspect)])


def viewport(points: core.Point, view_elevation: float, view_azimuth: float, n_pixels: int) -> solids.View:
    """Frames an orthographic `View` (`n_pixels` wide) enclosing `points` at `view_elevation` and `view_azimuth`."""
    view = solids.view([solids.origin + solids.context.include_from_algebra(points)], 0.0, view_elevation, view_azimuth, margin,
                       n_pixels)
    return view._replace(half=landscape_extents(view.half))


def sheet_draw(points: core.Point, topology: Topology, ink: np.ndarray, colour: meshes.Colour,
               view: solids.View) -> Draw:
    """Samples `points` (`[n_rows, n_columns] Point-`) over `topology` as a two-sided `Draw` pass."""
    V, E, F = topology.n_elements
    del V, E, F
    places = solids.origin + solids.context.include_from_algebra(points.reshape(-1))  # [V] Point-
    return meshes.samples(places, topology.elements[2], np.zeros(3), meshes.smooth(places, topology), view, face_values=ink,
                          colour=colour, gloss=solids.gloss)


def saddle_image(points: core.Point, topology: Topology, ink: np.ndarray, view: solids.View, scale: float) -> np.ndarray:
    """Renders a single gridded sheet to an RGB image with line pixel width scaled by `scale`."""
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(sheet_draw(points, topology, ink, grid_colour(front_colour, back_colour, scale * edge_width), view), 1.0)])


def backlund_image(transformed: core.Point, source: core.Point, topology: Topology, ink: np.ndarray, view: solids.View) -> np.ndarray:
    """Composites translucent `source` and `transformed` sheets with tangent connector rods and occluded ghost pass."""
    n_rows, n_columns = source.shape
    row_margin, column_margin = n_rows // 6, n_columns // 6
    rows = np.linspace(row_margin, n_rows - row_margin - 1, n_connector_rows, dtype=int)
    columns = np.linspace(column_margin, n_columns - column_margin - 1, n_connector_columns, dtype=int)
    starts = solids.origin + solids.context.include_from_algebra(source[rows[:, None], columns].reshape(-1))       # [n_connectors] Point-
    ends = solids.origin + solids.context.include_from_algebra(transformed[rows[:, None], columns].reshape(-1))    # [n_connectors] Point-
    connectors = [(solids.rods(starts, ends, connector_width / 2 * view.pixel), connector_colour),
                  (solids.ball(starts, endpoint_radius * view.pixel)[:, None], connector_colour),
                  (solids.ball(ends, endpoint_radius * view.pixel)[:, None], connector_colour)]
    sheets = [(sheet_draw(source, topology, ink * pair_grid_ink, grid_colour(source_colour, source_colour, edge_width), view), surface_opacity),
              (sheet_draw(transformed, topology, ink * pair_grid_ink, grid_colour(backlund_colour, backlund_colour, edge_width), view), surface_opacity)]
    scene = solids.layer(connectors, view, inks=[], translucent=[], meshes=sheets)
    # Composite a faint foreground pass of the connectors alone so occluded portions show through the sheets.
    segments, depths = solids.layer(connectors, view, inks=[], translucent=[], meshes=[])
    del depths
    ghost_depth = np.where(segments[..., 3] > 0, -np.inf, 0.0)
    return compose([(*scene, 1.0), (segments, ghost_depth, hidden_opacity)], background)


def saddle_animation(reference: core.Point, states: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Yields RGB frames of each saddle in `states` framed against `reference`."""
    topology, ink = sheet_topology(*reference.shape[-2:], saddle_line_spacing)
    view = viewport(reference, elevation, azimuth, round(saddle_scale * n_width))
    for state in states:
        yield saddle_image(state, topology, ink, view, saddle_scale)


def backlund_orbit_animation(transformed: core.Point, source: core.Point, azimuths: np.ndarray,
                             orbit_elevation: float) -> Iterator[np.ndarray]:
    """Yields RGB frames orbiting `source` and `transformed` across degree `azimuths` at `orbit_elevation`."""
    topology, ink = sheet_topology(*source.shape, 1)
    V, E, F = topology.n_elements
    del V, E, F
    points = concatenate([source.reshape(-1), transformed.reshape(-1)])         # [2 * V] Point-
    view_elevation, view_azimuths = np.radians(orbit_elevation), np.radians(azimuths + 90)
    views = [viewport(points, view_elevation, view_azimuth, n_width) for view_azimuth in view_azimuths]
    half = np.max([view.half for view in views], axis=0)
    for view in views:
        yield backlund_image(transformed, source, topology, ink, view._replace(half=half))


def saddle_figure(points: core.Point, reference: core.Point) -> plt.Figure:
    """Wraps a single K-surface `points` framed by `reference` in a borderless `Figure`."""
    image, = saddle_animation(reference, [points])
    return image_figure(image)


def backlund_figure(transformed: core.Point, source: core.Point, reference: core.Point) -> plt.Figure:
    """Wraps `source` and `transformed` with their tangent connectors in a borderless `Figure`."""
    topology, ink = sheet_topology(*source.shape, 1)
    view = viewport(reference, elevation, azimuth, n_width)
    return image_figure(backlund_image(transformed, source, topology, ink, view))
