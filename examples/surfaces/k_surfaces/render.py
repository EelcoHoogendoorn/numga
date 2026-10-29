"""Depth-tested views of saddle surfaces and their tangent partners, on the GPU.

A surface is a mesh drawn by `meshes`, its grid lines inked by its colour function from each face's edges and
how near a pixel lies to them; a pair of partners are two translucent sheets, with the segments between
corresponding points as rods traced by `solids`, all composited by depth sample by sample. The segments' hidden
parts show faintly through the sheets.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from examples import instantiate
from examples.surfaces.k_surfaces import core
from examples.util.wgsl import compose, image_figure, meshes, solids

Mesh = instantiate("examples.util.mesh", core.ga).Mesh

# The pictures' pixels across and up; the bending saddle's at a share of them per side, its widths in pixels scaled
# with it, and drawn with every so many of the grid's lines.
n_width = 640
n_height = 480
saddle_scale = 0.5
saddle_line_spacing = 2
elevation, azimuth = np.radians(15.0), np.radians(75.0)
margin = 0.1
front_colour = np.array([0.29, 0.61, 0.67])
back_colour = np.array([0.79, 0.48, 0.32])
background = np.ones(3)
# The grid lines' width in pixels, their darkness inside and along the boundary, and their share on a pair's sheets.
edge_width = 0.8
edge_darkness = 0.28
boundary_darkness = 0.6
pair_grid_ink = 0.6
source_colour = np.array([0.28, 0.58, 0.64])
partner_colour = np.array([0.78, 0.48, 0.28])
surface_opacity = 0.78
# The segments between partners: their colour, width and end dots' radius in pixels, the opacity of their hidden
# parts, and the rows and columns of the grid they join.
connector_colour = np.array([0.17, 0.17, 0.20])
connector_width = 1.8
endpoint_radius = 2.5
hidden_opacity = 0.35
n_connector_rows = 3
n_connector_columns = 3


# WGSL colouring a sheet by the side the camera sees, darkened along the grid's edges: each face's ink at the edge
# opposite each corner, by how many pixels away that edge lies.
gridded = """
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let edges = clamp(1 - corner / (edge_width * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    let ink = max(edges.x, max(edges.y, edges.z));
    return vec4<f32>(select(back_colour, front_colour, front) * (1 - ink), 1.0);
}
"""


def grid_colour(front: np.ndarray, back: np.ndarray, width: float) -> meshes.Colour:
    """A sheet coloured `front` on the side its normal points to and `back` on the other, gridded with lines `width`
    pixels wide."""
    return meshes.Colour(gridded, dict(edge_width=width, front_colour=front, back_colour=back))


# --- plumbing
def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the example's, flattened: their coordinates, read by pairing the
    intrinsic points with the drawing frame, are the one hand-off from this algebra to the renderer's."""
    mv = points.context.multivector
    return solids.mv.point((stack((mv.x, mv.y, mv.z)) & points.reshape(-1)[..., None]).to_array())


def mesh(n_rows: int, n_columns: int, spacing: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangles and their edge ink at the edge opposite each corner, with the diagonal of each quad left
    unmarked, and of the grid's lines inside every `spacing`-th row and column."""
    indices = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    quads = np.stack([indices[:-1, :-1], indices[1:, :-1],
                     indices[1:, 1:], indices[:-1, 1:]], axis=-1).reshape(-1, 4)
    grid = Mesh(n_rows * n_columns, quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3))
    # Of each face, whether the edge opposite each corner lies on the grid's boundary.
    boundary = np.isin(grid.face_edges, grid.boundary_edges)                    # [faces, corners]
    # Of each quad's two triangles, the edges opposite their corners: the next row and this column, then the next
    # column and this row, each drawn where its line is.
    rows, columns = np.meshgrid(np.arange(n_rows - 1), np.arange(n_columns - 1), indexing="ij")
    drawn_rows, drawn_columns = np.arange(n_rows) % spacing == 0, np.arange(n_columns) % spacing == 0
    zero = np.zeros(rows.shape, dtype=bool)
    lines = np.stack([np.stack([drawn_rows[rows + 1], zero, drawn_columns[columns]], axis=-1),
                      np.stack([drawn_columns[columns + 1], drawn_rows[rows], zero], axis=-1)], axis=-2).reshape(-1, 3)
    grid_edges = np.tile([[1, 0, 1], [1, 1, 0]], (len(quads), 1))
    ink = np.where(boundary, boundary_darkness * grid_edges, edge_darkness * lines)
    return grid.faces, ink


def landscape(half: np.ndarray) -> np.ndarray:
    """Half extents grown to the canvas's proportions."""
    aspect = n_width / n_height
    return np.array([max(half[0], half[1] * aspect), max(half[1], half[0] / aspect)])


def viewport(points: core.Point, view_elevation: float, view_azimuth: float, n_pixels: int) -> solids.View:
    """One orthographic view for an entire deformation, `n_pixels` across."""
    view = solids.view([gpu_points(points)], 0.0, view_elevation, view_azimuth, margin, n_pixels)
    return view._replace(half=landscape(view.half))


def sheet(points: core.Point, faces: np.ndarray, ink: np.ndarray, colour: meshes.Colour,
          view: solids.View) -> meshes.Samples:
    """One sheet `[rows, columns] Point-`, lit on both sides, at every sample, as `meshes.samples`."""
    places = gpu_points(points)
    return meshes.samples(places, faces, np.zeros(3), meshes.smooth(places, faces), view, face_values=ink,
                          colour=colour, gloss=solids.gloss)


def picture(points: core.Point, faces: np.ndarray, ink: np.ndarray, view: solids.View, scale: float) -> np.ndarray:
    """One gridded sheet, its lines' width in pixels scaled by `scale`."""
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(sheet(points, faces, ink, grid_colour(front_colour, back_colour, scale * edge_width), view), 1.0)])


def partner_picture(partner: core.Point, source: core.Point, faces: np.ndarray, ink: np.ndarray, view: solids.View) -> np.ndarray:
    """Two translucent sheets with common tangents and marked corresponding points."""
    n_rows, n_columns = source.shape
    row_margin, column_margin = n_rows // 6, n_columns // 6
    rows = np.linspace(row_margin, n_rows - row_margin - 1, n_connector_rows, dtype=int)
    columns = np.linspace(column_margin, n_columns - column_margin - 1, n_connector_columns, dtype=int)
    starts, ends = gpu_points(source[rows[:, None], columns]), gpu_points(partner[rows[:, None], columns])
    # Widths in pixels, as lengths in the scene.
    connectors = [(solids.rods(starts, ends, connector_width / 2 * view.pixel), connector_colour),
                  (solids.ball(starts, endpoint_radius * view.pixel)[:, None], connector_colour),
                  (solids.ball(ends, endpoint_radius * view.pixel)[:, None], connector_colour)]
    sheets = [(sheet(source, faces, ink * pair_grid_ink, grid_colour(source_colour, source_colour, edge_width), view), surface_opacity),
              (sheet(partner, faces, ink * pair_grid_ink, grid_colour(partner_colour, partner_colour, edge_width), view), surface_opacity)]
    scene = solids.layer(connectors, view, inks=[], translucent=[], meshes=sheets)
    # The segments' hidden parts show faintly: a copy of the segments alone, in front of everything.
    segments, _ = solids.layer(connectors, view, inks=[], translucent=[], meshes=[])
    ghost_depth = np.where(segments[..., 3] > 0, -np.inf, 0.0)
    return compose([(*scene, 1.0), (segments, ghost_depth, hidden_opacity)], background)


def frames(states: Iterable[core.Point], reference: core.Point) -> Iterator[np.ndarray]:
    """Draw each point grid with the camera fixed by a reference grid of the same rows and columns, at the saddle's
    share of the pixels and of its grid's lines."""
    faces, ink = mesh(*reference.shape[-2:], saddle_line_spacing)
    view = viewport(reference, elevation, azimuth, round(saddle_scale * n_width))
    for state in states:
        yield picture(state, faces, ink, view, saddle_scale)


def partner_frames(states: Iterable[core.Point], source: core.Point, reference: core.Point) -> list[np.ndarray]:
    """Draw a partner family against one source net and one fixed camera."""
    faces, ink = mesh(*source.shape, 1)
    view = viewport(reference, elevation, azimuth, n_width)
    return [partner_picture(state, source, faces, ink, view) for state in states]


def partner_orbit(partner: core.Point, source: core.Point, azimuths: np.ndarray, orbit_elevation: float
                  ) -> Iterator[np.ndarray]:
    """Orbit a fixed pair, framed with one scale fitted to every view; azimuths and elevation in degrees, the
    azimuth measured from x."""
    faces, ink = mesh(*source.shape, 1)
    points = concatenate([source.reshape(-1), partner.reshape(-1)])            # [vertices] Point-
    view_elevation, view_azimuths = np.radians(orbit_elevation), np.radians(azimuths + 90)
    views = [viewport(points, view_elevation, view_azimuth, n_width) for view_azimuth in view_azimuths]
    half = np.max([view.half for view in views], axis=0)
    for view in views:
        yield partner_picture(partner, source, faces, ink, view._replace(half=half))


def draw(points: core.Point, reference: core.Point) -> plt.Figure:
    """One surface filling its canvas, without axes or margins."""
    image, = frames([points], reference)
    return image_figure(image)


def draw_partner(partner: core.Point, source: core.Point, reference: core.Point) -> plt.Figure:
    """One partner with its source surface and common tangent segments."""
    image, = partner_frames([partner], source, reference)
    return image_figure(image)
