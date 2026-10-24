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

from numga import stack
from examples.surfaces.k_surfaces import core
from examples.util.wgsl import compose, image_figure, meshes, solids, vec3

n_width = 640
n_height = 480
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


def grid_colour(front: np.ndarray, back: np.ndarray) -> str:
    """WGSL colouring a sheet by the side the camera sees, darkened along the grid's edges: each face's ink at
    the edge opposite each corner, by how many pixels away that edge lies."""
    return f"""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {{
    let edges = clamp(1 - corner / ({edge_width} * oversampling * fwidth(corner)), vec3<f32>(0.0), vec3<f32>(1.0)) * face_values.xyz;
    let ink = max(edges.x, max(edges.y, edges.z));
    return vec4<f32>(select({vec3(back)}, {vec3(front)}, front) * (1 - ink), 1.0);
}}
"""


# --- plumbing
def coordinates(points: core.Point) -> np.ndarray:
    """Positions read by pairing with the planes across the axes."""
    mv = points.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()


def mesh(n_rows: int, n_columns: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangles and their edge ink at the edge opposite each corner, with the diagonal of each quad left
    unmarked."""
    indices = np.arange(n_rows * n_columns).reshape(n_rows, n_columns)
    quads = np.stack([indices[:-1, :-1], indices[1:, :-1],
                     indices[1:, 1:], indices[:-1, 1:]], axis=-1).reshape(-1, 4)
    faces = quads[:, [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)
    edge_vertices = np.sort(faces[:, [[1, 2], [2, 0], [0, 1]]], axis=-1)
    _, inverse, counts = np.unique(edge_vertices.reshape(-1, 2), axis=0,
                                  return_inverse=True, return_counts=True)
    boundary = counts[inverse].reshape(-1, 3) == 1
    grid_edges = np.tile([[1, 0, 1], [1, 1, 0]], (len(quads), 1))
    ink = np.where(boundary, boundary_darkness, edge_darkness) * grid_edges
    return faces, ink


def projective(points: np.ndarray) -> solids.Point:
    return solids.mv.point(points.reshape(-1, 3))


def landscape(half: np.ndarray) -> np.ndarray:
    """Half extents grown to the canvas's proportions."""
    aspect = n_width / n_height
    return np.array([max(half[0], half[1] * aspect), max(half[1], half[0] / aspect)])


def viewport(points: np.ndarray, view_elevation: float, view_azimuth: float) -> solids.View:
    """One orthographic view for an entire deformation."""
    view = solids.view([projective(points)], 0.0, view_elevation, view_azimuth, margin, n_width)
    return view._replace(half=landscape(view.half))


def sheet(points: np.ndarray, faces: np.ndarray, ink: np.ndarray, colour: str, view: solids.View) -> meshes.Samples:
    """One sheet, lit on both sides, at every sample, as `meshes.samples`."""
    places = projective(points)
    return meshes.samples(places, faces, np.zeros(3), meshes.smooth(places, faces), view, face_values=ink,
                          colour=colour, gloss=solids.gloss)


def picture(points: np.ndarray, faces: np.ndarray, ink: np.ndarray, view: solids.View) -> np.ndarray:
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(sheet(points, faces, ink, grid_colour(front_colour, back_colour), view), 1.0)])


def partner_picture(partner: np.ndarray, source: np.ndarray, faces: np.ndarray, ink: np.ndarray, view: solids.View) -> np.ndarray:
    """Two translucent sheets with common tangents and marked corresponding points."""
    n_rows, n_columns = source.shape[:2]
    row_margin, column_margin = n_rows // 6, n_columns // 6
    rows = np.linspace(row_margin, n_rows - row_margin - 1, n_connector_rows, dtype=int)
    columns = np.linspace(column_margin, n_columns - column_margin - 1, n_connector_columns, dtype=int)
    starts, ends = projective(source[rows[:, None], columns]), projective(partner[rows[:, None], columns])
    # Widths in pixels, as lengths in the scene.
    connectors = [(solids.rods(starts, ends, connector_width / 2 * view.pixel), connector_colour),
                  (solids.ball(starts, endpoint_radius * view.pixel)[:, None], connector_colour),
                  (solids.ball(ends, endpoint_radius * view.pixel)[:, None], connector_colour)]
    sheets = [(sheet(source, faces, ink * pair_grid_ink, grid_colour(source_colour, source_colour), view), surface_opacity),
              (sheet(partner, faces, ink * pair_grid_ink, grid_colour(partner_colour, partner_colour), view), surface_opacity)]
    scene = solids.layer(connectors, view, inks=[], translucent=[], meshes=sheets)
    # The segments' hidden parts show faintly: a copy of the segments alone, in front of everything.
    segments, _ = solids.layer(connectors, view, inks=[], translucent=[], meshes=[])
    ghost_depth = np.where(segments[..., 3] > 0, -np.inf, 0.0)
    return compose([(*scene, 1.0), (segments, ghost_depth, hidden_opacity)], background)


def frames(states: Iterable[core.Point], reference: core.Point) -> Iterator[np.ndarray]:
    """Draw each point grid with the camera fixed by a reference grid of the same rows and columns."""
    places = coordinates(reference)                                           # [..., rows, columns, 3]
    faces, ink = mesh(*places.shape[-3:-1])
    view = viewport(places, elevation, azimuth)
    for state in states:
        yield picture(coordinates(state), faces, ink, view)


def partner_frames(states: Iterable[core.Point], source: core.Point, reference: core.Point) -> list[np.ndarray]:
    """Draw a partner family against one source net and one fixed camera."""
    source = coordinates(source)
    faces, ink = mesh(*source.shape[:2])
    view = viewport(coordinates(reference), elevation, azimuth)
    return [partner_picture(coordinates(state), source, faces, ink, view) for state in states]


def partner_orbit(partner: core.Point, source: core.Point, azimuths: np.ndarray, orbit_elevation: float
                  ) -> Iterator[np.ndarray]:
    """Orbit a fixed pair, framed with one scale fitted to every view; azimuths and elevation in degrees, the
    azimuth measured from x."""
    source, partner = coordinates(source), coordinates(partner)
    faces, ink = mesh(*source.shape[:2])
    points = np.concatenate([source.reshape(-1, 3), partner.reshape(-1, 3)])
    view_elevation, view_azimuths = np.radians(orbit_elevation), np.radians(azimuths + 90)
    views = [viewport(points, view_elevation, view_azimuth) for view_azimuth in view_azimuths]
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
