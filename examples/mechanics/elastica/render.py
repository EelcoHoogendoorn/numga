"""Planar and three-dimensional rendering for static elastica curves, twist ribbons, and the dynamic telephone cord."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import stack
from numga.render import Camera, drawing, meshes, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology
from examples.mechanics.elastica import core
from examples.util.animation import capture
from examples.util.panels import image_figure, overlay, panel

mv = core.mv
rod_cmap = "viridis"
rod_colour = np.array(to_rgb("#1f6f99"))
thrust_colour = np.array(to_rgb("0.55"))
ribbon_colour = np.array(to_rgb("#7cc4e4"))
pin_colour = np.array(to_rgb("0.2"))
clamp_colour = np.full(3, 0.2)
background = np.ones(3)
margin = 0.08
helix_spacing = 1.4
helix_view = np.radians([12.0, 10.0])
uncoil_elevation = np.radians(16.0)
uncoil_azimuths = np.radians([-30.0, 150.0])
cord_view = np.radians([20.0, 26.0])
n_helix_pixels, n_uncoil_pixels, n_cord_pixels = 330, 160, 675
rod_width, clamp_radius = 3.0, 5.0
plane_rod_width, plane_clamp_width, thrust_width, pin_radius = 1.7, 3.0, 1.0, 3.0
n_rim_points = 4


# --- plumbing -------------------------------------------------------------------------
def extent(points: core.Point) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate pairings with the z and y planes bound the points in the bending plane (`[2]` low and `[2]` high)."""
    flat_points = points.reshape(-1)                                                    # [...] Point-
    readings = (stack((mv.z, mv.y))[:, None] & flat_points) / (mv.w & flat_points)      # [2, ...] Scalar+
    return readings.min(axis=-1).to_array(), readings.max(axis=-1).to_array()


def frame_plane(ax: plt.Axes, low: np.ndarray, high: np.ndarray) -> Camera:
    """Frames `ax` around `low` and `high` and pairs points in the bending plane against the z, y, and w planes."""
    ax.set(xlim=(low[0] - margin, high[0] + margin), ylim=(low[1] - margin, high[1] + margin), aspect="equal")
    ax.set_axis_off()
    return panel(ax, mv.z, mv.y, mv.w)


def draw_curls(centres: core.Point) -> plt.Figure:
    """Draws the strip curves (`[n_curls, n_points] Point-`) from straight to a double ring beside their clamp."""
    n_curls, n_points = centres.shape
    colours = plt.get_cmap(rod_cmap)(np.linspace(0.05, 0.9, n_curls))[:, :3]
    figure, ax = plt.subplots(figsize=(7.5, 4.0), dpi=55)
    ax.set_axis_off()
    figure.tight_layout()
    camera = frame_plane(ax, *extent(centres))
    # The clamp spans the full vertical extent along z = 0 where the strip starts.
    clamp = mv.point([[0, height, 0] for height in ax.get_ylim()])                      # [2] Point-
    overlay(ax, [
        simplices.strokes(
            centres.reshape(-1), Topology.chains([n_points] * n_curls),
            np.repeat(colours, n_points - 1, axis=0), plane_rod_width, camera,
        ),
        simplices.strokes(clamp, Topology.chains([2]), pin_colour, plane_clamp_width, camera, cap=simplices.Cap.BUTT),
    ])
    return figure


def rod_pictures(curve: core.Point, ax: plt.Axes, camera: Camera) -> list[np.ndarray]:
    """Renders the line of thrust, a single elastica curve, and its two end pins into overlay layers."""
    thrust = mv.point([[0, 0, z_coord] for z_coord in ax.get_xlim()])                   # [2] Point-
    return [
        simplices.strokes(thrust, Topology.chains([2]), thrust_colour, thrust_width, camera),
        simplices.strokes(curve, Topology.chains([len(curve)]), rod_colour, plane_rod_width, camera),
        simplices.markers(curve[[0, -1]], pin_colour, pin_radius, camera),
    ]


def animate_sweep(centres: core.Point) -> Iterator[np.ndarray]:
    """Yields RGB frames of Euler's elastica (`[n_frames, n_points] Point-`) across the sweep of departure angles."""
    low, high = extent(centres)
    aspect_height = 5.0 * (high[1] - low[1] + 2 * margin) / (high[0] - low[0] + 2 * margin)
    figure, ax = plt.subplots(figsize=(5.0, aspect_height), dpi=60)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    camera = frame_plane(ax, low, high)
    artist = overlay(ax, rod_pictures(centres[0], ax, camera))
    for curve in centres:
        artist.set_data(drawing.composited(rod_pictures(curve, ax, camera)))
        yield capture(figure)
    plt.close(figure)


def draw_helices(centres: core.Point, edges: core.Point) -> plt.Figure:
    """Places Kirchhoff's helices side by side in space with ribbons (`[n_tilts, n_points, 2] Point-`) showing cross-section twist."""
    n_tilts = centres.shape[0]
    readings = stack((mv.x, mv.y))[None, :, None] & edges.reshape(n_tilts, -1)[:, None]  # [n_tilts, 2, ...] Scalar+
    spans = readings.max(axis=-1) - readings.min(axis=-1)                                # [n_tilts, 2] Scalar+
    step_x = helix_spacing * spans.max()                                                 # [] Scalar+
    offsets = mv.x.dual() * (step_x * np.arange(n_tilts))                                # [n_tilts] Point-
    curves, ribbons = centres + offsets[:, None], edges + offsets[:, None, None]
    view = solids.view([ribbons], 0.0, *helix_view, margin, n_helix_pixels)
    return image_figure(scene(curves, ribbons, view))


def animate_uncoiling(centres: core.Point, edges: core.Point) -> Iterator[np.ndarray]:
    """Yields RGB frames as the wire sweeps between ring and twisted rod while the camera orbits a fixed bounding cylinder."""
    n_frames = centres.shape[0]
    places = edges.reshape(-1)                                                          # [...] Point-
    planes = stack((mv.x, mv.y, mv.z))                                                  # [3] Plane+
    readings = planes[:, None] & places                                                 # [3, ...] Scalar+
    low = planes & places[readings.argmin(axis=-1)]                                     # [3] Scalar+
    high = planes & places[readings.argmax(axis=-1)]                                    # [3] Scalar+
    # A vertical cylinder around the overall midpoint bounds every frame symmetrically across all camera azimuths.
    middle = mv.w.dual() + (planes.dual() * ((low + high) / 2)).sum()                   # [] Point-
    axis = (middle & mv.z.dual()).normalized()                                          # [] Forque-
    radii = (axis & edges).norm().reshape(-1)                                           # [...] Scalar+
    radius = radii.max()                                                                # [] Scalar+
    half_height = (high - low)[2] / 2                                                   # [] Scalar+
    rims = middle + mv.z.dual() * (half_height * np.array([-1.0, 1.0]))                 # [2] Point-
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_rim_points, endpoint=False) / 2)).exp()  # [n_rim_points] Motor+
    circles = rims[:, None] + (turns >> mv.x.dual() * radius)                           # [2, n_rim_points] Point-
    framing = solids.view([circles], 0.0, uncoil_elevation, 0.0, margin, n_uncoil_pixels)
    azimuths = np.linspace(*uncoil_azimuths, n_frames, endpoint=False)
    for curve, ribbon, azimuth in zip(centres, edges, azimuths):
        yield scene(curve[None], ribbon[None], framing._replace(camera=solids.aim(middle, uncoil_elevation, azimuth)))


def animate_cord(centres: core.Point, edges: core.Point) -> Iterator[np.ndarray]:
    """Yields RGB frames of the telephone cord rendered `n_cord_pixels` wide."""
    return sized_cord(centres, edges, n_cord_pixels)


def sized_cord(centres: core.Point, edges: core.Point, n_pixels: int) -> Iterator[np.ndarray]:
    """Yields RGB frames of the telephone cord and its end clamps (`[n_frames, n_samples] Point-`) rendered `n_pixels` wide."""
    n_frames = centres.shape[0]
    view = solids.view([edges], 0.0, *cord_view, margin, n_pixels)
    clamps = solids.ball(centres[:, [0, -1]], clamp_radius * view.pixel)[..., None]     # [n_frames, 2, 1] Plane+ <- Point-
    for curve, ribbon, ends in zip(centres, edges, clamps):
        yield solids.draw(
            [(ends, clamp_colour)], view, background, inks=[], translucent=[],
            meshes=wires(curve[None], ribbon[None], view),
        )


def wires(curves: core.Point, ribbons: core.Point, view: solids.View) -> list[tuple[Draw, float]]:
    """Assembles opaque triangulated ribbon strips and cylindrical rod tubes for a batch of curves."""
    n_rods, n_samples = curves.shape
    index = np.arange(np.prod(ribbons.shape)).reshape(ribbons.shape)                    # [n_rods, n_samples, 2]
    corners = np.stack([index[:, :-1, 0], index[:, 1:, 0], index[:, 1:, 1], index[:, :-1, 1]], axis=-1)
    faces = corners[..., [[0, 1, 2], [0, 2, 3]]].reshape(-1, 3)                         # [F, 3]
    vertices = ribbons.reshape(-1)                                                      # [V] Point-
    V = len(vertices)
    strips = meshes.samples(
        vertices, faces, ribbon_colour, meshes.smooth(vertices, Topology(V, faces)), view,
        face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss,
    )
    rods = simplices.tubes(
        curves.reshape(-1), Topology.chains([n_samples] * n_rods), rod_colour,
        rod_width, view, gloss=solids.gloss,
    )
    return [(strips, 1.0), (rods, 1.0)]


def scene(curves: core.Point, ribbons: core.Point, view: solids.View) -> np.ndarray:
    """Renders rod tubes and twist ribbons against a white background (`[n_rows, n_cols, 3]`)."""
    return solids.draw([], view, background, inks=[], translucent=[], meshes=wires(curves, ribbons, view))
