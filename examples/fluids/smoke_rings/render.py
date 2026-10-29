"""Drawing for the smoke ring flow, on the GPU: the Hashimoto surface swept by a curve, as shaded strips between the
curve at successive times; the tubes swept by several rings, cut open to show how they pass through each other;
curves flowing, as strokes coloured by how fast each part moves; and the turning angle along a curve through time.
Curves are polylines; a closed one repeats its first vertex at the end."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.fluids.smoke_rings import core
from examples.util import charts
from examples.util.wgsl import image_figure, meshes, solids, strokes

# The surface's colour from the first time to the last, and the colours of the rings' swept tubes, one per ring; the
# curves' colours by speed, from blue at rest to green at the fastest, the stretch of the map light enough to show its
# shading and short of its pale end.
early, late = np.array([0.93, 0.88, 0.78]), np.array([0.55, 0.69, 0.80])
speed_map, rest_share, fastest_share = "viridis", 0.3, 0.78
tubes = np.array([[0.85, 0.52, 0.32], [0.38, 0.58, 0.78]])
background = np.ones(3)
# The view, elevation and azimuth in radians, azimuth zero looking along +y, and the margin around the subject as
# a share of its larger side; the pictures' pixels across, and the gloss of the swept surfaces.
elevation, azimuth = np.radians([50.0, 40.0])
margin = 0.05
n_picture_pixels = 330
n_frame_pixels = 250
surface_gloss = 0.1
# A curve's width in pixels.
curve_width = 6.0
# The share of the largest motion above which a vertex counts as moving, for framing a curve's trail, and
# the trail's opacity, so that the curve shows through it.
moving_share = 0.01
trail_opacity = 0.6
sm = solids.mv
# The planes x = 0, y = 0 and z = 0 of the example's space, the drawing frame.
axes = stack((core.mv.x, core.mv.y, core.mv.z))                              # [3] Vector+


# --- plumbing -------------------------------------------------------------------------
def gpu_points(points: core.Point) -> solids.Point:
    """The GPU library's own points at the places of the example's: their coordinates, read by pairing the intrinsic
    points with the drawing frame, are the one hand-off from this algebra to the renderer's."""
    return sm.point((axes & points[..., None]).to_array())


def grid(points: solids.Point) -> np.ndarray:
    """The index of each of a batch of points, `[..., times, n]`, among them all, in their own order."""
    return np.arange(np.prod(points.shape)).reshape(points.shape)


def strips(index: np.ndarray) -> np.ndarray:
    """The triangles between successive curves, their vertices' indices `[..., times, n]`: two to each quad between an
    edge at one time and the same edge at the next, `[..., times - 1, n - 1, 2, 3]` indices."""
    corners = np.stack([index[..., :-1, :-1], index[..., 1:, :-1], index[..., 1:, 1:], index[..., :-1, 1:]], axis=-1)
    return corners[..., [[0, 1, 2], [0, 2, 3]]]                                 # [..., times - 1, n - 1, 2, 3]


def framed(points: solids.Point, pixels: int) -> tuple[solids.View, solids.Point]:
    """A view framing points, `pixels` across, its height fitted to whole rows, and the direction toward the
    viewer."""
    camera, half, _ = solids.view([points.reshape(-1)], 0.0, elevation, azimuth, margin, pixels)
    half = half * np.array([1, round(pixels * half[1] / half[0]) / pixels * half[0] / half[1]])
    return solids.View(camera, half, pixels), camera >> sm.z.dual()


def sheet(points: solids.Point, faces: np.ndarray, colours: np.ndarray,
          view: solids.View) -> meshes.Samples:
    """Triangles over vertices, coloured at each corner `[faces, 3, 3]`, smoothly lit, at every sample of the view."""
    return meshes.samples(points, faces, colours, meshes.smooth(points, faces), view, face_values=np.zeros(4),
                          colour=meshes.plain, gloss=surface_gloss)


def curve_strokes(points: solids.Point, speeds: np.ndarray, fastest: float, view: solids.View) -> meshes.Samples:
    """Polylines `[curves, n]` as strokes, each segment coloured by the mean speed at its ends, from the speeds at
    the vertices `[curves, n]`, against the fastest, at every sample of the view."""
    middles = (speeds[..., :-1] + speeds[..., 1:]) / 2                         # [curves, n - 1]
    shades = plt.get_cmap(speed_map)(rest_share + (fastest_share - rest_share) * middles / fastest)[..., :3]   # [curves, n - 1, 3]
    return strokes.segments(points[:, :-1].reshape(-1), points[:, 1:].reshape(-1), shades.reshape(-1, 3), curve_width,
                            view)


def draw_surface(curves: core.Point) -> plt.Figure:
    """The surface swept by a curve through time `[times, n]`, coloured from early to late."""
    points = gpu_points(curves)                                                # [times, n] Point-
    n_times, n_vertices = points.shape
    faces = strips(grid(points)).reshape(-1, 3)
    share = np.linspace(0, 1, n_times)[:, None, None] * np.ones((n_times, n_vertices, 1))
    colours = ((1 - share) * early + share * late).reshape(-1, 3)[faces]       # [faces, 3, 3]
    view, _ = framed(points, n_picture_pixels)
    surface = sheet(points.reshape(-1), faces, colours, view)
    return image_figure(solids.draw([], view, background, inks=[], translucent=[], meshes=[(surface, 1.0)]))


def draw_swept(curves: core.Point) -> plt.Figure:
    """The tubes swept by closed curves through time `[times, rings, n]`, each in its own colour, with the half of
    every tube nearer the viewer cut away, so that the tubes inside show."""
    points = gpu_points(curves)                                                # [times, rings, n] Point-
    view, toward = framed(points, n_picture_pixels)
    # The middle of each quad, between an edge's start at one time and its end at the next.
    middles = gpu_points((curves[:-1, :, :-1] + curves[1:, :, 1:]) / 2)        # [times - 1, rings, n - 1] Point-
    # Cut by the upright plane through the axis facing the viewer: keep the far half of each tube.
    level_toward = toward - sm.z.dual() * (sm.z & toward)                      # [] Point-
    cut = -level_toward.dual()                                                 # [] Plane+
    kept = np.moveaxis((cut & middles) < 0, 1, 0)                              # [rings, times - 1, n - 1]
    # Each ring's strips, ring after ring, its vertices among all the points.
    faces = strips(np.moveaxis(grid(points), 1, 0))[kept].reshape(-1, 3)       # [faces, 3]
    colours = np.broadcast_to(tubes[:, None, None, None, None, :], kept.shape + (2, 3, 3))[kept].reshape(-1, 3, 3)
    swept = sheet(points.reshape(-1), faces, colours, view)
    return image_figure(solids.draw([], view, background, inks=[], translucent=[], meshes=[(swept, 1.0)]))


def animate_curves(curves: core.Point, speeds: core.Scalar) -> Iterator[np.ndarray]:
    """Frames of curves flowing `[times, ..., n]`, coloured by their speeds `[times, ..., n]`, all from one view and to
    one scale."""
    points, rates = gpu_points(curves), speeds.to_array()                      # [times, ..., n] Point-, [times, ..., n]
    view, _ = framed(points, n_frame_pixels)
    n_vertices = points.shape[-1]
    for time in range(len(rates)):
        flowing = curve_strokes(points[time].reshape(-1, n_vertices), rates[time].reshape(-1, n_vertices),
                                rates.max(), view)
        yield solids.draw([], view, background, inks=[], translucent=[], meshes=[(flowing, 1.0)])


def animate_trail(curves: core.Point, speeds: core.Scalar) -> Iterator[np.ndarray]:
    """Frames of a curve flowing `[times, n]`, coloured by its speeds `[times, n]`, each over the surface the curve has
    swept so far, coloured from early to late, and framed on the part of space where the curve moves."""
    points, rates = gpu_points(curves), speeds.to_array()                      # [times, n] Point-, [times, n]
    # The vertices that move: how far each travels from where it starts, at the most.
    travel = (curves - curves[0]).norm().to_array().max(axis=0)               # [n]
    moving = points[:, travel > moving_share * travel.max()]                   # [times, moving] Point-
    # The box around their paths, with the margin, as the planes of its faces; what lies outside is left out.
    extent = (solids.axes & moving.reshape(-1)[:, None]).to_array()            # [moving points, 3]
    low, high = extent.min(axis=0), extent.max(axis=0)
    middle = solids.origin + (solids.axes.dual() * ((low + high) / 2)).sum()    # [] Point-
    box = solids.boxes(solids.translator(middle - solids.origin), (0.5 + margin) * (high - low))   # [6] Plane+
    inside = ((box & points[..., None]) <= 0).all(axis=-1)                     # [times, n]
    view, _ = framed(moving, n_frame_pixels)
    faces = strips(grid(points))                                               # [times - 1, n - 1, 2, 3]
    flat = inside.reshape(-1)
    kept = flat[faces].all(axis=-1)                                            # [times - 1, n - 1, 2]
    n_times = len(rates)
    share = np.linspace(0, 1, n_times)[:, None, None] * np.ones(rates.shape + (1,))
    colours = ((1 - share) * early + share * late).reshape(-1, 3)
    for time in range(n_times):
        swept = faces[:time][kept[:time]]                                      # [faces, 3]
        shown = [(curve_strokes(points[time][inside[time]][None], rates[time][inside[time]][None], rates.max(),
                                view), 1.0)]
        if len(swept):
            shown.append((sheet(points.reshape(-1), swept, colours[swept], view), trail_opacity))
        yield solids.draw([], view, background, inks=[], translucent=[], meshes=shown)


@charts.style()
def draw_turns(turns: core.Rotor) -> plt.Figure:
    """The turning angle at every inner vertex of a curve through time `[times, n]`: vertices across,
    times down, the angle as darkness."""
    # The angle of each turn: the size of its logarithm.
    angles = turns.log().norm().to_array()                                     # [times, n]
    # Half the column: a field of cells with two names and no numbers.
    figure, ax = charts.subplots(400, 270, layout="constrained")
    # Each vertex and time a cell of its own, drawn as the cell it is.
    ax.pcolormesh(np.arange(angles.shape[1] + 1) + 0.5, np.arange(angles.shape[0] + 1) - 0.5, angles, cmap="Greys")
    ax.invert_yaxis()
    ax.set(xlabel="vertex", ylabel="time", xticks=[], yticks=[])
    ax.spines[:].set_visible(False)
    return figure
