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
n_picture_pixels = 660
n_frame_pixels = 500
surface_gloss = 0.1
# A curve's width in pixels.
curve_width = 12.0
# The share of the largest motion above which a vertex counts as moving, for framing a curve's trail, and
# the trail's opacity, so that the curve shows through it.
moving_share = 0.01
trail_opacity = 0.6


# --- plumbing -------------------------------------------------------------------------
def coordinates(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def places(points: np.ndarray) -> solids.Point:
    """Coordinates `[..., 3]` as the drawing's points."""
    return solids.mv.point(points)


def strips(points: np.ndarray) -> np.ndarray:
    """The triangles between successive curves `[times, n]`: two to each quad between an edge at one time and the
    same edge at the next, `[times - 1, n - 1, 2, 3]` indices into the curves' vertices, flattened."""
    index = np.arange(points.shape[0] * points.shape[1]).reshape(points.shape[:2])
    corners = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)   # [times - 1, n - 1, 4]
    return corners[..., [[0, 1, 2], [0, 2, 3]]]


def framed(points: np.ndarray, pixels: int) -> tuple[solids.View, np.ndarray]:
    """A view framing points `[..., 3]`, `pixels` across, its height fitted to whole rows, and the direction toward
    the viewer `[3]`."""
    camera, half, _ = solids.view([places(points.reshape(-1, 3))], 0.0, elevation, azimuth, margin, pixels)
    half = half * np.array([1, round(pixels * half[1] / half[0]) / pixels * half[0] / half[1]])
    toward = (stack((solids.mv.x, solids.mv.y, solids.mv.z)) & (camera >> solids.mv.z.dual())).to_array()
    return solids.View(camera, half, pixels), toward


def sheet(points: np.ndarray, faces: np.ndarray, colours: np.ndarray,
          view: solids.View) -> meshes.Samples:
    """Triangles over vertices `[vertices, 3]`, coloured at each corner `[faces, 3, 3]`, smoothly lit, at every sample
    of the view."""
    corners = places(points)
    return meshes.samples(corners, faces, colours, meshes.smooth(corners, faces), view, gloss=surface_gloss)


def curve_strokes(points: np.ndarray, speeds: np.ndarray, fastest: float, view: solids.View) -> meshes.Samples:
    """Polylines `[curves, n, 3]` as strokes, each segment coloured by the mean speed at its ends, from the speeds at
    the vertices `[curves, n]`, against the fastest, at every sample of the view."""
    middles = (speeds[..., :-1] + speeds[..., 1:]) / 2                         # [curves, n - 1]
    shades = plt.get_cmap(speed_map)(rest_share + (fastest_share - rest_share) * middles / fastest)[..., :3]   # [curves, n - 1, 3]
    return strokes.segments(points[:, :-1].reshape(-1, 3), points[:, 1:].reshape(-1, 3), shades.reshape(-1, 3),
                            curve_width, view)


def draw_surface(curves: core.Point) -> plt.Figure:
    """The surface swept by a curve through time `[times, n]`, coloured from early to late."""
    points = coordinates(curves)                                               # [times, n, 3]
    faces = strips(points).reshape(-1, 3)
    share = np.linspace(0, 1, len(points))[:, None, None] * np.ones(points.shape[:2] + (1,))
    colours = ((1 - share) * early + share * late).reshape(-1, 3)[faces]       # [faces, 3, 3]
    view, _ = framed(points, n_picture_pixels)
    surface = sheet(points.reshape(-1, 3), faces, colours, view)
    return image_figure(solids.draw([], view, background, meshes=[(surface, 1.0)]))


def draw_swept(curves: core.Point) -> plt.Figure:
    """The tubes swept by closed curves through time `[times, rings, n]`, each in its own colour, with the half of
    every tube nearer the viewer cut away, so that the tubes inside show."""
    points = coordinates(curves)                                               # [times, rings, n, 3]
    view, toward = framed(points, n_picture_pixels)
    rings = np.moveaxis(points, 1, 0)                                          # [rings, times, n, 3]
    # Cut by the plane through the axis facing the viewer: keep the far half of each tube.
    centres = (rings[:, :-1, :-1] + rings[:, 1:, 1:]) / 2                      # [rings, times - 1, n - 1, 3]
    kept = (centres[..., :2] @ toward[:2] < 0)                                 # [rings, times - 1, n - 1]
    offsets = np.arange(len(rings))[:, None, None, None, None] * np.prod(rings.shape[1:3])
    faces = (strips(rings[0]) + offsets)[kept].reshape(-1, 3)                  # [faces, 3]
    colours = np.broadcast_to(tubes[:, None, None, None, None, :], kept.shape + (2, 3, 3))[kept].reshape(-1, 3, 3)
    swept = sheet(rings.reshape(-1, 3), faces, colours, view)
    return image_figure(solids.draw([], view, background, meshes=[(swept, 1.0)]))


def animate_curves(curves: core.Point, speeds: core.Scalar) -> Iterator[np.ndarray]:
    """Frames of curves flowing `[times, ..., n]`, coloured by their speeds `[times, ..., n]`, all from one view and to
    one scale."""
    points, rates = coordinates(curves), speeds.to_array()                     # [times, ..., n, 3], [times, ..., n]
    view, _ = framed(points, n_frame_pixels)
    for curve, rate in zip(points, rates):
        flowing = curve_strokes(curve.reshape(-1, *points.shape[-2:]), rate.reshape(-1, points.shape[-2]), rates.max(), view)
        yield solids.draw([], view, background, meshes=[(flowing, 1.0)])


def animate_trail(curves: core.Point, speeds: core.Scalar) -> Iterator[np.ndarray]:
    """Frames of a curve flowing `[times, n]`, coloured by its speeds `[times, n]`, each over the surface the curve has
    swept so far, coloured from early to late, and framed on the part of space where the curve moves."""
    points, rates = coordinates(curves), speeds.to_array()                     # [times, n, 3], [times, n]
    # The vertices that move, and the box around their paths, with the margin; what lies outside is left out.
    travel = np.linalg.norm(points - points[0], axis=-1).max(axis=0)          # [n]
    moving = points[:, travel > moving_share * travel.max()]                   # [times, moving, 3]
    low, high = moving.reshape(-1, 3).min(axis=0), moving.reshape(-1, 3).max(axis=0)
    inside = np.all((points >= low - margin * (high - low)) & (points <= high + margin * (high - low)), axis=-1)
    view, _ = framed(moving, n_frame_pixels)
    faces = strips(points)                                                     # [times - 1, n - 1, 2, 3]
    flat = inside.reshape(-1)
    kept = flat[faces].all(axis=-1)                                            # [times - 1, n - 1, 2]
    share = np.linspace(0, 1, len(points))[:, None, None] * np.ones(points.shape[:2] + (1,))
    colours = ((1 - share) * early + share * late).reshape(-1, 3)
    for time in range(len(points)):
        swept = faces[:time][kept[:time]]                                      # [faces, 3]
        shown = [(curve_strokes(points[time][inside[time]][None], rates[time][inside[time]][None], rates.max(), view), 1.0)]
        if len(swept):
            shown.append((sheet(points.reshape(-1, 3), swept, colours[swept], view), trail_opacity))
        yield solids.draw([], view, background, meshes=shown)


def draw_turns(turns: core.Rotor) -> plt.Figure:
    """The turning angle at every inner vertex of a curve through time `[times, n]`: vertices across,
    times down, the angle as darkness."""
    parts = turns.cast(core.ga.subspace("1 yz zx xy")).kernel                  # [times, n, 4]
    angles = np.arctan2(np.linalg.norm(parts[..., 1:], axis=-1), parts[..., 0])   # [times, n]
    figure = plt.figure(figsize=(6.0, 4.0), dpi=110)
    ax = figure.add_axes((0.1, 0.12, 0.86, 0.84))
    ax.imshow(angles, cmap="Greys", origin="upper", aspect="auto", interpolation="nearest",
              extent=(0.5, angles.shape[1] + 0.5, angles.shape[0] - 0.5, -0.5))
    ax.set(xlabel="vertex", ylabel="time", xticks=[], yticks=[])
    ax.spines[:].set_visible(False)
    return figure
