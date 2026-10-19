"""Drawing for connections on curved surfaces: latitude holonomy on the unit sphere; tangent vector fields
and vector heat diffusion on the torus; unit tangent fields as needles over shaded surfaces, with the
singular faces marked by discs, warm for positive indices and cool for negative, larger for larger
indices; and stripe patterns.

Stripe patterns are drawn as a renderer's shader would: every pixel of the surface interpolates the
phase from the corners of the triangle it falls in, with the whole turns a fork winds spread around the
triangle, and the stripes are the cosine of that phase, sharp at any resolution. The phase at a
triangle's corners is read as the angle from the stripe field to its line of the direction field, each
corner's lines matched to the first corner's, and unwrapped from the first corner along the edges'
targets. Lines are drawn as the stripes' middles, straight across each face, over the lit surface and
hidden where the surface is in front of them."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from examples.animation import capture
from examples.raster import rasterise, screen, view_frame
from examples.surfaces.connections import core

# Surface base colours, ring palette, needle colour, the colours of positive and negative singularities
# and their disc size per unit index, and the views as (elevation, azimuth) in degrees: of the transport
# figures, and of the designed fields and stripes.
SURFACE_COLOUR = np.array([0.92, 0.91, 0.88])
WARM_COLOUR = np.array([0.85, 0.36, 0.18])
RING_COLOURS = ("#c84b31", "#2d6a9f", "#2a8c6a")
NEEDLE_COLOUR = "#1d3557"
POSITIVE, NEGATIVE = "#c84b31", "#2d6a9f"
DISC = 70.0
TRANSPORT_VIEW = np.array([28.0, -36.0])
VIEW = np.array([24.0, -36.0])
# The extents and needle lengths of the holonomy sphere and the transport torus, the stride between
# needles along the rings, and those of the designed fields' sphere and torus.
HOLONOMY_LIMITS = np.array([[-1.15, 1.15], [-1.15, 1.15], [-1.15, 1.15]])
TRANSPORT_LIMITS = np.array([[-2.75, 2.75], [-2.75, 2.75], [-1.15, 1.15]])
HOLONOMY_STRIDE = 3
HOLONOMY_NEEDLE = 0.25
TRANSPORT_NEEDLE = 0.32
SPHERE_LIMITS = np.array([[-1.1, 1.1], [-1.1, 1.1], [-1.1, 1.1]])
TORUS_LIMITS = np.array([[-2.75, 2.75], [-2.75, 2.75], [-1.0, 1.0]])
SPHERE_NEEDLE = 0.09
TORUS_NEEDLE = 0.2
# The colours of the dark and the light stripes, the share of a stripe that is dark, and how sharply one
# turns into the other; the width and colour of drawn lines, how far they stand off the surface, and how
# far behind the surface a line may be and still show; the light's direction and the share of light
# that falls everywhere; the image's side in pixels and how many times finer each side is rendered
# before averaging; and how far around the centre the views of the sphere reach.
DARK, LIGHT = np.array([0.16, 0.24, 0.36]), np.array([0.93, 0.91, 0.86])
BANDS = 0.5
SHARPNESS = 12.0
LINE_WIDTH, STANDOFF, DEPTH_SLACK = 0.9, 0.004, 0.02
LIGHT_DIRECTION, AMBIENT = np.array([-0.3, -0.5, 0.8]), 0.4
PIXELS = 420
SUPERSAMPLE = 3
SPHERE_REACH = 1.1


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z components of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def positions(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def print_scalars(label: str, values: core.Scalar) -> None:
    """A label and the values of a batch of scalars."""
    print(label, values.cast(core.ga.subspace.scalar()).kernel[..., 0])


def facing(view: np.ndarray) -> np.ndarray:
    """The unit vector pointing towards the viewer, for a view as (elevation, azimuth)."""
    return view_frame(*view)[2]


def setup_axes(ax, limits: np.ndarray, zoom: float, view: np.ndarray) -> None:
    for set_limit, limit in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), limits):
        set_limit(*limit)
    ax.set_box_aspect(limits[:, 1] - limits[:, 0], zoom=zoom)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()


def needles(origins: np.ndarray, directions: np.ndarray, normals: np.ndarray, length: float) -> np.ndarray:
    """Line segments of arrows along the directions, with their heads in each point's tangent plane."""
    tips = origins + directions * length
    side = np.cross(normals, directions)
    base = origins + directions * (0.62 * length)
    return np.concatenate([np.stack([origins, tips], axis=1),
                           np.stack([base + side * (0.25 * length), tips], axis=1),
                           np.stack([base - side * (0.25 * length), tips], axis=1)])


# --- parallel transport ---------------------------------------------------------------
def animate_holonomy(sphere: core.Mesh, rings: core.Vector, states: Iterable[tuple[core.Vector, core.Vector]]
                     ) -> list[np.ndarray]:
    """The unit sphere and the three latitude circles `[steps + 1, rings]`, and each frame's points on
    them with the tangent vectors carried there `[rings]`: a trail of where the vectors have been, and
    where they are."""
    verts = positions(sphere.vertices)
    view_dir = facing(TRANSPORT_VIEW)
    ring_xyz = coordinates(rings)                                               # [steps + 1, rings, 3]
    points, carried = [], []
    frames = []
    for point, vector in states:
        points.append(coordinates(point))                                       # [rings, 3]
        carried.append(coordinates(vector))                                     # [rings, 3]
        trail = slice(None, None, HOLONOMY_STRIDE)
        figure = plt.figure(figsize=(5.4, 5.4), dpi=100)
        ax = figure.add_axes((0, 0, 1, 1), projection="3d")
        ax.add_collection3d(
            Poly3DCollection(verts[sphere.faces], facecolors=SURFACE_COLOUR, shade=True, linewidths=0, alpha=0.22)
        )
        for r, colour in enumerate(RING_COLOURS):
            curve = ring_xyz[:, r]
            in_front = curve @ view_dir >= -0.05
            ax.plot(*np.where(~in_front[:, None], curve, np.nan).T, color=colour, linewidth=1.0, alpha=0.35, linestyle="--")
            ax.plot(*np.where(in_front[:, None], curve, np.nan).T, color=colour, linewidth=1.8)
            past, along = np.array(points)[trail, r], np.array(carried)[trail, r]   # [trail, 3] each
            ax.add_collection3d(Line3DCollection(needles(past, along, past, HOLONOMY_NEEDLE),
                                                 colors=colour, linewidths=1.0, alpha=0.5))
            ax.add_collection3d(Line3DCollection(needles(points[-1][r:r + 1], carried[-1][r:r + 1], points[-1][r:r + 1],
                                                         HOLONOMY_NEEDLE * 1.15), colors=colour, linewidths=2.6))
        setup_axes(ax, HOLONOMY_LIMITS, 1.35, TRANSPORT_VIEW)
        frames.append(capture(figure))
        plt.close(figure)
    return frames


def _draw_torus_panel(ax, torus: core.Mesh, field: core.Vector, amplitudes: np.ndarray) -> None:
    """One shaded torus with face colours weighted by `amplitudes` and front-facing tangent needles."""
    verts = positions(torus.vertices)
    normals = coordinates(torus.vertex_normals)
    vecs = coordinates(field)
    norms = np.linalg.norm(vecs, axis=-1, keepdims=True)
    unit_vecs = vecs / np.maximum(norms, 1e-12)

    face_weight = np.clip(amplitudes[torus.faces].mean(axis=1), 0.0, 1.0)[:, None]
    face_colours = (1.0 - 0.58 * face_weight) * SURFACE_COLOUR + (0.58 * face_weight) * WARM_COLOUR
    ax.add_collection3d(Poly3DCollection(verts[torus.faces], facecolors=face_colours, shade=True, linewidths=0))

    visible = normals @ facing(TRANSPORT_VIEW) > 0.05
    scale = (0.25 + 0.75 * np.clip(amplitudes[visible], 0.0, 1.0))[:, None]
    segments = needles(
        verts[visible] + normals[visible] * 0.03,
        unit_vecs[visible] * scale,
        normals[visible],
        TRANSPORT_NEEDLE,
    )
    ax.add_collection3d(Line3DCollection(segments, colors=NEEDLE_COLOUR, linewidths=1.0))
    setup_axes(ax, TRANSPORT_LIMITS, 1.38, TRANSPORT_VIEW)


def draw_torus_fields(torus: core.Mesh, fields: list[core.Vector]) -> plt.Figure:
    """The torus side by side for each tangent vector field in `fields`."""
    figure = plt.figure(figsize=(3.8 * len(fields), 3.6), dpi=100)
    for i, field in enumerate(fields):
        vecs = coordinates(field)
        lengths = np.linalg.norm(vecs, axis=-1)
        amplitudes = lengths / max(float(np.percentile(lengths, 95.0)), 1e-12)
        ax = figure.add_subplot(1, len(fields), i + 1, projection="3d")
        _draw_torus_panel(ax, torus, field, amplitudes)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_diffusion(torus: core.Mesh, diffused: core.Vector) -> list[np.ndarray]:
    """Frames of a vector heat pulse spreading across the torus `[steps] Vector[V]`."""
    images = []
    all_vecs = coordinates(diffused)                                            # [steps, V, 3]
    lengths = np.linalg.norm(all_vecs, axis=-1)                                 # [steps, V]
    peak = np.percentile(lengths, 92.0, axis=-1, keepdims=True)
    scaled = np.pow(np.clip(lengths / np.maximum(peak, 1e-12), 0.0, 1.0), 0.45)
    for step, field in enumerate(diffused):
        figure = plt.figure(figsize=(4.2, 3.8), dpi=80)
        ax = figure.add_axes((0, 0, 1, 1), projection="3d")
        _draw_torus_panel(ax, torus, field, scaled[step])
        images.append(capture(figure))
        plt.close(figure)
    return images


# --- designed fields ------------------------------------------------------------------
def draw_field(ax, mesh: core.Mesh, field: core.Vector, singular: np.ndarray, indices: list[int], limits: np.ndarray,
               needle: float) -> None:
    """One surface with its unit tangent field on the vertices facing the viewer, and those of its
    singular faces `[S]` that face the viewer, drawn in that order rather than sorted by depth."""
    points, normals, directions = positions(mesh.vertices), coordinates(mesh.vertex_normals), coordinates(field)
    ax.computed_zorder = False
    ax.add_collection3d(Poly3DCollection(points[mesh.faces], facecolors=SURFACE_COLOUR, shade=True, linewidths=0))
    seen = normals @ facing(VIEW) > 0.1
    ax.add_collection3d(Line3DCollection(needles(points[seen] + 0.01 * normals[seen], directions[seen], normals[seen],
                                                 needle), colors=NEEDLE_COLOUR, linewidths=0.9))
    centres = positions(mesh.face_centers)[singular]                           # [S, 3]
    toward = normals[mesh.faces[singular, 0]] @ facing(VIEW) > 0.1             # [S]
    for centre, index in zip(centres[toward], np.asarray(indices)[toward]):
        ax.scatter(*centre, s=DISC * abs(index), color=POSITIVE if index > 0 else NEGATIVE, depthshade=False,
                   edgecolors="white", linewidths=0.8)
    setup_axes(ax, limits, 1.3, VIEW)


def draw_fields(mesh: core.Mesh, scenes: list[tuple[core.Vector, np.ndarray, list[int]]], limits: np.ndarray,
                needle: float) -> plt.Figure:
    """The surface side by side for each scene's field and singular faces."""
    width = (limits[0, 1] - limits[0, 0]) / (limits[2, 1] - limits[2, 0])
    figure = plt.figure(figsize=(3.6 * len(scenes) * min(width, 1.6), 3.6), dpi=110)
    for case, (field, singular, indices) in enumerate(scenes):
        draw_field(figure.add_subplot(1, len(scenes), case + 1, projection="3d"), mesh, field, singular, indices, limits,
                   needle)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_turning(mesh: core.Mesh, fields: core.Vector, singular: np.ndarray, indices: list[int]) -> list[np.ndarray]:
    """Frames of a parallel field on the sphere turned about every normal `[frames] Vector[V]`."""
    images = []
    for field in fields:
        figure = plt.figure(figsize=(4.0, 4.0), dpi=90)
        draw_field(figure.add_axes((0, 0, 1, 1), projection="3d"), mesh, field, singular, indices, SPHERE_LIMITS,
                   SPHERE_NEEDLE)
        images.append(capture(figure))
        plt.close(figure)
    return images


# --- stripes --------------------------------------------------------------------------
def corner_phases(mesh: core.Mesh, lines: core.Vector, stripes: core.Vector, advance: core.Scalar) -> tuple[np.ndarray, np.ndarray]:
    """The phase of each family at each corner of each face `[F, 3, n]`, and the whole turns it winds around
    the face `[F, n]`, for the lines of the families at every vertex `[V, n]`, their stripe fields `[V, n]`
    and the targets across every edge for each line at its head `[E, n]`. Each corner's lines are matched
    to the face's first corner's, by the line nearest each and the sense it points in; walking around the
    face along the edges' targets, each corner's phase is taken nearest the one before it plus the target;
    a face around which the walk comes back off by whole turns holds a fork, and the mismatch is spread
    evenly over its three edges, as in the reference implementation of the stripe patterns paper."""
    stripe, directions = stripes.batch(), lines.batch()                        # [V, n] Vector each
    sines = ((stripe ^ directions) | mesh.vertex_normals.batch()[:, None].dual()).to_array()
    phase = np.arctan2(sines, (stripe | directions).to_array())                # [V, n]
    steps = advance.batch().to_array()                                         # [E, n]
    along = coordinates(lines)[mesh.faces]                                     # [F, 3, n, 3]
    dots = np.einsum("fnd,fcmd->fcnm", along[:, 0], along)                     # [F, 3, n, n]
    picked = np.argmax(np.abs(dots), axis=-1)                                  # [F, 3, n]
    sense = np.sign(np.take_along_axis(dots, picked[..., None], axis=-1)[..., 0])
    local = sense * np.take_along_axis(phase[mesh.faces], picked, axis=-1)     # [F, 3, n]
    # The target from each corner to the next, read at the edge's head, the higher vertex.
    count = len(phase)
    tails, heads = mesh.faces, np.roll(mesh.faces, -1, axis=1)                 # [F, 3] each
    keys = mesh.edges[:, 0] * count + mesh.edges[:, 1]
    edge = np.searchsorted(keys, np.minimum(tails, heads) * count + np.maximum(tails, heads))
    forward = tails < heads                                                    # [F, 3]
    head = np.where(forward, np.roll(np.arange(3), -1), np.arange(3))          # [F, 3]: the corner at the edge's head
    at_head = np.take_along_axis(picked, head[..., None], axis=1)              # [F, 3, n]
    sense_head = np.take_along_axis(sense, head[..., None], axis=1)
    target = sense_head * np.take_along_axis(steps[edge], at_head, axis=-1) * np.where(forward, 1.0, -1.0)[..., None]
    walked = [local[:, 0]]
    for side in range(3):
        expected = walked[-1] + target[:, side]
        walked.append(expected + np.angle(np.exp(1j * (local[:, (side + 1) % 3] - expected))))
    walked = np.stack(walked, axis=1)                                          # [F, 4, n]
    turns = np.round((walked[:, 3] - walked[:, 0]) / (2 * np.pi))              # [F, n]
    spread = walked[:, :3] - 2 * np.pi * turns[:, None] * (np.arange(3) / 3)[:, None]
    return spread, turns


def winding(weights: np.ndarray) -> np.ndarray:
    """An angle around the middle of a triangle from its barycentric weights `[..., 3]`, rising by a whole
    turn once around: piecewise linear on the three sectors nearest each edge."""
    first, second, third = weights[..., 0], weights[..., 1], weights[..., 2]
    sector = np.argmin(weights, axis=-1)                                       # the corner whose weight is least
    least = np.min(weights, axis=-1)
    along = np.choose(sector, [third - second, first - third, second - first])
    start = np.choose(sector, [3.0, 5.0, 1.0])
    return (np.pi / 3) * (start + along / np.maximum(1 - 3 * least, 1e-12))


def crossings(places: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """The segments `[F, L, 2, 3]` across faces with corners at the given places `[F, 3, 3]` where a phase
    linear across each, given at its corners in whole turns `[F, 3]`, passes each whole number; `L` is
    the most whole numbers any face holds, and a face holding fewer repeats its first corner as an
    empty segment."""
    lowest = np.floor(phases.min(axis=-1))                                    # [F]
    levels = lowest[:, None] + 1 + np.arange(int(np.max(np.ceil(phases.max(axis=-1)) - lowest)))   # [F, L]
    start, end = phases[:, None, :, None], np.roll(phases, -1, axis=-1)[:, None, :, None]   # [F, 1, 3, 1]
    share = (levels[:, :, None, None] - start) / np.where(end == start, np.inf, end - start)  # [F, L, 3, 1]
    across = (share >= 0) & (share < 1)                                       # [F, L, 3, 1]
    at = places[:, None] + share * (np.roll(places, -1, axis=1) - places)[:, None]   # [F, L, 3, 3]
    # The two sides a level crosses, first and second; a level crossing none collapses onto the first corner.
    order = np.argsort(~across[..., 0], axis=-1, kind="stable")[..., :2]        # [F, L, 2]
    ends = np.take_along_axis(np.where(across, at, places[:, None, :1]), order[..., None], axis=2)   # [F, L, 2, 3]
    return ends


def lit(normals: np.ndarray) -> np.ndarray:
    """The share of light falling on surfaces of the given normals `[..., 3]`, either side alike."""
    normals = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    return AMBIENT + (1 - AMBIENT) * np.abs(normals @ (LIGHT_DIRECTION / np.linalg.norm(LIGHT_DIRECTION)))


def raster(mesh: core.Mesh, view: np.ndarray, reach: float, size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The mesh rasterised from the given view: each covered pixel, its face and barycentric weights, its
    depth, and the light on it from the vertex normals interpolated across the face."""
    frame = view_frame(*view)
    places = screen(positions(mesh.vertices), frame, np.zeros(3), reach, size)
    pixel, face, weights, depth = rasterise(places, mesh.faces, size)
    normals = np.einsum("pk,pkd->pd", weights, coordinates(mesh.vertex_normals)[mesh.faces[face]])
    return pixel, face, weights, depth, lit(normals)


def picture(pixel: np.ndarray, colours: np.ndarray, size: int) -> np.ndarray:
    """An RGB image of the given side, white where nothing covers it, from colours at covered pixels,
    averaged over the supersampled blocks."""
    image = np.ones((size * size, 3))
    image[pixel] = colours
    side = size // SUPERSAMPLE
    return image.reshape(side, SUPERSAMPLE, side, SUPERSAMPLE, 3).mean(axis=(1, 3))


def draw_bands(ax, mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], view: np.ndarray, reach: float) -> None:
    """The surface shaded with the stripes of every family of a pattern of lines `[V, n]`, stripe fields
    `[V, n]` and edge targets `[E, n]`, the given share of each stripe dark."""
    size = PIXELS * SUPERSAMPLE
    pixel, face, weights, _, light = raster(mesh, view, reach, size)
    corners, turns = corner_phases(mesh, *pattern)                             # [F, 3, n], [F, n]
    phases = np.einsum("pk,pkn->pn", weights, corners[face]) + winding(weights)[:, None] * turns[face]   # [P, n]
    share = np.prod(0.5 - 0.5 * np.tanh(SHARPNESS * (np.cos(phases) - np.cos(np.pi * BANDS))), axis=-1)  # [P]
    colours = (share[:, None] * LIGHT + (1 - share[:, None]) * DARK) * light[:, None]
    ax.imshow(picture(pixel, colours, size), origin="lower", extent=(0, PIXELS, 0, PIXELS), interpolation="none")
    ax.set_axis_off()


def draw_lines(ax, mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], view: np.ndarray, reach: float) -> None:
    """The lit surface with the middle line of every stripe of every family of a pattern drawn on it,
    straight across each face, ending in the faces where the stripes fork, and hidden where the surface
    is in front of them."""
    size = PIXELS * SUPERSAMPLE
    pixel, _, _, depth, light = raster(mesh, view, reach, size)
    ax.imshow(picture(pixel, light[:, None] * LIGHT, size), origin="lower", extent=(0, PIXELS, 0, PIXELS), interpolation="none")
    corners, turns = corner_phases(mesh, *pattern)                             # [F, 3, n], [F, n]
    points, normals = positions(mesh.vertices), coordinates(mesh.vertex_normals)
    lifted = (points + STANDOFF * normals)[mesh.faces]                         # [F, 3, 3]
    segments = np.concatenate([crossings(lifted, corners[..., family] / (2 * np.pi))[turns[:, family] == 0].reshape(-1, 2, 3)
                               for family in range(corners.shape[-1])])                # [S, 2, 3]
    # Each segment shows where its middle is no further behind the surface than the slack.
    frame = view_frame(*view)
    ends = screen(segments, frame, np.zeros(3), reach, size)                  # [S, 2, 3]
    middles = ends.mean(axis=1)
    nearest = np.full(size * size, -np.inf)
    nearest[pixel] = depth
    at = np.clip(middles[:, 1].astype(int), 0, size - 1) * size + np.clip(middles[:, 0].astype(int), 0, size - 1)
    shown = middles[:, 2] >= nearest[at] - DEPTH_SLACK * size / (2 * reach)
    ax.add_collection(LineCollection(ends[shown, :, :2] / SUPERSAMPLE, colors=[DARK], linewidths=LINE_WIDTH))
    ax.set_xlim(0, PIXELS)
    ax.set_ylim(0, PIXELS)
    ax.set_axis_off()


def draw_patterns(mesh: core.Mesh, patterns: list[tuple[core.Vector, core.Vector, core.Scalar]], draw, reach: float,
                  views: np.ndarray) -> plt.Figure:
    """The surface side by side for each pattern of lines, stripe fields and edge targets, each drawn by
    `draw_bands` or `draw_lines` and seen from its (elevation, azimuth) `[patterns, 2]`."""
    figure = plt.figure(figsize=(4.2 * len(patterns), 4.2), dpi=110)
    for case, (pattern, view) in enumerate(zip(patterns, views)):
        draw(figure.add_subplot(1, len(patterns), case + 1), mesh, pattern, view, reach)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_flow(mesh: core.Mesh, lines: core.Vector, stripe_fields: core.Vector, advance: core.Scalar, reach: float) -> list[np.ndarray]:
    """Frames of the stripes as the stripe field `[V, 1]` turns about every normal, `[frames]`."""
    images = []
    for stripe_field in stripe_fields:
        figure, ax = plt.subplots(figsize=(4.2, 4.2), dpi=100)
        draw_bands(ax, mesh, (lines, stripe_field, advance), VIEW, reach)
        figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
        images.append(capture(figure))
        plt.close(figure)
    return images
