"""Drawing for connections on curved surfaces, on the GPU: latitude holonomy on the unit sphere; tangent vector
fields and vector heat diffusion on the torus; unit tangent fields as needles over shaded surfaces, with the
singular faces marked by balls, warm for positive indices and cool for negative, larger for larger indices; and
stripe patterns.

The holonomy sphere is a translucent ball traced by `solids`; the other surfaces are meshes drawn by `meshes`;
needles, rings and stripe middles are strokes, all composited by depth sample by sample. Stripe patterns are drawn in the surface's own shader: every pixel interpolates the phase from the
corners of the triangle it falls in, with the whole turns a fork winds spread around the triangle, and the
stripes are the cosine of that phase, sharp at any resolution. The phase at a triangle's corners is read as the
angle from the stripe field to its line of the direction field, each corner's lines matched to the first
corner's, and unwrapped from the first corner along the edges' targets. Lines are drawn as the stripes'
middles, straight across each face, standing just off the surface."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.surfaces.connections import core
from examples.util.wgsl import fitted, meshes, show, solids, strokes, vec3
from examples.util.wgsl.scene import Scene

# Surface base colours, ring palette, needle colour, the colours of positive and negative singularities and their
# balls' radius per unit index, in pixels, and the views as (elevation, azimuth) in degrees, the azimuth measured
# from x: of the transport figures, and of the designed fields and stripes.
surface_colour = np.array([0.92, 0.91, 0.88])
warm_colour = np.array([0.85, 0.36, 0.18])
ring_colours = ("#c84b31", "#2d6a9f", "#2a8c6a")
needle_colour = to_rgb("#1d3557")
positive, negative = to_rgb("#c84b31"), to_rgb("#2d6a9f")
singular_radius = 3.0
transport_view = np.array([28.0, -36.0])
view = np.array([24.0, -36.0])
# The needle lengths of the holonomy sphere and the transport torus, the stride between needles along the rings,
# the designed fields' needle length as a share of their surface's mean edge length, and the widest a panel of them
# gets, as a share of its height.
holonomy_stride = 3
holonomy_needle = 0.25
transport_needle = 0.32
field_needle = 0.6
widest_panel = 1.6
# How much each surface of the holonomy sphere hides, and each surface of the trail of carried vectors.
sphere_opacity = 0.12
trail_opacity = 0.3
# The colours of the dark and the light stripes, the share of a stripe that is dark, and how sharply one turns
# into the other; the width of drawn lines and needles in pixels, and how far lines stand off the surface, as a
# share of its reach around the centre; the image's side in pixels, until a figure's view is fitted to its axes, and
# the margin around a framed surface as a share of its larger side.
dark, light = np.array([0.16, 0.24, 0.36]), np.array([0.93, 0.91, 0.86])
bands = 0.5
sharpness = 12.0
line_width, standoff = 1.4, 0.0066
pixels = 420
margin = 0.04
background = np.ones(3)


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z components of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def positions(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


# --- views and parts ------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, mesh: core.Mesh, reach: float) -> solids.View:
    """A square view framing a mesh, with `reach` to spare, from a view in degrees, the azimuth measured from x."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90)
    frame = solids.view([solids.mv.point(positions(mesh.vertices))], reach, elevation, azimuth, margin, pixels)
    return frame.squared()


def needle_length(mesh: core.Mesh) -> float:
    """The length of the designed fields' needles on a surface: a share of its mean edge length."""
    places = positions(mesh.vertices)
    return field_needle * float(np.linalg.norm(places[mesh.edges[:, 1]] - places[mesh.edges[:, 0]], axis=-1).mean())


def facing(view_degrees: np.ndarray) -> np.ndarray:
    """The unit vector pointing towards the viewer, for a view as (elevation, azimuth) in degrees from x."""
    elevation, azimuth = np.radians(view_degrees)
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def needles(origins: np.ndarray, directions: np.ndarray, normals: np.ndarray, length: float) -> tuple[np.ndarray, np.ndarray]:
    """Arrows along the directions, with their heads in each point's tangent plane: a shaft and two strokes to the
    tip, as segments, their starts and ends `[arrows * 3, 3]`."""
    tips = origins + directions * length
    side = np.cross(normals, directions)
    base = origins + directions * (0.62 * length)
    starts = np.concatenate([origins, base + side * (0.25 * length), base - side * (0.25 * length)])
    return starts, np.concatenate([tips, tips, tips])


def surface(mesh: core.Mesh, colours: np.ndarray, frame: solids.View, face_values: np.ndarray = np.zeros(4),
            colour: str = meshes.plain) -> meshes.Samples:
    """A mesh, smooth shaded, at every sample, as `meshes.samples`."""
    points = solids.mv.point(positions(mesh.vertices))
    return meshes.samples(points, mesh.faces, colours, meshes.smooth(points, mesh.faces), frame, face_values=face_values,
                          colour=colour)


# --- parallel transport ---------------------------------------------------------------
def animate_holonomy(sphere: core.Mesh, rings: core.Vector, states: Iterable[tuple[core.Vector, core.Vector]]
                     ) -> Iterator[np.ndarray]:
    """The unit sphere, translucent, and the three latitude circles `[steps + 1, rings]`, and each frame's points
    on them with the tangent vectors carried there `[rings]`: a trail of where the vectors have been, and where
    they are. The sphere is the traced unit ball, the circles' far halves showing through its near surface."""
    frame = square_view(transport_view, sphere, holonomy_needle)
    ring_places = coordinates(rings)                                            # [steps + 1, rings, 3]
    colours = np.array([to_rgb(colour) for colour in ring_colours])             # [rings, 3]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], surface_colour, sphere_opacity)]
    # The circles and the needles are strokes; the sphere alone is traced. The sphere and the circles stay put.
    circles = strokes.polylines(list(zip(np.moveaxis(ring_places, 1, 0), colours)), 4 * line_width, frame)
    scene = Scene(frame, background, [], [(circles, 1.0)], translucent=ball)
    points, carried = [], []
    for point, vector in states:
        points.append(coordinates(point))                                       # [rings, 3]
        carried.append(coordinates(vector))                                     # [rings, 3]
        past, along = np.array(points)[::holonomy_stride], np.array(carried)[::holonomy_stride]   # [trail, rings, 3]
        # Each needle of the trail in its ring's colour: a needle's three strokes, a run per stroke.
        trail = strokes.segments(*needles(past.reshape(-1, 3), along.reshape(-1, 3), past.reshape(-1, 3),
                                          holonomy_needle), np.tile(np.tile(colours, (len(past), 1)), (3, 1)),
                                 2 * line_width, frame)
        now = strokes.segments(*needles(points[-1], carried[-1], points[-1], holonomy_needle * 1.15),
                               np.tile(colours, (3, 1)), 5 * line_width, frame)
        yield scene.draw(meshes=[(now, 1.0), (trail, trail_opacity)])


def torus_panel(torus: core.Mesh, field: core.Vector, amplitudes: np.ndarray, frame: solids.View) -> np.ndarray:
    """One shaded torus with face colours weighted by `amplitudes` and front-facing tangent needles."""
    places = positions(torus.vertices)
    normals = coordinates(torus.vertex_normals)
    vectors = coordinates(field)
    units = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)
    face_weight = np.clip(amplitudes[torus.faces].mean(axis=1), 0.0, 1.0)[:, None]
    face_colours = (1.0 - 0.58 * face_weight) * surface_colour + (0.58 * face_weight) * warm_colour
    visible = normals @ facing(transport_view) > 0.05
    scale = (0.25 + 0.75 * np.clip(amplitudes[visible], 0.0, 1.0))[:, None]
    arrows = strokes.segments(*needles(places[visible] + normals[visible] * 0.03, units[visible] * scale,
                                       normals[visible], transport_needle), needle_colour, 2 * line_width, frame)
    return solids.draw([], frame, background,
                       meshes=[(surface(torus, np.zeros(3), frame, face_values=face_colours, colour=meshes.by_face), 1.0),
                               (arrows, 1.0)])


def draw_torus_fields(torus: core.Mesh, fields: list[core.Vector]) -> plt.Figure:
    """The torus side by side for each tangent vector field in `fields`."""
    figure, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 3.6), dpi=100)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(transport_view, torus, transport_needle), axes[0])
    for axis, field in zip(axes, fields):
        lengths = np.linalg.norm(coordinates(field), axis=-1)
        show(axis, torus_panel(torus, field, lengths / max(float(np.percentile(lengths, 95.0)), 1e-12), frame), frame)
    return figure


def animate_diffusion(torus: core.Mesh, diffused: Iterable[core.Vector]) -> Iterator[np.ndarray]:
    """Frames of a vector heat pulse spreading across the torus, a `Vector[V]` each, each scaled by its own
    lengths."""
    frame = square_view(transport_view, torus, transport_needle)
    for field in diffused:
        lengths = np.linalg.norm(coordinates(field), axis=-1)                  # [V]
        peak = np.percentile(lengths, 92.0)
        scaled = np.pow(np.clip(lengths / np.maximum(peak, 1e-12), 0.0, 1.0), 0.45)
        yield torus_panel(torus, field, scaled, frame)


# --- designed fields ------------------------------------------------------------------
def field_marks(mesh: core.Mesh, singular: np.ndarray, indices: list[int], frame: solids.View
                ) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Balls on a surface's singular faces `[S]`, larger for larger indices."""
    centres = solids.mv.point(positions(mesh.face_centers)[singular])          # [S] Point
    return [(solids.ball(centres[index:index + 1], singular_radius * abs(value) * frame.pixel)[:, None],
             positive if value > 0 else negative) for index, value in enumerate(indices)]


def field_arrows(mesh: core.Mesh, field: core.Vector, needle: float, frame: solids.View) -> meshes.Samples:
    """A unit tangent field on the vertices of a surface facing the viewer, as strokes."""
    places, normals, directions = positions(mesh.vertices), coordinates(mesh.vertex_normals), coordinates(field)
    seen = normals @ facing(view) > 0.1
    return strokes.segments(*needles(places[seen] + 0.01 * normals[seen], directions[seen], normals[seen], needle),
                            needle_colour, 2 * line_width, frame)


def field_panel(mesh: core.Mesh, field: core.Vector, singular: np.ndarray, indices: list[int], needle: float,
                frame: solids.View) -> np.ndarray:
    """One surface with its unit tangent field on the vertices facing the viewer, and balls on its singular faces
    `[S]`, larger for larger indices."""
    arrows = field_arrows(mesh, field, needle, frame)
    return solids.draw(field_marks(mesh, singular, indices, frame), frame, background,
                       meshes=[(surface(mesh, surface_colour, frame), 1.0), (arrows, 1.0)])


def draw_fields(mesh: core.Mesh, scenes: list[tuple[core.Vector, np.ndarray, list[int]]]) -> plt.Figure:
    """The surface side by side for each scene's field and singular faces, each panel as wide as the surface across
    x over its height, up to the widest a panel gets."""
    extent = np.ptp(positions(mesh.vertices), axis=0)                          # [3]
    width = min(extent[0] / extent[2], widest_panel)
    needle = needle_length(mesh)
    figure, axes = plt.subplots(1, len(scenes), figsize=(3.6 * len(scenes) * width, 3.6), dpi=110)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(view, mesh, needle), axes[0])
    for axis, (field, singular, indices) in zip(axes, scenes):
        show(axis, field_panel(mesh, field, singular, indices, needle, frame), frame)
    return figure


def animate_turning(mesh: core.Mesh, fields: Iterable[core.Vector], singular: np.ndarray, indices: list[int]
                    ) -> Iterator[np.ndarray]:
    """Frames of a parallel field on the sphere turned about every normal, a `Vector[V]` each, among the surface
    and the balls on its singular faces, which stay put."""
    needle = needle_length(mesh)
    frame = square_view(view, mesh, needle)
    scene = Scene(frame, background, field_marks(mesh, singular, indices, frame),
                  [(surface(mesh, surface_colour, frame), 1.0)])
    for field in fields:
        yield scene.draw(meshes=[(field_arrows(mesh, field, needle, frame), 1.0)])


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


def stripe_colour(n_families: int) -> str:
    """WGSL colouring a surface by the stripes of up to four families: each family's phase, the corners'
    blended across the face, plus the whole turns a fork winds around it, spread by an angle about the face's
    middle that rises by a whole turn once around; light where every family's stripe is light."""
    return f"""
// An angle around the middle of a triangle from how near a pixel lies to each corner, rising by a whole turn once
// around: piecewise linear on the three sectors nearest each edge.
fn winding(near: vec3<f32>) -> f32 {{
    let least = min(near.x, min(near.y, near.z));
    let along = select(select(near.y - near.x, near.x - near.z, near.y == least), near.z - near.y, near.x == least);
    let start = select(select(1.0, 5.0, near.y == least), 3.0, near.x == least);
    return (3.14159265 / 3) * (start + along / max(1 - 3 * least, 1e-6));
}}

fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {{
    let phases = values + winding(corner) * face_values;
    let shares = 0.5 - 0.5 * tanh({sharpness} * (cos(phases) - cos(3.14159265 * {bands})));
    var share = 1.0;
    for (var family = 0u; family < {n_families}u; family++) {{
        share *= shares[family];
    }}
    return vec4<f32>(mix({vec3(dark)}, {vec3(light)}, share), 1.0);
}}
"""


def bands_image(mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], frame: solids.View) -> np.ndarray:
    """The surface shaded with the stripes of every family of a pattern of lines `[V, n]`, stripe fields `[V, n]`
    and edge targets `[E, n]`, the given share of each stripe dark."""
    corners, turns = corner_phases(mesh, *pattern)                             # [F, 3, n], [F, n]
    return solids.draw([], frame, background,
                       meshes=[(surface(mesh, corners, frame, face_values=turns, colour=stripe_colour(corners.shape[-1])), 1.0)])


def lines_image(mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], frame: solids.View) -> np.ndarray:
    """The lit surface with the middle line of every stripe of every family of a pattern drawn on it, straight
    across each face, ending in the faces where the stripes fork, standing just off the surface."""
    corners, turns = corner_phases(mesh, *pattern)                             # [F, 3, n], [F, n]
    places, normals = positions(mesh.vertices), coordinates(mesh.vertex_normals)
    # Off the surface by a share of its reach around the centre, so the straight segments clear it between its
    # vertices.
    reach = np.linalg.norm(places, axis=-1).max()
    lifted = (places + standoff * reach * normals)[mesh.faces]                 # [F, 3, 3]
    segments = np.concatenate([crossings(lifted, corners[..., family] / (2 * np.pi))[turns[:, family] == 0].reshape(-1, 2, 3)
                               for family in range(corners.shape[-1])])         # [S, 2, 3]
    drawn = segments[np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1) > 0]
    lines = strokes.segments(drawn[:, 0], drawn[:, 1], dark, line_width, frame)
    return solids.draw([], frame, background, meshes=[(surface(mesh, light, frame), 1.0), (lines, 1.0)])


def pattern_figure(mesh: core.Mesh, patterns: list[tuple[core.Vector, core.Vector, core.Scalar]],
                   image: Callable[[core.Mesh, tuple[core.Vector, core.Vector, core.Scalar], solids.View], np.ndarray]
                   ) -> plt.Figure:
    """The surface side by side for each pattern, as `image` draws it in the square view fitted to a panel."""
    figure, axes = plt.subplots(1, len(patterns), figsize=(4.2 * len(patterns), 4.2), dpi=110)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(view, mesh, 0.0), axes[0])
    for axis, pattern in zip(axes, patterns):
        show(axis, image(mesh, pattern, frame), frame)
    return figure


def draw_bands(mesh: core.Mesh, patterns: list[tuple[core.Vector, core.Vector, core.Scalar]]) -> plt.Figure:
    """The surface side by side for each pattern of lines, stripe fields and edge targets, shaded with its
    stripes."""
    return pattern_figure(mesh, patterns, bands_image)


def draw_lines(mesh: core.Mesh, patterns: list[tuple[core.Vector, core.Vector, core.Scalar]]) -> plt.Figure:
    """The surface side by side for each pattern of lines, stripe fields and edge targets, with the middle lines
    of its stripes drawn on it."""
    return pattern_figure(mesh, patterns, lines_image)


def animate_flow(mesh: core.Mesh, lines: core.Vector, stripe_fields: Iterable[core.Vector], advance: core.Scalar
                 ) -> Iterator[np.ndarray]:
    """Frames of the stripes as the stripe field `[V, 1]` turns about every normal, one frame each."""
    frame = square_view(view, mesh, 0.0)
    for stripe_field in stripe_fields:
        yield bands_image(mesh, (lines, stripe_field, advance), frame)
