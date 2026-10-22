"""Drawing for connections on curved surfaces, on the GPU: latitude holonomy on the unit sphere; tangent vector
fields and vector heat diffusion on the torus; unit tangent fields as needles over shaded surfaces, with the
singular faces marked by balls, warm for positive indices and cool for negative, larger for larger indices; and
stripe patterns.

Surfaces are meshes drawn by `meshes`; needles, rings and stripe middles are rods traced by `solids`, all blended
by depth. Stripe patterns are drawn in the surface's own shader: every pixel interpolates the phase from the
corners of the triangle it falls in, with the whole turns a fork winds spread around the triangle, and the
stripes are the cosine of that phase, sharp at any resolution. The phase at a triangle's corners is read as the
angle from the stripe field to its line of the direction field, each corner's lines matched to the first
corner's, and unwrapped from the first corner along the edges' targets. Lines are drawn as the stripes'
middles, straight across each face, standing just off the surface."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from examples.surfaces.connections import core
from examples.wgsl import compose, meshes, show, solids, vec3

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
# and the extents and needle lengths of the designed fields' sphere and torus.
holonomy_stride = 3
holonomy_needle = 0.25
transport_needle = 0.32
sphere_limits = np.array([[-1.1, 1.1], [-1.1, 1.1], [-1.1, 1.1]])
torus_limits = np.array([[-2.75, 2.75], [-2.75, 2.75], [-1.0, 1.0]])
sphere_needle = 0.09
torus_needle = 0.2
# The holonomy sphere's opacity, that of the rings' hidden parts, and that of the trail of carried vectors.
sphere_opacity = 0.22
hidden_opacity = 0.35
trail_opacity = 0.5
# The colours of the dark and the light stripes, the share of a stripe that is dark, and how sharply one turns
# into the other; the width of drawn lines and needles in pixels, and how far lines stand off the surface, as a
# share of its reach; the image's side in pixels, how far around the centre the views of the sphere reach, and the
# margin around a framed surface as a share of its larger side.
dark, light = np.array([0.16, 0.24, 0.36]), np.array([0.93, 0.91, 0.86])
bands = 0.5
sharpness = 12.0
line_width, standoff = 1.4, 0.006
pixels = 420
sphere_reach = 1.1
margin = 0.04
background = np.ones(3)


def coordinates(vectors: core.Vector) -> np.ndarray:
    """The x, y and z components of vectors."""
    return vectors.cast(core.ga.subspace("x y z")).kernel


def positions(points: core.Point) -> np.ndarray:
    """The x, y and z coordinates of points, antivectors."""
    return points.cast(core.ga.subspace("yz zx xy")).kernel


def print_scalars(label: str, values: core.Scalar) -> None:
    """A label and the values of a batch of scalars."""
    print(label, values.cast(core.ga.subspace.scalar()).kernel[..., 0])


# --- views and parts ------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, mesh: core.Mesh, reach: float) -> solids.View:
    """A square view framing a mesh, with `reach` to spare, from a view in degrees, the azimuth measured from x."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90)
    frame = solids.view([solids.mv.point(positions(mesh.vertices))], reach, elevation, azimuth, margin, pixels)
    return frame.squared()


def facing(view_degrees: np.ndarray) -> np.ndarray:
    """The unit vector pointing towards the viewer, for a view as (elevation, azimuth) in degrees from x."""
    elevation, azimuth = np.radians(view_degrees)
    return np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def needles(origins: np.ndarray, directions: np.ndarray, normals: np.ndarray, length: float, radius: float) -> solids.Quadric:
    """Arrows along the directions, with their heads in each point's tangent plane: a shaft and two strokes to the
    tip, as rods, `[arrows * 3, 3]` quadrics."""
    tips = origins + directions * length
    side = np.cross(normals, directions)
    base = origins + directions * (0.62 * length)
    starts = np.concatenate([origins, base + side * (0.25 * length), base - side * (0.25 * length)])
    return solids.rods(solids.mv.point(starts), solids.mv.point(np.concatenate([tips, tips, tips])), radius)


def surface(mesh: core.Mesh, colours: np.ndarray, frame: solids.View, face_values: np.ndarray = np.zeros(4),
            colour: str = meshes.plain) -> tuple[np.ndarray, np.ndarray]:
    """A mesh, smooth shaded, as a layer."""
    points = solids.mv.point(positions(mesh.vertices))
    return meshes.layer(points, mesh.faces, colours, meshes.smooth(points, mesh.faces), frame, face_values=face_values,
                        colour=colour)


# --- parallel transport ---------------------------------------------------------------
def animate_holonomy(sphere: core.Mesh, rings: core.Vector, states: Iterable[tuple[core.Vector, core.Vector]]
                     ) -> list[np.ndarray]:
    """The unit sphere, translucent, and the three latitude circles `[steps + 1, rings]`, and each frame's points
    on them with the tangent vectors carried there `[rings]`: a trail of where the vectors have been, and where
    they are. The circles' far halves show faintly through the sphere."""
    frame = square_view(transport_view, sphere, holonomy_needle)
    width = line_width * frame.pixel
    ring_places = coordinates(rings)                                            # [steps + 1, rings, 3]
    sphere_layer = surface(sphere, surface_colour, frame)
    circles = [(solids.rods(solids.mv.point(ring_places[:-1, ring]), solids.mv.point(ring_places[1:, ring]), 2 * width),
                to_rgb(colour)) for ring, colour in enumerate(ring_colours)]
    circle_layer = solids.layer(circles, frame)
    points, carried = [], []
    frames = []
    for point, vector in states:
        points.append(coordinates(point))                                       # [rings, 3]
        carried.append(coordinates(vector))                                     # [rings, 3]
        past, along = np.array(points)[::holonomy_stride], np.array(carried)[::holonomy_stride]   # [trail, rings, 3]
        trail = solids.layer([(needles(past[:, ring], along[:, ring], past[:, ring], holonomy_needle, width), to_rgb(colour))
                              for ring, colour in enumerate(ring_colours)], frame)
        now = solids.layer([(needles(points[-1][ring:ring + 1], carried[-1][ring:ring + 1], points[-1][ring:ring + 1],
                                     holonomy_needle * 1.15, 2.5 * width), to_rgb(colour))
                            for ring, colour in enumerate(ring_colours)], frame)
        ghost = np.where(circle_layer[0][..., 3] > 0, -np.inf, 0.0)
        frames.append(compose([(*sphere_layer, sphere_opacity), (*circle_layer, 1.0), (circle_layer[0], ghost, hidden_opacity),
                               (*trail, trail_opacity), (*now, 1.0)], background))
    return frames


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
    arrows = needles(places[visible] + normals[visible] * 0.03, units[visible] * scale, normals[visible], transport_needle,
                     line_width * frame.pixel)
    return compose([(*surface(torus, np.zeros(3), frame, face_values=face_colours, colour=meshes.by_face), 1.0),
                    (*solids.layer([(arrows, needle_colour)], frame), 1.0)], background)


def draw_torus_fields(torus: core.Mesh, fields: list[core.Vector]) -> plt.Figure:
    """The torus side by side for each tangent vector field in `fields`."""
    frame = square_view(transport_view, torus, transport_needle)
    figure, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 3.6), dpi=100)
    for axis, field in zip(np.atleast_1d(axes), fields):
        lengths = np.linalg.norm(coordinates(field), axis=-1)
        show(axis, torus_panel(torus, field, lengths / max(float(np.percentile(lengths, 95.0)), 1e-12), frame), frame)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_diffusion(torus: core.Mesh, diffused: core.Vector) -> list[np.ndarray]:
    """Frames of a vector heat pulse spreading across the torus `[steps] Vector[V]`."""
    lengths = np.linalg.norm(coordinates(diffused), axis=-1)                    # [steps, V]
    peak = np.percentile(lengths, 92.0, axis=-1, keepdims=True)
    scaled = np.pow(np.clip(lengths / np.maximum(peak, 1e-12), 0.0, 1.0), 0.45)
    frame = square_view(transport_view, torus, transport_needle)
    return [torus_panel(torus, field, scaled[step], frame) for step, field in enumerate(diffused)]


# --- designed fields ------------------------------------------------------------------
def field_panel(mesh: core.Mesh, field: core.Vector, singular: np.ndarray, indices: list[int], needle: float,
                frame: solids.View) -> np.ndarray:
    """One surface with its unit tangent field on the vertices facing the viewer, and balls on its singular faces
    `[S]`, larger for larger indices."""
    places, normals, directions = positions(mesh.vertices), coordinates(mesh.vertex_normals), coordinates(field)
    seen = normals @ facing(view) > 0.1
    arrows = needles(places[seen] + 0.01 * normals[seen], directions[seen], normals[seen], needle,
                     line_width * frame.pixel)
    centres = solids.mv.point(positions(mesh.face_centers)[singular])          # [S] Point
    marks = [(solids.ball(centres[index:index + 1], singular_radius * abs(value) * frame.pixel)[:, None],
              positive if value > 0 else negative) for index, value in enumerate(indices)]
    return compose([(*surface(mesh, surface_colour, frame), 1.0),
                    (*solids.layer([(arrows, needle_colour)] + marks, frame), 1.0)], background)


def draw_fields(mesh: core.Mesh, scenes: list[tuple[core.Vector, np.ndarray, list[int]]], limits: np.ndarray,
                needle: float) -> plt.Figure:
    """The surface side by side for each scene's field and singular faces."""
    width = (limits[0, 1] - limits[0, 0]) / (limits[2, 1] - limits[2, 0])
    frame = square_view(view, mesh, needle)
    figure, axes = plt.subplots(1, len(scenes), figsize=(3.6 * len(scenes) * min(width, 1.6), 3.6), dpi=110)
    for axis, (field, singular, indices) in zip(np.atleast_1d(axes), scenes):
        show(axis, field_panel(mesh, field, singular, indices, needle, frame), frame)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_turning(mesh: core.Mesh, fields: core.Vector, singular: np.ndarray, indices: list[int]) -> list[np.ndarray]:
    """Frames of a parallel field on the sphere turned about every normal `[frames] Vector[V]`."""
    frame = square_view(view, mesh, sphere_needle)
    return [field_panel(mesh, field, singular, indices, sphere_needle, frame) for field in fields]


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

fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec3<f32> {{
    let phases = values + winding(corner) * face_values;
    let shares = 0.5 - 0.5 * tanh({sharpness} * (cos(phases) - cos(3.14159265 * {bands})));
    var share = 1.0;
    for (var family = 0u; family < {n_families}u; family++) {{
        share *= shares[family];
    }}
    return mix({vec3(dark)}, {vec3(light)}, share);
}}
"""


def draw_bands(axis: plt.Axes, mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], view_degrees: np.ndarray,
               reach: float) -> None:
    """The surface shaded with the stripes of every family of a pattern of lines `[V, n]`, stripe fields `[V, n]`
    and edge targets `[E, n]`, the given share of each stripe dark."""
    frame = square_view(view_degrees, mesh, 0.0)
    show(axis, bands_image(mesh, pattern, frame), frame)


def bands_image(mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], frame: solids.View) -> np.ndarray:
    corners, turns = corner_phases(mesh, *pattern)                             # [F, 3, n], [F, n]
    return compose([(*surface(mesh, corners, frame, face_values=turns, colour=stripe_colour(corners.shape[-1])), 1.0)],
                   background)


def draw_lines(axis: plt.Axes, mesh: core.Mesh, pattern: tuple[core.Vector, core.Vector, core.Scalar], view_degrees: np.ndarray,
               reach: float) -> None:
    """The lit surface with the middle line of every stripe of every family of a pattern drawn on it, straight
    across each face, ending in the faces where the stripes fork, standing just off the surface."""
    frame = square_view(view_degrees, mesh, 0.0)
    corners, turns = corner_phases(mesh, *pattern)                             # [F, 3, n], [F, n]
    places, normals = positions(mesh.vertices), coordinates(mesh.vertex_normals)
    # Off the surface by a share of its size, so the straight segments clear it between its vertices.
    lifted = (places + standoff * reach * normals)[mesh.faces]                 # [F, 3, 3]
    segments = np.concatenate([crossings(lifted, corners[..., family] / (2 * np.pi))[turns[:, family] == 0].reshape(-1, 2, 3)
                               for family in range(corners.shape[-1])])         # [S, 2, 3]
    drawn = segments[np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1) > 0]
    lines = solids.rods(solids.mv.point(drawn[:, 0]), solids.mv.point(drawn[:, 1]), line_width / 2 * frame.pixel)
    show(axis, compose([(*surface(mesh, light, frame), 1.0), (*solids.layer([(lines, dark)], frame), 1.0)], background),
         frame)


def draw_patterns(mesh: core.Mesh, patterns: list[tuple[core.Vector, core.Vector, core.Scalar]], draw, reach: float,
                  views: np.ndarray) -> plt.Figure:
    """The surface side by side for each pattern of lines, stripe fields and edge targets, each drawn by
    `draw_bands` or `draw_lines` and seen from its (elevation, azimuth) `[patterns, 2]`."""
    figure, axes = plt.subplots(1, len(patterns), figsize=(4.2 * len(patterns), 4.2), dpi=110)
    for axis, pattern, view_degrees in zip(np.atleast_1d(axes), patterns, views):
        draw(axis, mesh, pattern, view_degrees, reach)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    return figure


def animate_flow(mesh: core.Mesh, lines: core.Vector, stripe_fields: core.Vector, advance: core.Scalar, reach: float) -> list[np.ndarray]:
    """Frames of the stripes as the stripe field `[V, 1]` turns about every normal, `[frames]`."""
    frame = square_view(view, mesh, 0.0)
    return [bands_image(mesh, (lines, stripe_field, advance), frame) for stripe_field in stripe_fields]
