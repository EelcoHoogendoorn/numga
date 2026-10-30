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

from numga import concatenate, stack, where
from examples.surfaces.connections import core
from examples.util.wgsl import fitted, meshes, show, solids, strokes
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
holonomy_stride = 5
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
# A pattern panel's side in inches and its dots per inch, at which the stripes' middle lines are `line_width` pixels
# wide, and narrower in proportion on a smaller panel.
panel_inches, panel_dpi = 4.2, 110
background = np.ones(3)


# The unit volume, an intrinsic scalar: a normal's dual, the plane its tangent vectors turn in, times it is the unit
# direction square to the tangent plane.
unit_volume = core.mv.antipseudoscalar([1.0])                                   # [] Antipseudoscalar-


# --- views and parts ------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, vertices: core.Point, spare: float) -> solids.View:
    """A square view framing a mesh's vertices, with `spare` room around them, from a view in degrees, the azimuth measured
    from x."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90)
    frame = solids.view([solids.origin + solids.context.include_from_algebra(vertices.batch())], spare, elevation, azimuth, margin, pixels)
    return frame.squared()


def needle_length(mesh: core.Mesh, vertices: core.Point) -> core.Scalar:
    """The length of the designed fields' needles on a surface: a share of its mean edge length."""
    lengths = (mesh.d[0] * vertices).norm()                                     # Scalar+[edges]
    return field_needle * lengths.batch().mean()


def facing(view_degrees: np.ndarray) -> core.Point:
    """The unit direction toward the viewer, for a view as (elevation, azimuth) in degrees from x: x raised toward z
    by the elevation and turned about z by the azimuth."""
    mv = core.mv
    elevation, azimuth = np.radians(view_degrees)
    rise = (mv.zx * (elevation / 2)).exp()                                      # [] Rotor+
    spin = (mv.xy * (-azimuth / 2)).exp()                                       # [] Rotor+
    return (spin * rise) >> mv.x.dual()                                         # [] Point-


def needles(origins: core.Point, directions: core.Point, normals: core.Vector,
            length: float) -> tuple[solids.Point, solids.Point]:
    """Arrows along the directions `[arrows]`, with their heads in each point's tangent plane: a shaft and two strokes
    to the tip, as segments, their starts and ends `[arrows * 3]`."""
    tips = origins + directions * length                                        # [arrows] Point-
    # The heads spread to either side of the shaft by a quarter turn of it in the tangent plane: its commutator with
    # the plane a tangent vector turns in, the normal's dual.
    side = directions.commutator(normals.dual())                                # [arrows] Point-
    base = origins + directions * (0.62 * length)                               # [arrows] Point-
    starts = concatenate([origins, base + side * (0.25 * length), base - side * (0.25 * length)])
    return solids.origin + solids.context.include_from_algebra(starts), solids.origin + solids.context.include_from_algebra(concatenate([tips, tips, tips]))


def surface(mesh: core.Mesh, vertices: core.Point, colours: np.ndarray, frame: solids.View, face_values: np.ndarray,
            colour: meshes.Colour) -> meshes.Samples:
    """A mesh at its vertices, smooth shaded, at every sample, as `meshes.samples`."""
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())               # [vertices] Point-
    return meshes.samples(points, mesh.faces, colours, meshes.smooth(points, mesh), frame, face_values=face_values,
                          colour=colour, gloss=solids.gloss)


# --- parallel transport ---------------------------------------------------------------
def animate_holonomy(sphere: core.Mesh, vertices: core.Point, rings: core.Vector, states: Iterable[tuple[core.Vector, core.Point]]
                     ) -> Iterator[np.ndarray]:
    """The unit sphere, translucent, and the three latitude circles `[steps + 1, rings]`, and each frame's points
    on them with the tangent vectors carried there `[rings]`: a trail of where the vectors have been, and where
    they are. The sphere is the traced unit ball, the circles' far halves showing through its near surface."""
    frame = square_view(transport_view, vertices, holonomy_needle)
    # On the unit sphere about the origin, each point is the unit direction along its own normal.
    ring_paths = solids.origin + solids.context.include_from_algebra(rings.dual() * unit_volume)   # [steps + 1, rings] Point-
    colours = np.array([to_rgb(colour) for colour in ring_colours])             # [rings, 3]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], surface_colour, sphere_opacity)]
    # The circles and the needles are strokes; the sphere alone is traced. The sphere and the circles stay put.
    circles = strokes.polylines([(ring_paths[:, ring], colour) for ring, colour in enumerate(colours)], 4 * line_width,
                                frame)
    scene = Scene(frame, background, [], [(circles, 1.0)], inks=[], translucent=ball)
    normals, carried = [], []
    for normal, vector in states:
        normals.append(normal)                                                  # [rings] Vector-
        carried.append(vector)                                                  # [rings] Point-
        past, along = stack(normals[::holonomy_stride]), stack(carried[::holonomy_stride])   # [trail, rings] Vector-, Point-
        # Each needle of the trail in its ring's colour: a needle's three strokes, a run per stroke.
        trail = strokes.segments(*needles((past.dual() * unit_volume).reshape(-1), along.reshape(-1), past.reshape(-1),
                                          holonomy_needle), np.tile(np.tile(colours, (past.shape[0], 1)), (3, 1)),
                                 2 * line_width, frame)
        now = strokes.segments(*needles(normal.dual() * unit_volume, vector, normal, holonomy_needle * 1.15),
                               np.tile(colours, (3, 1)), 5 * line_width, frame)
        yield scene.draw([], inks=[], translucent=[], meshes=[(now, 1.0), (trail, trail_opacity)])


def torus_panel(torus: core.Mesh, vertices: core.Point, field: core.Point, amplitudes: core.Scalar,
                frame: solids.View) -> np.ndarray:
    """One shaded torus with face colours weighted by `amplitudes` `Scalar+[vertices]` and front-facing tangent
    needles."""
    places = vertices.batch()                                                   # [vertices] Point-
    normals = core.vertex_normals(torus, vertices).batch()                      # [vertices] Vector-
    units = field.batch().normalized()                                          # [vertices] Point-
    # Each face weighted by the mean of its corners' amplitudes, read through the mesh.
    face_amplitudes = torus.incidence[2, 0].mean(axis=0) * amplitudes           # Scalar+[faces]
    face_weight = np.clip(face_amplitudes.batch().to_array(), 0.0, 1.0)[:, None]
    face_colours = (1.0 - 0.58 * face_weight) * surface_colour + (0.58 * face_weight) * warm_colour
    visible = (normals & facing(transport_view)) > 0.05
    scale = 0.25 + 0.75 * np.clip(amplitudes.batch().to_array()[visible], 0.0, 1.0)
    # The needles stand just off the surface, along each vertex's normal.
    lifted = places + normals.dual() * (0.03 * unit_volume)                     # [vertices] Point-
    arrows = strokes.segments(*needles(lifted[visible], units[visible] * scale, normals[visible], transport_needle),
                              needle_colour, 2 * line_width, frame)
    return solids.draw([], frame, background,
                       inks=[], translucent=[],
                       meshes=[(surface(torus, vertices, np.zeros(3), frame, face_values=face_colours, colour=meshes.by_face), 1.0),
                               (arrows, 1.0)])


def draw_torus_fields(torus: core.Mesh, vertices: core.Point, fields: list[core.Point]) -> plt.Figure:
    """The torus side by side for each tangent vector field in `fields`."""
    figure, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 3.6), dpi=100)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(transport_view, vertices, transport_needle), axes[0])
    for axis, field in zip(axes, fields):
        lengths = field.norm()                                                 # Scalar+[vertices]
        peak = max(float(np.percentile(lengths.batch().to_array(), 95.0)), 1e-12)
        show(axis, torus_panel(torus, vertices, field, lengths / peak, frame), frame)
    return figure


def animate_diffusion(torus: core.Mesh, vertices: core.Point, diffused: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Frames of a vector heat pulse spreading across the torus, a `Point-[vertices]` each, each scaled by its own
    lengths."""
    frame = square_view(transport_view, vertices, transport_needle)
    for field in diffused:
        lengths = field.norm().batch().to_array()                              # [vertices]
        peak = np.percentile(lengths, 92.0)
        scaled = np.pow(np.clip(lengths / np.maximum(peak, 1e-12), 0.0, 1.0), 0.45)
        # The amplitudes, brightened, at the vertices, as a field the mesh carries to its faces.
        amplitudes = core.mv.scalar(scaled[:, None]).field()                   # Scalar+[vertices]
        yield torus_panel(torus, vertices, field, amplitudes, frame)


# --- designed fields ------------------------------------------------------------------
def field_marks(mesh: core.Mesh, vertices: core.Point, singular: np.ndarray, indices: list[int], frame: solids.View
                ) -> list[tuple[solids.Quadric, np.ndarray]]:
    """Balls on a surface's singular faces `[S]`, at the mean of their corners, larger for larger indices."""
    middles = (mesh.incidence[2, 0].mean(axis=0) * vertices).batch()[singular]   # [S] Point-
    centres = solids.origin + solids.context.include_from_algebra(middles)                       # [S] Point-
    return [(solids.ball(centres[index:index + 1], singular_radius * abs(value) * frame.pixel)[:, None],
             positive if value > 0 else negative) for index, value in enumerate(indices)]


def field_arrows(mesh: core.Mesh, vertices: core.Point, field: core.Point, needle: float, frame: solids.View
                 ) -> meshes.Samples:
    """A unit tangent field on the vertices of a surface facing the viewer, as strokes."""
    places, normals = vertices.batch(), core.vertex_normals(mesh, vertices).batch()   # [vertices] Point-, Vector-
    seen = (normals & facing(view)) > 0.1
    # The needles stand just off the surface, along each vertex's normal.
    lifted = places + normals.dual() * (0.01 * unit_volume)                     # [vertices] Point-
    return strokes.segments(*needles(lifted[seen], field.batch()[seen], normals[seen], needle),
                            needle_colour, 2 * line_width, frame)


def field_panel(mesh: core.Mesh, vertices: core.Point, field: core.Point, singular: np.ndarray, indices: list[int], needle: float,
                frame: solids.View) -> np.ndarray:
    """One surface with its unit tangent field on the vertices facing the viewer, and balls on its singular faces
    `[S]`, larger for larger indices."""
    arrows = field_arrows(mesh, vertices, field, needle, frame)
    return solids.draw(field_marks(mesh, vertices, singular, indices, frame), frame, background,
                       inks=[], translucent=[],
                       meshes=[(surface(mesh, vertices, surface_colour, frame, face_values=np.zeros(4), colour=meshes.plain), 1.0),
                               (arrows, 1.0)])


def draw_fields(mesh: core.Mesh, vertices: core.Point, scenes: list[tuple[core.Point, np.ndarray, list[int]]]) -> plt.Figure:
    """The surface side by side for each scene's field and singular faces, each panel as wide as the surface across
    x over its height, up to the widest a panel gets."""
    mv = core.mv
    # The surface's extent across x and up z, read by pairing its vertices with those planes, sets the panels' shape.
    across, up = np.ptp((stack((mv.x, mv.z)) & vertices.batch()[..., None]).to_array(), axis=0)
    width = min(across / up, widest_panel)
    needle = needle_length(mesh, vertices)
    figure, axes = plt.subplots(1, len(scenes), figsize=(3.6 * len(scenes) * width, 3.6), dpi=110)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(view, vertices, solids.context.include_from_algebra(needle)), axes[0])
    for axis, (field, singular, indices) in zip(axes, scenes):
        show(axis, field_panel(mesh, vertices, field, singular, indices, needle, frame), frame)
    return figure


def animate_turning(mesh: core.Mesh, vertices: core.Point, fields: Iterable[core.Point], singular: np.ndarray, indices: list[int]
                    ) -> Iterator[np.ndarray]:
    """Frames of a parallel field on the sphere turned about every normal, a `Point-[vertices]` each, among the surface
    and the balls on its singular faces, which stay put."""
    needle = needle_length(mesh, vertices)
    frame = square_view(view, vertices, solids.context.include_from_algebra(needle))
    scene = Scene(frame, background, field_marks(mesh, vertices, singular, indices, frame),
                  [(surface(mesh, vertices, surface_colour, frame, face_values=np.zeros(4), colour=meshes.plain), 1.0)],
                  inks=[], translucent=[])
    for field in fields:
        yield scene.draw([], inks=[], translucent=[], meshes=[(field_arrows(mesh, vertices, field, needle, frame), 1.0)])


# --- stripes --------------------------------------------------------------------------
def corner_phases(mesh: core.Mesh, vertices: core.Point, lines: core.Point, stripes: core.Point, advance: core.Scalar) -> tuple[core.Scalar, np.ndarray]:
    """The phase of each family at each corner of each face `[F, 3, n]`, and the whole turns it winds around
    the face `[F, n]`, for the lines of the families at every vertex `[n] Point-[V]`, their stripe fields
    `[n] Point-[V]` and the targets across every edge for each line at its head `[n] Scalar+[E]`. Each corner's
    lines are matched to the face's first corner's, by the line nearest each and the sense it points in; walking
    around the face along the edges' targets, each corner's phase is taken nearest the one before it plus the
    target; a face around which the walk comes back off by whole turns holds a fork, and the mismatch is spread
    evenly over its three edges, as in the reference implementation of the stripe patterns paper."""
    corners = mesh.incidence[2, 0]                                             # [3] Scalar+[faces <- vertices]
    corner_lines = (corners[:, None] * lines).batch()                          # [3, n, faces] Point-
    corner_stripes = (corners[:, None] * stripes).batch()                      # [3, n, faces] Point-
    corner_normals = (corners * core.vertex_normals(mesh, vertices)).batch()   # [3, faces] Vector-
    # The angle from each corner's stripe field to its line: the product of the two is the turn between them in the
    # tangent plane, and its logarithm's pairing with that plane is the angle.
    between = (corner_stripes * corner_lines.reverse()).normalized()           # [3, n, faces] Rotor+
    phases = between.log().scalar_product(corner_normals[:, None].dual())      # [3, n, faces] Scalar+
    # How far each line at each corner runs along each line at the face's first corner.
    alignment = corner_lines[:1, :, None].dot(corner_lines[:, None, :])        # [3, n, n, faces] Scalar+
    face, corner, family = np.indices((phases.shape[-1], 3, phases.shape[1]))
    phase = phases[corner, family, face]                                     # [faces, corners, families] Scalar+
    dots = alignment[corner[..., None], family[..., None], np.arange(phases.shape[1]), face[..., None]]
    picked = dots.abs().argmax(axis=-1)
    agreements = dots[face, corner, family, picked]
    sense = np.asarray(agreements > 0, dtype=int) - np.asarray(agreements < 0, dtype=int)
    local = phase[face, corner, picked] * sense                              # [faces, corners, families] Scalar+
    # The target from each corner to the next, read at the edge's head, the higher vertex. The side from a corner to
    # the next faces the corner after them, and runs along its edge where the face runs along it.
    edge = mesh.face_edges[:, [2, 0, 1]]                                       # [F, 3]
    forward = mesh.face_edge_orientation[:, [2, 0, 1]] > 0                     # [F, 3]
    head = np.where(forward, np.roll(np.arange(3), -1), np.arange(3))          # [F, 3]: the corner at the edge's head
    at_head = np.take_along_axis(picked, head[..., None], axis=1)              # [F, 3, n]
    sense_head = np.take_along_axis(sense, head[..., None], axis=1)
    target = advance.batch()[at_head, edge[..., None]] * sense_head * np.where(forward, 1.0, -1.0)[..., None]
    walked = [local[:, 0]]
    for side in range(3):
        expected = walked[-1] + target[:, side]
        # The next corner's phase taken nearest the one expected, by whole turns: the turn through their difference in
        # that corner's tangent plane, whose logarithm's angle lies within half a turn.
        plane = corner_normals[(side + 1) % 3].dual()[:, None]                  # [faces, 1] Bivector+
        mismatch = (plane * (local[:, (side + 1) % 3] - expected)).exp()        # [faces, n] Rotor+
        walked.append(expected + mismatch.log().dot(plane))
    walked = stack(walked, axis=1)                                          # [F, 4, n]
    turns = np.rint(((walked[:, 3] - walked[:, 0]) / (2 * np.pi)).to_array()).astype(int)              # [F, n]
    spread = walked[:, :3] - 2 * np.pi * turns[:, None] * (np.arange(3) / 3)[:, None]
    return spread, turns


def crossings(corners: core.Point, phases: core.Scalar) -> core.Point:
    """The segments `[2, faces, levels]` across faces with corners `[3, faces]` where a phase linear across each,
    given at its corners in whole turns `[3, faces]`, passes each whole number; `levels` is the most whole numbers
    any face holds, and a face holding fewer repeats its first corner as an empty segment."""
    low = phases.min(axis=0)
    high = phases.max(axis=0)
    # Integer stripe labels bound the batch of level sets; their interpolation stays in the scalar field.
    lowest, highest = np.floor(low.to_array()).astype(int), np.ceil(high.to_array()).astype(int)
    levels = lowest[:, None] + 1 + np.arange((highest - lowest).max())
    start, end = phases[..., None], phases[np.roll(np.arange(3), -1), :, None]
    mv = phases.context.multivector
    share = (levels - start) / where((end - start).abs() > 0, end - start, mv.scalar([np.inf]))
    across = (share >= 0) & (share < 1)                                       # [3, faces, levels]
    following = corners[np.roll(np.arange(3), -1)]                            # [3, faces] Point-
    # Where each level meets each side: that share of the way from the side's corner to the next.
    meeting = corners[..., None] + (following - corners)[..., None] * share   # [3, faces, levels] Point-
    # The two sides a level crosses, first and second; a level crossing none collapses onto the first corner.
    order = np.argsort(~across, axis=0, kind="stable")[:2]                    # [2, faces, levels]
    ends = where(across, meeting, corners[:1, :, None])                        # [3, faces, levels] Point-
    face, level = np.indices(order.shape[1:])
    return ends[order, face, level]                                            # [2, faces, levels] Point-


# WGSL colouring a surface by the stripes of up to four families, `n_families`: each family's phase, the corners'
# blended across the face, plus the whole turns a fork winds around it, spread by an angle about the face's middle
# that rises by a whole turn once around; light where every family's stripe is light.
striped = """
// An angle around the middle of a triangle from how near a pixel lies to each corner, rising by a whole turn once
// around: piecewise linear on the three sectors nearest each edge.
fn winding(near: vec3<f32>) -> f32 {
    let least = min(near.x, min(near.y, near.z));
    let along = select(select(near.y - near.x, near.x - near.z, near.y == least), near.z - near.y, near.x == least);
    let start = select(select(1.0, 5.0, near.y == least), 3.0, near.x == least);
    return (3.14159265 / 3) * (start + along / max(1 - 3 * least, 1e-6));
}

fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let phases = values + winding(corner) * face_values;
    let shares = 0.5 - 0.5 * tanh(sharpness * (cos(phases) - cos(3.14159265 * bands)));
    var share = 1.0;
    for (var family = 0u; family < n_families; family++) {
        share *= shares[family];
    }
    return vec4<f32>(mix(dark_stripe, light_stripe, share), 1.0);
}
"""


def bands_image(mesh: core.Mesh, vertices: core.Point, pattern: tuple[core.Point, core.Point, core.Scalar], frame: solids.View) -> np.ndarray:
    """The surface shaded with the stripes of every family of a pattern of lines `[n] Point-[V]`, stripe fields
    `[n] Point-[V]` and edge targets `[n] Scalar+[E]`, the given share of each stripe dark."""
    corners, turns = corner_phases(mesh, vertices, *pattern)                   # [F, 3, n], [F, n]
    stripes = meshes.Colour(striped, dict(n_families=corners.shape[-1], sharpness=sharpness, bands=bands,
                                          dark_stripe=dark, light_stripe=light))
    return solids.draw([], frame, background,
                       inks=[], translucent=[],
                       meshes=[(surface(mesh, vertices, corners.to_array(), frame, face_values=turns, colour=stripes), 1.0)])


def lines_image(mesh: core.Mesh, vertices: core.Point, pattern: tuple[core.Point, core.Point, core.Scalar], frame: solids.View) -> np.ndarray:
    """The lit surface with the middle line of every stripe of every family of a pattern drawn on it, straight
    across each face, ending in the faces where the stripes fork, standing just off the surface."""
    corners, turns = corner_phases(mesh, vertices, *pattern)                   # [F, 3, n], [F, n]
    places, normals = vertices.batch(), core.vertex_normals(mesh, vertices).batch()   # [vertices] Point-, Vector-
    # Off the surface by a share of its radius about the centre, so the straight segments clear it between its
    # vertices.
    radii = places.norm()
    radius = radii.max()
    lifted = places + normals.dual() * (standoff * radius * unit_volume)        # [vertices] Point-
    face_corners = (mesh.incidence[2, 0] * lifted.field()).batch()              # [3, faces] Point-
    # The faces where a family forks hold no middle lines of it.
    segments = concatenate([crossings(face_corners[:, turns[:, family] == 0],
                                      stack([corners[turns[:, family] == 0, corner, family] for corner in range(3)]) / (2 * np.pi)).reshape(2, -1)
                            for family in range(corners.shape[-1])], axis=1)   # [2, segments] Point-
    drawn = solids.origin + solids.context.include_from_algebra(segments[:, (segments[1] - segments[0]).norm() > 0])   # [2, segments] Point-
    lines = strokes.segments(drawn[0], drawn[1], dark, line_width * frame.pixels / (panel_inches * panel_dpi), frame)
    return solids.draw([], frame, background, inks=[], translucent=[],
                       meshes=[(surface(mesh, vertices, light, frame, face_values=np.zeros(4), colour=meshes.plain), 1.0),
                               (lines, 1.0)])


def pattern_figure(mesh: core.Mesh, vertices: core.Point, patterns: list[tuple[core.Point, core.Point, core.Scalar]],
                   image: Callable[[core.Mesh, core.Point, tuple[core.Point, core.Point, core.Scalar], solids.View], np.ndarray],
                   scale: float) -> plt.Figure:
    """The surface side by side for each pattern, as `image` draws it in the square view fitted to a panel, the panels
    `scale` times the side of the full size."""
    side = scale * panel_inches
    figure, axes = plt.subplots(1, len(patterns), figsize=(side * len(patterns), side), dpi=panel_dpi)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(view, vertices, 0.0), axes[0])
    for axis, pattern in zip(axes, patterns):
        show(axis, image(mesh, vertices, pattern, frame), frame)
    return figure


def draw_bands(mesh: core.Mesh, vertices: core.Point, patterns: list[tuple[core.Point, core.Point, core.Scalar]]) -> plt.Figure:
    """The surface side by side for each pattern of lines, stripe fields and edge targets, shaded with its
    stripes."""
    return pattern_figure(mesh, vertices, patterns, bands_image, 1.0)


def draw_lines(mesh: core.Mesh, vertices: core.Point, patterns: list[tuple[core.Point, core.Point, core.Scalar]],
               scale: float) -> plt.Figure:
    """The surface side by side for each pattern of lines, stripe fields and edge targets, with the middle lines
    of its stripes drawn on it, the panels `scale` times the side of the full size."""
    return pattern_figure(mesh, vertices, patterns, lines_image, scale)


def animate_flow(mesh: core.Mesh, vertices: core.Point, lines: core.Point, stripe_fields: Iterable[core.Point], advance: core.Scalar
                 ) -> Iterator[np.ndarray]:
    """Frames of the stripes as the stripe field `[1] Point-[V]` turns about every normal, one frame each."""
    frame = square_view(view, vertices, 0.0)
    for stripe_field in stripe_fields:
        yield bands_image(mesh, vertices, (lines, stripe_field, advance), frame)
