"""GPU rendering for surface connections: spherical holonomy trails, tangent surface grain, and stripes.

The holonomy scene traces a translucent unit ball via `solids` while compositing latitude rings and
chevron-tipped tangent needles in depth order. Tangent vector fields and vector heat diffusion draw
as oriented surface grain via `flow`, while stripe patterns interpolate unwrapped corner phases and
fork winding numbers directly in a fragment shader (`meshes.Colour`) or trace zero-level stripe
centrelines lifted slightly off the triangulated surface.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

from numga import concatenate, stack, where
from numga.render import Scene, flow, meshes, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology
from examples.surfaces.connections import core
from examples.util.panels import fitted, show

# Surface palette, ring colours, and camera elevations and azimuths in degrees from x.
surface_colour = np.array([0.92, 0.91, 0.88])
ring_colours = ("#c84b31", "#2d6a9f", "#2a8c6a")
transport_view = np.array([28.0, -36.0])
view = np.array([24.0, -36.0])
holonomy_stride = 5
holonomy_needle = 0.25
needle_chevron = 3.0
transport_margin = 0.32
widest_panel = 1.6
sphere_opacity = 0.12
trail_opacity = 0.3
dark, light = np.array([0.16, 0.24, 0.36]), np.array([0.93, 0.91, 0.86])
band_fraction = 0.5
sharpness = 12.0
line_width, standoff = 1.4, 0.0066
n_pixels = 420
margin = 0.04
panel_inches, panel_dpi = 4.2, 110
background = np.ones(3)
flow_colour = np.array([0.88, 0.83, 0.72])

# Intrinsic unit volume converting dual tangent planes into outward normal directions.
antiscalar = core.mv.antipseudoscalar([1])                                      # [] Antiscalar-


# --- views and parts ------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, vertices: core.Point, spare: float) -> solids.View:
    """An orthographic camera at `view_degrees` frames `vertices` (`Point-[V]`) inside a square
    viewport with `spare` padding."""
    elevation, azimuth = np.radians(view_degrees[0]), np.radians(view_degrees[1] + 90)
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())                # [V] Point-
    frame = solids.view([points], spare, elevation, azimuth, margin, n_pixels)
    return frame.squared()


def needles(origins: core.Point, directions: core.Point, colours: np.ndarray,
            width: float, frame: solids.View) -> Draw:
    """Chevron-tipped cylindrical tubes from `origins` along `directions` form tangent vector
    needles in `frame`."""
    tips = origins + directions                                                                   # [...] Point-
    ends = solids.origin + solids.context.include_from_algebra(stack((origins, tips), axis=-1))   # [..., 2] Point-
    n_needles = ends.shape[0]
    return simplices.tubes(ends.reshape(-1), Topology.chains([2] * n_needles), colours, width, frame,
                           solids.gloss, chevrons=np.ones(n_needles), chevron=needle_chevron * width)


def shaded_surface(topology: Topology, vertices: core.Point, colours: np.ndarray, frame: solids.View,
                   face_values: np.ndarray, colour: meshes.Colour) -> Draw:
    """Smooth-shaded triangles of `topology` at `vertices` (`Point-[V]`) evaluate fragment `colour`
    over interpolated `colours` and `face_values`."""
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())                # [V] Point-
    return meshes.samples(points, topology.elements[2], colours, meshes.smooth(points, topology), frame,
                          face_values=face_values, colour=colour, gloss=solids.gloss)


# --- parallel transport ---------------------------------------------------------------
def holonomy_animation(sphere: Topology, vertices: core.Point, rings: core.Patch,
                       states: Iterable[tuple[core.Patch, core.Point]]) -> Iterator[np.ndarray]:
    """Yields frames of tangent needles and fading trails carried around latitude `rings` on `sphere`."""
    frame = square_view(transport_view, vertices, holonomy_needle)
    n_points, n_rings = rings.shape
    ring_paths = solids.origin + solids.context.include_from_algebra(rings.dual() * antiscalar)   # [n_points, n_rings] Point-
    colours = np.array([to_rgb(colour) for colour in ring_colours])                               # [n_rings, 3]
    ball = [(solids.ball(solids.origin, 1.0)[None, None], surface_colour, sphere_opacity)]
    circles = simplices.tubes(concatenate([ring_paths[:, ring] for ring in range(n_rings)]),
                              Topology.chains([n_points] * n_rings),
                              np.repeat(colours, n_points - 1, axis=0), 4 * line_width, frame, solids.gloss)
    scene = Scene(frame, background, [], [(circles, 1.0)], inks=[], translucent=ball)
    ring_patches, ring_tangents = [], []
    for patch, vector in states:
        ring_patches.append(patch)                                                                # [n_rings] Patch-
        ring_tangents.append(vector)                                                              # [n_rings] Point-
        past_patches = stack(ring_patches[::holonomy_stride])                                     # [..., n_rings] Patch-
        past_vectors = stack(ring_tangents[::holonomy_stride])                                    # [..., n_rings] Point-
        n_trail = past_patches.shape[0]
        trail_colours = np.tile(colours, (n_trail, 1))                                            # [n_trail * n_rings, 3]
        trail = needles((past_patches.dual() * antiscalar).reshape(-1),
                        past_vectors.reshape(-1) * holonomy_needle, trail_colours, 2 * line_width, frame)
        current = needles(patch.dual() * antiscalar, vector * (holonomy_needle * 1.15),
                          colours, 5 * line_width, frame)
        yield scene.draw([], inks=[], translucent=[], meshes=[(current, 1.0), (trail, trail_opacity)])


def torus_panel(torus: Topology, vertices: core.Point, field: core.Point,
                frame: solids.View) -> np.ndarray:
    """Symmetric surface streaks depict the unit direction of `field` (`Point-[V]`) on `torus`."""
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())                # [V] Point-
    directions = solids.context.include_from_algebra(field.batch().normalized())                  # [V] Point-
    return flow.draw(points, torus, directions, frame, flow_colour, 1.0, downstream_fraction=0.5)


def torus_fields_figure(torus: Topology, vertices: core.Point, fields: list[core.Point]) -> plt.Figure:
    """Side-by-side torus panels display each tangent vector field in `fields`."""
    figure, axes = plt.subplots(1, len(fields), figsize=(3.8 * len(fields), 3.6), dpi=100)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(transport_view, vertices, transport_margin), axes[0])
    for axis, field in zip(axes, fields):
        show(axis, torus_panel(torus, vertices, field, frame), frame)
    return figure


def diffusion_animation(torus: Topology, vertices: core.Point,
                        fields: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Yields symmetric surface-grain frames as vector heat `fields` spread across `torus`."""
    frame = square_view(transport_view, vertices, transport_margin)
    for field in fields:
        yield torus_panel(torus, vertices, field, frame)


# --- designed fields ------------------------------------------------------------------
def field_panel(topology: Topology, vertices: core.Point, field: core.Point,
                frame: solids.View) -> np.ndarray:
    """Directed surface streaks depict the unit tangent `field` (`Point-[V]`) on `topology`."""
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())                # [V] Point-
    directions = solids.context.include_from_algebra(field.batch())                               # [V] Point-
    return flow.draw(points, topology, directions, frame, flow_colour, 1.0)


def fields_figure(topology: Topology, vertices: core.Point,
                  scenes: list[tuple[core.Point, np.ndarray, list[int]]]) -> plt.Figure:
    """Side-by-side surface-grain panels display `scenes` sized to the horizontal-to-vertical aspect
    of `vertices`."""
    mv = core.mv
    readings = stack((mv.x, mv.z))[:, None] & vertices.batch()                                    # [2, V] Scalar+
    width_span, height_span = (readings.max(axis=-1) - readings.min(axis=-1)).to_array()
    width = min(width_span / height_span, widest_panel)
    figure, axes = plt.subplots(1, len(scenes), figsize=(3.6 * len(scenes) * width, 3.6), dpi=110)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(view, vertices, 0.0), axes[0])
    for axis, (field, _, _) in zip(axes, scenes):
        show(axis, field_panel(topology, vertices, field, frame), frame)
    return figure


def turning_animation(topology: Topology, vertices: core.Point,
                      fields: Iterable[core.Point]) -> Iterator[np.ndarray]:
    """Yields surface-grain frames as `fields` rotate continuously around every vertex normal."""
    frame = square_view(view, vertices, 0.0)
    for field in fields:
        yield field_panel(topology, vertices, field, frame)


# --- stripes --------------------------------------------------------------------------
def corner_phases(topology: Topology, vertices: core.Point, lines: core.Point,
                  stripes: core.Point, advance: core.Scalar) -> tuple[core.Scalar, np.ndarray]:
    """Walking around every triangle of `topology` unwraps each stripe family's corner angles
    (`[F, 3, n_families] Scalar+`) and integer fork turns (`[F, n_families]`)."""
    V, E, F = topology.n_elements
    n_families = lines.shape[0]
    corners = topology.incidence[2, 0]                                                            # [3] Scalar+[F <- V]
    corner_lines = (corners[:, None] * lines).batch()                                             # [3, n_families, F] Point-
    corner_stripes = (corners[:, None] * stripes).batch()                                         # [3, n_families, F] Point-
    corner_patches = (corners * core.vertex_patches(topology, vertices)).batch()                  # [3, F] Patch-
    phase_rotors = (corner_stripes * corner_lines.reverse()).normalized()                         # [3, n_families, F] Rotor+
    phases = phase_rotors.log().scalar_product(corner_patches[:, None].dual())                    # [3, n_families, F] Scalar+
    alignment = -(corner_lines[:1, :, None] | corner_lines[:, None, :])                           # [3, n_families, n_families, F] Scalar+
    face, corner, family = np.indices((F, 3, n_families))
    phase = phases[corner, family, face]                                                          # [F, 3, n_families] Scalar+
    dots = alignment[corner[..., None], family[..., None], np.arange(n_families), face[..., None]]
    matched_family = dots.abs().argmax(axis=-1)
    agreements = dots[face, corner, family, matched_family]
    sense = np.asarray(agreements > 0, dtype=int) - np.asarray(agreements < 0, dtype=int)
    local_phase = phase[face, corner, matched_family] * sense                                     # [F, 3, n_families] Scalar+
    edge = topology.boundary_indices[1][:, [2, 0, 1]]                                             # [F, 3]
    edge_positive = topology.orientation[1][:, [2, 0, 1]] > 0                                     # [F, 3]
    head = np.where(edge_positive, np.roll(np.arange(3), -1), np.arange(3))                       # [F, 3]
    head_family = np.take_along_axis(matched_family, head[..., None], axis=1)                     # [F, 3, n_families]
    sense_head = np.take_along_axis(sense, head[..., None], axis=1)
    target = advance.batch()[head_family, edge[..., None]] * sense_head * np.where(edge_positive, 1.0, -1.0)[..., None]
    unwrapped_phases = [local_phase[:, 0]]
    for side in range(3):
        expected = unwrapped_phases[-1] + target[:, side]
        plane = corner_patches[(side + 1) % 3].dual()[:, None]                                    # [F, 1] Bivector+
        mismatch = (plane * (local_phase[:, (side + 1) % 3] - expected)).exp()                    # [F, n_families] Rotor+
        unwrapped_phases.append(expected + mismatch.log().scalar_product(~plane))
    phase_walk = stack(unwrapped_phases, axis=1)                                                  # [F, 4, n_families] Scalar+
    turns = np.rint(((phase_walk[:, 3] - phase_walk[:, 0]) / (2 * np.pi)).to_array()).astype(int)  # [F, n_families]
    corner_angles = phase_walk[:, :3] - 2 * np.pi * turns[:, None] * (np.arange(3) / 3)[:, None]  # [F, 3, n_families] Scalar+
    return corner_angles, turns


def crossings(corners: core.Point, phases: core.Scalar) -> core.Point:
    """Linear interpolation across triangle `corners` (`[3, ...] Point-`) locates straight segments
    (`[2, ..., n_levels] Point-`) where `phases` (`[3, ...] Scalar+` in whole turns) crosses integer
    levels."""
    low = phases.min(axis=0)
    high = phases.max(axis=0)
    lowest, highest = np.floor(low.to_array()).astype(int), np.ceil(high.to_array()).astype(int)
    n_levels = int((highest - lowest).max())
    levels = lowest[:, None] + 1 + np.arange(n_levels)
    start, end = phases[..., None], phases[np.roll(np.arange(3), -1), :, None]
    mv = phases.context.multivector
    share = (levels - start) / where((end - start).abs() > 0, end - start, mv.scalar([np.inf]))
    edge_crossings = (share >= 0) & (share < 1)                                                   # [3, ..., n_levels]
    next_corners = corners[np.roll(np.arange(3), -1)]                                             # [3, ...] Point-
    crossing_points = corners[..., None] + (next_corners - corners)[..., None] * share            # [3, ..., n_levels] Point-
    order = np.argsort(~edge_crossings, axis=0, kind="stable")[:2]                                # [2, ..., n_levels]
    ends = where(edge_crossings, crossing_points, corners[:1, :, None])                           # [3, ..., n_levels] Point-
    face, level = np.indices(order.shape[1:])
    return ends[order, face, level]                                                               # [2, ..., n_levels] Point-


stripe_shader = '''
// Piecewise-linear angle around the triangle centroid rising by 2 * pi once around its three sectors.
fn winding(near: vec3<f32>) -> f32 {
    let least = min(near.x, min(near.y, near.z));
    let sector_offset = select(select(near.y - near.x, near.x - near.z, near.y == least), near.z - near.y, near.x == least);
    let start = select(select(1.0, 5.0, near.y == least), 3.0, near.x == least);
    return (3.14159265 / 3) * (start + sector_offset / max(1 - 3 * least, 1e-6));
}

fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    let phases = values + winding(corner) * face_values;
    let shares = 0.5 - 0.5 * tanh(sharpness * (cos(phases) - cos(3.14159265 * band_fraction)));
    var share = 1.0;
    for (var family = 0u; family < n_families; family++) {
        share *= shares[family];
    }
    return vec4<f32>(mix(dark_stripe, light_stripe, share), 1.0);
}
'''


def bands_image(topology: Topology, vertices: core.Point,
                pattern: tuple[core.Point, core.Point, core.Scalar], frame: solids.View) -> np.ndarray:
    """Smooth anti-aliased stripe bands interpolated from `pattern` shade `topology` in `frame`."""
    corners, turns = corner_phases(topology, vertices, *pattern)
    stripes = meshes.Colour(stripe_shader, dict(n_families=corners.shape[-1], sharpness=sharpness,
                                                band_fraction=band_fraction, dark_stripe=dark, light_stripe=light))
    return solids.draw(
        [], frame, background, inks=[], translucent=[],
        meshes=[(shaded_surface(topology, vertices, corners.to_array(), frame, face_values=turns, colour=stripes), 1.0)],
    )


def lines_image(topology: Topology, vertices: core.Point,
                pattern: tuple[core.Point, core.Point, core.Scalar], frame: solids.View) -> np.ndarray:
    """Zero-level stripe centrelines of `pattern` trace thin tubes lifted slightly off `topology`."""
    corners, turns = corner_phases(topology, vertices, *pattern)
    places, patches = vertices.batch(), core.vertex_patches(topology, vertices).batch()           # [V] Point-, Patch-
    radii = places.norm()                                                                         # [V] Scalar+
    radius = radii.max()                                                                          # [] Scalar+
    lifted_places = places + patches.dual() * (standoff * radius * antiscalar)                    # [V] Point-
    face_corners = (topology.incidence[2, 0] * lifted_places.field()).batch()                     # [3, F] Point-
    segments = concatenate([
        crossings(face_corners[:, turns[:, family] == 0],
                  stack([corners[turns[:, family] == 0, corner, family] for corner in range(3)]) / (2 * np.pi)).reshape(2, -1)
        for family in range(corners.shape[-1])
    ], axis=1)                                                                                    # [2, ...] Point-
    valid_segments = (segments[1] - segments[0]).norm() > 0
    tube_points = solids.origin + solids.context.include_from_algebra(segments[:, valid_segments])  # [2, ...] Point-
    n_segments = tube_points.shape[1]
    lines = simplices.tubes(stack((tube_points[0], tube_points[1]), axis=-1).reshape(-1),
                            Topology.chains([2] * n_segments), dark,
                            line_width * frame.pixels / (panel_inches * panel_dpi), frame, solids.gloss)
    return solids.draw(
        [], frame, background, inks=[], translucent=[],
        meshes=[(shaded_surface(topology, vertices, light, frame, face_values=np.zeros(4), colour=meshes.plain), 1.0),
                (lines, 1.0)],
    )


def pattern_figure(topology: Topology, vertices: core.Point,
                   patterns: list[tuple[core.Point, core.Point, core.Scalar]],
                   image: Callable[[Topology, core.Point, tuple[core.Point, core.Point, core.Scalar], solids.View], np.ndarray],
                   scale: float) -> plt.Figure:
    """Side-by-side panels display `patterns` on `topology` using `image` at panel `scale`."""
    side = scale * panel_inches
    figure, axes = plt.subplots(1, len(patterns), figsize=(side * len(patterns), side), dpi=panel_dpi)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0)
    axes = np.atleast_1d(axes)
    frame = fitted(square_view(view, vertices, 0.0), axes[0])
    for axis, pattern in zip(axes, patterns):
        show(axis, image(topology, vertices, pattern, frame), frame)
    return figure


def bands_figure(topology: Topology, vertices: core.Point,
                 patterns: list[tuple[core.Point, core.Point, core.Scalar]]) -> plt.Figure:
    """Side-by-side panels display shaded stripe bands for `patterns` on `topology`."""
    return pattern_figure(topology, vertices, patterns, bands_image, 1.0)


def lines_figure(topology: Topology, vertices: core.Point,
                 patterns: list[tuple[core.Point, core.Point, core.Scalar]],
                 scale: float) -> plt.Figure:
    """Side-by-side panels display stripe centrelines for `patterns` on `topology` at `scale`."""
    return pattern_figure(topology, vertices, patterns, lines_image, scale)


def sliding_animation(topology: Topology, vertices: core.Point, lines: core.Point,
                      stripe_fields: Iterable[core.Point], advance: core.Scalar) -> Iterator[np.ndarray]:
    """Yields stripe-band frames as `stripe_fields` rotate about every vertex normal."""
    frame = square_view(view, vertices, 0.0)
    for stripe_field in stripe_fields:
        yield bands_image(topology, vertices, (lines, stripe_field, advance), frame)
