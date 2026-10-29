"""Oriented patches, turning arcs, ribbons, and a director sphere display rotor fields."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.tri import LinearTriInterpolator, Triangulation
import numpy as np

from numga import stack
from examples.surfaces.rotor_whitney import core
from examples.util.wgsl import image_figure, meshes, solids, strokes

sm = solids.mv
background = np.ones(3)
surface_colour = np.array([0.91, 0.93, 0.95])
ink = np.array([0.16, 0.23, 0.30])
coral = np.array([0.91, 0.24, 0.16])
teal = np.array([0.02, 0.65, 0.55])
blue = np.array([0.12, 0.40, 0.88])
gold = np.array([0.99, 0.67, 0.10])
violet = np.array([0.57, 0.27, 0.84])
path_colours = np.stack((coral, gold, teal, blue, violet, [0.90, 0.20, 0.57]))
pixels = 300
n_curve_samples = 96
n_profile_samples = 7
n_ribbons = 5
n_meridians = len(path_colours)
triangle_view = np.array([38.0, -48.0])
disk_view = np.array([40.0, -42.0])
saddle_view = np.array([32.0, -38.0])


# --- plumbing -------------------------------------------------------------------------
def coordinates(elements: core.Point) -> np.ndarray:
    """Coordinates are read by pairing intrinsic points with the drawing frame."""
    mv = elements.context.multivector
    return (stack((mv.x, mv.y, mv.z)) & elements.batch()[..., None]).to_array()


def square_view(view_degrees: np.ndarray, vertices: core.Point, spare: float) -> solids.View:
    """A square orthographic view frames the supplied geometric points."""
    elevation, azimuth = np.radians(view_degrees + [0, 90])
    return solids.view([sm.point(coordinates(vertices))], spare, elevation, azimuth, 0.06, pixels).squared()


def sheet(places: np.ndarray, faces: np.ndarray, colours: np.ndarray,
          view: solids.View) -> meshes.Samples:
    """A lit sheet carries its vertex colours at the image's own pixel size."""
    points = sm.point(places)
    return meshes.samples(points, faces, colours, meshes.smooth(points, faces), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def boundary_lines(places: np.ndarray, edges: np.ndarray, view: solids.View) -> meshes.Samples:
    """Thin outlines keep the reference geometry visible beneath the field."""
    return strokes.segments(sm.point(places[edges[:, 0]]), sm.point(places[edges[:, 1]]), ink, 1.1, view)


def panels(images: list[np.ndarray]) -> plt.Figure:
    """Separate geometric views share a narrow white gutter."""
    gutter = np.full((pixels, 8, 3), 255, dtype=np.uint8)
    pieces = [piece for i, picture in enumerate(images) for piece in ([gutter, picture] if i else [picture])]
    return image_figure(np.concatenate(pieces, axis=1))


def frame_patches(starts: np.ndarray, rotors: core.Rotor, size: float,
                  view: solids.View) -> tuple[meshes.Samples, meshes.Samples]:
    """Coloured rigid patches expose a frame's plane and its in-plane orientation."""
    mv = rotors.context.multivector
    axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    directions = coordinates(rotors.batch()[..., None] >> axes)
    profile = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, 0]])
    local = size * profile
    places = starts[:, None, :] + local[None, :, :1] * directions[:, None, 0] + local[None, :, 1:] * directions[:, None, 1]
    # A shallow centre ridge lets light reveal an almost edge-on patch.
    places[:, -1] += 0.16 * size * directions[:, 2]
    local_faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    faces = (local_faces[None] + np.arange(len(starts))[:, None, None] * len(profile)).reshape(-1, 3)
    colours = np.broadcast_to(np.stack((coral, gold, teal, blue))[None, :, None, :], (len(starts), 4, 3, 3)).reshape(-1, 3, 3)
    patches = sheet(places.reshape(-1, 3), faces, colours, view)
    perimeter = np.concatenate((places[:, :4], places[:, :1]), axis=1)
    seams = strokes.polylines([(sm.point(path), ink) for path in perimeter], 1.0, view)
    return patches, seams


def draw_darboux_whitney(
    corners: core.Point, points: core.Point,
    fine_mesh: core.Mesh, fine_vertices: core.Point, fine_norms_squared: core.Scalar,
    corner_rotors: core.Rotor, unit_rotors: core.Rotor,
    whitney_rates: core.RateMap,
) -> plt.Figure:
    """Frame patches, rate-plane arcs, and a chord-norm bowl expose the derivative's grade split."""
    view = square_view(triangle_view, corners, 0.30)
    corner_xyz, sample_xyz = coordinates(corners), coordinates(points)
    fine_xyz = coordinates(fine_vertices)
    normal = np.cross(corner_xyz[1] - corner_xyz[0], corner_xyz[2] - corner_xyz[0])
    normal /= np.linalg.norm(normal)
    outline = boundary_lines(fine_xyz, fine_mesh.edges[fine_mesh.boundary_edges], view)
    base = sheet(fine_xyz, fine_mesh.faces, surface_colour, view)

    n_inner_patches = 4
    selected = np.linspace(0, len(sample_xyz) - 1, n_inner_patches, dtype=int)
    corner_tiles, corner_seams = frame_patches(corner_xyz + 0.04 * normal, corner_rotors, 0.13, view)
    tiles, seams = frame_patches(sample_xyz[selected] + 0.07 * normal, unit_rotors[selected], 0.085, view)
    orientation = solids.draw([], view, background, inks=[], translucent=[],
                             meshes=[(base, 1.0), (outline, 1.0), (corner_tiles, 1.0), (corner_seams, 1.0), (tiles, 1.0), (seams, 1.0)])

    # The visible probe fixes the input of the rate map; the arcs lie in its returned bivector planes.
    probe = (corners[1] - corners[0]).normalized()
    rates = whitney_rates(probe)[selected]
    ray = probe.commutator(rates).normalized()
    spans = 2.4 * rates.norm().to_array()
    phases = np.linspace(0, 1, n_curve_samples)
    turns = (rates.normalized()[:, None] * (spans[:, None] * phases / 2)).exp()
    arc_xyz = coordinates(turns >> ray[:, None]) * 0.115
    centres = sample_xyz[selected] + 0.14 * normal
    arc_xyz += centres[:, None]
    progression = phases[:-1, None] * gold + (1 - phases[:-1, None]) * coral
    arc_colours = np.broadcast_to(progression, arc_xyz[:, :-1].shape)
    arcs = strokes.segments(sm.point(arc_xyz[:, :-1].reshape(-1, 3)), sm.point(arc_xyz[:, 1:].reshape(-1, 3)), arc_colours.reshape(-1, 3), 3.0, view)
    probe_xyz = coordinates(probe)
    probe_starts = sample_xyz[selected] + 0.025 * normal - 0.08 * probe_xyz
    probe_ends = probe_starts + 0.18 * probe_xyz
    probes = strokes.segments(sm.point(probe_starts), sm.point(probe_ends), blue, 2.0, view)
    probe_ends_mark = solids.ball(sm.point(probe_ends), 0.018)[:, None]
    arc_ends = solids.ball(sm.point(arc_xyz[:, -1]), 0.018)[:, None]
    rates_image = solids.draw([(probe_ends_mark, blue), (arc_ends, gold)], view, background, inks=[], translucent=[],
                             meshes=[(base, 1.0), (outline, 1.0), (probes, 1.0), (arcs, 1.0)])

    # Height makes the chord's loss of norm a geometric bowl, with its corners pinned to unit norm.
    norms = fine_norms_squared.batch().to_array()
    bowl_xyz = fine_xyz + 1.3 * (norms - 1)[:, None] * normal
    depth = np.clip((1 - norms) / (1 - norms.min()), 0, 1)
    bowl_colours = (1 - depth[:, None]) * np.array([0.50, 0.83, 0.98]) + depth[:, None] * blue
    bowl = sheet(bowl_xyz, fine_mesh.faces, bowl_colours[fine_mesh.faces], view)
    corner_marks = solids.ball(sm.point(corner_xyz), 0.025)[:, None]
    norm_image = solids.draw([(corner_marks, gold)], view, background, inks=[], translucent=[], meshes=[(bowl, 1.0), (outline, 1.0)])
    return panels([orientation, rates_image, norm_image])


def concentric_paths(places: np.ndarray) -> np.ndarray:
    """Circular display paths stay inside the disk's polygonal boundary."""
    radius = np.hypot(places[:, 0], places[:, 1]).max()
    radii = radius * np.linspace(0.18, 0.90, n_ribbons)
    angles = np.linspace(0, 2 * np.pi, n_curve_samples)
    return radii[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def sample_paths(triangulation: Triangulation, paths: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Linear drawing interpolation samples a vertex field along fixed display paths."""
    return np.stack([LinearTriInterpolator(triangulation, component)(paths[..., 0], paths[..., 1]).data
                     for component in values.T], axis=-1)


def draw_skyrmion(mesh: core.Mesh, vertices: core.Point, directors: core.Point) -> plt.Figure:
    """Matched paths show the disk's centre reaching the north pole and its rim collapsing to the south pole."""
    places, directions = coordinates(vertices), coordinates(directors)
    disk_frame = square_view(disk_view, vertices, 0.04)
    sphere_frame = square_view(np.array([0.0, -42.0]), directors, 0.04)
    triangulation = Triangulation(places[:, 0], places[:, 1], mesh.faces)
    rings = concentric_paths(places)
    boundary = mesh.exterior(0)
    angles = np.arctan2(places[boundary, 1], places[boundary, 0])
    ends = [boundary[np.argmin(np.abs(np.angle(np.exp(1j * (angles - angle)))))]
            for angle in np.linspace(-np.pi, np.pi, n_meridians, endpoint=False)]
    # Rays stop just inside their boundary vertices so triangle lookup is unambiguous.
    radii = np.linspace(0, 1 - 1e-8, n_curve_samples)
    meridians = places[ends, None, :2] * radii[None, :, None]

    paths = np.concatenate((rings, meridians))
    domain_paths = sample_paths(triangulation, paths, places) + [0, 0, 0.012]
    sphere_paths = sample_paths(triangulation, paths, directions)
    sphere_paths *= 1.012 / np.linalg.norm(sphere_paths, axis=-1, keepdims=True)
    colours = np.concatenate((np.broadcast_to([0.56, 0.64, 0.72], (len(rings), 3)), path_colours))
    domain_lines = [(sm.point(path), colour) for path, colour in zip(domain_paths, colours)]
    sphere_lines = [(sm.point(path), colour) for path, colour in zip(sphere_paths, colours)]

    disk = sheet(places, mesh.faces, surface_colour, disk_frame)
    domain_grid = strokes.polylines(domain_lines, 2.4, disk_frame)
    centre = solids.ball(sm.point([[0, 0, 0.018]]), 0.045)[:, None]
    domain = solids.draw([(centre, coral)], disk_frame, background, inks=[], translucent=[], meshes=[(disk, 1.0), (domain_grid, 1.0)])
    sphere = solids.ball(sm.point([[0, 0, 0]]), 1.0)[:, None]
    sphere_grid = strokes.polylines(sphere_lines, 2.8, sphere_frame)
    poles = solids.ball(sm.point([[0, 0, 1.02], [0, 0, -1.02]]), 0.045)[:, None]
    image = solids.draw([(sphere, surface_colour), (poles, np.stack((coral, blue)))], sphere_frame, background,
                        inks=[], translucent=[], meshes=[(sphere_grid, 1.0)])
    return panels([domain, image])


def ribbon_panel(mesh: core.Mesh, vertices: core.Point, rotors: core.Rotor,
                 paths: np.ndarray, view: solids.View) -> np.ndarray:
    """Fixed centre curves carry cross-sections turned by the rotor field."""
    places = coordinates(vertices)
    mv = rotors.context.multivector
    frame_axes = stack((mv.x.dual(), mv.y.dual(), mv.z.dual()))
    axes = coordinates(rotors.batch()[:, None] >> frame_axes)
    triangulation = Triangulation(places[:, 0], places[:, 1], mesh.faces)
    centres = sample_paths(triangulation, paths, places) + [0, 0, 0.09]
    radial = paths / np.linalg.norm(paths, axis=-1, keepdims=True)
    widths = (radial[..., :1] * sample_paths(triangulation, paths, axes[:, 0])
              + radial[..., 1:] * sample_paths(triangulation, paths, axes[:, 1]))
    widths /= np.linalg.norm(widths, axis=-1, keepdims=True)
    heights = sample_paths(triangulation, paths, axes[:, 2])
    heights /= np.linalg.norm(heights, axis=-1, keepdims=True)
    profile = np.linspace(-1, 1, n_profile_samples)
    n_segments = n_curve_samples - 1
    first = np.arange(n_segments)[:, None] * n_profile_samples + np.arange(n_profile_samples - 1)
    faces = np.concatenate((np.stack((first, first + 1, first + n_profile_samples), axis=-1).reshape(-1, 3),
                            np.stack((first + 1, first + n_profile_samples + 1, first + n_profile_samples), axis=-1).reshape(-1, 3)))
    base = sheet(places, mesh.faces, surface_colour, view)
    layers = [(base, 0.34), (boundary_lines(places, mesh.edges[mesh.boundary_edges], view), 0.65)]
    for centre, width, height in zip(centres, widths, heights):
        cross_section = 0.065 * profile[None, :, None] * width[:, None] + 0.022 * (1 - profile**2)[None, :, None] * height[:, None]
        ribbon = centre[:, None] + cross_section
        across_colours = (1 - (profile + 1)[:, None] / 2) * coral + ((profile + 1)[:, None] / 2) * blue
        colours = np.broadcast_to(across_colours, (n_curve_samples, n_profile_samples, 3)).reshape(-1, 3)
        layers.append((sheet(ribbon.reshape(-1, 3), faces, colours[faces], view), 1.0))
        seam = strokes.polylines([(sm.point(ribbon[:, 0]), gold), (sm.point(ribbon[:, -1]), teal)], 1.3, view)
        layers.append((seam, 1.0))
    return solids.draw([], view, background, inks=[], translucent=[], meshes=layers)


def draw_rotor_frames(mesh: core.Mesh, vertices: core.Point, harmonic_rotors: core.Rotor) -> plt.Figure:
    """Ribbon cross-sections compare orientation fields with different boundary prescriptions."""
    view = square_view(saddle_view, vertices, 0.20)
    paths = concentric_paths(coordinates(vertices))
    return panels([ribbon_panel(mesh, vertices, rotors.field(), paths, view) for rotors in harmonic_rotors.batch()])


def animate_relaxation(mesh: core.Mesh, vertices: core.Point,
                       states: Iterable[tuple[core.Rotor, core.Scalar]]) -> Iterator[np.ndarray]:
    """Ribbon cross-sections settle as the unit rotor field smooths over its fixed surface."""
    view = square_view(saddle_view, vertices, 0.20)
    paths = concentric_paths(coordinates(vertices))
    for rotors, _ in states:
        yield ribbon_panel(mesh, vertices, rotors, paths, view)
