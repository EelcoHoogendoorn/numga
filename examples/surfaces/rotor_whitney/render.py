"""Oriented patches, turning arcs, ribbons, and a director sphere display rotor fields."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.sparse import SparseExtensor, spdiag
from examples.surfaces.rotor_whitney import core
from examples.util.wgsl import image_figure, meshes, solids, strokes

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
# Straight up, and the unit volume, an intrinsic scalar: a patch's dual, the plane its directions turn in, times it is
# the unit direction square to the patch.
vertical = core.mv.z.dual()                                                  # [] Point-
unit_volume = core.mv.antipseudoscalar([1.0])                                # [] Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, vertices: core.Point, spare: float) -> solids.View:
    """A square orthographic view frames the supplied geometric points."""
    elevation, azimuth = np.radians(view_degrees + [0, 90])
    places = solids.origin + solids.context.include_from_algebra(vertices.batch())             # [vertices] Point-
    return solids.view([places], spare, elevation, azimuth, 0.06, pixels).squared()


def sheet(places: core.Point, faces: np.ndarray, colours: np.ndarray,
          view: solids.View) -> meshes.Samples:
    """A lit sheet carries its vertex colours at the image's own pixel size."""
    points = solids.origin + solids.context.include_from_algebra(places.batch())                 # [vertices] Point-
    return meshes.samples(points, faces, colours, meshes.smooth(points, meshes.Mesh(len(points), faces)), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def boundary_lines(mesh: core.Mesh, vertices: core.Point, view: solids.View) -> meshes.Samples:
    """Thin outlines keep the reference geometry visible beneath the field."""
    tails, heads = mesh.reading(mesh.boundary_edges, 1) * (mesh.incidence[1, 0] * vertices)   # Point-[rim], Point-[rim]
    return strokes.segments(solids.origin + solids.context.include_from_algebra(tails.batch()),
                            solids.origin + solids.context.include_from_algebra(heads.batch()), ink, 1.1, view)


def panels(images: list[np.ndarray]) -> plt.Figure:
    """Separate geometric views share a narrow white gutter."""
    gutter = np.full((pixels, 8, 3), 255, dtype=np.uint8)
    pieces = [piece for i, picture in enumerate(images) for piece in ([gutter, picture] if i else [picture])]
    return image_figure(np.concatenate(pieces, axis=1))


def frame_patches(starts: core.Point, rotors: core.Rotor, size: float,
                  view: solids.View) -> tuple[meshes.Samples, meshes.Samples]:
    """Coloured rigid patches expose a frame's plane and its in-plane orientation."""
    mv = rotors.context.multivector
    # A square in the frame's own x and y about its middle, the middle raised a little along its z: a shallow ridge
    # lets light reveal an almost edge-on patch.
    profile = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0.16]])
    shape = (stack((mv.x.dual(), mv.y.dual(), mv.z.dual())) * (size * profile)).sum(axis=-1)   # [profile] Point-
    places = starts[:, None] + (rotors.batch()[:, None] >> shape)             # [patches, profile] Point-
    n_patches, n_profile = places.shape
    local_faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    faces = (local_faces[None] + np.arange(n_patches)[:, None, None] * n_profile).reshape(-1, 3)
    colours = np.broadcast_to(np.stack((coral, gold, teal, blue))[None, :, None, :], (n_patches, 4, 3, 3)).reshape(-1, 3, 3)
    patches = sheet(places.reshape(-1), faces, colours, view)
    perimeters = solids.origin + solids.context.include_from_algebra(places[:, [0, 1, 2, 3, 0]])   # [patches, 5] Point-
    seams = strokes.polylines([(perimeters[patch], ink) for patch in range(n_patches)], 1.0, view)
    return patches, seams


def draw_darboux_whitney(
    corners: core.Point, points: core.Point,
    fine_mesh: core.Mesh, fine_vertices: core.Point, fine_norms_squared: core.Scalar,
    corner_rotors: core.Rotor, unit_rotors: core.Rotor,
    whitney_rates: core.RateMap,
) -> plt.Figure:
    """Frame patches, rate-plane arcs, and a chord-norm bowl expose the derivative's grade split."""
    view = square_view(triangle_view, corners, 0.30)
    # The unit direction square to the triangle: the dual of its patch, the join of two of its sides.
    normal = ((corners[1] - corners[0]) & (corners[2] - corners[0])).normalized()   # [] Patch-
    lift = normal.dual() * unit_volume                                        # [] Point-
    outline = boundary_lines(fine_mesh, fine_vertices, view)
    base = sheet(fine_vertices, fine_mesh.faces, surface_colour, view)

    n_inner_patches = 4
    selected = np.linspace(0, points.shape[0] - 1, n_inner_patches, dtype=int)
    corner_tiles, corner_seams = frame_patches(corners + lift * 0.04, corner_rotors, 0.13, view)
    tiles, seams = frame_patches(points[selected] + lift * 0.07, unit_rotors[selected], 0.085, view)
    orientation = solids.draw([], view, background, inks=[], translucent=[],
                             meshes=[(base, 1.0), (outline, 1.0), (corner_tiles, 1.0), (corner_seams, 1.0), (tiles, 1.0), (seams, 1.0)])

    # The visible probe fixes the input of the rate map; the arcs lie in its returned bivector planes.
    probe = (corners[1] - corners[0]).normalized()
    rates = whitney_rates(probe)[selected]
    ray = probe.commutator(rates).normalized()
    # Each arc turns the ray in its rate's plane, through an angle 2.4 times the rate.
    phases = np.linspace(0, 1, n_curve_samples)
    turns = (rates[:, None] * (2.4 * phases / 2)).exp()                       # [selected, samples] Rotor+
    centres = points[selected] + lift * 0.14                                  # [selected] Point-
    arc_points = centres[:, None] + (turns >> ray[:, None]) * 0.115           # [selected, samples] Point-
    progression = phases[:-1, None] * gold + (1 - phases[:-1, None]) * coral
    arc_colours = np.broadcast_to(progression, (len(selected), n_curve_samples - 1, 3))
    arc_places = solids.origin + solids.context.include_from_algebra(arc_points)               # [selected, samples] Point-
    arcs = strokes.segments(arc_places[:, :-1].reshape(-1), arc_places[:, 1:].reshape(-1), arc_colours.reshape(-1, 3),
                            3.0, view)
    probe_starts = points[selected] + lift * 0.025 - probe * 0.08             # [selected] Point-
    probe_ends = probe_starts + probe * 0.18                                  # [selected] Point-
    probe_tips = solids.origin + solids.context.include_from_algebra(probe_ends)               # [selected] Point-
    probes = strokes.segments(solids.origin + solids.context.include_from_algebra(probe_starts), probe_tips, blue, 2.0, view)
    probe_ends_mark = solids.ball(probe_tips, 0.018)[:, None]
    arc_ends = solids.ball(arc_places[:, -1], 0.018)[:, None]
    rates_image = solids.draw([(probe_ends_mark, blue), (arc_ends, gold)], view, background, inks=[], translucent=[],
                             meshes=[(base, 1.0), (outline, 1.0), (probes, 1.0), (arcs, 1.0)])

    # Height makes the chord's loss of norm a geometric bowl, with its corners pinned to unit norm.
    bowl_points = fine_vertices.batch() + lift * (1.3 * (fine_norms_squared.batch() - 1))   # [vertices] Point-
    norms = fine_norms_squared.batch().to_array()
    depth = np.clip((1 - norms) / (1 - norms.min()), 0, 1)
    bowl_colours = (1 - depth[:, None]) * np.array([0.50, 0.83, 0.98]) + depth[:, None] * blue
    bowl = sheet(bowl_points, fine_mesh.faces, bowl_colours[fine_mesh.faces], view)
    corner_marks = solids.ball(solids.origin + solids.context.include_from_algebra(corners), 0.025)[:, None]
    norm_image = solids.draw([(corner_marks, gold)], view, background, inks=[], translucent=[], meshes=[(bowl, 1.0), (outline, 1.0)])
    return panels([orientation, rates_image, norm_image])


def shadow(vertices: core.Point) -> core.Point:
    """The vertices' shadows on the level plane through the origin, straight below or above them."""
    return vertices - vertical * (core.mv.z & vertices)


def concentric_paths(vertices: core.Point) -> core.Point:
    """Circular display paths `[ribbons, samples]` on the level plane, inside the shadow of the disk's polygonal
    boundary: the farthest shadow's distance from the middle, scaled, turned about the vertical."""
    mv = core.mv
    radii = shadow(vertices.batch()).norm()
    radius = radii.max()
    radii = radius * np.linspace(0.18, 0.90, n_ribbons)
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_curve_samples) / 2)).exp()   # [samples] Rotor+
    return turns >> (mv.x.dual() * radii[:, None])                            # [ribbons, samples] Point-


def sampling(mesh: core.Mesh, vertices: core.Point, samples: core.Point) -> SparseExtensor:
    """The map carrying a field on the vertices to samples on the level plane `[samples] Point-`, linear across each
    face of the mesh's shadow there: a sample's hats in the face that holds it, the face where its least hat is
    largest, weight that face's corners."""
    corners = mesh.incidence[2, 0]                                            # [3] Scalar+[faces <- vertices]
    flat = shadow(vertices)                                                   # Point-[vertices]
    sides = mesh.boundary[2, 1] * (mesh.d[0] * flat)                          # [3] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                       # Patch-[faces]
    # A hat rises from zero along the side facing its corner to one at the corner.
    gradients = ((sides | patches.inverse()) / 2).batch()                     # [3, faces] Plane+
    hats = 1 + (gradients[:, None] & (samples[:, None] - (corners * flat).batch()[:, None]))   # [3, samples, faces] Scalar+
    least_hats = hats.min(axis=0)
    holding = least_hats.argmax(axis=-1)                    # [samples]
    weights = hats[:, np.arange(samples.shape[0]), holding].field()           # [3] Scalar+[samples]
    return (spdiag(weights) * (mesh.reading(holding, 2) * corners)).sum(axis=0)   # Scalar+[samples <- vertices]


def draw_skyrmion(mesh: core.Mesh, vertices: core.Point, directors: core.Point) -> plt.Figure:
    """Matched paths show the disk's centre reaching the north pole and its rim collapsing to the south pole."""
    mv = core.mv
    disk_frame = square_view(disk_view, vertices, 0.04)
    sphere_frame = square_view(np.array([0.0, -42.0]), directors, 0.04)
    rings = concentric_paths(vertices)                                        # [ribbons, samples] Point-
    # The meridians run out to the boundary vertices nearest in direction to evenly spaced turns of x about the
    # vertical.
    rim = vertices.batch()[mesh.exterior(0)]                                  # [rim] Point-
    headings = (mv.xy * (-np.linspace(-np.pi, np.pi, n_meridians, endpoint=False) / 2)).exp() >> mv.x.dual()   # [meridians] Point-
    ends = rim[headings[:, None].dot(rim.normalized()[None]).argmax(axis=-1)]   # [meridians] Point-
    meridians = ends[:, None] * np.linspace(0, 1, n_curve_samples)            # [meridians, samples] Point-

    paths = concatenate((rings, meridians))                                   # [paths, samples] Point-
    along_paths = sampling(mesh, vertices, paths.reshape(-1))                 # Scalar+[paths * samples <- vertices]
    domain_points = (along_paths * vertices).batch().reshape(paths.shape) + vertical * 0.012   # [paths, samples] Point-
    sphere_points = (along_paths * directors).batch().reshape(paths.shape).normalized() * 1.012   # [paths, samples] Point-
    domain_paths = solids.origin + solids.context.include_from_algebra(domain_points)          # [paths, samples] Point-
    sphere_paths = solids.origin + solids.context.include_from_algebra(sphere_points)          # [paths, samples] Point-
    colours = np.concatenate((np.broadcast_to([0.56, 0.64, 0.72], (rings.shape[0], 3)), path_colours))
    domain_lines = [(domain_paths[path], colour) for path, colour in enumerate(colours)]
    sphere_lines = [(sphere_paths[path], colour) for path, colour in enumerate(colours)]

    disk = sheet(vertices, mesh.faces, surface_colour, disk_frame)
    domain_grid = strokes.polylines(domain_lines, 2.4, disk_frame)
    centre = solids.ball((solids.origin + solids.context.include_from_algebra(vertical * 0.018))[None], 0.045)[:, None]
    domain = solids.draw([(centre, coral)], disk_frame, background, inks=[], translucent=[], meshes=[(disk, 1.0), (domain_grid, 1.0)])
    sphere = solids.ball(solids.origin[None], 1.0)[:, None]
    sphere_grid = strokes.polylines(sphere_lines, 2.8, sphere_frame)
    poles = solids.ball(solids.origin + solids.context.include_from_algebra(stack([vertical, -vertical]) * 1.02), 0.045)[:, None]
    image = solids.draw([(sphere, surface_colour), (poles, np.stack((coral, blue)))], sphere_frame, background,
                        inks=[], translucent=[], meshes=[(sphere_grid, 1.0)])
    return panels([domain, image])


def ribbon_panel(mesh: core.Mesh, vertices: core.Point, rotors: core.Rotor, rings: core.Point,
                 along_rings: SparseExtensor, view: solids.View) -> np.ndarray:
    """Fixed centre curves `[ribbons, samples]` carry cross-sections turned by the rotor field, its chords across
    each face made unit again: the radial direction turned gives the width, and the vertical the height."""
    centres = (along_rings * vertices).batch().reshape(rings.shape) + vertical * 0.09   # [ribbons, samples] Point-
    frames = (along_rings * rotors).batch().reshape(rings.shape).normalized()  # [ribbons, samples] Rotor+
    widths = frames >> rings.normalized()                                     # [ribbons, samples] Point-
    heights = frames >> vertical                                              # [ribbons, samples] Point-
    profile = np.linspace(-1, 1, n_profile_samples)
    ribbons = (centres[..., None] + widths[..., None] * (0.065 * profile)
               + heights[..., None] * (0.022 * (1 - profile**2)))             # [ribbons, samples, profile] Point-
    n_segments = n_curve_samples - 1
    first = np.arange(n_segments)[:, None] * n_profile_samples + np.arange(n_profile_samples - 1)
    faces = np.concatenate((np.stack((first, first + 1, first + n_profile_samples), axis=-1).reshape(-1, 3),
                            np.stack((first + 1, first + n_profile_samples + 1, first + n_profile_samples), axis=-1).reshape(-1, 3)))
    across_colours = (1 - (profile + 1)[:, None] / 2) * coral + ((profile + 1)[:, None] / 2) * blue
    colours = np.broadcast_to(across_colours, (n_curve_samples, n_profile_samples, 3)).reshape(-1, 3)
    base = sheet(vertices, mesh.faces, surface_colour, view)
    layers = [(base, 0.34), (boundary_lines(mesh, vertices, view), 0.65)]
    for ribbon in range(ribbons.shape[0]):
        layers.append((sheet(ribbons[ribbon].reshape(-1), faces, colours[faces], view), 1.0))
        edges = solids.origin + solids.context.include_from_algebra(ribbons[ribbon][:, [0, -1]])   # [samples, 2] Point-
        seam = strokes.polylines([(edges[:, 0], gold), (edges[:, 1], teal)], 1.3, view)
        layers.append((seam, 1.0))
    return solids.draw([], view, background, inks=[], translucent=[], meshes=layers)


def draw_rotor_frames(mesh: core.Mesh, vertices: core.Point, harmonic_rotors: core.Rotor) -> plt.Figure:
    """Ribbon cross-sections compare orientation fields with different boundary prescriptions."""
    view = square_view(saddle_view, vertices, 0.20)
    rings = concentric_paths(vertices)                                        # [ribbons, samples] Point-
    along_rings = sampling(mesh, vertices, rings.reshape(-1))                 # Scalar+[ribbons * samples <- vertices]
    return panels([ribbon_panel(mesh, vertices, rotors.field(), rings, along_rings, view)
                   for rotors in harmonic_rotors.batch()])


def animate_relaxation(mesh: core.Mesh, vertices: core.Point,
                       states: Iterable[tuple[core.Rotor, core.Scalar]]) -> Iterator[np.ndarray]:
    """Ribbon cross-sections settle as the unit rotor field smooths over its fixed surface."""
    view = square_view(saddle_view, vertices, 0.20)
    rings = concentric_paths(vertices)                                        # [ribbons, samples] Point-
    along_rings = sampling(mesh, vertices, rings.reshape(-1))                 # Scalar+[ribbons * samples <- vertices]
    for rotors, _ in states:
        yield ribbon_panel(mesh, vertices, rotors, rings, along_rings, view)
