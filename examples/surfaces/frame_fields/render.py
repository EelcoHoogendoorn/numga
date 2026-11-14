"""Oriented patches, turning arcs, ribbons, and a director sphere display rotor fields."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.sparse import SparseExtensor, spdiag
from examples.surfaces.frame_fields import core
from numga.topology import Topology
from numga.render.drawing import Draw
from numga.render import meshes, simplices, solids
from examples.util.panels import image_figure

background = np.ones(3)
surface_colour = np.array([0.91, 0.93, 0.95])
ink = np.array([0.16, 0.23, 0.30])
coral = np.array([0.91, 0.24, 0.16])
teal = np.array([0.02, 0.65, 0.55])
blue = np.array([0.12, 0.40, 0.88])
gold = np.array([0.99, 0.67, 0.10])
violet = np.array([0.57, 0.27, 0.84])
path_colours = np.stack((coral, gold, teal, blue, violet, [0.90, 0.20, 0.57]))
n_pixels = 300
n_curve_samples = 96
n_profile_samples = 7
n_ribbons = 5
n_meridians = len(path_colours)
triangle_view = np.array([38.0, -48.0])
disk_view = np.array([40.0, -42.0])
saddle_view = np.array([32.0, -38.0])
# Upward unit direction and intrinsic unit volume element used to turn patch duals into normal directions.
vertical = core.mv.z.dual()                                                  # [] Point-
unit_volume = core.mv.antipseudoscalar([1])                                  # [] Antipseudoscalar-


# --- plumbing -------------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, vertices: core.Point, spare: float) -> solids.View:
    """Frames the supplied geometric points inside a square orthographic view."""
    elevation, azimuth = np.radians(view_degrees + [0, 90])
    places = solids.origin + solids.context.include_from_algebra(vertices.batch())  # [...] Point-
    return solids.view([places], spare, elevation, azimuth, 0.06, n_pixels).squared()


def lit_sheet(points: solids.Point, faces: np.ndarray, colours: np.ndarray,
              view: solids.View) -> Draw:
    """Draws a smooth-shaded triangulated sheet at the view's pixel resolution."""
    return meshes.samples(points, faces, colours, meshes.smooth(points, Topology(len(points), faces)), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def boundary_lines(topology: Topology, points: solids.Point, view: solids.View) -> Draw:
    """Draws thin tubes along the boundary edges of `topology`."""
    rim = topology.border
    return simplices.tubes(points, rim, ink, 1.1, view, solids.gloss)


def joined_panels(images: list[np.ndarray]) -> plt.Figure:
    """Joins rendered panel images side by side with a narrow white gutter."""
    gutter = np.full((n_pixels, 8, 3), 255, dtype=np.uint8)
    pieces = [piece for i, picture in enumerate(images) for piece in ([gutter, picture] if i else [picture])]
    return image_figure(np.concatenate(pieces, axis=1))


def frame_patches(starts: core.Point, rotors: core.Rotor, size: float,
                  view: solids.View) -> tuple[Draw, Draw]:
    """Builds coloured rigid square patches and perimeter tubes showing each rotor's plane and in-plane orientation."""
    mv = rotors.context.multivector
    # Raised centre apex lets lighting distinguish edge-on patches.
    profile = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0], [0, 0, 0.16]])
    shape = (stack((mv.x.dual(), mv.y.dual(), mv.z.dual())) * (size * profile)).sum(axis=-1)  # [5] Point-
    places = starts[:, None] + (rotors.batch()[:, None] >> shape)             # [..., 5] Point-
    embedded = solids.origin + solids.context.include_from_algebra(places)    # [..., 5] Point-
    n_patches, n_profile = places.shape
    local_faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    faces = (local_faces[None] + np.arange(n_patches)[:, None, None] * n_profile).reshape(-1, 3)
    colours = np.broadcast_to(np.stack((coral, gold, teal, blue))[None, :, None, :], (n_patches, 4, 3, 3)).reshape(-1, 3, 3)
    patches = lit_sheet(embedded.reshape(-1), faces, colours, view)
    perimeters = embedded[:, :4]                                              # [n_patches, 4] Point-
    seams = simplices.tubes(perimeters.reshape(-1), Topology.loops([4] * n_patches), ink, 1.0, view, solids.gloss)
    return patches, seams


def darboux_figure(
    corners: core.Point, points: core.Point,
    topology: Topology, fine_vertices: core.Point, fine_norms_squared: core.Scalar,
    corner_rotors: core.Rotor, unit_rotors: core.Rotor,
    whitney_rates: core.RateMap,
) -> plt.Figure:
    """Renders the three-panel Darboux–Whitney triangle view of frame patches, rate-plane arcs, and chord-norm relief."""
    view = square_view(triangle_view, corners, 0.30)
    normal = ((corners[1] - corners[0]) & (corners[2] - corners[0])).normalized()  # [] Patch-
    lift = normal.dual() * unit_volume                                       # [] Point-
    fine_places = solids.origin + solids.context.include_from_algebra(fine_vertices.batch())  # [...] Point-
    outline = boundary_lines(topology, fine_places, view)
    base = lit_sheet(fine_places, topology.elements[2], surface_colour, view)

    n_selected = 4
    selected_indices = np.linspace(0, points.shape[0] - 1, n_selected, dtype=int)
    corner_tiles, corner_seams = frame_patches(corners + lift * 0.04, corner_rotors, 0.13, view)
    tiles, seams = frame_patches(points[selected_indices] + lift * 0.07, unit_rotors[selected_indices], 0.085, view)
    orientation_image = solids.draw(
        [], view, background, inks=[], translucent=[],
        meshes=[(base, 1.0), (outline, 1.0), (corner_tiles, 1.0), (corner_seams, 1.0), (tiles, 1.0), (seams, 1.0)],
    )

    probe = (corners[1] - corners[0]).normalized()                          # [] Point-
    rates = whitney_rates(probe)[selected_indices]                           # [n_selected] Bivector+
    ray = probe.commutator(rates).normalized()                               # [n_selected] Point-
    phases = np.linspace(0, 1, n_curve_samples)
    turns = (rates[:, None] * (2.4 * phases / 2)).exp()                      # [n_selected, n_curve_samples] Rotor+
    centres = points[selected_indices] + lift * 0.14                         # [n_selected] Point-
    arc_points = centres[:, None] + (turns >> ray[:, None]) * 0.115          # [n_selected, n_curve_samples] Point-
    progression = phases[:-1, None] * gold + (1 - phases[:-1, None]) * coral
    arc_colours = np.broadcast_to(progression, (n_selected, n_curve_samples - 1, 3))
    arc_places = solids.origin + solids.context.include_from_algebra(arc_points)  # [n_selected, n_curve_samples] Point-
    arcs = simplices.tubes(arc_places.reshape(-1), Topology.chains([n_curve_samples] * n_selected),
                           arc_colours.reshape(-1, 3), 3.0, view, solids.gloss)
    probe_starts = points[selected_indices] + lift * 0.025 - probe * 0.08    # [n_selected] Point-
    probe_ends = probe_starts + probe * 0.18                                 # [n_selected] Point-
    probe_tips = solids.origin + solids.context.include_from_algebra(probe_ends)  # [n_selected] Point-
    probe_segments = stack((solids.origin + solids.context.include_from_algebra(probe_starts), probe_tips), axis=-1)  # [n_selected, 2] Point-
    probes = simplices.tubes(probe_segments.reshape(-1), Topology.chains([2] * n_selected), blue, 2.0, view,
                             solids.gloss)
    probe_ends_mark = solids.ball(probe_tips, 0.018)[:, None]
    arc_ends = solids.ball(arc_places[:, -1], 0.018)[:, None]
    rates_image = solids.draw([(probe_ends_mark, blue), (arc_ends, gold)], view, background, inks=[], translucent=[],
                              meshes=[(base, 1.0), (outline, 1.0), (probes, 1.0), (arcs, 1.0)])

    bowl_points = fine_vertices.batch() + lift * (1.3 * (fine_norms_squared.batch() - 1))  # [...] Point-
    norms = fine_norms_squared.batch().to_array()
    depth = np.clip((1 - norms) / (1 - norms.min()), 0, 1)
    bowl_colours = (1 - depth[:, None]) * np.array([0.50, 0.83, 0.98]) + depth[:, None] * blue
    bowl = lit_sheet(solids.origin + solids.context.include_from_algebra(bowl_points), topology.elements[2],
                     bowl_colours[topology.elements[2]], view)
    corner_marks = solids.ball(solids.origin + solids.context.include_from_algebra(corners), 0.025)[:, None]
    norm_image = solids.draw([(corner_marks, gold)], view, background, inks=[], translucent=[], meshes=[(bowl, 1.0), (outline, 1.0)])
    return joined_panels([orientation_image, rates_image, norm_image])


def horizontal_shadow(vertices: core.Point) -> core.Point:
    """Horizontal projection of `vertices` onto the `z = 0` plane."""
    return vertices - vertical * (core.mv.z & vertices)


def concentric_paths(vertices: core.Point) -> core.Point:
    """Concentric circular sample curves (`[n_ribbons, n_curve_samples] Point-`) inside the horizontal footprint of `vertices`."""
    mv = core.mv
    radii = horizontal_shadow(vertices.batch()).norm()
    max_radius = radii.max()
    ring_radii = max_radius * np.linspace(0.18, 0.90, n_ribbons)
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_curve_samples) / 2)).exp()  # [n_curve_samples] Rotor+
    return turns >> (mv.x.dual() * ring_radii[:, None])                       # [n_ribbons, n_curve_samples] Point-


def surface_sampler(topology: Topology, vertices: core.Point, sample_points: core.Point) -> SparseExtensor:
    """Linear simplex-gradient interpolation map from surface vertices `V` to planar `sample_points` (`[...] Point-`)."""
    n_samples = sample_points.shape[0]
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    flat_vertices = horizontal_shadow(vertices)                               # Point-[V]
    sides = topology.boundary[2, 1] * (topology.d[0] * flat_vertices)         # [3] Point-[F]
    patches = (sides[0] & sides[1]) / 2                                       # Patch-[F]
    gradients = ((sides | patches.inverse()) / 2).batch()                     # [3, F] Plane+
    corner_weights = 1 + (gradients[:, None] & (sample_points[:, None] - (corners * flat_vertices).batch()[:, None]))  # [3, n_samples, F] Scalar+
    least_weights = corner_weights.min(axis=0)
    containing_faces = least_weights.argmax(axis=-1)                          # [n_samples]
    weights = corner_weights[:, np.arange(n_samples), containing_faces].field()  # [3] Scalar+[n_samples]
    return (spdiag(weights) * (topology.reading(containing_faces, 2) * corners)).sum(axis=0)  # Scalar+[n_samples <- V]


def skyrmion_figure(topology: Topology, vertices: core.Point, directors: core.Point) -> plt.Figure:
    """Renders matched radial and circular curves on the planar disk and on the sphere of directors."""
    mv = core.mv
    disk_frame = square_view(disk_view, vertices, 0.04)
    sphere_frame = square_view(np.array([0.0, -42.0]), directors, 0.04)
    rings = concentric_paths(vertices)                                        # [n_ribbons, n_curve_samples] Point-
    rim = (topology.reading(topology.exterior[0], 0) * vertices).batch()      # [...] Point-
    heading_turns = (mv.xy * (-np.linspace(-np.pi, np.pi, n_meridians, endpoint=False) / 2)).exp()  # [n_meridians] Rotor+
    headings = heading_turns >> mv.x.dual()                                   # [n_meridians] Point-
    ends = rim[headings[:, None].dot(rim.normalized()[None]).argmax(axis=-1)]  # [n_meridians] Point-
    meridians = ends[:, None] * np.linspace(0, 1, n_curve_samples)            # [n_meridians, n_curve_samples] Point-

    paths = concatenate((rings, meridians))                                   # [n_paths, n_curve_samples] Point-
    n_paths = paths.shape[0]
    along_paths = surface_sampler(topology, vertices, paths.reshape(-1))      # Scalar+[n_paths * n_curve_samples <- V]
    domain_points = (along_paths * vertices).batch().reshape(paths.shape) + vertical * 0.012  # [n_paths, n_curve_samples] Point-
    sphere_points = (along_paths * directors).batch().reshape(paths.shape).normalized() * 1.012  # [n_paths, n_curve_samples] Point-
    domain_paths = solids.origin + solids.context.include_from_algebra(domain_points)  # [n_paths, n_curve_samples] Point-
    sphere_paths = solids.origin + solids.context.include_from_algebra(sphere_points)  # [n_paths, n_curve_samples] Point-
    colours = np.concatenate((np.broadcast_to([0.56, 0.64, 0.72], (n_ribbons, 3)), path_colours))
    chains = Topology.chains([n_curve_samples] * n_paths)
    chain_colours = np.repeat(colours, n_curve_samples - 1, axis=0)

    disk = lit_sheet(solids.origin + solids.context.include_from_algebra(vertices.batch()), topology.elements[2], surface_colour,
                     disk_frame)
    domain_grid = simplices.tubes(domain_paths.reshape(-1), chains, chain_colours, 2.4, disk_frame, solids.gloss)
    centre = solids.ball((solids.origin + solids.context.include_from_algebra(vertical * 0.018))[None], 0.045)[:, None]
    domain_image = solids.draw([(centre, coral)], disk_frame, background, inks=[], translucent=[], meshes=[(disk, 1.0), (domain_grid, 1.0)])
    sphere = solids.ball(solids.origin[None], 1.0)[:, None]
    sphere_grid = simplices.tubes(sphere_paths.reshape(-1), chains, chain_colours, 2.8, sphere_frame, solids.gloss)
    poles = solids.ball(solids.origin + solids.context.include_from_algebra(stack([vertical, -vertical]) * 1.02), 0.045)[:, None]
    sphere_image = solids.draw([(sphere, surface_colour), (poles, np.stack((coral, blue)))], sphere_frame, background,
                               inks=[], translucent=[], meshes=[(sphere_grid, 1.0)])
    return joined_panels([domain_image, sphere_image])


def ribbon_panel(topology: Topology, vertices: core.Point, rotors: core.Rotor, rings: core.Point,
                 along_rings: SparseExtensor, view: solids.View) -> np.ndarray:
    """Renders twisted ribbons along `rings` carried by the interpolated rotor field over `vertices`."""
    centres = (along_rings * vertices).batch().reshape(rings.shape) + vertical * 0.09  # [n_ribbons, n_curve_samples] Point-
    frames = (along_rings * rotors).batch().reshape(rings.shape).normalized()  # [n_ribbons, n_curve_samples] Rotor+
    widths = frames >> rings.normalized()                                     # [n_ribbons, n_curve_samples] Point-
    heights = frames >> vertical                                              # [n_ribbons, n_curve_samples] Point-
    profile = np.linspace(-1, 1, n_profile_samples)
    ribbons = (centres[..., None] + widths[..., None] * (0.065 * profile)
               + heights[..., None] * (0.022 * (1 - profile**2)))             # [n_ribbons, n_curve_samples, n_profile_samples] Point-
    n_segments = n_curve_samples - 1
    first = np.arange(n_segments)[:, None] * n_profile_samples + np.arange(n_profile_samples - 1)
    faces = np.concatenate((np.stack((first, first + 1, first + n_profile_samples), axis=-1).reshape(-1, 3),
                            np.stack((first + 1, first + n_profile_samples + 1, first + n_profile_samples), axis=-1).reshape(-1, 3)))
    across_colours = (1 - (profile + 1)[:, None] / 2) * coral + ((profile + 1)[:, None] / 2) * blue
    colours = np.broadcast_to(across_colours, (n_curve_samples, n_profile_samples, 3)).reshape(-1, 3)
    points = solids.origin + solids.context.include_from_algebra(vertices.batch())  # [...] Point-
    ribbon_places = solids.origin + solids.context.include_from_algebra(ribbons)  # [n_ribbons, n_curve_samples, n_profile_samples] Point-
    base = lit_sheet(points, topology.elements[2], surface_colour, view)
    layers = [(base, 0.34), (boundary_lines(topology, points, view), 0.65)]
    for ribbon_index in range(n_ribbons):
        layers.append((lit_sheet(ribbon_places[ribbon_index].reshape(-1), faces, colours[faces], view), 1.0))
        edges = ribbon_places[ribbon_index][:, [0, -1]]                       # [n_curve_samples, 2] Point-
        seam = simplices.tubes(concatenate((edges[:, 0], edges[:, 1])), Topology.chains([n_curve_samples] * 2),
                               np.repeat(np.stack((gold, teal)), n_curve_samples - 1, axis=0), 1.3, view, solids.gloss)
        layers.append((seam, 1.0))
    return solids.draw([], view, background, inks=[], translucent=[], meshes=layers)


def rotor_frames_figure(topology: Topology, vertices: core.Point, harmonic_rotors: core.Rotor) -> plt.Figure:
    """Renders side-by-side ribbon panels comparing harmonic rotor fields under distinct boundary prescriptions."""
    view = square_view(saddle_view, vertices, 0.20)
    rings = concentric_paths(vertices)                                        # [n_ribbons, n_curve_samples] Point-
    along_rings = surface_sampler(topology, vertices, rings.reshape(-1))      # Scalar+[n_ribbons * n_curve_samples <- V]
    return joined_panels([ribbon_panel(topology, vertices, rotors.field(), rings, along_rings, view)
                          for rotors in harmonic_rotors.batch()])


def relaxation_animation(topology: Topology, vertices: core.Point,
                         states: Iterable[core.Rotor]) -> Iterator[np.ndarray]:
    """Yields rendered ribbon panels as the unit rotor field smooths across the saddle surface."""
    view = square_view(saddle_view, vertices, 0.20)
    rings = concentric_paths(vertices)                                        # [n_ribbons, n_curve_samples] Point-
    along_rings = surface_sampler(topology, vertices, rings.reshape(-1))      # Scalar+[n_ribbons * n_curve_samples <- V]
    for rotors in states:
        yield ribbon_panel(topology, vertices, rotors, rings, along_rings, view)
