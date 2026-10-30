"""Lit surface sheets, domain parameter curves, and normal pins display minimal surfaces and their conjugates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate
from numga.sparse import SparseExtensor, spdiag
from examples.surfaces.minimal_surfaces import core
from examples.util.wgsl import image_figure, meshes, solids, strokes

mv = core.mv
background = np.ones(3)
surface_colour = np.array([0.90, 0.93, 0.96])
ink = np.array([0.16, 0.23, 0.30])
coral = np.array([0.91, 0.24, 0.16])
teal = np.array([0.02, 0.65, 0.55])
blue = np.array([0.12, 0.40, 0.88])
gold = np.array([0.99, 0.67, 0.10])
violet = np.array([0.57, 0.27, 0.84])
meridian_colours = np.stack((coral, gold, teal, blue, violet, [0.90, 0.20, 0.57]))
pixels = 300
n_curve_samples = 80
n_rings = 4
# The domain rings' radii, in shares of the disk's radius.
ring_shares = np.linspace(0.22, 0.86, n_rings)
# How far each normal pin reaches behind and in front of its vertex, and the size of its head.
pin_depth, pin_reach, pin_head_radius = 0.16, 0.30, 0.048
# A volume, the join of three points, is an intrinsic scalar; its unit carries an intrinsic patch to the plane square
# to it, whose dual is the direction square to the patch.
unit_volume = mv.antipseudoscalar([1])                                       # [] Antipseudoscalar-
plateau_view = np.array([28.0, -38.0])
conjugate_view = np.array([30.0, -36.0])
bonnet_view = np.array([24.0, -36.0])


# --- plumbing -------------------------------------------------------------------------
def square_view(view_degrees: np.ndarray, surfaces: list[core.Point], spare: float) -> solids.View:
    """A square orthographic view frames the supplied surfaces."""
    elevation, azimuth = np.radians(view_degrees + [0, 90])
    places = [solids.origin + solids.context.include_from_algebra(surface.batch()) for surface in surfaces]   # [surfaces] Point-[vertices]
    return solids.view(places, spare, elevation, azimuth, 0.06, pixels).squared()


def sheet(mesh: core.Mesh, surface: core.Point, corner_colours: np.ndarray, view: solids.View) -> meshes.Samples:
    """A lit sheet carries its colours at the corners of its faces at the image's own pixel size."""
    points = solids.origin + solids.context.include_from_algebra(surface.batch())               # [vertices] Point-
    return meshes.samples(points, mesh.faces, corner_colours, meshes.smooth(points, mesh), view,
                          face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss)


def boundary_lines(mesh: core.Mesh, surface: core.Point, width: float, view: solids.View) -> meshes.Samples:
    """Crisp outlines mark the boundary wire of a surface."""
    tails, heads = mesh.reading(mesh.exterior(1), 1) * (mesh.incidence[1, 0] * surface)   # Point-[rim], Point-[rim]
    return strokes.segments(solids.origin + solids.context.include_from_algebra(tails.batch()),
                            solids.origin + solids.context.include_from_algebra(heads.batch()), ink, width, view)


def disk_paths(rim: core.Point) -> core.Point:
    """Concentric rings and radial spokes inside a planar disk domain, its rim vertices given, `[curves, samples]`:
    each ring a point on the x-axis turned through a full turn, each spoke a rim vertex drawn in towards the centre,
    the vertex whose direction lies closest to the spoke's."""
    radius = rim.norm().sites.mean()                                         # [] Scalar+
    turns = (mv.xy * (-np.linspace(0, 2 * np.pi, n_curve_samples) / 2)).exp()   # [samples] Rotor+
    rings = (radius * ring_shares)[:, None] * (turns >> mv.x.dual())         # [rings, samples] Point-

    spoke_angles = np.linspace(-np.pi, np.pi, len(meridian_colours), endpoint=False)
    spoke_directions = (mv.xy * (-spoke_angles / 2)).exp() >> mv.x.dual()    # [spokes] Point-
    rim_points = rim.batch()                                                 # [rim] Point-
    # The rim vertex whose unit direction pairs best with the spoke's lies closest to it in angle.
    alignments = spoke_directions[:, None].dot(rim_points.normalized()[None])   # [spokes, rim] Scalar+
    spoke_ends = rim_points[alignments.argmax(axis=-1)]                      # [spokes] Point-
    spokes = np.linspace(0, 1, n_curve_samples) * spoke_ends[:, None]        # [spokes, samples] Point-
    return concatenate((rings, spokes))


def lifting(mesh: core.Mesh, flat_vertices: core.Point, paths: core.Point) -> SparseExtensor:
    """The map carrying a surface over the planar domain onto its samples along the domain paths, linear across each
    triangle: a sample's hats in the domain triangle that holds it weight that triangle's corners."""
    corners = mesh.incidence[2, 0]                                           # [corners] Scalar+[faces <- vertices]
    domain_corners = corners * flat_vertices                                 # [corners] Point-[faces]
    sides = mesh.boundary[2, 1] * (mesh.d[0] * flat_vertices)                # [corners] Point-[faces]
    patches = (sides[0] & sides[1]) / 2                                      # Patch-[faces]
    # Each corner's hat rises from zero along the facing side to one at the corner.
    corner_gradients = (sides | patches.inverse()) / 2                       # [corners] Plane+[faces]
    # A sample's three hats in any triangle sum to one, and their sizes do too only in a triangle that holds it.
    hats_everywhere = 1 + (corner_gradients.batch()[:, None, None]
                           & (paths[..., None] - domain_corners.batch()[:, None, None]))   # [corners, curves, samples, faces] Scalar+
    holding = mesh.reading(hats_everywhere.abs().sum(axis=0).argmin(axis=-1), 2)   # [curves] Scalar+[samples <- faces]
    hats = 1 + ((holding[None] * corner_gradients[:, None])
                & (paths.field() - holding[None] * domain_corners[:, None]))   # [corners, curves] Scalar+[samples]
    return (spdiag(hats) * (holding[None] * corners[:, None])).sum(axis=0)   # [curves] Scalar+[samples <- vertices]


def surface_with_curves(
    mesh: core.Mesh, lift: SparseExtensor, surface: core.Point, corner_colours: np.ndarray, view: solids.View,
) -> np.ndarray:
    """A lit surface sheet overlaid with its boundary wire and its lifted domain curves."""
    curves = solids.origin + solids.context.include_from_algebra((lift * surface).batch())    # [curves, samples] Point-
    colours = np.concatenate((np.broadcast_to(teal, (n_rings, 3)), meridian_colours))
    curve_strokes = strokes.polylines(list(zip(curves, colours)), 2.0, view)
    base = sheet(mesh, surface, corner_colours, view)
    rim = boundary_lines(mesh, surface, 2.2, view)
    return solids.draw([], view, background, inks=[], translucent=[],
                       meshes=[(base, 1.0), (curve_strokes, 1.0), (rim, 1.0)])


def animate_plateau(
    mesh: core.Mesh, flat_vertices: core.Point,
    states: Iterable[tuple[core.Point, core.Scalar, core.Curvature]],
) -> Iterator[np.ndarray]:
    """The domed surface sinks onto the four-arch minimal soap film while its mean-curvature tint fades."""
    states = list(states)
    # Every frame moved by the minimal film's mean point, so that the wire stays put.
    centre = states[-1][0].sites.mean()                                      # [] Point-
    surfaces = [vertices - centre for vertices, _, _ in states]              # [frames] Point-[vertices]
    view = square_view(plateau_view, surfaces, 0.12)
    lift = lifting(mesh, flat_vertices, disk_paths(mesh.reading(mesh.exterior(0)) * flat_vertices))
    # The tint's scale: most of the domed surface's curvature reaches full coral.
    scale = max(float(np.percentile(np.abs(states[0][2].batch().to_array()), 95)), 1e-6)
    for surface, (_, _, curvature) in zip(surfaces, states):
        # The curvature's size at the corners of each face, read through the mesh, sets the tint there.
        corner_curvatures = np.abs((mesh.incidence[2, 0] * curvature).batch().to_array()).T   # [faces, 3]
        intensity = np.clip(corner_curvatures / scale, 0.0, 1.0)[..., None]
        corner_colours = (1 - intensity) * surface_colour + intensity * coral
        yield surface_with_curves(mesh, lift, surface, corner_colours, view)


def conjugate_pair(mesh: core.Mesh, flat_vertices: core.Point, vertices: core.Point, conjugate: core.Point) -> np.ndarray:
    """A minimal surface beside its conjugate, each moved by its own mean point, framed together, with matched domain
    curves."""
    surfaces = [vertices - vertices.sites.mean(), conjugate - conjugate.sites.mean()]   # [2] Point-[vertices]
    view = square_view(conjugate_view, surfaces, 0.12)
    lift = lifting(mesh, flat_vertices, disk_paths(mesh.reading(mesh.exterior(0)) * flat_vertices))
    left, right = (surface_with_curves(mesh, lift, surface, surface_colour, view) for surface in surfaces)
    return np.concatenate((left, np.full((pixels, 8, 3), 255, dtype=np.uint8), right), axis=1)


def draw_conjugates(
    plateau_mesh: core.Mesh, plateau_flat: core.Point,
    plateau_vertices: core.Point, plateau_conjugate: core.Point,
    enneper_mesh: core.Mesh, enneper_flat: core.Point,
    enneper_vertices: core.Point, enneper_conjugate: core.Point,
) -> plt.Figure:
    """Two minimal surfaces beside their recovered harmonic conjugates, with matched domain curves."""
    top_row = conjugate_pair(plateau_mesh, plateau_flat, plateau_vertices, plateau_conjugate)
    bottom_row = conjugate_pair(enneper_mesh, enneper_flat, enneper_vertices, enneper_conjugate)
    h_gutter = np.full((8, top_row.shape[1], 3), 255, dtype=np.uint8)
    return image_figure(np.concatenate((top_row, h_gutter, bottom_row), axis=0))


def animate_bonnet(
    mesh: core.Mesh, rings: np.ndarray, meridians: np.ndarray,
    surfaces: Iterable[tuple[core.Point, core.Point]],
) -> Iterator[np.ndarray]:
    """The slit catenoid uncurls isometrically into the helicoid while its Gauss normals keep their directions; its
    rings and meridians, vertex indices `[rings, azimuths]` and `[meridians, heights]`, are drawn on it."""
    frames = [(surface - surface.sites.mean(), normals) for surface, normals in surfaces]
    view = square_view(bonnet_view, [surface for surface, _ in frames], 0.14)

    n_heights, n_azimuths = len(rings), len(meridians)
    ring_reading = mesh.reading(rings[np.linspace(2, n_heights - 3, 5, dtype=int)])   # Scalar+[rings, azimuths <- vertices]
    meridian_reading = mesh.reading(meridians[np.linspace(3, n_azimuths - 4, 6, dtype=int)])   # Scalar+[meridians, heights <- vertices]
    # Pins where a sparse set of rings crosses a sparse set of meridians.
    pin_rings = rings[np.linspace(3, n_heights - 4, 3, dtype=int)]
    pin_reading = mesh.reading(pin_rings[:, np.linspace(5, n_azimuths - 6, 4, dtype=int)].reshape(-1))   # Scalar+[pins <- vertices]

    for surface, normals in frames:
        ring_curves = solids.origin + solids.context.include_from_algebra((ring_reading * surface).batch())   # [rings, azimuths] Point-
        meridian_curves = solids.origin + solids.context.include_from_algebra((meridian_reading * surface).batch())   # [meridians, heights] Point-
        curves = [(points, teal) for points in ring_curves] + [(points, coral) for points in meridian_curves]
        curve_strokes = strokes.polylines(curves, 2.0, view)
        base = sheet(mesh, surface, surface_colour, view)
        rim = boundary_lines(mesh, surface, 1.8, view)

        # Outward normal pins at the crossings show that the Gauss normals stay fixed in space.
        pin_points = pin_reading * surface                                   # Point-[pins]
        normal_directions = (pin_reading * normals * unit_volume).dual()     # Point-[pins]
        roots = solids.origin + solids.context.include_from_algebra((pin_points - pin_depth * normal_directions).batch())   # [pins] Point-
        tips = solids.origin + solids.context.include_from_algebra((pin_points + pin_reach * normal_directions).batch())    # [pins] Point-
        stems = strokes.segments(roots, tips, gold, 2.2, view)
        heads = solids.ball(tips, pin_head_radius)[:, None]
        yield solids.draw([(heads, gold)], view, background, inks=[], translucent=[],
                          meshes=[(base, 1.0), (curve_strokes, 1.0), (rim, 1.0), (stems, 1.0)])
