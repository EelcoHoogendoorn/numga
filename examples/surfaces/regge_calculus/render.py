"""The drawings show Regge edge metric cones, metric ellipses, vertex curvature defects, and intrinsic Laplace–Beltrami flow."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import stack
from examples.util import charts
from examples.util.wgsl import contours, flow, simplices
from examples.util.wgsl.camera import Camera, flat_panel
from examples.surfaces.regge_calculus import core

primal_colour = "#286580"
accent_colour = "#d77542"
radial_ellipse_colour = "#c84b31"
hoop_ellipse_colour = "#2d6a9f"
mesh_colour = "#b8c4c9"
wall_colour = "#52616a"
flow_colour = "#173c4b"
hat_colours = LinearSegmentedColormap.from_list(
    "regge_metric", ("#f6f4ee", "#8fb8c9", "#286580"))
diverging_colours = LinearSegmentedColormap.from_list(
    "regge_diverging", ("#286580", "#f5f2eb", "#d77542"))
indicatrix_colours = LinearSegmentedColormap.from_list(
    "regge_indicatrix", (hoop_ellipse_colour, wall_colour, radial_ellipse_colour))
n_contours = 7
n_ellipse_grid = 25
# In pixels: the triangles' outlines, thin and regular, their highlighted edges and the disks' boundaries; the mesh's
# edges and how much they cover; the corners' dots, small, regular and highlighted, and the centroid's; the white rings
# about the highlighted dots.
thin_width, outline_width, edge_width, wall_width = 1.5, 1.7, 3.6, 1.8
mesh_width, mesh_opacity = 1.0, 0.38
small_dot, corner_dot, highlight_dot, centroid_dot = 2.9, 4.2, 4.6, 3.25
halo = 1.25
# The pixel sizes of a row of four triangles and of a row of three disks.
row_size, disk_row_size = (780, 195), (780, 260)


# --- plumbing -------------------------------------------------------------------------
def frame_panel(ax: plt.Axes, points: core.Point, margin_ratio: float) -> None:
    """Square equal-aspect viewport around the points with no axes or ticks: their extent read by pairing with the
    planes x and y."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & points.batch().reshape(-1)      # [2, points] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                  # [2] Scalar+, Scalar+
    centre = (low + high) / 2
    half_span = (1 + margin_ratio) * (high - low).max() / 2
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()


def ink(colour: str) -> np.ndarray:
    """A colour's red, green and blue."""
    return np.array(to_rgb(colour))


def ringed(points: core.Point, colour: str, radius: float, camera: Camera) -> list[np.ndarray]:
    """Dots at points, `radius` pixels, ringed in white: their pictures."""
    return [simplices.markers(points, np.ones(3), radius + halo, camera),
            simplices.markers(points, ink(colour), radius, camera)]


def walls(topology: core.Topology, points: core.Point, camera: Camera) -> np.ndarray:
    """The boundary of a mesh at its points `[vertices]`, the mesh of its boundary's edges: their picture."""
    rim = topology.border
    return simplices.strokes(points, rim, ink(wall_colour), wall_width, camera)


def spaced_faces(centres: core.Point, max_radius: float, min_separation: float) -> list[int]:
    """Well-spaced interior face indices ordered from the origin outward: each face's centre `Point-[faces]` within
    `max_radius` of the origin and further than `min_separation` from those of the faces chosen before it."""
    centres = centres.batch()                                                  # [faces] Point-
    reach = centres.norm()                                                   # [faces] Scalar+
    separations = (centres[:, None] - centres[None, :]).norm()                 # [faces, faces] Scalar+
    chosen: list[int] = []
    for face in reach.argsort():
        if reach[face] < max_radius and all(separations[face, other] > min_separation for other in chosen):
            chosen.append(int(face))
    return chosen


def draw_metric_basis(corners: core.Point, points: core.Point,
                      basis: core.Metric, metric: core.Metric) -> plt.Figure:
    """The three hyperbolic Regge edge metric basis forms and their sum into a positive-definite metric ellipse."""
    # The samples triangulated, for the pictures of the forms across them.
    topology = core.Topology.triangulate(points)
    places = points.batch()                                                   # [samples] Point-
    centroid = corners.mean()                                                  # [] Point-
    offsets = points - centroid

    basis_forms = np.asarray((basis[:, None](offsets[None, :]) & offsets[None, :]).to_array())
    metric_form = np.asarray((metric(offsets) & offsets).to_array())
    reference_form = np.asarray(offsets.norm_squared().to_array())
    basis_bound = np.abs(basis_forms).max()
    # Six levels at even steps either side of zero, the outermost three quarters of the way to the bound.
    nonzero_levels = contours.Lines(0.3 * basis_bound, 0.15 * basis_bound, np.append(to_rgb(wall_colour), 0.55), 0.9)
    # The zero level through the centroid consists of the two null lines parallel to the other two edges.
    null_lines = contours.Level(0.0, np.append(to_rgb(primal_colour), 1.0), 1.5)

    figure, axes = charts.subplots(*row_size, ncols=4)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    tails, heads = core.triangle.corners[2, 1]
    for index in range(3):
        ax = axes[index]
        frame_panel(ax, corners, 0.22)
        camera = flat_panel(ax, places.algebra)
        values = basis_forms[index]
        colours = diverging_colours(Normalize(-basis_bound, basis_bound)(values))[:, :3]
        simplices.overlay(ax, [
            contours.picture(places, topology, values, colours, camera, lines=(nonzero_levels,), levels=(null_lines,)),
            simplices.strokes(corners, core.triangle, ink(wall_colour), thin_width, camera),
            simplices.strokes(corners[[tails[index], heads[index]]], simplices.Topology.chains([2]),
                              ink(accent_colour), edge_width, camera),
            simplices.markers(corners, ink(wall_colour), small_dot, camera),
            *ringed(corners[index][None], primal_colour, highlight_dot, camera)])

    ax = axes[3]
    frame_panel(ax, corners, 0.22)
    camera = flat_panel(ax, places.algebra)
    max_metric = metric_form.max()
    ellipse_levels = np.linspace(0.03 * max_metric, 0.28 * max_metric, 4)
    ellipses = tuple(contours.Level(level, np.append(to_rgb(accent_colour), 1.0), 1.9) for level in ellipse_levels)
    # The reference circles, the same levels of a second field over the same samples, a clear picture over the first.
    circles = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.65), 0.9) for level in ellipse_levels)
    simplices.overlay(ax, [
        contours.picture(places, topology, metric_form, hat_colours(Normalize(0.0, max_metric)(metric_form))[:, :3],
                         camera, levels=ellipses),
        contours.picture(places, topology, reference_form, np.zeros(3), camera, levels=circles, opacity=0.0),
        simplices.strokes(corners, core.triangle, ink(wall_colour), outline_width, camera),
        *ringed(corners, accent_colour, corner_dot, camera),
        simplices.markers(centroid[None], ink(wall_colour), centroid_dot, camera)])
    return figure


def draw_curvature_defects(topology: core.Topology, vertices: core.Point, centres: core.Point,
                           metrics: core.Metric, curvatures: core.Bivector) -> plt.Figure:
    """Vertex curvature shades the fixed reference disks, with equal-length displacement contours at sampled faces."""
    mv = vertices.context.multivector
    # The curvature at the interior vertices, and nothing on the boundary, where the disk is cut off.
    interior = topology.reading(topology.interior[0])                         # Scalar+[interior <- vertices]
    curvature_values = np.asarray(((~interior * (interior * curvatures)).batch() & mv.xy.dual()).to_array())
    points = vertices.batch()                                                 # [vertices] Point-
    bound = np.abs(curvature_values).max()

    ellipse_faces = spaced_faces(centres, 0.84, 0.24)
    radius = 0.082
    span = np.linspace(-1.6 * radius, 1.6 * radius, n_ellipse_grid)
    grid_x, grid_y = np.meshgrid(span, span, indexing="ij")
    grid_offsets = mv.x.dual() * grid_x + mv.y.dual() * grid_y                 # [grid, grid] Point-
    sampled = topology.reading(np.array(ellipse_faces), 2)                    # Scalar+[samples <- faces]
    sampled_metrics = (sampled * metrics).batch()
    # Implicit evaluation of each face's Regge metric quadratic form on the local displacement grid.
    quadrics = np.asarray((sampled_metrics[:, :, None, None](grid_offsets) & grid_offsets).to_array())
    reference_quadric = grid_offsets.norm_squared().to_array()                # [grid, grid]
    sampled_centres = (sampled * centres).batch()                             # [samples] Point-
    # Each local grid placed at its face's centre, where its ellipses are drawn, its points numbered row by row; and
    # all the grids together, one mesh for the reference circles of them all.
    n_samples = len(ellipse_faces)
    grid_points = (sampled_centres[:, None, None] + grid_offsets
                   ).reshape(n_samples, -1)                                   # [samples, grid * grid] Point-
    grid = contours.Topology.grid(n_ellipse_grid, n_ellipse_grid)
    grids = contours.Topology(n_samples * grid.n_vertices,
                          (grid.elements[2] + grid.n_vertices * np.arange(n_samples)[:, None, None]).reshape(-1, 3))
    reference_circles = contours.Level(radius**2, np.append(to_rgb(wall_colour), 0.65), 0.75)

    # Log-ratio of radial to circumferential metric stretch at each sampled face centroid for indicatrix colouring.
    radial_dirs = sampled_centres.normalized()                                # [samples] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()
    hoop_dirs = quarter_turn >> radial_dirs
    radial_stretch = np.asarray((sampled_metrics(radial_dirs[None, :]) & radial_dirs[None, :]).to_array())
    hoop_stretch = np.asarray((sampled_metrics(hoop_dirs[None, :]) & hoop_dirs[None, :]).to_array())
    anisotropy = np.log(radial_stretch / hoop_stretch)
    anisotropy_bound = np.abs(anisotropy).max()
    indicatrix_norm = Normalize(-anisotropy_bound, anisotropy_bound)

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for case_index, ax in enumerate(axes):
        frame_panel(ax, vertices, 0.08)
        camera = flat_panel(ax, points.algebra)
        field = curvature_values[case_index]
        # Each face's ellipse its own clear picture, filled translucent inside and outlined in its colour: the fills
        # under the reference circles, the outlines over them.
        fills, outlines = [], []
        for places_at, quadric, stretch in zip(grid_points, quadrics[case_index], anisotropy[case_index]):
            colour = np.array(indicatrix_colours(indicatrix_norm(stretch)))
            inside = contours.Bands(0.0, radius**2, np.array([np.append(colour[:3], 0.22), np.zeros(4)]))
            fills.append(contours.picture(places_at, grid, quadric.reshape(-1), np.zeros(3), camera, bands=(inside,),
                                          opacity=0.0))
            outline = contours.Level(radius**2, colour, 1.5)
            outlines.append(contours.picture(places_at, grid, quadric.reshape(-1), np.zeros(3), camera,
                                             levels=(outline,), opacity=0.0))
        simplices.overlay(ax, [
            contours.picture(points, topology, field, diverging_colours(Normalize(-bound, bound)(field))[:, :3], camera),
            simplices.strokes(points, topology, ink(mesh_colour), mesh_width, camera, opacity=mesh_opacity),
            *fills,
            contours.picture(grid_points.reshape(-1), grids, np.tile(reference_quadric.reshape(-1), len(grid_points)),
                             np.zeros(3), camera, levels=(reference_circles,), opacity=0.0),
            *outlines,
            walls(topology, points, camera)])
    return figure


def draw_intrinsic_flow(topology: core.Topology, vertices: core.Point,
                        potential: core.Scalar,
                        current: core.Point) -> plt.Figure:
    """Potential colours and current streaks on one shared magnitude scale across the Regge metrics."""
    potentials = np.asarray(potential.batch().to_array())
    potential_levels = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.45), 1.0)
                             for level in np.linspace(-0.85, 0.85, n_contours + 2)[1:-1])
    # Area-weighted vertex currents give a continuous drawing of the facewise solution.
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)
    areas = ((sides[0] & sides[1]) / 2).norm()                              # Scalar+[faces]
    corners = ~topology.incidence[2, 0].mean(axis=0)
    vertex_current = (corners * (areas * current)) / (corners * areas)       # [cases] Point-[vertices]
    reference_speed = float(np.percentile(vertex_current.norm().batch().to_array(), 95))
    points = vertices.batch()                                                 # [vertices] Point-
    directions = vertex_current.batch()                                       # [cases, vertices] Point-

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index, ax in enumerate(axes):
        frame_panel(ax, vertices, 0.08)
        camera = flat_panel(ax, points.algebra)
        colours = diverging_colours(Normalize(-1.0, 1.0)(potentials[index]))[:, :3]
        simplices.overlay(ax, [
            contours.picture(points, topology, potentials[index], colours, camera, levels=potential_levels),
            simplices.strokes(points, topology, ink(mesh_colour), mesh_width, camera, opacity=0.45),
            flow.layer(points, topology, directions[index], camera, reference_speed, ink(flow_colour)),
            walls(topology, points, camera)])
    return figure
