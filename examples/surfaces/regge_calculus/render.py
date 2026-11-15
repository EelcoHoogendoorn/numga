"""The figures display Regge edge-metric basis forms, metric ellipses, vertex curvature defects, and intrinsic conduction streaks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb

from numga import stack
from numga.render import Camera, contours, flow, simplices
from numga.topology import Topology
from examples.util import charts, shapes
from examples.util.panels import offset_panel, overlay
from examples.surfaces.regge_calculus import core

primal_colour = "#286580"
accent_colour = "#d77542"
radial_ellipse_colour = "#c84b31"
hoop_ellipse_colour = "#2d6a9f"
edge_colour = "#b8c4c9"
wall_colour = "#52616a"
flow_colour = "#173c4b"
metric_colours = LinearSegmentedColormap.from_list(
    "regge_metric", ("#f6f4ee", "#8fb8c9", "#286580"))
diverging_colours = LinearSegmentedColormap.from_list(
    "regge_diverging", ("#286580", "#f5f2eb", "#d77542"))
indicatrix_colours = LinearSegmentedColormap.from_list(
    "regge_indicatrix", (hoop_ellipse_colour, wall_colour, radial_ellipse_colour))
n_contours = 7
n_ellipse_grid = 25
# Pixel widths and marker radii for triangle outlines, highlighted edges, disk rims, and interior edges.
thin_width, outline_width, highlight_width, wall_width = 1.5, 1.7, 3.6, 1.8
stroke_width, stroke_opacity = 1.0, 0.38
small_dot, corner_dot, highlight_dot, centroid_dot = 2.9, 4.2, 4.6, 3.25
halo = 1.25
row_size, disk_row_size = (780, 195), (780, 260)


# --- plumbing -------------------------------------------------------------------------
def panel_frame(ax: plt.Axes, points: core.Point, margin_ratio: float) -> None:
    """Point extents along `x` and `y` set a square equal-aspect viewport on `ax` with no axes or ticks."""
    mv = points.context.multivector
    readings = stack((mv.x, mv.y))[:, None] & points.batch().reshape(-1)      # [2, ...] Scalar+
    low, high = readings.min(axis=-1), readings.max(axis=-1)                  # [2] Scalar+, Scalar+
    centre = (low + high) / 2
    half_span = (1 + margin_ratio) * (high - low).max() / 2
    limits = stack((centre - half_span, centre + half_span), axis=-1).to_array()
    ax.set(xlim=limits[0], ylim=limits[1], aspect="equal")
    ax.set_axis_off()


def ink(colour: str) -> np.ndarray:
    """A colour specification converts to a three-channel RGB float array (`[3]`)."""
    return np.array(to_rgb(colour))


def ringed_markers(points: core.Point, colour: str, radius: float, camera: Camera) -> list[np.ndarray]:
    """Two concentric marker layers put a white halo around coloured dots of pixel `radius` at `points`."""
    return [simplices.markers(points, np.ones(3), radius + halo, camera),
            simplices.markers(points, ink(colour), radius, camera)]


def rim_strokes(topology: core.Topology, points: core.Point, camera: Camera) -> np.ndarray:
    """Outer boundary edges of `topology` render as dark rim strokes at `points`."""
    return simplices.strokes(points, topology.border, ink(wall_colour), wall_width, camera)


def spaced_faces(centres: core.Point, max_radius: float, min_separation: float) -> list[int]:
    """Greedy radial filtering selects interior face indices within `max_radius` of the origin separated by at least `min_separation`."""
    face_centres = centres.batch()                                             # [...] Point-
    n_faces = len(face_centres)
    reach = face_centres.antinorm()                                            # [n_faces] Scalar+
    separations = (face_centres[:, None] - face_centres[None, :]).antinorm()   # [n_faces, n_faces] Scalar+
    face_indices: list[int] = []
    for face in reach.argsort():
        if reach[face] < max_radius and all(separations[face, other] > min_separation for other in face_indices):
            face_indices.append(int(face))
    return face_indices


def metric_basis_figure(corners: core.Point, sample_points: core.Point,
                        basis: core.Metric, metric: core.Metric) -> plt.Figure:
    """Four panels display the three hyperbolic edge-metric basis forms and their sum into a positive-definite metric ellipse."""
    sample_topology, _ = shapes.delaunay((stack((core.mv.x, core.mv.y)) & sample_points[..., None]).to_array())
    corner_points = corners.batch()                                            # [3] Point-
    triangle_basis = basis.batch()[:, 0]                                       # [3] Plane+ <- Point-
    triangle_metric = metric.batch()[0]                                        # [] Plane+ <- Point-
    centroid = corner_points.mean()                                            # [] Point-
    offsets = sample_points - centroid                                         # [...] Point-

    basis_forms = np.asarray((triangle_basis[:, None](offsets[None, :]) & offsets[None, :]).to_array())
    metric_form = np.asarray((triangle_metric(offsets) & offsets).to_array())
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
        panel_frame(ax, corner_points, 0.22)
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        values = basis_forms[index]
        colours = diverging_colours(Normalize(-basis_bound, basis_bound)(values))[:, :3]
        overlay(ax, [
            contours.picture(sample_points, sample_topology, values, colours, camera,
                             lines=(nonzero_levels,), levels=(null_lines,)),
            simplices.strokes(corner_points, core.triangle, ink(wall_colour), thin_width, camera),
            simplices.strokes(corner_points[[tails[index], heads[index]]], Topology.chains([2]),
                              ink(accent_colour), highlight_width, camera),
            simplices.markers(corner_points, ink(wall_colour), small_dot, camera),
            *ringed_markers(corner_points[index][None], primal_colour, highlight_dot, camera)])

    ax = axes[3]
    panel_frame(ax, corner_points, 0.22)
    camera = offset_panel(ax, core.mv.x, core.mv.y)
    max_metric = metric_form.max()
    ellipse_levels = np.linspace(0.03 * max_metric, 0.28 * max_metric, 4)
    ellipses = tuple(contours.Level(level, np.append(to_rgb(accent_colour), 1.0), 1.9) for level in ellipse_levels)
    circles = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.65), 0.9) for level in ellipse_levels)
    overlay(ax, [
        contours.picture(sample_points, sample_topology, metric_form,
                         metric_colours(Normalize(0.0, max_metric)(metric_form))[:, :3],
                         camera, levels=ellipses),
        contours.picture(sample_points, sample_topology, reference_form, np.zeros(3),
                         camera, levels=circles, opacity=0.0),
        simplices.strokes(corner_points, core.triangle, ink(wall_colour), outline_width, camera),
        *ringed_markers(corner_points, accent_colour, corner_dot, camera),
        simplices.markers(centroid[None], ink(wall_colour), centroid_dot, camera)])
    return figure


def curvature_defects_figure(topology: core.Topology, vertices: core.Point, centres: core.Point,
                             metrics: core.Metric, curvatures: core.Bivector) -> plt.Figure:
    """Three disk panels shade interior Gaussian curvature and overlay equal-length metric ellipses at sampled faces."""
    V, E, F = topology.n_elements
    mv = vertices.context.multivector
    interior = topology.reading(topology.interior[0], 0)                       # Scalar+[V_interior <- V]
    curvature_values = np.asarray(((interior.adjugate() * (interior * curvatures)).batch() & mv.xy.dual()).to_array())
    points = vertices.batch()                                                  # [V] Point-
    bound = np.abs(curvature_values).max()

    ellipse_faces = spaced_faces(centres, 0.84, 0.24)
    n_samples = len(ellipse_faces)
    radius = 0.082
    span = np.linspace(-1.6 * radius, 1.6 * radius, n_ellipse_grid)
    grid_x, grid_y = np.meshgrid(span, span, indexing="ij")
    grid_offsets = mv.x.dual() * grid_x + mv.y.dual() * grid_y                 # [n_ellipse_grid, n_ellipse_grid] Point-
    sample_reading = topology.reading(np.array(ellipse_faces), 2)              # Scalar+[F_sampled <- F]
    sample_metrics = (sample_reading * metrics).batch()                        # [3, n_samples] Plane+ <- Point-
    # Implicit evaluation of each face's Regge metric quadratic form on the local displacement grid.
    quadrics = np.asarray((sample_metrics[:, :, None, None](grid_offsets) & grid_offsets).to_array())
    reference_quadric = grid_offsets.norm_squared().to_array()                 # [n_ellipse_grid, n_ellipse_grid]
    sample_centres = (sample_reading * centres).batch()                        # [n_samples] Point-
    # Place a local sample grid at each chosen face centroid and combine all local grids into one triangulation.
    grid = Topology.grid(n_ellipse_grid, n_ellipse_grid)
    V_grid, E_grid, F_grid = grid.n_elements
    grid_points = (sample_centres[:, None, None] + grid_offsets).reshape(n_samples, -1)  # [n_samples, V_grid] Point-
    grids = Topology(n_samples * V_grid,
                     (grid.elements[2] + V_grid * np.arange(n_samples)[:, None, None]).reshape(-1, 3))
    reference_circles = contours.Level(radius**2, np.append(to_rgb(wall_colour), 0.65), 0.75)

    # Log-ratio of radial to circumferential metric stretch colours each indicatrix ellipse.
    radial_directions = sample_centres.normalized()                            # [n_samples] Point-
    quarter_turn = (mv.xy * (-np.pi / 4)).exp()                                # [] Rotor+
    hoop_directions = quarter_turn >> radial_directions                        # [n_samples] Point-
    radial_stretch = np.asarray((sample_metrics(radial_directions[None, :]) & radial_directions[None, :]).to_array())
    hoop_stretch = np.asarray((sample_metrics(hoop_directions[None, :]) & hoop_directions[None, :]).to_array())
    anisotropy = np.log(radial_stretch / hoop_stretch)
    anisotropy_bound = max(float(np.abs(anisotropy).max()), 1e-12)
    indicatrix_norm = Normalize(-anisotropy_bound, anisotropy_bound)
    curvature_bound = max(float(bound), 1e-12)

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for case_index, ax in enumerate(axes):
        panel_frame(ax, vertices, 0.08)
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        field = curvature_values[case_index]
        fills, outlines = [], []
        for local_points, quadric, stretch in zip(grid_points, quadrics[case_index], anisotropy[case_index]):
            colour = np.array(indicatrix_colours(indicatrix_norm(stretch)))
            fill_band = contours.Bands(0.0, radius**2, np.array([np.append(colour[:3], 0.22), np.zeros(4)]))
            fills.append(contours.picture(local_points, grid, quadric.reshape(-1), np.zeros(3), camera,
                                          bands=(fill_band,), opacity=0.0))
            outline = contours.Level(radius**2, colour, 1.5)
            outlines.append(contours.picture(local_points, grid, quadric.reshape(-1), np.zeros(3), camera,
                                             levels=(outline,), opacity=0.0))
        overlay(ax, [
            contours.picture(points, topology, field,
                             diverging_colours(Normalize(-curvature_bound, curvature_bound)(field))[:, :3], camera),
            simplices.strokes(points, topology, ink(edge_colour), stroke_width, camera, opacity=stroke_opacity),
            *fills,
            contours.picture(grid_points.reshape(-1), grids, np.tile(reference_quadric.reshape(-1), len(grid_points)),
                             np.zeros(3), camera, levels=(reference_circles,), opacity=0.0),
            *outlines,
            rim_strokes(topology, points, camera)])
    return figure


def intrinsic_flow_figure(topology: core.Topology, vertices: core.Point,
                          potential: core.Scalar, current: core.Point) -> plt.Figure:
    """Three disk panels display potential contours and current streaks on one shared speed scale across the Regge metrics."""
    V, E, F = topology.n_elements
    potentials = np.asarray(potential.batch().to_array())
    potential_levels = tuple(contours.Level(level, np.append(to_rgb(wall_colour), 0.45), 1.0)
                             for level in np.linspace(-0.85, 0.85, n_contours + 2)[1:-1])
    # Area-weighted vertex currents give a continuous streak field from the facewise solution.
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)               # [3] Point-[F]
    areas = ((sides[0] & sides[1]) / 2).antinorm()                             # Scalar+[F]
    vertex_weights = topology.incidence[2, 0].mean(axis=0).adjugate()          # Scalar+[V <- F]
    vertex_current = (vertex_weights * (areas * current)) / (vertex_weights * areas)  # [3] Point-[V]
    reference_speed = float(np.percentile(vertex_current.antinorm().batch().to_array(), 95))
    points = vertices.batch()                                                  # [V] Point-
    directions = vertex_current.batch()                                        # [3, V] Point-

    figure, axes = charts.subplots(*disk_row_size, ncols=3)
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.04)

    for index, ax in enumerate(axes):
        panel_frame(ax, vertices, 0.08)
        camera = offset_panel(ax, core.mv.x, core.mv.y)
        colours = diverging_colours(Normalize(-1.0, 1.0)(potentials[index]))[:, :3]
        overlay(ax, [
            contours.picture(points, topology, potentials[index], colours, camera, levels=potential_levels),
            simplices.strokes(points, topology, ink(edge_colour), stroke_width, camera, opacity=0.45),
            flow.layer(points, topology, directions[index], camera, reference_speed, ink(flow_colour)),
            rim_strokes(topology, points, camera)])
    return figure
