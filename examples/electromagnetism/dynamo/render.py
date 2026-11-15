"""Cutaway, top-view field-line, and energy-history views for the planetary dynamo."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import chain

import numpy as np
from matplotlib.colors import to_rgb
from scipy.spatial import cKDTree

from numga import Algebra, concatenate, stack
from numga.extensor import Extensor
from numga.render import flow, meshes, simplices, solids
from numga.render.drawing import Draw
from numga.topology import Topology
from examples.electromagnetism.dynamo import core
from examples.util import charts, shapes
from examples.util.panels import image_figure

mv = core.mv
# Palette colours for the cut face, planet shell, magnetic field lines, streamlines, and backgrounds.
weak_colour, strong_colour = np.array(to_rgb("#1b1b3a")), np.array(to_rgb("#f59f00"))
surface_colour = np.array(to_rgb("#dee2e6"))
line_colour = np.array(to_rgb("#f1f3f5"))
stream_colour = np.array(to_rgb("#15aabf"))
background = np.ones(3)
dark_background = np.array(to_rgb("#1b1b3a"))
# Top-view camera angles, resolution, seed radius, translucent core shell, and downstream chevron geometry.
above_degrees = np.array([89.0, -30.0])
n_lines_pixels = 256
lines_seed_radii = np.array([0.6])
core_colour, core_opacity = np.array(to_rgb("#dee2e6")), 0.14
chevron_length = 5.0
n_chevron_steps = 15
# Cutaway camera framing, panel resolutions, cut orientation, and disk/sphere subdivisions.
view_degrees = np.array([18.0, -30.0])
surface_reach = 0.04
margin = 0.06
n_pixels = 320
n_mode_pixels = 160
cut_degrees = -120.0
n_rings, n_sectors = 24, 96
n_surface_levels = 3
# Field-line integration, Gaussian smoothing, and tube styling parameters.
n_seed_levels = 0
seed_radii = np.array([0.45, 0.75])
stream_seed_radii = np.array([0.45, 0.75])
n_trace_steps, trace_step = 60, 0.025
softening = 1e-2
n_neighbours, smoothing_reach = 16, 0.18
line_width = 1.6
# Surface-streak parameters on the cut plane and chart curve colours.
reference_percentile = 95
streak_pixels = 160.0
grain_pixels = 3.0
upstream_share = 0.9
energy_colour = "#e8590c"
flow_colour = "#15aabf"
axis_colour = "0.8"

antiscalar = mv.antipseudoscalar([1])                                           # Antiscalar-
cut_rotor = (mv.xy * (-np.radians(cut_degrees) / 2)).exp()                      # Rotor+
cut_plane = cut_rotor >> mv.y                                                   # Plane+
DiskPlane = core.Plane.derive.within(Algebra("x+z+"))                           # Vector+
cut_projection = core.Point - cut_plane.dual() * (cut_plane & core.Point)       # Point- <- Point-

strength_shader = meshes.Colour("""
fn colour(values: vec4<f32>, face_values: vec4<f32>, corner: vec3<f32>, front: bool) -> vec4<f32> {
    return vec4<f32>(mix(weak_colour, strong_colour, pow(clamp(values.x, 0.0, 1.0), 0.8)), 1.0);
}
""", dict(weak_colour=weak_colour, strong_colour=strong_colour))


# --- plumbing -------------------------------------------------------------------------
def camera_view(points: solids.Point, n_view_pixels: int, degrees: np.ndarray) -> solids.View:
    """Square camera framing encloses `points` from an elevation and azimuth in degrees."""
    elevation, azimuth = np.radians(degrees[0]), np.radians(degrees[1] + 90.0)
    return solids.view([points], surface_reach, elevation, azimuth, margin, n_view_pixels).squared()


def solid_points(points: core.Point) -> solids.Point:
    """Spatial directions embed into the solid renderer's projective space around the origin."""
    return solids.origin + solids.context.include_from_algebra(points.batch())


def line_tubes(lines: core.Point, inside_mask: np.ndarray, colour: np.ndarray, frame: solids.View,
               chevron: float) -> Draw:
    """Polylines (`[n_lines, n_line_points] Point-`) render as tubes on interior segments with periodic downstream chevrons."""
    n_lines, n_line_points = lines.shape
    chains = Topology.chains([n_line_points] * n_lines)
    V_chain, E_chain = chains.n_elements
    kept_edges = np.all(inside_mask[chains.elements[1]], axis=-1)               # [E_chain]
    chevron_flags = (chains.elements[1][:, 0] % n_line_points % n_chevron_steps == n_chevron_steps - 1).astype(float)  # [E_chain]
    return simplices.tubes(
        solid_points(lines.reshape(-1)), chains.subset(kept_edges, 1), colour, line_width, frame, solids.gloss,
        chevrons=chevron_flags[kept_edges], chevron=chevron,
    )


def nearest_indices(points: core.Point, centres: core.Point) -> np.ndarray:
    """Spatial KD-tree lookup finds the `n_neighbours` nearest tetrahedron indices (`[..., n_neighbours]`) for each query point."""
    places = (stack((mv.x, mv.y, mv.z)) & centres[..., None]).to_array()        # [..., 3]
    queries = (stack((mv.x, mv.y, mv.z)) & points[..., None]).to_array()        # [..., 3]
    return cKDTree(places).query(queries, n_neighbours)[1]


def smooth_field(points: core.Point, centres: core.Point, fields: Extensor) -> Extensor:
    """Gaussian distance weighting averages tetrahedral values over the `n_neighbours` nearest centroids."""
    near_indices = nearest_indices(points, centres)                             # [..., n_neighbours]
    offsets = points[..., None] - centres[near_indices]                         # [..., n_neighbours] Point-
    distances = offsets.antinorm()                                              # [..., n_neighbours] Scalar+
    weights = (-(distances * distances) / smoothing_reach**2).exp()             # [..., n_neighbours] Scalar+
    return (weights * fields[near_indices]).sum(axis=-1) / weights.sum(axis=-1)


def field_lines(centres: core.Point, fields: core.Point, seeds: core.Point) -> core.Point:
    """Fourth-order Runge–Kutta integration traces smoothed tangent curves (`[n_lines, 2 * n_trace_steps + 1] Point-`) both ways from `seeds`."""
    n_lines = seeds.shape[0]
    senses = np.array([-1.0, 1.0])[:, None]                                     # [2, 1]
    floor = (softening * fields.antinorm().max()) ** 2                          # Scalar+

    def unit_tangent(points: core.Point) -> core.Point:
        local_field = smooth_field(points, centres, fields) * senses            # [2, n_lines] Point-
        norms = local_field.antinorm()                                          # [2, n_lines] Scalar+
        return local_field / (norms * norms + floor).square_root()

    points = stack((seeds, seeds))                                              # [2, n_lines] Point-
    steps = [points]
    for _ in range(n_trace_steps):
        k1 = unit_tangent(points)
        k2 = unit_tangent(points + k1 * (trace_step / 2))
        k3 = unit_tangent(points + k2 * (trace_step / 2))
        k4 = unit_tangent(points + k3 * trace_step)
        points = points + (k1 + 2 * k2 + 2 * k3 + k4) * (trace_step / 6)
        steps.append(points)
    traced_points = stack(steps, axis=-1)                                       # [2, n_lines, n_trace_steps + 1] Point-
    return concatenate((traced_points[0, :, :0:-1], traced_points[1]), axis=-1)


def cut_disk(radius: float) -> tuple[Topology, core.Point]:
    """Polar disk topology and vertices rotated into the vertical cut plane span the cutaway cross-section."""
    disk, coordinates = shapes.disk(radius, n_rings, n_sectors)
    V_disk, E_disk, F_disk = disk.n_elements
    unturned_points = mv(DiskPlane, coordinates).dual()                         # [V_disk] Point-
    disk_points = cut_rotor >> unturned_points                                  # [V_disk] Point-
    return disk, disk_points


def back_shell(radius: float, n_view_pixels: int) -> tuple[Draw, solids.View]:
    """Back hemisphere behind the cut plane and matching square camera view frame the planet."""
    sphere, coordinates = shapes.sphere(n_surface_levels)
    V_sphere, E_sphere, F_sphere = sphere.n_elements
    surface_points = mv(core.Plane, radius * coordinates).dual()                # [V_sphere] Point-
    back_mask = np.all(((cut_plane & surface_points) < 0)[sphere.elements[2]], axis=-1)  # [F_sphere]
    hemisphere = sphere.subset(back_mask, -1)
    frame = camera_view(solid_points(surface_points), n_view_pixels, view_degrees)
    shell_draw = meshes.samples(
        solid_points(surface_points), hemisphere.elements[2], surface_colour,
        meshes.smooth(solid_points(surface_points), hemisphere), frame,
        face_values=np.zeros(1), colour=meshes.plain, gloss=solids.gloss,
    )
    return shell_draw, frame


def front_seeds(radii_values: np.ndarray) -> core.Point:
    """Icosahedral seed points at `radii_values` in front of the cut plane anchor cutaway tubes."""
    _, coordinates = shapes.sphere(n_seed_levels)
    candidates = mv(core.Plane, (radii_values[:, None, None] * coordinates).reshape(-1, 3)).dual()  # [...] Point-
    return candidates[(cut_plane & candidates) > 0.1]


def cutaway_tubes(lines: core.Point, radius: float, colour: np.ndarray, frame: solids.View) -> Draw:
    """Tube geometry clips `lines` to the removed front hemisphere inside `radius`."""
    n_lines, n_line_points = lines.shape
    places = lines.reshape(-1)                                                  # [n_lines * n_line_points] Point-
    inside_mask = (places.antinorm() < radius) & ((cut_plane & places) > 0)     # [n_lines * n_line_points]
    return line_tubes(lines, inside_mask, colour, frame, 0.0)


def cut_velocity(disk_points: core.Point, nearest_tets: np.ndarray, centres: core.Point,
                 velocity: core.Point) -> tuple[core.Point, np.ndarray]:
    """Projected velocity on the cut disk (`[...] Point-`) and fluid-shell mask mark moving disk vertices."""
    batch_velocity = velocity.batch()                                           # [...] Point-
    moving_mask = batch_velocity[nearest_tets].antinorm() > 0                   # [...]
    return cut_projection(smooth_field(disk_points, centres, batch_velocity)), moving_mask


def lines_animation(centres: core.Point,
                    states: Iterable[tuple[core.Magnetic, core.Point]]) -> Iterator[np.ndarray]:
    """Yields top-view frames of outer-core magnetic field lines and chevroned streamlines inside a translucent unit sphere."""
    batch_centres = centres.batch()                                             # [...] Point-
    _, seed_coordinates = shapes.sphere(n_seed_levels)
    seeds = mv(core.Plane, (lines_seed_radii[:, None, None] * seed_coordinates).reshape(-1, 3)).dual()  # [...] Point-
    _, sphere_coordinates = shapes.sphere(n_surface_levels)
    frame = camera_view(solid_points(mv(core.Plane, sphere_coordinates).dual()), n_lines_pixels, above_degrees)
    core_surface = solids.ball(solids.origin, 1.0)[None, None]
    for magnetic, velocity in states:
        field_directions = (magnetic.batch() ^ mv.t) * antiscalar               # [...] Point-
        magnetic_curves = field_lines(batch_centres, field_directions, seeds)   # [..., 2 * n_trace_steps + 1] Point-
        stream_curves = field_lines(batch_centres, velocity.batch(), seeds)     # [..., 2 * n_trace_steps + 1] Point-
        inside_masks = [(curves.reshape(-1).antinorm() < 1) for curves in (magnetic_curves, stream_curves)]
        yield solids.draw(
            [], frame, dark_background, inks=[],
            translucent=[(core_surface, core_colour, core_opacity)],
            meshes=[
                (line_tubes(magnetic_curves, inside_masks[0], line_colour, frame, 0.0), 1.0),
                (line_tubes(stream_curves, inside_masks[1], stream_colour, frame, chevron_length), 1.0),
            ],
        )


def cutaway_animation(centres: core.Point, radius: float,
                      states: Iterable[tuple[core.Magnetic, core.Point]]) -> Iterator[np.ndarray]:
    """Yields cutaway frames compositing downstream flow streaks on the cut plane with front-hemisphere magnetic field tubes."""
    batch_centres = centres.batch()                                             # [...] Point-
    disk, disk_points = cut_disk(radius)
    nearest_tets = nearest_indices(disk_points, batch_centres)[:, 0]            # [...]
    shell_draw, frame = back_shell(radius, n_pixels)
    bare_disk = meshes.samples(
        solid_points(disk_points), disk.elements[2], weak_colour,
        meshes.flat(solid_points(disk_points), disk.elements[2]), frame,
        face_values=np.zeros(1), colour=meshes.plain, gloss=0.0,
    )
    seeds = front_seeds(seed_radii)                                             # [...] Point-
    iterator = iter(states)
    first_state = next(iterator)
    first_tangent, first_moving = cut_velocity(disk_points, nearest_tets, batch_centres, first_state[1])
    reference_speed = float(np.percentile(first_tangent.antinorm().to_array()[first_moving], reference_percentile))
    base_image = solids.draw([], frame, background, inks=[], translucent=[],
                             meshes=[(bare_disk, 1.0), (shell_draw, 1.0)]) / 255
    for magnetic, velocity in chain((first_state,), iterator):
        field_directions = (magnetic.batch() ^ mv.t) * antiscalar               # [...] Point-
        tangent_velocity, moving_mask = cut_velocity(disk_points, nearest_tets, batch_centres, velocity)
        fluid_disk = disk.subset(np.all(moving_mask[disk.elements[2]], axis=-1), -1)
        streak_coverage = flow.texture(
            solid_points(disk_points), fluid_disk, solids.context.include_from_algebra(tangent_velocity),
            frame, np.ones(3), reference_speed, flow.surface_grain, grain_pixels,
            upstream_share, streak_pixels, np.ones(3),
        )["coverage"]                                                           # [n_pixels, n_pixels, 1]
        streaked_face = base_image * (1 - streak_coverage) + stream_colour * streak_coverage
        tube_layer, _ = solids.layer(
            [], frame, [], [],
            [(cutaway_tubes(field_lines(batch_centres, field_directions, seeds), radius, line_colour, frame), 1.0)],
        )                                                                       # [n_pixels, n_pixels, 4]
        composite = streaked_face * (1 - tube_layer[..., 3:]) + tube_layer[..., :3]
        yield np.round(np.clip(composite, 0, 1) * 255).astype(np.uint8)


def modes_figure(centres: core.Point, radius: float, velocities: core.Point) -> "plt.Figure":
    """Side-by-side cutaway panels colour the cut plane by flow speed and trace front-hemisphere streamlines for selected modes."""
    batch_centres = centres.batch()                                             # [...] Point-
    disk, disk_points = cut_disk(radius)
    shell_draw, frame = back_shell(radius, n_mode_pixels)
    seeds = front_seeds(stream_seed_radii)                                      # [...] Point-
    panels = []
    for velocity in velocities:
        batch_velocity = velocity.batch()                                       # [...] Point-
        strength = smooth_field(disk_points, batch_centres, batch_velocity).antinorm().to_array()
        levels = strength / strength.max()
        coloured_disk = meshes.samples(
            solid_points(disk_points), disk.elements[2], levels[disk.elements[2]][..., None],
            meshes.flat(solid_points(disk_points), disk.elements[2]), frame,
            face_values=np.zeros(1), colour=strength_shader, gloss=0.0,
        )
        stream_tubes = cutaway_tubes(field_lines(batch_centres, batch_velocity, seeds), radius, stream_colour, frame)
        panels.append(solids.draw([], frame, background, inks=[], translucent=[],
                                  meshes=[(coloured_disk, 1.0), (shell_draw, 1.0), (stream_tubes, 1.0)]))
    return image_figure(np.concatenate(panels, axis=1))


@charts.style()
def energies_figure(times: np.ndarray, kinetic_energy: core.Scalar, magnetic_energy: core.Scalar) -> "plt.Figure":
    """Logarithmic time-series chart plots kinetic energy in cyan and magnetic energy in orange from kinematic growth through saturation."""
    figure, axis = charts.subplots(300, 200, layout="constrained")
    axis.semilogy(times, kinetic_energy.to_array(), color=flow_colour)
    axis.semilogy(times, magnetic_energy.to_array(), color=energy_colour)
    axis.set(
        xlim=(times[0], times[-1]),
        xticks=[times[0], times[-1]],
        xticklabels=[f"{times[0]:g}", f"{times[-1]:g}"],
        xlabel="time",
        ylabel="energy",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(axis_colour)
    return figure
