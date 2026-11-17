"""Smoothed field-line tubes show magnetic linkage and selective decay."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import matplotlib.pyplot as plt
import numpy as np

from numga import concatenate, stack
from numga.render import meshes, simplices, solids
from numga.topology import Topology
from examples.electromagnetism.beltrami_fields import core
from examples.util import charts
from examples.util.panels import show_pixels

# The observer sees the point where each event's worldline meets the rest-frame spatial slice.
space_slice = core.mv.t ^ (core.Point & core.time)                              # [] Point- <- Point-

n_torus_pixels = 320
n_anim_pixels = 270
n_spectrum_pixels = charts.DESIGN_WIDTH - 2 * n_torus_pixels
spectrum_margins = np.array([44, 26, 8, 14])                                    # [4]
shown_mode_size = charts.points(2 * charts.MARKER_PX) ** 2
background = np.ones(3)                                                         # [3]
sapphire = np.array([0.20, 0.48, 0.86])                                         # [3]
teal = np.array([0.08, 0.68, 0.56])                                             # [3]
gold = np.array([0.95, 0.68, 0.16])                                             # [3]
vermilion = np.array([0.88, 0.32, 0.16])                                        # [3]
cream = np.array([0.93, 0.94, 0.96])                                            # [3]
torus_elevation, torus_azimuth = np.radians(28), np.radians(36)
top_elevation, top_azimuth = np.radians(82), np.radians(36)
vertical_axis = core.mv.z.dual()                                                # [] Point-

tube_major, tube_minor = 1.0, 0.33
seed_shares = np.array([0.24, 0.52, 0.78])                                      # [3]
seed_counts = np.array([3, 4, 5])                                               # [3]
n_lines = int(seed_counts.sum())
seed_angles_around_tube = np.array([0.20, 1.10, 2.15])                          # [3]
seed_angles_around_axis = np.array([0.0, np.pi / 4, np.pi / 5])                 # [3]
n_trace_steps, trace_step, trace_stride = 170, 0.018, 2
smoothing_turn, smoothing_reach = 0.58, 0.20
speed_floor, wall_pull = 0.14, 0.08
line_width, taper_share = 4.6, 0.15


# --- plumbing -------------------------------------------------------------------------
def field_lines(centres: core.Point, volumes: core.Volume, magnetic: core.Field) -> tuple[core.Point, np.ndarray]:
    """Runge-Kutta integration through the wedge-averaged, wall-tapered field traces confined field lines (`[n_lines, ...] Point-`)."""
    mv = core.mv
    shells = np.repeat(np.arange(len(seed_counts)), seed_counts)                # [n_lines]
    spacing = np.concatenate([np.arange(count) * (2 * np.pi / count) for count in seed_counts])  # [n_lines]
    tube_rotors = (mv.xz * (-(seed_angles_around_tube[shells] + spacing) / 2)).exp()  # [n_lines] Rotor+
    axis_rotors = (mv.xy * (-(seed_angles_around_axis[shells] + spacing) / 2)).exp()  # [n_lines] Rotor+
    radial_unit = mv.x.dual()                                                   # [] Point-
    tube_directions = tube_rotors >> radial_unit                                # [n_lines] Point-
    section_seeds = radial_unit * tube_major + tube_directions * (seed_shares[shells] * tube_minor)  # [n_lines] Point-
    seeds = axis_rotors >> section_seeds                                        # [n_lines] Point-

    # Symmetrize the two outer tetrahedra of each 3-tet prism wedge to remove alternating diagonal split bias.
    unit_volumes = volumes / (volumes & core.time).norm()                       # Volume-[T]
    n_wedges = len(centres.batch()) // 3
    wedge_sites = centres.batch().reshape(n_wedges, 3)                          # [n_wedges, 3] Point-
    wedge_fields = (magnetic * unit_volumes).batch().reshape(n_wedges, 3)       # [n_wedges, 3] Point-
    n_samples = 2 * n_wedges
    sample_sites = concatenate(((wedge_sites[:, 2] + wedge_sites[:, 1]) / 2, wedge_sites[:, 0]))    # [n_samples] Point-
    sample_fields = concatenate(((wedge_fields[:, 2] + wedge_fields[:, 1]) / 2, wedge_fields[:, 0]))  # [n_samples] Point-

    sample_heights = mv.z & sample_sites                                        # [n_samples] Scalar+
    sample_levels = sample_sites - vertical_axis * sample_heights               # [n_samples] Point-
    sample_radii = sample_levels.antinorm()                                     # [n_samples] Scalar+
    sample_radials = sample_levels / sample_radii                               # [n_samples] Point-
    sample_frames = stack(
        [sample_radials, sample_radials.commutator(mv.xy), vertical_axis.broadcast_to(sample_radials.shape)],
        axis=-1,
    )                                                                           # [n_samples, 3] Point-
    sample_readings = sample_frames | sample_fields[:, None]                    # [n_samples, 3] Scalar+
    typical_strength = sample_fields.squared().mean(axis=0).square_root()       # [] Scalar+

    def tangent_direction(points: core.Point, senses: np.ndarray) -> core.Point:
        """Cylindrical kernel smoothing and wall tapering give unit streamline tangents (`[2, n_lines] Point-`)."""
        heights = mv.z & points                                                 # [2, n_lines] Scalar+
        levels = points - vertical_axis * heights                               # [2, n_lines] Point-
        radii = levels.antinorm()                                               # [2, n_lines] Scalar+
        radials = levels / radii                                                # [2, n_lines] Point-
        chords = 2 - 2 * (radials[..., None] | sample_radials)                  # [2, n_lines, n_samples] Scalar+
        cross_distances = (radii[..., None] - sample_radii).squared() + (heights[..., None] - sample_heights).squared()
        weights = (-chords / smoothing_turn**2 - cross_distances / smoothing_reach**2).exp()
        mean_readings = (weights[..., None] * sample_readings).sum(axis=-2) / weights.sum(axis=-1)[..., None]
        directed_readings = senses[..., None] * mean_readings                   # [2, n_lines, 3] Scalar+
        frames = stack([radials, radials.commutator(mv.xy), vertical_axis.broadcast_to(radials.shape)], axis=-1)  # [2, n_lines, 3] Point-
        field_vectors = (frames * directed_readings).sum(axis=-1)               # [2, n_lines] Point-

        # Taper outward motion near the conducting wall so streamlines stay confined inside the cavity.
        tube_offsets = points - radials * tube_major                            # [2, n_lines] Point-
        tube_radii = tube_offsets.antinorm()                                    # [2, n_lines] Scalar+
        tube_radials = tube_offsets / tube_radii                                # [2, n_lines] Point-
        radial_shares = (tube_radii / tube_minor).clip(0, 1)                    # [2, n_lines] Scalar+
        outward_components = field_vectors | tube_radials                       # [2, n_lines] Scalar+
        tapered_components = outward_components * (0.5 * np.pi * radial_shares).cos().squared() - wall_pull * radial_shares**3 * typical_strength
        confined_vectors = field_vectors + tube_radials * (tapered_components - outward_components)  # [2, n_lines] Point-
        return confined_vectors / (confined_vectors.squared() + (speed_floor * typical_strength).squared()).square_root()

    senses = np.array([-1.0, 1.0])[:, None]                                     # [2, 1]
    points = seeds[None].broadcast_to((2, n_lines))                             # [2, n_lines] Point-
    traced_samples = [points]
    for step in range(1, n_trace_steps + 1):
        stage_1 = tangent_direction(points, senses)
        stage_2 = tangent_direction(points + stage_1 * (0.5 * trace_step), senses)
        stage_3 = tangent_direction(points + stage_2 * (0.5 * trace_step), senses)
        stage_4 = tangent_direction(points + stage_3 * trace_step, senses)
        points = points + (stage_1 + 2 * stage_2 + 2 * stage_3 + stage_4) * (trace_step / 6.0)
        if step % trace_stride == 0:
            traced_samples.append(points)
    n_half_steps = len(traced_samples)
    traced_branches = stack(traced_samples, axis=-1)                            # [2, n_lines, n_half_steps] Point-
    return concatenate((traced_branches[0, :, :0:-1], traced_branches[1]), axis=-1), shells


def segment_colours(climb_rates: np.ndarray, shells: np.ndarray) -> np.ndarray:
    """Radial shell colours blend toward warm gold where field lines climb poloidally (`[n_lines, ..., 3]`)."""
    base_colours = np.array([vermilion, teal, sapphire])[shells]                # [n_lines, 3]
    pitch_weight = np.clip(0.5 + 0.45 * climb_rates, 0.0, 1.0)[..., None]
    blended_colours = (1.0 - 0.38 * pitch_weight) * base_colours[:, None, :] + (0.38 * pitch_weight) * gold[None, None, :]
    return np.clip(blended_colours, 0.0, 1.0)


def torus_field_image(
    topology: Topology,
    vertices: core.Point,
    lines: core.Point,
    shells: np.ndarray,
    view: solids.View,
) -> np.ndarray:
    """Field-line tubes sit inside a smooth translucent toroidal cavity wall."""
    torus_points = solids.origin + solids.context.include_from_algebra(space_slice(vertices).batch())
    wall = topology.border
    wall_colours = np.broadcast_to(cream, wall.elements[2].shape + (3,))
    wall_surface = meshes.samples(
        torus_points, wall.elements[2], wall_colours, meshes.smooth(torus_points, wall), view,
        face_values=np.zeros(4), colour=meshes.plain, gloss=solids.gloss,
    )

    n_curves, n_steps = lines.shape
    starts, ends = lines[:, :-1], lines[:, 1:]                                  # [n_curves, n_steps - 1] Point-
    steps = ends - starts                                                       # [n_curves, n_steps - 1] Point-
    climb_rates = (core.mv.z & steps) / steps.antinorm()                        # [n_curves, n_steps - 1] Scalar+
    colours = segment_colours(climb_rates.to_array(), shells)                   # [n_curves, n_steps - 1, 3]
    scene_lines = solids.origin + solids.context.include_from_algebra(space_slice(lines).batch())
    chains = Topology.chains([n_steps] * n_curves)
    # Each open line's tube thins to a point over the last `taper_share` of its length at either end, along a quarter
    # circle of the share of its length reached at each point.
    reached = np.cumsum(steps.antinorm().to_array(), axis=1)                    # [n_curves, n_steps - 1]
    shares = np.concatenate((np.zeros((n_curves, 1)), reached), axis=1) / reached[:, -1:]   # [n_curves, n_steps]
    ends = np.minimum(np.minimum(shares, 1 - shares) / taper_share, 1)          # [n_curves, n_steps]
    widths = line_width * np.sqrt(ends * (2 - ends))                            # [n_curves, n_steps]
    tubes = simplices.tubes(scene_lines.reshape(-1), chains, colours.reshape(-1, 3), widths.reshape(-1), view,
                            gloss=solids.gloss)

    return solids.draw([], view, background, inks=[], translucent=[], meshes=[(wall_surface, 0.16), (tubes, 1.0)])


@charts.style()
def eigenmodes_figure(
    topology: Topology,
    vertices: core.Point,
    centres: core.Point,
    volumes: core.Volume,
    inverse_eigenvalues: core.Scalar,
    mode_fields: core.Field,
    shown_indices: np.ndarray,
) -> plt.Figure:
    """The chiral Beltrami spectrum sits beside 3D field-line views of the Taylor state and an excited mode."""
    inverse_values = np.asarray(inverse_eigenvalues.to_array())
    n_modes = len(inverse_values)
    mode_indices = np.arange(n_modes)

    view = solids.View(solids.aim(solids.origin, torus_elevation, torus_azimuth), np.full(2, 1.38), n_torus_pixels)
    taylor_image = torus_field_image(topology, vertices, *field_lines(centres, volumes, mode_fields[0]), view)
    excited_image = torus_field_image(topology, vertices, *field_lines(centres, volumes, mode_fields[1]), view)

    width, height = charts.DESIGN_WIDTH, n_torus_pixels
    figure = plt.figure(figsize=(width / charts.DPI, height / charts.DPI))
    left_margin, bottom_margin, right_margin, top_margin = spectrum_margins
    ax_spectrum = figure.add_axes([
        left_margin / width,
        bottom_margin / height,
        (n_spectrum_pixels - left_margin - right_margin) / width,
        (height - bottom_margin - top_margin) / height,
    ])
    ax_taylor = figure.add_axes([n_spectrum_pixels / width, 0.0, n_torus_pixels / width, 1.0])
    ax_excited = figure.add_axes([(n_spectrum_pixels + n_torus_pixels) / width, 0.0, n_torus_pixels / width, 1.0])

    ax_spectrum.axhline(0.0, color="#6c7a82", linewidth=charts.FRAME_WIDTH, zorder=1)
    negative_mask = inverse_values < 0
    positive_mask = inverse_values > 0
    ax_spectrum.scatter(mode_indices[negative_mask], inverse_values[negative_mask], color=sapphire, alpha=0.82, zorder=2)
    ax_spectrum.scatter(mode_indices[positive_mask], inverse_values[positive_mask], color=vermilion, alpha=0.82, zorder=2)
    shown_positions = shown_indices % n_modes
    ax_spectrum.scatter(
        shown_positions, inverse_values[shown_positions], s=shown_mode_size, color=[vermilion, teal],
        edgecolor="white", zorder=4,
    )
    max_inverse = float(np.abs(inverse_values).max())
    ax_spectrum.set_xlim(-4, n_modes + 3)
    ax_spectrum.set_ylim(-1.15 * max_inverse, 1.15 * max_inverse)
    ax_spectrum.set_xticks([0, n_modes - 1])
    ax_spectrum.set_yticks([-round(max_inverse, 2), 0.0, round(max_inverse, 2)])
    ax_spectrum.spines["top"].set_visible(False)
    ax_spectrum.spines["right"].set_visible(False)
    ax_spectrum.spines["left"].set_color("#52616a")
    ax_spectrum.spines["bottom"].set_color("#52616a")
    ax_spectrum.tick_params(colors="#52616a")

    for ax, image in ((ax_taylor, taylor_image), (ax_excited, excited_image)):
        show_pixels(ax, image)
    return figure


def relaxation_animation(
    topology: Topology,
    vertices: core.Point,
    centres: core.Point,
    volumes: core.Volume,
    states: Iterable[tuple[core.Scalar, core.Field]],
) -> Iterator[np.ndarray]:
    """Yields paired oblique and axial 3D views of magnetic field lines relaxing toward the Taylor state."""
    oblique_view = solids.View(solids.aim(solids.origin, torus_elevation, torus_azimuth), np.full(2, 1.38), n_anim_pixels)
    axial_view = solids.View(solids.aim(solids.origin, top_elevation, top_azimuth), np.full(2, 1.38), n_anim_pixels)
    separator = np.full((n_anim_pixels, 8, 3), 255, dtype=np.uint8)
    for _, magnetic in states:
        lines, shells = field_lines(centres, volumes, magnetic)
        left_panel = torus_field_image(topology, vertices, lines, shells, oblique_view)
        right_panel = torus_field_image(topology, vertices, lines, shells, axial_view)
        yield np.concatenate((left_panel, separator, right_panel), axis=1)
