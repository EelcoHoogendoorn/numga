"""Scenes for discrete K-surfaces deform a saddle patch at constant edge length and construct a Bäcklund surface joined by equal-length tangent connectors."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.surfaces.k_surfaces import core

mv = core.mv

# Half-grid extent, angular arc span, curvature radius, bending sweep, and camera orbit settings.
n_half = 12
n_vertices = 2 * n_half + 1
extent = 0.8
curvature_radius = 1
n_frames = 84
n_bend_states = n_frames // 2 + 1
frame_duration_ms = 40
neutral_angle = np.pi / 2
bend_amplitude = np.pi / 6
seed_azimuth = 1.6
seed_turn = 0.65
n_orbit_frames = 180
n_orbit_views = n_orbit_frames // 2 + 1
orbit_elevation = 26
orbit_azimuth_centre = -70
orbit_azimuth_amplitude = 50
# Ordered volume pseudoscalar sets the oriented radius of curvature.
radius = curvature_radius * mv.xyz.dual()                                           # [] Antiscalar-


# --- math -----------------------------------------------------------------------------
def bending_envelope() -> core.Point:
    """Extreme bent saddle patches (`[2, n_vertices, n_vertices] Point-`) fix the orthographic camera bounds across the bend."""
    corner_extremes = neutral_angle + bend_amplitude * np.array([-1, 1])            # [2]
    return stack(list(core.saddle_states(n_half, extent, corner_extremes, radius))) # [2, n_vertices, n_vertices] Point-


def bending_states() -> Iterator[core.Point]:
    """Yields centred saddle vertex nets (`[n_vertices, n_vertices] Point-`) across one half-cycle of corner-angle bending."""
    phases = np.linspace(-np.pi / 2, np.pi / 2, n_bend_states)                      # [n_bend_states]
    corner_angles = neutral_angle + bend_amplitude * np.sin(phases)                 # [n_bend_states]
    return core.saddle_states(n_half, extent, corner_angles, radius)


def backlund_pair() -> tuple[core.Plane, core.Point, core.Plane, core.Point]:
    """Propagating a corner normal seed across the neutral saddle constructs its Bäcklund sheet (`normals`, `points`, `transformed_normals`, `transformed_points`)."""
    normals, points = core.saddle_patch(n_half, extent, neutral_angle, radius)      # [n_vertices, n_vertices] Plane+, Point-
    corner_normal = normals[0, 0]                                                   # [] Plane+
    reference_plane = (corner_normal ^ mv.x).normalized()                           # [] Bivector+
    spin_plane = corner_normal | mv.xyz                                             # [] Bivector+
    azimuth_turn = (spin_plane * (-seed_azimuth / 2)).exp()                         # [] Rotor+
    seed_plane = azimuth_turn >> reference_plane                                    # [] Bivector+
    elevation_turn = (seed_plane * (-seed_turn / 2)).exp()                          # [] Rotor+
    starting_normal = elevation_turn >> corner_normal                               # [] Plane+
    transformed_normals = core.backlund_normals(normals, starting_normal, core.quad_normal)  # [n_vertices, n_vertices] Plane+
    transformed_points = points + radius * transformed_normals.commutator(normals)  # [n_vertices, n_vertices] Point-
    return normals, points, transformed_normals, transformed_points


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.k_surfaces import render

    reference = bending_envelope()                                                  # [2, n_vertices, n_vertices] Point-
    recorded: list[core.Point] = []

    def streaming() -> Iterator[core.Point]:
        for state in bending_states():
            recorded.append(state)
            yield state

    save_animation(render.saddle_animation(reference, streaming()), "k_surfaces_saddle", frame_duration_ms,
                   forward_then_back=True)

    normals, points, transformed_normals, transformed_points = backlund_pair()
    orbit_phases = np.linspace(0, np.pi, n_orbit_views)                             # [n_orbit_views]
    orbit_azimuths = orbit_azimuth_centre + orbit_azimuth_amplitude * np.cos(orbit_phases)  # [n_orbit_views]
    save_animation(render.backlund_orbit_animation(transformed_points, points, orbit_azimuths, orbit_elevation),
                   "k_surfaces_backlund", frame_duration_ms, forward_then_back=True)

    # checks
    assert len(recorded) == n_bend_states
    # Bending preserves every grid edge length across the deformation, and Bäcklund connectors have uniform length
    # while lying in the tangent planes of both sheets.
    bent_patches = stack([recorded[0], recorded[-1]])                               # [2, n_vertices, n_vertices] Point-
    edge_lengths = (bent_patches[:, 1:, :] - bent_patches[:, :-1, :]).antinorm()    # [2, n_vertices - 1, n_vertices] Scalar+
    assert ((edge_lengths - edge_lengths[0, 0, 0]).abs() < 1e-10).all()
    connectors = transformed_points - points                                        # [n_vertices, n_vertices] Point-
    assert ((connectors.antinorm() - connectors[0, 0].antinorm()).abs() < 1e-10).all()
    assert ((normals & connectors).abs() < 1e-10).all()
    assert ((transformed_normals & connectors).abs() < 1e-10).all()


if __name__ == "__main__":
    main()
