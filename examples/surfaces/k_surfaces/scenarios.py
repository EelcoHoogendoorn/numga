"""A bending saddle and its Bäcklund partners, built from crossing curves of unit normals."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.surfaces.k_surfaces import core

mv = core.mv


# --- math

def saddle(n_half: int, extent: float, corner_angle: float,
           radius: core.Antipseudoscalar) -> tuple[core.Normal, core.Point]:
    """A centred patch grows from two crossing normal curves, great-circle arcs."""
    n_vertices = 2 * n_half + 1
    angles = np.linspace(-extent, extent, n_vertices)
    alternating = 1 - 2 * ((np.arange(n_vertices) - n_half) % 2)
    reference_normal = mv.vector([0, 0, 1])                              # Vector+
    second_plane = (mv.xy * (-corner_angle / 2)).exp() >> mv.yz          # Bivector+
    mid_u = (mv.yz * (-angles / 2)).exp() >> reference_normal             # [vertices] Vector+
    # Opposite signs on neighboring columns give one completion rule for both directions.
    mid_v = alternating * ((second_plane * (-angles / 2)).exp() >> reference_normal)
    normals = core.normal_net(mid_u, mid_v, core.complete)               # [vertices, vertices] Vector+
    points = core.positions(normals, radius, reference_normal.dual() * 0)  # [vertices, vertices] Antivector-
    return normals, points - points[n_half, n_half]


def deforming_saddle(n_half: int, extent: float, corner_angles: np.ndarray,
                     radius: core.Antipseudoscalar) -> Iterator[core.Point]:
    """The angle between the two normal arcs varies while their sample spacing is kept."""
    for corner_angle in corner_angles:
        _, points = saddle(n_half, extent, corner_angle, radius)
        yield points


def backlund_family(normals: core.Normal, points: core.Point,
                    radius: core.Antipseudoscalar,
                    starting_normals: core.Normal) -> Iterator[core.Point]:
    """Each starting normal propagates to a partner surface, joined to the saddle at corresponding points."""
    for starting_normal in starting_normals:
        partner_normals = core.partner(normals, starting_normal, core.complete)  # [vertices, vertices] Normal+
        yield points + radius * partner_normals.commutator(normals)


# --- plumbing

def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.k_surfaces import render

    n_half = 12
    extent = 0.8
    n_frames = 84
    duration_ms = 40
    neutral_angle = np.pi / 2
    bend_amplitude = np.pi / 6
    seed_azimuth = 1.6
    seed_turn = 0.65
    n_orbit_frames = 180
    orbit_elevation = 26
    orbit_azimuth_centre = -70
    orbit_azimuth_amplitude = 50
    # The bend from one extreme to the other, and the orbit's swing one way; the animations play them forward and
    # back.
    phases = np.linspace(-np.pi / 2, np.pi / 2, n_frames // 2 + 1)
    corner_angles = neutral_angle + bend_amplitude * np.sin(phases)
    corner_extremes = neutral_angle + bend_amplitude * np.array([-1, 1])
    orbit_phases = np.linspace(0, np.pi, n_orbit_frames // 2 + 1)
    orbit_azimuths = orbit_azimuth_centre + orbit_azimuth_amplitude * np.cos(orbit_phases)
    # The ordered frame gives the length scale its orientation.
    radius = mv.xyz.dual()                                              # [] Antipseudoscalar-

    reference = stack(deforming_saddle(n_half, extent, corner_extremes, radius))
    states = deforming_saddle(n_half, extent, corner_angles, radius)
    save_animation(render.frames(states, reference), "k_surfaces_saddle", duration_ms, forward_then_back=True)

    normals, points = saddle(n_half, extent, neutral_angle, radius)
    corner_normal = normals[0, 0]                                       # Vector+
    reference_plane = (corner_normal ^ mv.x).normalized()                # Bivector+
    spin_plane = corner_normal | mv.xyz                                 # [] Bivector+
    seed_plane = (spin_plane * (-seed_azimuth / 2)).exp() >> reference_plane  # Bivector+
    starting_normal = (seed_plane * (-seed_turn / 2)).exp() >> corner_normal  # Vector+
    partner_normals = core.partner(normals, starting_normal, core.complete)  # [vertices, vertices] Vector+
    partner_points = points + radius * partner_normals.commutator(normals)  # [vertices, vertices] Antivector-
    save_animation(render.partner_orbit(partner_points, points, orbit_azimuths, orbit_elevation),
                   "k_surfaces_backlund", duration_ms, forward_then_back=True)

    # checks
    # Bending keeps every grid edge's length, and the segments to the partner all have one length and lie
    # in the tangent planes of both sheets.
    lengths = (reference[:, 1:, :] - reference[:, :-1, :]).norm().to_array()   # [extremes, rows - 1, columns]
    np.testing.assert_allclose(lengths, lengths[0, 0, 0], atol=1e-10)
    connectors = partner_points - points                                # [vertices, vertices] Antivector-
    np.testing.assert_allclose(connectors.norm().to_array(), connectors[0, 0].norm().to_array(), atol=1e-10)
    np.testing.assert_allclose((normals & connectors).to_array(), 0, atol=1e-10)
    np.testing.assert_allclose((partner_normals & connectors).to_array(), 0, atol=1e-10)


if __name__ == "__main__":
    main()
