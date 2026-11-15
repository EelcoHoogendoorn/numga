"""Scene setups for double contact and Penrose's deforming eight-conic cube."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.math.eight_conics import core

n_default_frames = 216
frame_duration_ms = 40
closing_edges = np.array([9, 10, 11])


def double_contact_pair() -> tuple[core.Quadric, core.Quadric, core.Plane, core.Point]:
    """A unit circle and a plane dyad produce a second conic in double contact at two real crossings (`[2] Point-`)."""
    strength = 2.5
    pair_offset = 0.5
    pair_weight = 0.7
    center = core.mv.w.dual()                                               # [] Point-
    semi_axes = stack((core.mv.x.dual(), core.mv.y.dual()))                 # [2] Point-
    sphere = core.ellipsoid(center, semi_axes)                              # [] Plane+ <- Point-
    pair_plane = (core.mv.x - core.mv.w * pair_offset) * pair_weight        # [] Plane+
    double_plane = pair_plane * (pair_plane & core.Point)                   # [] Plane+ <- Point-
    contact_quadric = sphere + strength * double_plane                      # [] Plane+ <- Point-
    pair_contacts = core.intersections(sphere, pair_plane)                  # [2] Point-
    return sphere, contact_quadric, pair_plane, pair_contacts


def cube_states(n_frames: int) -> Iterator[tuple[core.Quadric, core.Plane, core.Point, core.Quadric]]:
    """Yields `(quadrics, edge_planes, contacts, completions)` across `n_frames` phases as three face couplings orbit a loop."""
    mean_couplings = np.array([0.30, -0.24, 0.27])
    coupling_amplitude = 0.25
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    plane_angles = np.arange(3) * (2 * np.pi / 3)
    couplings = mean_couplings + coupling_amplitude * np.cos(phases[:, None] + plane_angles)
    plane_offsets = np.array([0.20, 0.24, 0.16])
    plane_weight = 1.3

    center = core.mv.w.dual()                                               # [] Point-
    semi_axes = stack((core.mv.x.dual(), core.mv.y.dual()))                 # [2] Point-
    sphere = core.ellipsoid(center, semi_axes)                              # [] Plane+ <- Point-
    plane_rotors = (core.mv.xy * (-plane_angles / 2)).exp()                 # [3] Rotor+
    normals = plane_rotors >> core.mv.x                                     # [3] Plane+
    contact_planes = (normals - core.mv.w * plane_offsets) * plane_weight   # [3] Plane+
    return core.deformation(sphere, contact_planes, couplings)


def main() -> None:
    import matplotlib.pyplot as plt
    from examples.math.eight_conics import render
    from examples.util.animation import save_animation, save_figure

    sphere, contact_quadric, pair_plane, pair_contacts = double_contact_pair()
    pair_figure = render.double_contact_figure(sphere, contact_quadric, pair_contacts)
    save_figure(pair_figure, "eight_conics_double_contact")
    plt.close(pair_figure)

    recorded_states = []

    def streamed_states() -> Iterator[tuple[core.Quadric, core.Plane, core.Point, core.Quadric]]:
        for state in cube_states(n_default_frames):
            recorded_states.append(state)
            yield state

    frames = render.eighth_conic_animation(core.edges, closing_edges, streamed_states())
    save_animation(frames, "eight_conics_eighth", frame_duration_ms)

    # checks
    basis_points = stack((core.mv.x.dual(), core.mv.y.dual(), core.mv.w.dual()))  # [3] Point-
    assert ((sphere(pair_contacts) & pair_contacts).abs() < 1e-9).all()
    assert ((contact_quadric(pair_contacts) & pair_contacts).abs() < 1e-9).all()
    tangent_residual = (contact_quadric(pair_contacts) - sphere(pair_contacts))[..., None] & basis_points  # [2, 3] Scalar+
    assert (tangent_residual.abs() < 1e-9).all()

    streamed_quadrics = stack([state[0] for state in recorded_states])      # [216, 8] Plane+ <- Point-
    streamed_contacts = stack([state[2] for state in recorded_states])      # [216, 12, 2] Point-
    streamed_completions = stack([state[3] for state in recorded_states])   # [216, 3] Plane+ <- Point-

    completion_residual = (
        (streamed_completions[:, 1:] - streamed_completions[:, :1])[..., None](basis_points)[..., None]
        & basis_points
    )                                                                       # [216, 2, 3, 3] Scalar+
    assert (completion_residual.abs() < 1e-9).all()

    for end in range(2):
        edge_quadrics = streamed_quadrics[:, core.edges[:, end]][..., None]  # [216, 12, 1] Plane+ <- Point-
        assert ((edge_quadrics(streamed_contacts) & streamed_contacts).abs() < 1e-9).all()


if __name__ == "__main__":
    main()
