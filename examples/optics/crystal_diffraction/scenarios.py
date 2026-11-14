"""Three planar deformations and a smooth deformation cycle scatter circularly polarized light from
a finite square crystal."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.optics.crystal_diffraction import core

n_shells = 3
n_orders = 3
n_atoms = (2 * n_shells + 1) ** 2
n_reflections = (2 * n_orders + 1) ** 2
spacing = 1.0
max_stretch = 1.7
max_shear = 0.6
max_angle = np.pi / 9
# Unstrained square layer, uniaxial x-stretch, and combined stretch, shear, and rotation.
stretches = np.array([1.0, max_stretch, max_stretch])
shears = np.array([0.0, 0.0, max_shear])
angles = np.array([0.0, 0.0, max_angle])
n_cases = len(stretches)
half_width = 10.0
n_pixels = 101
n_frames = 60
n_movie_pixels = 211
frame_duration_ms = 40
# Null wave propagating along +z with initial electric field along x.
wavevector = core.mv.t - core.mv.z                                           # [] Wavevector+
incident = wavevector ^ core.mv.x                                            # [] Field+


# --- math -----------------------------------------------------------------------------
def comparison_cases() -> tuple[
    core.Event, core.Wavevector, core.Event, core.Wavevector, core.Wavevector, core.Scalar
]:
    """Three crystal states (`[n_cases, n_atoms] Event-`) and their carried reciprocal lattices
    (`[n_cases, n_reflections] Wavevector+`) scatter the incident wave into brightness maps
    (`[n_cases, n_pixels, n_pixels] Scalar+`)."""
    atoms, lattice = core.square_lattice(n_shells, n_orders, spacing)
    deformations = core.layer_deformation(stretches, shears, angles)         # [n_cases] Event- <- Event-
    positions = deformations[:, None](atoms)                                 # [n_cases, n_atoms] Event-
    reciprocal = deformations.adjugate()[:, None].solve(lattice)             # [n_cases, n_reflections] Wavevector+
    transfers = core.transfer_grid(half_width, n_pixels)                     # [n_pixels, n_pixels] Wavevector+
    fields = core.scattered_field(positions, transfers, incident)            # [n_cases, n_pixels, n_pixels] Field+
    return atoms, lattice, positions, reciprocal, transfers, core.energy_density(fields, fields)


def deformation_frames(
    transfers: core.Wavevector,
) -> Iterator[tuple[core.Event, core.Wavevector, core.Scalar]]:
    """Yields the deformed atom events (`[n_atoms] Event-`), reciprocal wavevectors
    (`[n_reflections] Wavevector+`), and scattered brightness (`[n_movie_pixels, n_movie_pixels] Scalar+`)
    across a half-cycle of stretch, shear, and rotation."""
    atoms, lattice = core.square_lattice(n_shells, n_orders, spacing)
    for phase in np.linspace(0, np.pi, n_frames // 2 + 1):
        weight = np.array((1 - np.cos(phase)) / 2)
        frame_map = core.layer_deformation(
            1 + (max_stretch - 1) * weight, max_shear * weight, max_angle * weight
        )                                                                    # [] Event- <- Event-
        positions = frame_map(atoms)                                         # [n_atoms] Event-
        reciprocal = frame_map.adjugate().solve(lattice)                     # [n_reflections] Wavevector+
        field = core.scattered_field(positions, transfers, incident)         # [n_movie_pixels, n_movie_pixels] Field+
        yield positions, reciprocal, core.energy_density(field, field)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Saves the three-case comparison figure and deformation animation to the shared plots
    directory and verifies phase invariance."""
    import matplotlib.pyplot as plt

    from examples.optics.crystal_diffraction import render
    from examples.util.animation import save_animation, save_figure

    atoms, lattice, positions, reciprocal, transfers, brightness = comparison_cases()
    figure = render.comparison_figure(positions, reciprocal, transfers, brightness)
    save_figure(figure, "crystal_diffraction")
    plt.close(figure)

    movie_transfers = core.transfer_grid(half_width, n_movie_pixels)         # [n_movie_pixels, n_movie_pixels] Wavevector+
    save_animation(
        render.diffraction_animation(deformation_frames(movie_transfers), movie_transfers),
        "crystal_diffraction",
        frame_duration_ms,
        forward_then_back=True,
    )

    # checks
    phases = reciprocal[:, :, None] & positions[:, None, :]                  # [n_cases, n_reflections, n_atoms] Scalar+
    reference_phases = lattice[:, None] & atoms[None, :]                     # [n_reflections, n_atoms] Scalar+
    assert ((phases - reference_phases).abs() < 1e-6).all()
    assert (brightness >= 0).all()


if __name__ == "__main__":
    main()
