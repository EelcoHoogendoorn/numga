"""Uniaxial tension around a circular hole produces stress-concentration contours and polariscope fringes."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.optics.photoelasticity import core

# Hole radius and half-width of the square viewing window in mm, pixel count per side, and remote tension in MPa.
hole_radius = 10.0
extent = 30.0
n_pixels = 200
far_tension = 30.0
# Plate thickness and wavelength in mm, and stress-optic coefficient in inverse MPa.
thickness = 1.0
wavelength = 550e-6
stress_optic = 5.5e-5
delay_per_stress = 2 * np.pi * stress_optic * thickness / wavelength
polarizer_angles = np.deg2rad([0, 22.5])
n_frames = 180
n_load_steps = n_frames // 2 + 1
duration_ms = 40
# Banded greyscale fringes compress cleanly into a small shared palette.
n_colors = 12
# Smooth half-cycle load profile played forward and back over one loading cycle.
loads = (1 - np.cos(np.linspace(0, np.pi, n_load_steps))) / 2


# --- math -----------------------------------------------------------------------------
def stress_field() -> tuple[core.Planar, core.Stress, core.Scalar, core.Planar, core.Scalar]:
    """Kirsch's exterior solution on the `[n_pixels, n_pixels]` pixel grid gives the positions, stress map, principal
    stresses and directions, and principal stress difference."""
    centres = (np.arange(n_pixels) + 0.5) * (2 * extent / n_pixels) - extent
    positions = core.mv.x.dual() * centres[None, :] + core.mv.y.dual() * centres[:, None]   # [n_pixels, n_pixels] Planar-
    pull_direction = core.mv.x.dual()                                             # [] Planar-
    remote = far_tension * pull_direction * (core.Cut & pull_direction)           # [] Traction- <- Cut+
    stress = core.kirsch_stress(positions, hole_radius, remote)                   # [n_pixels, n_pixels] Traction- <- Cut+
    principal, directions = core.principal_stresses(stress)                       # [n_pixels, n_pixels, 2] Scalar+, Planar-
    difference = principal[..., 1] - principal[..., 0]                            # [n_pixels, n_pixels] Scalar+
    return positions, stress, principal, directions, difference


def polariscopes(plate: core.OpticalMap) -> core.Scalar:
    """Two crossed linear polariscopes at `polarizer_angles` and one circular polariscope produce
    `[3, n_pixels, n_pixels] Scalar+` transmitted intensities."""
    rotors = (core.mv.xy * (-polarizer_angles / 2)).exp()                         # [2] Rotor+
    axes = rotors >> core.mv.x.dual()                                             # [2] Planar-
    linear = core.linear_polariscope(plate, axes[:, None, None])                  # [2, n_pixels, n_pixels] Scalar+
    circular = core.circular_polariscope(plate, core.mv.x.dual())                 # [n_pixels, n_pixels] Scalar+
    return concatenate((linear, circular[None]))                                  # [3, n_pixels, n_pixels] Scalar+


def loading_frames(principal: core.Scalar, directions: core.Planar) -> Iterator[core.Scalar]:
    """Yields `[n_pixels, n_pixels] Scalar+` circular-polariscope intensities as the remote tension rises across
    `loads`."""
    principal_polarizers = core.polarizer(directions, directions)                 # [n_pixels, n_pixels, 2] Field+ <- Field+
    loaded_plates = (((principal * (1j * delay_per_stress * load)).exp() * principal_polarizers).sum(axis=-1)
                     for load in loads)
    return (core.circular_polariscope(plate, core.mv.x.dual()) for plate in loaded_plates)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    import matplotlib.pyplot as plt
    from examples.util.animation import save_animation, save_figure
    from examples.optics.photoelasticity import render

    positions, stress, principal, directions, difference = stress_field()
    plate = core.retarder(stress, delay_per_stress)                               # [n_pixels, n_pixels] Field+ <- Field+
    views = polariscopes(plate)                                                   # [3, n_pixels, n_pixels] Scalar+

    figure = render.stress_figure(positions, difference, hole_radius, far_tension)
    save_figure(figure, "photoelastic_stress")
    plt.close(figure)

    figure = render.polariscope_figure(positions, views, hole_radius)
    save_figure(figure, "photoelastic_polariscopes")
    plt.close(figure)

    save_animation(render.polariscope_animation(positions, loading_frames(principal, directions), hole_radius),
                   "photoelastic_loading", duration_ms, n_colors, forward_then_back=True)

    # checks
    outside = positions.antinorm().real() >= hole_radius
    symmetry_residual = ((core.mv.x & stress(core.mv.y)) - (core.mv.y & stress(core.mv.x))).real()  # [n_pixels, n_pixels] Scalar+
    assert (symmetry_residual[outside].abs() < 1e-10).all()

    expected_circular = (difference.real() * (delay_per_stress / 2)).sin().squared()  # [n_pixels, n_pixels] Scalar+
    assert ((views[2].real()[outside] - expected_circular[outside]).abs() < 1e-10).all()
    assert (views.real()[:, outside] >= -1e-10).all() and (views.real()[:, outside] <= 1 + 1e-10).all()
    assert (views.real()[:2, outside] <= views[2].real()[outside] + 1e-10).all()


if __name__ == "__main__":
    main()
