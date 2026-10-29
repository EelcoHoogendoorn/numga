"""A stretched plate viewed between crossed linear and circular polarizers."""

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.optics.photoelasticity import core

# The hole's radius and the half width of the window into the infinite plate, in mm; the pixels across the
# window, each way; and the tension pulling along x, in MPa.
hole_radius = 10.0
extent = 30.0
n_pixels = 200
far_tension = 30.0
# The plate's thickness and the light's wavelength, in mm, and the stress-optic coefficient, an illustrative
# sensitivity in inverse MPa: together the angle of the period a unit of stress delays the light by.
thickness = 1.0
wavelength = 550e-6
stress_optic = 5.5e-5
delay_per_stress = 2 * np.pi * stress_optic * thickness / wavelength
polarizer_angles = np.deg2rad([0, 22.5])
n_frames = 180
duration_ms = 40
# Fringes are bands, not shades: a few grey levels draw them.
n_colors = 12
# A smooth rise from zero to the full tension; the animation plays it forward and back.
loads = (1 - np.cos(np.linspace(0, np.pi, n_frames // 2 + 1))) / 2


# --- math -----------------------------------------------------------------------------
def stress_field() -> tuple[core.Planar, core.Stress, core.Scalar]:
    """The exterior stress and its principal difference at the pixel centres of the plate's window;
    the pixels inside the hole carry the formula's values and are covered when drawn."""
    centres = (np.arange(n_pixels) + 0.5) * (2 * extent / n_pixels) - extent
    positions = core.mv.x.dual() * centres[None, :] + core.mv.y.dual() * centres[:, None]   # [rows, columns] Planar-
    # Tension along x: the traction along x on each cut, by how far x crosses it.
    pull_direction = core.mv.x.dual()                                             # [] Planar-
    remote = far_tension * pull_direction * (core.Cut & pull_direction)           # [] Traction- <- Cut+
    stress = core.kirsch(positions, hole_radius, remote)                          # [rows, columns] Traction- <- Cut+
    principal, _ = core.principal_stresses(stress)                                # [rows, columns, modes] Scalar+, Planar-
    difference = principal[..., 1] - principal[..., 0]                            # [rows, columns] Scalar+
    return positions, stress, difference


def polariscopes(plate: core.OpticalMap) -> core.Scalar:
    """Two crossed linear polariscopes and a circular one, `[views, rows, columns]`: without stress, every
    view is dark."""
    turns = (core.mv.xy * (-polarizer_angles / 2)).exp()                         # [linear views] Rotor+
    axes = turns >> core.mv.x.dual()                                              # [linear views] Planar-
    linear = core.linear_polariscope(plate, axes[:, None, None])                  # [linear views, rows, columns] Scalar+
    circular = core.circular_polariscope(plate, core.mv.x.dual())                 # [rows, columns] Scalar+
    return concatenate((linear, circular[None]))


def loading(stress: core.Stress) -> Iterator[core.Scalar]:
    """The circular polariscope as the load rises, `[rows, columns]` per frame: the fringes alone."""
    plates = (core.retarder(stress * load, delay_per_stress) for load in loads)
    return (core.circular_polariscope(plate, core.mv.x.dual()) for plate in plates)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.photoelasticity import render

    positions, stress, difference = stress_field()
    plate = core.retarder(stress, delay_per_stress)                               # [rows, columns] Field+ <- Field+
    save_figure(render.draw_stress(positions, difference, hole_radius, far_tension), "photoelastic_stress")
    save_figure(render.draw_polariscope(positions, polariscopes(plate), hole_radius), "photoelastic_polariscopes")
    save_animation(render.animate_polariscope(positions, loading(stress), hole_radius), "photoelastic_loading",
                   duration_ms, n_colors, forward_then_back=True)


if __name__ == "__main__":
    main()
