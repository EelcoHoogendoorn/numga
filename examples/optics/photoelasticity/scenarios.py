"""A stretched plate viewed between crossed linear and circular polarizers."""

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.optics.photoelasticity import core

hole_radius = 10.0                                        # mm
extent = 30.0                                        # mm, window into an infinite plate
n_pixels = 200                                         # across the window, each way
far_tension = 30.0                                       # MPa, along x
thickness = 1.0                                      # mm
wavelength = 550e-6                                  # mm
stress_optic = 5.5e-5                                # inverse MPa, illustrative sensitivity
stress_phase = 2 * np.pi * stress_optic * thickness / wavelength
polarizer_angles = np.deg2rad([0, 22.5])
n_frames = 80
duration_ms = 90
# Fringes are bands, not shades: a few grey levels draw them.
n_colors = 16
# A smooth cycle from zero to the full tension and back, without duplicating the endpoint.
loads = (1 - np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))) / 2


# --- math -----------------------------------------------------------------------------
def stress_field() -> tuple[core.Planar, core.Stress, core.Scalar]:
    """The exterior stress and its principal difference at the pixel centres of the plate's window;
    the pixels inside the hole carry the formula's values and are covered when drawn."""
    centres = (np.arange(n_pixels) + 0.5) * (2 * extent / n_pixels) - extent
    positions = core.mv.x.dual() * centres[None, :] + core.mv.y.dual() * centres[:, None]   # [rows, columns] Planar
    # Tension along x: the traction along x on each cut, by how far x crosses it.
    along = core.mv.x.dual()                                                         # [] Planar
    remote = far_tension * along * (core.Cut & along)                                # [] Traction <- Cut
    stress = core.kirsch(positions, hole_radius, remote)                               # [rows, columns] Traction <- Cut
    principal, _ = core.principal_stresses(stress)                                # [rows, columns, modes] Scalar, Planar
    difference = principal[..., 1] - principal[..., 0]                            # [rows, columns] Scalar
    return positions, stress, difference


def polariscopes(plate: core.OpticalMap) -> core.Scalar:
    """Two crossed linear polariscopes and a circular one, `[views, rows, columns]`: without stress, every
    view is dark."""
    axes = (core.mv.xy * (-polarizer_angles / 2)).exp() >> core.mv.x.dual()      # [linear views] Planar
    linear = core.linear_polariscope(plate, axes[:, None, None])                   # [linear views, rows, columns] Scalar
    circular = core.circular_polariscope(plate, core.mv.x.dual())                  # [rows, columns] Scalar
    return concatenate((linear, circular[None]))


def loading(stress: core.Stress) -> Iterator[core.Scalar]:
    """The circular polariscope through the load cycle, `[rows, columns]` per frame: the fringes alone."""
    return (core.circular_polariscope(core.retarder(stress * load, stress_phase), core.mv.x.dual()) for load in loads)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.optics.photoelasticity import render

    positions, stress, difference = stress_field()
    plate = core.retarder(stress, stress_phase)                                    # [rows, columns] Field <- Field
    save_figure(render.draw_stress(positions, difference, hole_radius, far_tension), "photoelastic_stress")
    save_figure(render.draw_polariscope(positions, polariscopes(plate), hole_radius), "photoelastic_polariscopes")
    save_animation(render.animate_polariscope(positions, loading(stress), hole_radius), "photoelastic_loading", duration_ms, n_colors)


if __name__ == "__main__":
    main()
