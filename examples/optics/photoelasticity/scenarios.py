"""A stretched plate viewed between crossed linear and circular polarizers."""

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.optics.photoelasticity import core

RADIUS = 10.0                                        # mm
EXTENT = 30.0                                        # mm, window into an infinite plate
N_PIXELS = 200                                         # across the window, each way
TENSION = 30.0                                       # MPa, along x
THICKNESS = 1.0                                      # mm
WAVELENGTH = 550e-6                                  # mm
STRESS_OPTIC = 5.5e-5                                # inverse MPa, illustrative sensitivity
STRESS_PHASE = 2 * np.pi * STRESS_OPTIC * THICKNESS / WAVELENGTH
POLARIZER_ANGLES = np.deg2rad([0, 22.5])
N_FRAMES = 80
DURATION_MS = 90
# Fringes are bands, not shades: a few grey levels draw them.
N_COLORS = 16
# A smooth cycle from zero to the full tension and back, without duplicating the endpoint.
LOADS = (1 - np.cos(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False))) / 2


# --- math -----------------------------------------------------------------------------
def stress_field() -> tuple[core.Planar, core.Stress, core.Scalar]:
    """The exterior stress and its principal difference at the pixel centres of the plate's window;
    the pixels inside the hole carry the formula's values and are covered when drawn."""
    centres = (np.arange(N_PIXELS) + 0.5) * (2 * EXTENT / N_PIXELS) - EXTENT
    positions = core.mv.x.dual() * centres[None, :] + core.mv.y.dual() * centres[:, None]   # [rows, columns] Planar
    # Averaging identity and reflection in y keeps axial traction and cancels transverse traction.
    remote = TENSION * (core.Planar + (core.mv.y >> core.Planar)) / 2             # [] Planar <- Planar
    stress = core.kirsch(positions, RADIUS, remote)                               # [rows, columns] Planar <- Planar
    principal, _ = stress.eigh()                                                  # [rows, columns, modes] Scalar, Planar
    difference = principal[..., 1] - principal[..., 0]                            # [rows, columns] Scalar
    return positions, stress, difference


def polariscopes(plate: core.OpticalMap) -> core.Scalar:
    """Two crossed linear polariscopes and a circular one, `[views, rows, columns]`: without stress, every
    view is dark."""
    axes = (core.mv.xy * (-POLARIZER_ANGLES / 2)).exp() >> core.mv.x.dual()      # [linear views] Planar
    linear = core.linear_polariscope(plate, axes[:, None, None])                   # [linear views, rows, columns] Scalar
    circular = core.circular_polariscope(plate, core.mv.x.dual())                  # [rows, columns] Scalar
    return concatenate((linear, circular[None]))


def loading(stress: core.Stress) -> Iterator[core.Scalar]:
    """The circular polariscope through the load cycle, `[rows, columns]` per frame: the fringes alone."""
    return (core.circular_polariscope(core.retarder(stress * load, STRESS_PHASE), core.mv.x.dual()) for load in LOADS)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.optics.photoelasticity import render

    positions, stress, difference = stress_field()
    plate = core.retarder(stress, STRESS_PHASE)                                    # [rows, columns] Field <- Field
    save_figure(render.draw_stress(positions, difference, RADIUS, TENSION), "photoelastic_stress")
    save_figure(render.draw_polariscope(positions, polariscopes(plate), RADIUS), "photoelastic_polariscopes")
    save_animation(render.animate_polariscope(positions, loading(stress), RADIUS), "photoelastic_loading", DURATION_MS, N_COLORS)


if __name__ == "__main__":
    main()
