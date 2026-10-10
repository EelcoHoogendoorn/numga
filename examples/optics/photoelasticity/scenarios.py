"""A stretched plate viewed between crossed linear and circular polarizers."""

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.optics.photoelasticity import core

RADIUS = 10.0                                        # mm
EXTENT = 30.0                                        # mm, window into an infinite plate
PIXELS = 200                                         # across the window, each way
TENSION = 30.0                                       # MPa, along x
THICKNESS = 1.0                                      # mm
WAVELENGTH = 550e-6                                  # mm
STRESS_OPTIC = 5.5e-5                                # inverse MPa, illustrative sensitivity
STRESS_PHASE = 2 * np.pi * STRESS_OPTIC * THICKNESS / WAVELENGTH
POLARIZER_ANGLES = np.deg2rad([0, 22.5])
FRAMES = 80
DURATION_MS = 90
# Fringes are bands, not shades: a few grey levels draw them.
COLORS = 16
# A smooth cycle from zero to the full tension and back, without duplicating the endpoint.
LOADS = (1 - np.cos(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False))) / 2


# --- math -----------------------------------------------------------------------------
def stress_field() -> tuple[core.Planar, core.Stress, core.Scalar]:
    """The exterior stress and its principal difference at the pixel centres of the plate's window;
    the pixels inside the hole carry the formula's values and are covered when drawn."""
    centres = (np.arange(PIXELS) + 0.5) * (2 * EXTENT / PIXELS) - EXTENT
    positions = core.mv.x * centres[None, :] + core.mv.y * centres[:, None]      # [rows, columns] Planar
    # Averaging identity and reflection in x keeps axial traction and cancels transverse traction.
    remote = TENSION * (core.Planar + (core.mv.x >> core.Planar)) / 2             # [] Planar <- Planar
    stress = core.kirsch(positions, RADIUS, remote)                               # [rows, columns] Planar <- Planar
    principal, _ = stress.eigh()                                                  # [rows, columns, modes] Scalar, Planar
    difference = principal[..., 1] - principal[..., 0]                            # [rows, columns] Scalar
    return positions, stress, difference


def polariscopes(turn: core.Bivector) -> core.Scalar:
    """Two crossed linear polariscopes and a circular one, `[views, rows, columns]`: the light enters
    in one state and leaves through its opposite."""
    # Linear states lie on the sphere's equator, at twice their angle in the plate.
    linear = core.polarization((core.mv.xy * (-POLARIZER_ANGLES / 2)).exp() >> core.mv.x)   # [linear views] Polarization
    # Circular light sits at the sphere's pole.
    incident = concatenate((linear, core.mv.z[None]))                              # [views] Polarization
    # Each analyser passes the opposite state: without stress, every view is dark.
    return core.polariscope(turn, incident[:, None, None], -incident[:, None, None])


def loading(turn: core.Bivector) -> Iterator[core.Scalar]:
    """The circular polariscope through the load cycle, `[rows, columns]` per frame: the fringes alone."""
    return (core.polariscope(turn * load, core.mv.z, -core.mv.z) for load in LOADS)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.optics.photoelasticity import render

    positions, stress, difference = stress_field()
    turn = core.half_turn(stress, STRESS_PHASE)                                   # [rows, columns] Bivector
    save_figure(render.draw_stress(positions, difference, RADIUS, TENSION), "photoelastic_stress")
    save_figure(render.draw_polariscope(positions, polariscopes(turn), RADIUS), "photoelastic_polariscopes")
    save_animation(render.animate_polariscope(positions, loading(turn), RADIUS), "photoelastic_loading", DURATION_MS, COLORS)


if __name__ == "__main__":
    main()
