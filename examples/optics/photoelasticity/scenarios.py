"""A stretched plate viewed between crossed linear and circular polarizers."""

from collections.abc import Iterator

import numpy as np

from numga import concatenate
from examples.optics.photoelasticity import core

RADIUS = 10.0                                        # mm
EXTENT = 30.0                                        # mm, window into an infinite plate
RADIAL_SAMPLES = 200
ANGULAR_SAMPLES = 512
TENSION = 30.0                                       # MPa, along x
THICKNESS = 1.0                                      # mm
WAVELENGTH = 550e-6                                  # mm
STRESS_OPTIC = 5.5e-5                                # inverse MPa, illustrative sensitivity
POLARIZER_ANGLES = np.deg2rad([0, 22.5])
FRAMES = 80
DURATION_MS = 90
# Fringes are bands, not shades: a few grey levels draw them.
COLORS = 16
# A smooth cycle from zero to the full tension and back, without duplicating the endpoint.
LOADS = (1 - np.cos(np.linspace(0, 2 * np.pi, FRAMES, endpoint=False))) / 2


# --- math -----------------------------------------------------------------------------
def stress_field() -> tuple[core.Planar, core.Stress, core.Scalar]:
    """The exterior stress and its principal difference on the plate's sampling window."""
    positions = core.square_grid(RADIUS, EXTENT, RADIAL_SAMPLES, ANGULAR_SAMPLES)  # [radii, angles + 1] Planar
    # Averaging identity and reflection in x keeps axial traction and cancels transverse traction.
    remote = TENSION * (core.Planar + (core.mv.x >> core.Planar)) / 2             # [] Planar <- Planar
    stress = core.kirsch(positions, RADIUS, remote)                               # [radii, angles + 1] Planar <- Planar
    principal, _ = stress.eigh()                                                  # [radii, angles + 1, modes] Scalar, Planar
    difference = principal[..., 1] - principal[..., 0]                            # [radii, angles + 1] Scalar
    return positions, stress, difference


def polariscopes(stress: core.Stress, loads: np.ndarray) -> Iterator[core.Scalar]:
    """Two crossed linear polariscopes and a circular one under each load, yielding
    [views, radii, angles + 1]: the light enters in one state and leaves through its opposite."""
    stress_phase = 2 * np.pi * STRESS_OPTIC * THICKNESS / WAVELENGTH
    # Linear states lie on the sphere's equator, at twice their angle in the plate.
    linear = core.polarization((core.mv.xy * (-POLARIZER_ANGLES / 2)).exp() >> core.mv.x)   # [linear views] Polarization
    # Circular light sits at the sphere's pole.
    incident = concatenate((linear, core.mv.z[None]))                              # [views] Polarization
    # Each analyser passes the opposite state: without stress, every view is dark.
    yield from core.loading(stress, stress_phase, incident, -incident, loads)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.optics.photoelasticity import render

    positions, stress, difference = stress_field()
    intensities = next(polariscopes(stress, np.ones(1)))                          # [views, radii, angles + 1] Scalar
    save_figure(render.draw_stress(positions, difference, RADIUS, TENSION), "photoelastic_stress")
    save_figure(render.draw_polariscope(positions, intensities, RADIUS), "photoelastic_polariscopes")
    # The circular polariscope, the last view, shows the fringes alone as the load grows.
    save_animation(render.animate_polariscope(positions, (frame[-1] for frame in polariscopes(stress, LOADS)), RADIUS),
                   "photoelastic_loading", DURATION_MS, COLORS)


if __name__ == "__main__":
    main()
