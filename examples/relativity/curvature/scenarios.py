"""Scenes of the plane-wave curvature example: one function per figure.

Each runs the mathematics in `core` and returns the geometry its figure draws.
"""

from __future__ import annotations

import numpy as np

from examples.relativity.curvature import core
from examples.relativity.curvature.core import Curvature, Scalar, Tidal, Vector, t, x, z

# Wave packet and detector parameters (c = 1):
DURATION = 6.0          # packet length; the carrier wavelength is DURATION / CYCLES = 2
CYCLES = 3
AMPLITUDE = 1e-4        # peak strain
SAMPLES = 1201
RADIUS = 0.01           # detector radius, much smaller than the wavelength
BEADS = 24
AMPLIFICATION = 4000    # display magnification of the displacements


def curvature_map_scenario():
    """The plus wave at one event, with the rest observer, the wave direction and a transverse edge."""
    plus, _ = core.polarizations()

    # --- checks -------------------------------------------------------------
    # Nonzero, yet applying the curvature twice gives zero.
    np.testing.assert_allclose(plus(plus).kernel, 0.0, atol=1e-14)
    assert np.abs(plus.kernel).max() > 0.0
    return plus, t, t + z, x


def detector_scenario():
    """Integrate a ring of beads through the packet in the plus, cross and circular polarizations.

    Returns the samples, rest separations, displacements and accelerations, and the display
    magnification of the displacements.
    """
    plus, cross = core.polarizations()
    time = np.linspace(0.0, DURATION, SAMPLES)
    _, second = core.wave_packet(time, DURATION, CYCLES, AMPLITUDE)
    waves: Curvature = core.polarized_waves(plus, cross, second)          # [n_time, n_polarizations] Bivector <- Bivector
    response: Tidal = core.tidal_map(waves, t)                            # [n_time, n_polarizations] Vector <- Vector
    reference: Vector = core.detector_ring(BEADS) * RADIUS                # [n_beads] Vector
    acceleration: Vector = response[:, :, None](reference)                # [n_time, n_polarizations, n_beads] Vector
    displacement: Vector = core.integrate_acceleration(time, acceleration)

    # --- checks -------------------------------------------------------------
    # The integrated ring lands on the strain map applied to the rest separations.
    strain, _ = core.wave_packet(time, DURATION, CYCLES, AMPLITUDE)
    plus_strain, cross_strain = core.strain_patterns()
    predicted = core.polarized_strain(plus_strain, cross_strain, strain)[:, :, None](reference)
    np.testing.assert_allclose(displacement.kernel, predicted.kernel, atol=1e-12)
    return time, reference, displacement, acceleration, AMPLIFICATION


def doppler_scenario() -> tuple[np.ndarray, Scalar]:
    """Tidal amplitude of the unit plus wave as seen by observers boosted along the wave."""
    plus, _ = core.polarizations()
    rapidities = np.linspace(-0.7, 0.7, 15)
    observers: Vector = core.boosted_observers(rapidities, z)             # [n] Vector
    responses: Tidal = core.tidal_map(plus, observers)                    # [n] Vector <- Vector
    amplitudes: Scalar = responses.svdvals()[..., 0]                      # [n] Scalar

    # --- checks -------------------------------------------------------------
    # A chasing observer sees the frequency redshifted; the tide scales with its square.
    np.testing.assert_allclose(amplitudes.to_array(), np.exp(-2 * rapidities), atol=1e-12)
    return rapidities, amplitudes


if __name__ == "__main__":
    import argparse
    from examples.animation import save_animation, save_figure
    from examples.relativity.curvature import render

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also animate the detector rings.")
    args = parser.parse_args()
    save_figure(render.draw_curvature_map(*curvature_map_scenario()), "curvature_map")
    save_figure(render.draw_doppler(*doppler_scenario()), "curvature_doppler")
    detector = detector_scenario()
    save_figure(render.draw_detector(*detector), "curvature")
    if args.animate:
        save_animation(render.animate_detector(*detector), "curvature", 50)
