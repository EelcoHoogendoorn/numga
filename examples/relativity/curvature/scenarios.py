"""Curvature of a gravitational wave: the curvature map at one event, boosted observers, and a ring of beads
through a wave packet in three polarizations."""

from __future__ import annotations

import numpy as np

from examples.relativity.curvature import core
from examples.relativity.curvature.core import mv, time_direction, travel, wave_vector

# Wave packet and detector parameters, with the speed of light one.
# Packet window; the carrier wavelength is `DURATION / N_CYCLES`, two thirds.
DURATION = 6.0
N_CYCLES = 9
# Peak strain.
AMPLITUDE = 1e-4
N_SAMPLES = 1201
# Which of the plus and cross profiles each polarization carries: plus, cross, and circular.
POLARIZATIONS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
# Detector radius, much smaller than the wavelength.
RADIUS = 0.01
N_BEADS = 24
# Display magnification of the displacements.
AMPLIFICATION = 4000
# Observers boosted along the wave.
MAX_RAPIDITY = 0.7
N_OBSERVERS = 15
# The animation's frame duration in milliseconds.
FRAME_DURATION = 50


# --- math -----------------------------------------------------------------------------
def curvature_map() -> tuple[core.Curvature, core.Antivector, core.Antivector, core.Antivector]:
    """The plus wave at one event, with the observer at rest, the wave's direction and a transverse
    separation."""
    plus, _ = core.polarizations()
    return plus, time_direction, travel, mv.x.dual()


def detector() -> tuple[np.ndarray, np.ndarray, core.Antivector, core.Antivector, core.Antivector]:
    """A ring of beads through the packet in the plus, cross and circular polarizations: the samples, the
    profiles of each polarization `[times, polarizations, 2]`, the rest separations, and the displacements
    and accelerations, integrated from rest."""
    plus, cross = core.polarizations()
    time = np.linspace(0.0, DURATION, N_SAMPLES)
    profiles, second = core.wave_packet(time, DURATION, N_CYCLES, AMPLITUDE)          # [times, 2] each
    # The weak-wave curvature is minus half the strain's second derivative times the unit maps.
    waves = core.polarized(plus, cross, -0.5 * second[:, None] * POLARIZATIONS)        # [times, polarizations] Curvature
    response = core.tidal_map(waves, time_direction)                                  # [times, polarizations] Tidal
    # Each bead is x turned toward y by its angle; x and y square to minus one, which turns the sense.
    angles = np.linspace(0, 2 * np.pi, N_BEADS, endpoint=False)
    reference = (mv.xy * (angles / 2)).exp() >> mv.x.dual() * RADIUS                  # [beads] Antivector
    # The dual, the metric, turns each tidal force into an acceleration.
    acceleration = -response[:, :, None](reference).dual()                            # [times, polarizations, beads] Antivector
    displacement = core.integrate_acceleration(time, acceleration)                    # [times, polarizations, beads] Antivector
    return time, profiles[:, None] * POLARIZATIONS, reference, displacement, acceleration


def doppler() -> tuple[np.ndarray, core.Scalar]:
    """The tidal amplitude of the unit plus wave as seen by observers boosted along the wave."""
    plus, _ = core.polarizations()
    rapidities = np.linspace(-MAX_RAPIDITY, MAX_RAPIDITY, N_OBSERVERS)
    observers = core.boosted_observers(rapidities, mv.z.dual())                       # [observers] Antivector
    accelerations = -core.tidal_map(plus, observers).dual()                           # [observers] Antivector <- Antivector
    return rapidities, accelerations.svdvals()[..., 0]                                # [observers] Scalar


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.relativity.curvature import render

    plus, observer, wave, edge = curvature_map()
    save_figure(render.draw_curvature_map(plus, observer, wave, edge), "curvature_map")
    rapidities, amplitudes = doppler()
    save_figure(render.draw_doppler(rapidities, amplitudes), "curvature_doppler")
    time, profiles, reference, displacement, acceleration = detector()
    save_figure(render.draw_detector(time, reference, displacement, acceleration, AMPLIFICATION), "curvature")
    # The middle half of the window, three packet widths either side of the peak, where the ring moves.
    shown = slice(N_SAMPLES // 4, 3 * N_SAMPLES // 4 + 1)
    save_animation(render.animate_detector(time[shown], reference, displacement[shown], acceleration[shown], AMPLIFICATION),
                   "curvature", FRAME_DURATION)

    # checks
    # Type N: every plane the curvature gives contains the wave's direction. Spin two: a quarter turn
    # across the wave reverses the pattern. A chasing observer sees the wave redshifted, and the tide
    # scales with the square. The integrated ring lands on the strain map applied to the rest separations.
    area = mv.vector([0.3, -1.1, 0.7, 0.2]).dual() & mv.vector([0.9, 0.4, -0.5, 1.3]).dual()
    np.testing.assert_allclose((wave_vector ^ plus(area)).kernel, 0.0, atol=1e-12)
    quarter_turn = (mv.xy * (np.pi / 4)).exp()
    np.testing.assert_allclose(((quarter_turn >> plus(quarter_turn << core.Antibivector)) + plus).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(amplitudes.to_array(), np.exp(-2 * rapidities), atol=1e-12)
    plus_strain, cross_strain = core.strain_patterns()
    predicted = core.polarized(plus_strain, cross_strain, 0.5 * profiles)[:, :, None](reference)
    np.testing.assert_allclose((displacement - predicted).kernel, 0.0, atol=1e-12)


if __name__ == "__main__":
    main()
