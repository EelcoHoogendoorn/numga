"""Plane-wave curvature scenarios evaluate one spacetime event, boosted observers, and a three-polarization bead ring."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.relativity.curvature import core
from examples.relativity.curvature.core import mv, time_direction, travel, wave_vector

# Packet window and carrier cycles with unit speed of light.
wave_duration = 6.0
n_cycles = 9
wave_amplitude = 1e-4
n_samples = 1201
# Plus, cross, and circular combinations of the two quadrature carrier profiles.
polarization_weights = np.array([[1, 0], [0, 1], [1, 1]])
# Transverse detector ring much smaller than the carrier wavelength.
ring_radius = 0.01
n_beads = 24
amplification = 4000
# Boosted observers travelling along the wave direction.
max_rapidity = 0.7
n_observers = 15
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def curvature_event() -> tuple[core.Curvature, core.Antivector, core.Antivector, core.Antivector]:
    """Unit plus curvature at one event pairs with the rest observer, the null wave direction, and a unit x separation."""
    plus, _ = core.polarization_curvatures()
    return plus, time_direction, travel, mv.x.dual()


def detector_response(
    n_time_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, core.Antivector, core.Antivector, core.Antivector]:
    """Time-integrated bead displacements and tidal accelerations track the wave packet across plus, cross, and circular polarizations."""
    plus, cross = core.polarization_curvatures()
    time = np.linspace(0.0, wave_duration, n_time_samples)
    profiles, second_derivatives = core.wave_packet(time, wave_duration, n_cycles, wave_amplitude)
    # Weak-wave curvature weights the unit polarization maps by minus half the strain's second time derivative.
    curvature_weights = -0.5 * second_derivatives[:, None] * polarization_weights
    waves = curvature_weights[..., 0] * plus + curvature_weights[..., 1] * cross   # [n_time_samples, 3] Bivector+ <- Antibivector-
    response = core.tidal_map(waves, time_direction)                               # [n_time_samples, 3] Vector+ <- Antivector-
    angles = np.linspace(0, 2 * np.pi, n_beads, endpoint=False)
    bead_turns = (mv.xy * (-angles / 2)).exp()                                     # [n_beads] Rotor+
    reference = bead_turns >> (mv.x.dual() * ring_radius)                          # [n_beads] Antivector-
    # Spacetime duality converts each tidal force covector into a relative acceleration.
    acceleration = -response[:, :, None](reference).dual()                         # [n_time_samples, 3, n_beads] Antivector-
    displacement = core.integrated_displacements(time, acceleration)               # [n_time_samples, 3, n_beads] Antivector-
    return time, profiles, second_derivatives, reference, displacement, acceleration


def doppler_amplitudes(n_observer_samples: int) -> tuple[np.ndarray, core.Scalar]:
    """Boosted observers along +z measure tidal singular values that scale with the squared Doppler factor."""
    plus, _ = core.polarization_curvatures()
    rapidities = np.linspace(-max_rapidity, max_rapidity, n_observer_samples)
    observers = core.boosted_observers(rapidities, mv.z.dual())                    # [n_observer_samples] Antivector-
    accelerations = -core.tidal_map(plus, observers).dual()                        # [n_observer_samples] Antivector- <- Antivector-
    return rapidities, accelerations.svdvals()[..., 0]                             # [n_observer_samples] Scalar+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.relativity.curvature import render
    from examples.util.animation import save_animation, save_figure

    plus, observer, wave, edge = curvature_event()
    map_fig = render.curvature_map_figure(plus, observer, wave, edge)
    save_figure(map_fig, "curvature_map")
    plt.close(map_fig)

    rapidities, amplitudes = doppler_amplitudes(n_observers)
    doppler_fig = render.doppler_figure(rapidities, amplitudes)
    save_figure(doppler_fig, "curvature_doppler")
    plt.close(doppler_fig)

    time, profiles, second_derivatives, reference, displacement, acceleration = detector_response(n_samples)
    packet_fig = render.packet_figure(time, profiles, -0.5 * second_derivatives)
    save_figure(packet_fig, "curvature_packet")
    plt.close(packet_fig)

    detector_fig = render.detector_figure(time, reference, displacement, acceleration, amplification)
    save_figure(detector_fig, "curvature_detector")
    plt.close(detector_fig)

    # Middle half of the window spanning three packet widths on each side of the envelope peak.
    shown = slice(n_samples // 4, 3 * n_samples // 4 + 1)
    save_animation(
        render.detector_animation(time[shown], reference, displacement[shown], acceleration[shown], amplification),
        "curvature",
        frame_duration_ms,
    )

    # checks
    assert ((wave_vector ^ plus).svdvals() < 1e-10).all()
    quarter_turn = (mv.xy * (np.pi / 4)).exp()
    spin_two = (quarter_turn >> plus(quarter_turn << core.Antibivector)) + plus
    assert (spin_two.svdvals() < 1e-10).all()
    assert ((amplitudes - np.exp(-2 * rapidities)).abs() < 1e-10).all()
    plus_strain, cross_strain = core.strain_patterns()
    strain_weights = 0.5 * profiles[:, None] * polarization_weights
    strain_map = strain_weights[..., 0] * plus_strain + strain_weights[..., 1] * cross_strain
    predicted = strain_map[:, :, None](reference).dual()
    displacement_error = (stack((mv.t, mv.x, mv.y, mv.z)) & (displacement - predicted)[..., None]).abs()
    assert (displacement_error < 1e-7).all()


if __name__ == "__main__":
    main()
