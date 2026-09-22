"""Scenarios and canonical figures for the plane-wave curvature example.

Executes the mathematics in `core` and hands the geometry to `render`. Run from rewrite/:
    PYTHONPATH=src:. python -m examples.relativity.curvature.scenarios [--animate]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from examples import PLOT_DIR
from examples.relativity.curvature import core, render
from examples.relativity.curvature.core import Curvature, Scalar, Tidal, Vector, t, x, z

# Wave packet and detector parameters (c = 1):
DURATION = 6.0          # packet length; the carrier wavelength is DURATION / CYCLES = 2
CYCLES = 3
AMPLITUDE = 1e-4        # peak strain
SAMPLES = 1201
RADIUS = 0.01           # detector radius, much smaller than the wavelength
BEADS = 24
AMPLIFICATION = 4000    # display magnification of the displacements


def detector_scenario() -> tuple[np.ndarray, Vector, Vector, Vector]:
    """Integrate a ring of beads through the packet in the plus, cross and circular polarizations."""
    plus, cross = core.polarizations()
    time = np.linspace(0.0, DURATION, SAMPLES)
    _, second = core.wave_packet(time, DURATION, CYCLES, AMPLITUDE)
    waves: Curvature = core.polarized_waves(plus, cross, second)          # [n_time, 3] Bivector <- Bivector
    response: Tidal = core.tidal_map(waves, t)                            # [n_time, 3] Vector <- Vector
    reference: Vector = core.detector_ring(BEADS) * RADIUS                # [n_beads] Vector
    acceleration: Vector = response[:, :, None](reference)                # [n_time, 3, n_beads] Vector
    displacement: Vector = core.integrate_acceleration(time, acceleration)
    return time, reference, displacement, acceleration


def doppler_scenario() -> tuple[np.ndarray, Scalar]:
    """Tidal amplitude of the unit plus wave as seen by observers boosted along the wave."""
    plus, _ = core.polarizations()
    rapidities = np.linspace(-0.7, 0.7, 15)
    observers: Vector = core.boosted_observers(rapidities, z)             # [n] Vector
    responses: Tidal = core.tidal_map(plus, observers)                    # [n] Vector <- Vector
    return rapidities, responses.svdvals()[..., 0]                        # [n] Scalar


def main(plot_path: str = str(PLOT_DIR / "curvature.png"), animation_path: str = "") -> plt.Figure:
    """Draw the curvature map, the Doppler check and the detector; optionally save the animation."""
    plot_path = Path(plot_path)
    plus, _ = core.polarizations()
    render.draw_curvature_map(
        plus, observer=t, wave=t + z, edge=x,
        plot_path=plot_path.with_name(f"{plot_path.stem}_map{plot_path.suffix}"),
    )
    rapidities, amplitudes = doppler_scenario()
    render.draw_doppler(
        rapidities, amplitudes,
        plot_path=plot_path.with_name(f"{plot_path.stem}_doppler{plot_path.suffix}"),
    )
    time, reference, displacement, acceleration = detector_scenario()
    figure = render.draw_detector(time, reference, displacement, acceleration, AMPLIFICATION, plot_path)
    print(f"Figure saved to {plot_path}")
    if animation_path:
        render.save_animation(time, reference, displacement, acceleration, AMPLIFICATION, animation_path)
        print(f"Animation saved to {animation_path}")

    # --- checks -------------------------------------------------------------
    np.testing.assert_allclose(plus(plus).kernel, 0.0, atol=1e-14)
    assert np.abs(plus.kernel).max() > 0.0
    np.testing.assert_allclose(amplitudes.kernel[..., 0], np.exp(-2 * rapidities), atol=1e-12)
    return figure


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animate", action="store_true", help="Also save plots/curvature.gif.")
    args = parser.parse_args()
    main(animation_path=str(PLOT_DIR / "curvature.gif") if args.animate else "")
    plt.show()
