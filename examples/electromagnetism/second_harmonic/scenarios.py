"""Scenes for the frequency-doubling crystal: the doubled waveform, the crystal turned about the
beam, the linear map a fixed pump leaves, and the light grown along a uniform, a mismatched and a
periodically flipped crystal."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.second_harmonic import core


# --- math -----------------------------------------------------------------------------
def waveform(samples: int) -> tuple[np.ndarray, core.Vector, core.Vector]:
    """Two pump periods: the pump and the doubled-frequency polarization it drives across the beam,
    its static half removed."""
    crystal = core.response(core.BONDS)                                        # [] Vector <- (Vector, Vector)
    phase = np.linspace(0, 4 * np.pi, samples)
    pump = core.HORIZONTAL * np.cos(phase)                                     # [times] Vector
    # The square of the cosine is half static and half at twice the frequency.
    harmonic = core.TRANSVERSE(crystal(pump, pump)) - core.doubled(crystal, core.HORIZONTAL)   # [times] Vector
    return phase, pump, harmonic


def turning(frames: int, samples: int) -> tuple[core.Vector, Iterator[tuple[core.Vector, core.Vector]]]:
    """Pump polarizations all around the beam, and per turn of the crystal about the beam its bonds
    and the doubled-frequency polarization each pump drives."""
    crystal = core.response(core.BONDS)                                        # [] Vector <- (Vector, Vector)
    headings = np.linspace(0, 2 * np.pi, samples)
    pumps = core.HORIZONTAL * np.cos(headings) + core.VERTICAL * np.sin(headings)   # [pumps] Vector
    turns = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    rotations = ((core.HORIZONTAL ^ core.VERTICAL) * (-turns / 2)).exp()       # [frames] Rotor

    # --- checks
    # Turning the response as an extensor is the same as building it from turned bonds.
    rebuilt = core.response(rotations[:, None] >> core.BONDS)                  # [frames] Vector <- (Vector, Vector)
    turned = core.turned(crystal, rotations)                                   # [frames] Vector <- (Vector, Vector)
    np.testing.assert_allclose((turned(pumps[:, None], pumps[:, None]) - rebuilt(pumps[:, None], pumps[:, None])).kernel, 0.0, atol=1e-12)
    return pumps, ((rotation >> core.BONDS, core.doubled(core.turned(crystal, rotation), pumps)) for rotation in rotations)


def mixing(samples: int) -> tuple[core.Vector, core.Vector, core.Vector]:
    """Two fixed pumps, a circle of weak probe fields, and the doubled-frequency polarization that
    the linear map each pump leaves makes of the probes."""
    crystal = core.response(core.BONDS)                                        # [] Vector <- (Vector, Vector)
    pumps = stack((core.HORIZONTAL, (core.HORIZONTAL + core.VERTICAL) / np.sqrt(2)))   # [cases] Vector
    headings = np.linspace(0, 2 * np.pi, samples)
    probes = 0.2 * (core.HORIZONTAL * np.cos(headings) + core.VERTICAL * np.sin(headings))   # [probes] Vector
    # Binding a pump into one input leaves a linear map on the other.
    probe_map = core.TRANSVERSE(crystal(pumps[:, None], core.Vector))          # [cases, 1] Vector <- Vector

    # --- checks
    # The map is the change a probe makes to the doubled-frequency polarization, less the probe's own:
    # the two cross terms of the quadratic cancel the half.
    change = core.doubled(crystal, pumps[:, None] + probes) - core.doubled(crystal, pumps[:, None]) - core.doubled(crystal, probes)
    np.testing.assert_allclose((change - probe_map(probes)).kernel, 0.0, atol=1e-12)
    return pumps, probes, probe_map(probes)


def phase_matching(slices: int) -> tuple[np.ndarray, core.Phasor]:
    """The light grown along a matched crystal, a mismatched one, and a mismatched one flipped every
    time the slip reaches half a turn: the depth of each slice and the amplitude after it."""
    depths = (np.arange(slices) + 0.5) / slices
    # The slip over the whole crystal: none, and two full turns.
    mismatch = np.array([[0.0], [4 * np.pi], [4 * np.pi]])
    # Inverting the bonds reverses the response, so a flipped slice adds with the opposite sign; the
    # slip reaches half a turn every quarter of the crystal.
    flips = (-1.0) ** np.floor(depths * 4)
    orientation = np.stack([np.ones(slices), np.ones(slices), flips])
    amplitude = core.growth(orientation, mismatch, depths)                     # [cases, slices] Phasor

    # --- checks
    # Flipped bonds drive the opposite polarization. Matched, the amplitude grows with the depth;
    # mismatched by two turns it cancels; flipped every half turn it reaches two over pi of matched.
    crystal, flipped = core.response(core.BONDS), core.response(-core.BONDS)
    np.testing.assert_allclose((core.doubled(flipped, core.HORIZONTAL) + core.doubled(crystal, core.HORIZONTAL)).kernel, 0.0, atol=1e-15)
    power = amplitude.symmetric_reverse_product().to_array()                   # [cases, slices]
    np.testing.assert_allclose(power[:, -1], [1.0, 0.0, (2 / np.pi) ** 2], atol=1e-4)
    return depths, amplitude


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.second_harmonic import render

    save_figure(render.draw_waveform(*waveform(401)), "second_harmonic_waveform")
    pumps, frames = turning(72, 241)
    save_animation(render.animate(pumps, frames), "second_harmonic", 80)
    save_figure(render.draw_mixing(*mixing(241)), "second_harmonic_mixing")
    save_figure(render.draw_growth(*phase_matching(256)), "second_harmonic_phase_matching")
