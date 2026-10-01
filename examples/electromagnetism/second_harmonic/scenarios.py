"""Scenes for the frequency-doubling crystal: the doubled waveform, the crystal turned about the
beam, the linear map a fixed pump leaves, and the light grown along a uniform, a mismatched and a
periodically flipped crystal."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.second_harmonic import core


# --- math -----------------------------------------------------------------------------
def waveform(samples: int) -> tuple[np.ndarray, core.Bivector, core.Bivector]:
    """Two pump periods: the pump and the doubled-frequency polarization it drives across the beam,
    its static half removed."""
    crystal = core.response(core.BONDS)
    phase = np.linspace(0, 4 * np.pi, samples)
    pump = core.HORIZONTAL * np.cos(phase)
    # The square of the cosine is half static and half at twice the frequency.
    harmonic = core.TRANSVERSE(crystal(pump, pump)) - core.doubled(crystal, core.HORIZONTAL)
    return phase, pump, harmonic


def turning(frames: int, samples: int) -> tuple[core.Bivector, Iterator[tuple[core.Bivector, core.Bivector]]]:
    """Pump polarizations all around the beam, and per turn of the crystal about the beam its bonds
    and the doubled-frequency polarization each pump drives."""
    crystal = core.response(core.BONDS)
    headings = np.linspace(0, 2 * np.pi, samples)
    turn_plane = core.HORIZONTAL.commutator(core.VERTICAL)
    pumps = (turn_plane * (-headings / 2)).exp() >> core.HORIZONTAL
    turns = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    rotations = (turn_plane * (-turns / 2)).exp()
    return pumps, ((rotation >> core.BONDS, core.doubled(core.turned(crystal, rotation), pumps)) for rotation in rotations)


def mixing(samples: int) -> tuple[core.Bivector, core.Bivector, core.Bivector]:
    """Two fixed pumps, a circle of weak probe fields, and the doubled-frequency polarization that
    the linear map each pump leaves makes of the probes."""
    crystal = core.response(core.BONDS)
    pumps = stack((core.HORIZONTAL, (core.HORIZONTAL + core.VERTICAL) / np.sqrt(2)))
    headings = np.linspace(0, 2 * np.pi, samples)
    probes = 0.2 * ((core.HORIZONTAL.commutator(core.VERTICAL) * (-headings / 2)).exp() >> core.HORIZONTAL)
    # Binding a pump into one input leaves a linear map on the other.
    probe_map = core.TRANSVERSE(crystal(pumps[:, None], core.Bivector))
    return pumps, probes, probe_map(probes)


def phase_matching(slices: int) -> tuple[np.ndarray, core.Bivector]:
    """The light grown along a matched crystal, a mismatched one, and a mismatched one flipped every
    time the slip reaches half a turn: the depth of each slice and the amplitude after it."""
    depths = (np.arange(slices) + 0.5) / slices
    # The slip over the whole crystal: none, and two full turns.
    mismatch = np.array([[0.0], [4 * np.pi], [4 * np.pi]])
    # Inverting the bonds reverses the response, so a flipped slice adds with the opposite sign; the
    # slip reaches half a turn every quarter of the crystal.
    flips = (-1.0) ** np.floor(depths * 4)
    orientation = np.stack([np.ones(slices), np.ones(slices), flips])
    amplitude = core.growth(core.VERTICAL, orientation, mismatch, depths)
    return depths, amplitude


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.electromagnetism.second_harmonic import render

    save_figure(render.draw_waveform(*waveform(401)), "second_harmonic_waveform")
    pumps, frames = turning(72, 241)
    save_animation(render.animate(pumps, frames), "second_harmonic", 80)
    save_figure(render.draw_mixing(*mixing(241)), "second_harmonic_mixing")
    save_figure(render.draw_growth(*phase_matching(256)), "second_harmonic_phase_matching")
