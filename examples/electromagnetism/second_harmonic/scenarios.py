"""Scenes for the frequency-doubling crystal: the doubled waveform, the crystal turned about the
beam, the linear map a fixed pump leaves, and the light grown along a uniform, a mismatched and a
periodically flipped crystal."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.second_harmonic import core

# The pump periods of the waveform, and its samples.
N_PERIODS = 2
N_WAVEFORM_SAMPLES = 401
# Turns of the crystal about the beam, pump headings across it, and the duration of a frame in milliseconds.
N_TURNS = 72
N_HEADINGS = 241
FRAME_DURATION = 80
# The strength of the weak probes, as a share of the pump's.
PROBE_AMPLITUDE = 0.2
# Slices through a crystal of unit length, and the phase slip over the whole of the mismatched crystals.
N_SLICES = 256
SLIP = 4 * np.pi


# --- math -----------------------------------------------------------------------------
def waveform(samples: int) -> tuple[np.ndarray, core.Field, core.Polarization]:
    """The pump over its periods, and the doubled-frequency polarization it drives across the beam,
    its static half removed."""
    crystal = core.bond_response(core.bonds)
    phase = np.linspace(0, 2 * np.pi * N_PERIODS, samples)
    pump = core.horizontal_field * np.cos(phase)
    # The square of the cosine is half static and half at twice the frequency.
    harmonic = core.transverse(crystal(pump, pump)) - core.second_harmonic(crystal, core.horizontal_field)
    return phase, pump, harmonic


def turning(frames: int, samples: int) -> tuple[core.Field, Iterator[tuple[core.Direction, core.Polarization]]]:
    """Pump fields all around the beam, and per turn of the crystal about the beam its bonds'
    directions and the doubled-frequency polarization each pump drives."""
    crystal = core.bond_response(core.bonds)
    headings = np.linspace(0, 2 * np.pi, samples)
    turn_plane = core.horizontal_field.commutator(core.vertical_field)
    pumps = (turn_plane * (-headings / 2)).exp() >> core.horizontal_field
    turns = np.linspace(0, 2 * np.pi, frames, endpoint=False)
    rotations = (turn_plane * (-turns / 2)).exp()
    return pumps, ((rotation >> core.bond_directions, core.second_harmonic(core.turned_response(crystal, rotation), pumps)) for rotation in rotations)


def mixing(samples: int) -> tuple[core.Field, core.Field, core.Polarization]:
    """Two fixed pumps, a circle of weak probe fields, and the doubled-frequency polarization that
    the linear map each pump leaves makes of the probes."""
    crystal = core.bond_response(core.bonds)
    pumps = stack((core.horizontal_field, (core.horizontal_field + core.vertical_field) / np.sqrt(2)))
    headings = np.linspace(0, 2 * np.pi, samples)
    probes = PROBE_AMPLITUDE * ((core.horizontal_field.commutator(core.vertical_field) * (-headings / 2)).exp() >> core.horizontal_field)
    # Binding a pump into one input leaves a linear map on the other.
    probe_map = core.transverse(crystal(pumps[:, None], core.Field))
    return pumps, probes, probe_map(probes)


def phase_matching(n_slices: int) -> tuple[np.ndarray, core.Polarization]:
    """The light grown along a matched crystal, a mismatched one, and a mismatched one flipped every
    time the slip reaches half a turn: the depth of each slice and the amplitude after it."""
    depths = (np.arange(n_slices) + 0.5) / n_slices
    # The slip over the whole crystal: none, and the mismatched crystals' slip.
    mismatch = np.array([[0.0], [SLIP], [SLIP]])
    # Inverting the bonds reverses the response, so a flipped slice adds with the opposite sign; the
    # flips come every half turn of slip.
    flips = (-1.0) ** np.floor(depths * SLIP / np.pi)
    bond_orientation = np.stack([np.ones(n_slices), np.ones(n_slices), flips])
    amplitude = core.growth(core.vertical_polarization, bond_orientation, mismatch, depths)
    return depths, amplitude


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.second_harmonic import render

    save_figure(render.draw_waveform(*waveform(N_WAVEFORM_SAMPLES)), "second_harmonic_waveform")
    pumps, frames = turning(N_TURNS, N_HEADINGS)
    save_animation(render.animate(pumps, frames), "second_harmonic", FRAME_DURATION)
    pump_fields, probes, generated = mixing(N_HEADINGS)
    save_figure(render.draw_mixing(pump_fields, probes, generated), "second_harmonic_mixing")
    depths, amplitude = phase_matching(N_SLICES)
    save_figure(render.draw_growth(depths, amplitude), "second_harmonic_phase_matching")

    # checks
    # A horizontal pump drives a vertical doubled polarization; the pump-bound map is the mixed part of
    # the doubled field of pump and probe together; the matched crystal's light grows in step, the
    # mismatched crystal's cancels after whole turns of slip, and the flipped crystal keeps the share
    # (2 / pi) ** 2 of the matched power.
    crystal = core.bond_response(core.bonds)
    np.testing.assert_allclose((core.second_harmonic(crystal, core.horizontal_field) + core.vertical_polarization / 2).kernel,
                               0, atol=1e-12)
    mixed = (core.second_harmonic(crystal, pump_fields[:, None] + probes)
             - core.second_harmonic(crystal, pump_fields[:, None]) - core.second_harmonic(crystal, probes))
    np.testing.assert_allclose((mixed - generated).kernel, 0, atol=1e-12)
    # The power: the squared pairings of the fields across the beam with both quadratures.
    fields = stack((core.horizontal_field, core.vertical_field))
    power = (fields & amplitude[..., None]).squared().sum(axis=(-2, -1))
    np.testing.assert_allclose(power[:, -1].to_array(), [1.0, 0.0, (2 / np.pi) ** 2], atol=1e-4)


if __name__ == "__main__":
    main()
