"""Scenes for the frequency-doubling crystal: the doubled waveform, the crystal turned about the
beam, the linear map a fixed pump leaves, and the light grown along a uniform, a mismatched and a
periodically flipped crystal."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.second_harmonic import core

# The pump periods of the waveform, and its samples.
n_periods = 2
n_waveform_samples = 401
# Turns of the crystal about the beam, pump headings across it, and the duration of a frame in milliseconds.
n_turns = 72
n_headings = 241
frame_duration = 80
# The strength of the weak probes, as a share of the pump's.
probe_amplitude = 0.2
# Slices through a crystal of unit length, and the phase slip over the whole of the mismatched crystals.
n_slices = 256
slip = 4 * np.pi


# --- math -----------------------------------------------------------------------------
def waveform(n_samples: int) -> tuple[np.ndarray, core.Field, core.Polarization]:
    """The pump over its periods, and the doubled-frequency polarization it drives across the beam,
    its static half removed."""
    phase = np.linspace(0, 2 * np.pi * n_periods, n_samples)
    pump = core.horizontal_field * np.cos(phase)
    # The static half of the response equals the amplitude at twice the frequency of a unit pump.
    harmonic = (core.transverse(core.crystal(pump, pump))
                - core.second_harmonic(core.horizontal_field, core.horizontal_field))
    return phase, pump, harmonic


def turning(n_turns: int, n_headings: int) -> tuple[core.Field, Iterator[tuple[core.Direction, core.Polarization]]]:
    """Pump fields all around the beam, and per turn of the crystal about the beam its bonds'
    directions and the doubled-frequency polarization each pump drives."""
    headings = np.linspace(0, 2 * np.pi, n_headings)
    heading_turns = (core.turn_plane * (-headings / 2)).exp()
    pumps = heading_turns >> core.horizontal_field
    turns = np.linspace(0, 2 * np.pi, n_turns, endpoint=False)
    rotations = (core.turn_plane * (-turns / 2)).exp()
    # The turned crystal's response: the pumps carried into the crystal's frame and the doubled light carried
    # back. A turn about the beam keeps the doubled light across it, so the response across the beam turns as
    # the crystal's does.
    return pumps, ((rotation >> core.bond_directions,
                    rotation >> core.second_harmonic(rotation << pumps, rotation << pumps)) for rotation in rotations)


def mixing(n_headings: int) -> tuple[core.Field, core.Field, core.Polarization]:
    """Two fixed pumps, a circle of weak probe fields, and the doubled-frequency polarization that
    the linear map each pump leaves makes of the probes."""
    pumps = stack((core.horizontal_field, (core.horizontal_field + core.vertical_field) / np.sqrt(2)))
    headings = np.linspace(0, 2 * np.pi, n_headings)
    heading_turns = (core.turn_plane * (-headings / 2)).exp()
    probes = probe_amplitude * (heading_turns >> core.horizontal_field)
    # Binding a pump into one input leaves a linear map on the other.
    probe_map = core.transverse(core.crystal(pumps[:, None], core.Field))
    return pumps, probes, probe_map(probes)


def phase_matching(n_slices: int) -> tuple[np.ndarray, core.Polarization]:
    """The light grown along a matched crystal, a mismatched one, and a mismatched one flipped every
    time the slip reaches half a turn: the depth of each slice and the amplitude after it."""
    depths = (np.arange(n_slices) + 0.5) / n_slices
    # The slip over the whole crystal: none, and the mismatched crystals' slip.
    mismatch = np.array([[0], [slip], [slip]])
    # Inverting the bonds reverses the response, so a flipped slice adds with the opposite sign; the
    # flips come every half turn of slip.
    flips = (-1.0) ** np.floor(depths * slip / np.pi)
    bond_orientation = np.stack([np.ones(n_slices), np.ones(n_slices), flips])
    amplitude = core.growth(core.vertical_polarization, bond_orientation, mismatch, depths)
    return depths, amplitude


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.second_harmonic import render

    save_figure(render.draw_waveform(*waveform(n_waveform_samples)), "second_harmonic_waveform")
    pumps, frames = turning(n_turns, n_headings)
    save_animation(render.animate(pumps, frames), "second_harmonic", frame_duration)
    pump_fields, probes, generated = mixing(n_headings)
    save_figure(render.draw_mixing(pump_fields, probes, generated), "second_harmonic_mixing")
    depths, amplitude = phase_matching(n_slices)
    save_figure(render.draw_growth(depths, amplitude), "second_harmonic_phase_matching")

    # checks
    # A horizontal pump drives a vertical doubled polarization; the pump-bound map is the mixed part of
    # the doubled field of pump and probe together; the matched crystal's light grows in step, the
    # mismatched crystal's cancels after whole turns of slip, and the flipped crystal keeps the share
    # (2 / pi) ** 2 of the matched power.
    np.testing.assert_allclose(
        (core.second_harmonic(core.horizontal_field, core.horizontal_field) + core.vertical_polarization / 2).kernel,
        0, atol=1e-12)
    pump_and_probe = pump_fields[:, None] + probes
    mixed = (core.second_harmonic(pump_and_probe, pump_and_probe)
             - core.second_harmonic(pump_fields[:, None], pump_fields[:, None]) - core.second_harmonic(probes, probes))
    np.testing.assert_allclose((mixed - generated).kernel, 0, atol=1e-12)
    # The power: the squared pairings of the fields across the beam with both quadratures.
    fields = stack((core.horizontal_field, core.vertical_field))
    power = (fields & amplitude[..., None]).squared().sum(axis=(-2, -1))
    np.testing.assert_allclose(power[:, -1].to_array(), [1.0, 0.0, (2 / np.pi) ** 2], atol=1e-4)


if __name__ == "__main__":
    main()
