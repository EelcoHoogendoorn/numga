"""Scenarios for second-harmonic generation cover the doubled-frequency waveform, crystal rotation
about the beam, linear mixing of weak probes on a fixed pump, and coherent growth across matched,
mismatched, and periodically poled crystals."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.second_harmonic import core

# Waveform period count and sample resolution.
n_periods = 2
n_waveform_samples = 401
# Crystal rotation steps, pump heading angles across the beam, and frame duration in milliseconds.
n_turns = 144
n_headings = 241
frame_duration_ms = 40
# Weak probe amplitude relative to the pump.
probe_amplitude = 0.2
# Slice count across a unit-length crystal and total phase slip across mismatched crystals.
n_slices = 256
slip = 4 * np.pi


# --- math -----------------------------------------------------------------------------
def waveform(n_samples: int) -> tuple[np.ndarray, core.Field, core.Polarization]:
    """Samples `n_samples` phases over `n_periods` pump cycles, returning the phase array, the
    oscillating pump field (`[n_samples] Field+`), and the doubled-frequency polarization
    (`[n_samples] Polarization-`) with its static offset subtracted."""
    phase = np.linspace(0, 2 * np.pi * n_periods, n_samples)
    pump = core.horizontal_field * np.cos(phase)                               # [n_samples] Field+
    # Subtracting the static half leaves the pure doubled-frequency oscillation.
    harmonic = (core.transverse(core.crystal(pump, pump))
                - core.second_harmonic(core.horizontal_field, core.horizontal_field))  # [n_samples] Polarization-
    return phase, pump, harmonic


def turning(n_turns: int, n_headings: int) -> tuple[core.Field, Iterator[tuple[core.Direction, core.Polarization]]]:
    """Forms `n_headings` pump fields (`[n_headings] Field+`) around the beam and yields, across
    `n_turns` crystal rotations, the rotated bond directions (`[4] Direction-`) and doubled-frequency
    polarizations (`[n_headings] Polarization-`)."""
    headings = np.linspace(0, 2 * np.pi, n_headings)
    heading_turns = (core.turn_plane * (-headings / 2)).exp()                  # [n_headings] Rotor+
    pumps = heading_turns >> core.horizontal_field                             # [n_headings] Field+
    turns = np.linspace(0, 2 * np.pi, n_turns, endpoint=False)
    rotations = (core.turn_plane * (-turns / 2)).exp()                         # [n_turns] Rotor+

    def states() -> Iterator[tuple[core.Direction, core.Polarization]]:
        # Carrying the pumps into the crystal frame and rotating the resulting transverse polarization back
        # evaluates the turned crystal's response across the beam.
        for rotation in rotations:
            back_pumps = rotation << pumps                                     # [n_headings] Field+
            yield rotation >> core.bond_directions, rotation >> core.second_harmonic(back_pumps, back_pumps)

    return pumps, states()


def mixing(n_headings: int) -> tuple[core.Field, core.Field, core.Polarization]:
    """Binds a horizontal pump and a diagonal pump (`[2] Field+`) into one argument of `core.crystal`
    and applies the resulting linear map to `n_headings` weak probes (`[n_headings] Field+`),
    producing `[2, n_headings] Polarization-`."""
    pumps = stack((core.horizontal_field, (core.horizontal_field + core.vertical_field) / np.sqrt(2)))  # [2] Field+
    headings = np.linspace(0, 2 * np.pi, n_headings)
    heading_turns = (core.turn_plane * (-headings / 2)).exp()                  # [n_headings] Rotor+
    probes = probe_amplitude * (heading_turns >> core.horizontal_field)        # [n_headings] Field+
    # Binding a fixed pump into one input leaves a linear map on probe fields.
    probe_map = core.transverse(core.crystal(pumps[:, None], core.Field))      # [2, 1] Polarization- <- Field+
    return pumps, probes, probe_map(probes)


def phase_matching(n_slices: int) -> tuple[np.ndarray, core.Polarization]:
    """Integrates the doubled-frequency quadratures across `n_slices` slices for a phase-matched
    crystal, a mismatched crystal, and a periodically poled crystal (`[3, n_slices, 2] Polarization-`)."""
    depths = (np.arange(n_slices) + 0.5) / n_slices
    mismatch = np.array([[0], [slip], [slip]])
    # Inverting the tetrahedral bonds reverses the quadratic response every half turn of phase slip.
    flips = (-1.0) ** np.floor(depths * slip / np.pi)
    bond_orientation = np.stack([np.ones(n_slices), np.ones(n_slices), flips])
    amplitude = core.growth(core.vertical_polarization, bond_orientation, mismatch, depths)  # [3, n_slices, 2] Polarization-
    return depths, amplitude


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.electromagnetism.second_harmonic import render

    save_figure(render.draw_waveform(*waveform(n_waveform_samples)), "second_harmonic_waveform")
    pumps, states = turning(n_turns, n_headings)
    kept_states = []

    def streamed() -> Iterator[tuple[core.Direction, core.Polarization]]:
        for state in states:
            kept_states.append(state)
            yield state

    save_animation(render.animate(pumps, streamed()), "second_harmonic", frame_duration_ms)
    first_bonds, first_harmonic = kept_states[0]
    save_figure(render.draw_response(pumps, first_bonds, first_harmonic), "second_harmonic_response")
    pump_fields, probes, generated = mixing(n_headings)
    save_figure(render.draw_mixing(pump_fields, probes, generated), "second_harmonic_mixing")
    depths, amplitude = phase_matching(n_slices)
    save_figure(render.draw_growth(depths, amplitude), "second_harmonic_phase_matching")

    # checks
    # Unrotated horizontal pump drives a negative half-unit vertical polarization; a quarter turn extinguishes it
    # and a half turn reverses its sign.
    _, first_harmonic = kept_states[0]
    _, quarter_turn_harmonic = kept_states[n_turns // 4]
    _, half_turn_harmonic = kept_states[n_turns // 2]
    assert ((first_harmonic[0] + core.vertical_polarization / 2).norm_squared() < 1e-12).all()
    assert (quarter_turn_harmonic[0].norm_squared() < 1e-12).all()
    assert ((half_turn_harmonic[0] - core.vertical_polarization / 2).norm_squared() < 1e-12).all()
    # Pump-bound linear response equals the cross term of pump plus probe.
    pump_and_probe = pump_fields[:, None] + probes
    mixed = (core.second_harmonic(pump_and_probe, pump_and_probe)
             - core.second_harmonic(pump_fields[:, None], pump_fields[:, None])
             - core.second_harmonic(probes, probes))
    assert ((mixed - generated).norm_squared() < 1e-12).all()
    # Matched power reaches 1, mismatched cancels at whole-turn slip, and quasi-phase-matched reaches (2 / pi)^2.
    power = amplitude.scalar_norm_squared().sum(axis=-1)
    assert ((power[:, -1] - np.array([1.0, 0.0, (2 / np.pi) ** 2])).abs() < 1e-4).all()


if __name__ == "__main__":
    main()
