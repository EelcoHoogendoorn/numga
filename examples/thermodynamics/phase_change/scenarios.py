"""Heat taken in at a steady rate by a fixed amount of fluid under a loaded piston: the liquid warms, boils at fixed
temperature and pressure, and the vapour warms."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.phase_change import core

capacity, attraction, exclusion = 1.5, 9 / 8, 1 / 3
n_newton = 8


# --- math -----------------------------------------------------------------------------
def pressure_at(volume: core.Scalar, temperature: np.ndarray) -> core.Scalar:
    return core.pressure(volume, temperature, attraction, exclusion)


def enthalpy_at(volume: core.Scalar, temperature: np.ndarray) -> core.Scalar:
    return core.enthalpy(volume, temperature, capacity, attraction, exclusion)


def coexistence_at(temperature: np.ndarray) -> core.State:
    return core.coexistence(temperature, capacity, attraction, exclusion, n_newton)


def isobar_at(volume: core.Scalar, isobar_pressure: core.Scalar) -> core.Scalar:
    return core.isobar_temperature(volume, isobar_pressure, attraction, exclusion)


def volume_at(temperature: np.ndarray, isobar_pressure: core.Scalar, guess: core.Scalar) -> core.Scalar:
    return core.volume_at(temperature, isobar_pressure, guess, attraction, exclusion, n_newton)


def heating(boiling_temperature: float, coldest: float, hottest: float, n_frames: int
            ) -> Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar]]:
    """The fluid's volume, temperature, enthalpy and the volume its liquid takes up, in equal steps of heat at the
    saturation pressure of the boiling temperature: from liquid at the coldest temperature, through boiling, to
    vapour at the hottest."""
    coexisting = coexistence_at(np.array(boiling_temperature))
    liquid, vapour = core.liquid_volume(coexisting), core.vapour_volume(coexisting)
    saturation_pressure = pressure_at(liquid, boiling_temperature)                        # [] Scalar+
    # The enthalpies where each stage starts and ends: the steps of heat each stage takes are in their proportion.
    ends = stack([enthalpy_at(volume_at(coldest, saturation_pressure, liquid), coldest),
                  enthalpy_at(liquid, boiling_temperature), enthalpy_at(vapour, boiling_temperature),
                  enthalpy_at(volume_at(hottest, saturation_pressure, vapour), hottest)])  # [stages + 1] Scalar+
    counts = stage_frames(ends, n_frames)
    for temperature in np.linspace(coldest, boiling_temperature, counts[0], endpoint=False):
        volume = volume_at(temperature, saturation_pressure, liquid)
        yield volume, core.mv.scalar([temperature]), enthalpy_at(volume, temperature), volume
    for fraction in np.linspace(0, 1, counts[1], endpoint=False):
        volume, shared_enthalpy, liquid_share = core.mixture(coexisting, fraction, boiling_temperature, capacity,
                                                             attraction, exclusion)
        yield volume, core.mv.scalar([boiling_temperature]), shared_enthalpy, liquid_share
    for temperature in np.linspace(boiling_temperature, hottest, counts[2]):
        volume = volume_at(temperature, saturation_pressure, vapour)
        yield volume, core.mv.scalar([temperature]), enthalpy_at(volume, temperature), volume * 0


# --- plumbing -------------------------------------------------------------------------
def stage_frames(ends: core.Scalar, n_frames: int) -> np.ndarray:
    """The frames each stage takes, in proportion to its span between the ends `[stages + 1]`."""
    spans = np.diff(ends.to_array())
    return np.round(n_frames * spans / spans.sum()).astype(int)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.phase_change import render

    boiling_temperature, coldest, hottest = 0.85, 0.7, 0.98
    n_frames, frame_duration_ms = 105, 40
    save_figure(render.draw_charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, boiling_temperature),
                "phase_change_charts")
    save_animation(render.heating(pressure_at, enthalpy_at, coexistence_at, isobar_at, boiling_temperature,
                                  heating(boiling_temperature, coldest, hottest, n_frames)),
                   "phase_change", frame_duration_ms)
