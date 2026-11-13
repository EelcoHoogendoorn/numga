"""Steady heating under a loaded piston warms a van der Waals liquid, boils it at fixed temperature and pressure, and
warms the resulting vapour."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.thermodynamics.phase_change import core

capacity, attraction, exclusion = 1.5, 9 / 8, 1 / 3
n_newton = 8
boiling_temperature, cold_temperature, hot_temperature = 0.85, 0.7, 0.98
n_frames, frame_duration_ms = 105, 40


# --- math -----------------------------------------------------------------------------
def pressure_at(volume: core.Scalar, temperature: np.ndarray | float) -> core.Scalar:
    return core.pressure(volume, temperature, attraction, exclusion)


def enthalpy_at(volume: core.Scalar, temperature: np.ndarray | float) -> core.Scalar:
    return core.enthalpy(volume, temperature, capacity, attraction, exclusion)


def coexistence_at(temperature: np.ndarray | float) -> core.State:
    return core.coexistence(temperature, capacity, attraction, exclusion, n_newton)


def isobar_at(volume: core.Scalar, isobar_pressure: core.Scalar) -> core.Scalar:
    return core.isobar_temperature(volume, isobar_pressure, attraction, exclusion)


def volume_at(temperature: np.ndarray | float, isobar_pressure: core.Scalar, guess: core.Scalar) -> core.Scalar:
    return core.volume_at(temperature, isobar_pressure, guess, attraction, exclusion, n_newton)


def heating(boiling_temperature: float, cold_temperature: float, hot_temperature: float, n_frames: int
            ) -> Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar]]:
    """Steady heat input at the saturation pressure advances the fluid in equal enthalpy steps from cold liquid through
    boiling to hot vapour, yielding volume, temperature, enthalpy and liquid volume at each frame."""
    saturation_state = coexistence_at(boiling_temperature)                                # [] State-
    liquid, vapour = core.liquid_volume(saturation_state), core.vapour_volume(saturation_state)
    saturation_pressure = pressure_at(liquid, boiling_temperature)                        # [] Scalar+
    # Each stage receives frames in proportion to the enthalpy jump between its endpoints.
    stage_enthalpies = stack([
        enthalpy_at(volume_at(cold_temperature, saturation_pressure, liquid), cold_temperature),
        enthalpy_at(liquid, boiling_temperature),
        enthalpy_at(vapour, boiling_temperature),
        enthalpy_at(volume_at(hot_temperature, saturation_pressure, vapour), hot_temperature),
    ])                                                                                    # [4] Scalar+
    counts = stage_frames(stage_enthalpies, n_frames)
    for temperature in np.linspace(cold_temperature, boiling_temperature, counts[0], endpoint=False):
        volume = volume_at(temperature, saturation_pressure, liquid)
        yield volume, core.mv.scalar([temperature]), enthalpy_at(volume, temperature), volume
    for vapour_fraction in np.linspace(0, 1, counts[1], endpoint=False):
        volume, mixture_enthalpy, liquid_share = core.mixture(saturation_state, vapour_fraction, boiling_temperature,
                                                              capacity, attraction, exclusion)
        yield volume, core.mv.scalar([boiling_temperature]), mixture_enthalpy, liquid_share
    for temperature in np.linspace(boiling_temperature, hot_temperature, counts[2]):
        volume = volume_at(temperature, saturation_pressure, vapour)
        yield volume, core.mv.scalar([temperature]), enthalpy_at(volume, temperature), volume * 0


# --- plumbing -------------------------------------------------------------------------
def stage_frames(stage_enthalpies: core.Scalar, n_frames: int) -> np.ndarray:
    """Each heating stage receives a frame count proportional to its enthalpy span."""
    spans = np.diff(stage_enthalpies.to_array())
    return np.round(n_frames * spans / spans.sum()).astype(int)


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.phase_change import render

    save_figure(render.draw_charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, boiling_temperature),
                "phase_change_charts")
    save_animation(render.heating(pressure_at, enthalpy_at, coexistence_at, isobar_at, boiling_temperature,
                                  heating(boiling_temperature, cold_temperature, hot_temperature, n_frames)),
                   "phase_change", frame_duration_ms)


def pinned() -> None:
    """Saves the pinned gallery animation cropping the pressure–volume and temperature–enthalpy charts."""
    from examples.util.animation import pin_animation
    from examples.thermodynamics.phase_change import render

    # Crop out the cylinder on the left and the phase diagram on the right, keeping the two middle charts.
    gallery_columns = slice(render.cylinder_pixels, render.cylinder_pixels + 2 * render.n_columns)
    pin_animation((frame[:, gallery_columns]
                   for frame in render.heating(pressure_at, enthalpy_at, coexistence_at, isobar_at,
                                               boiling_temperature,
                                               heating(boiling_temperature, cold_temperature, hot_temperature,
                                                       n_frames))),
                  "boiling", frame_duration_ms)


if __name__ == "__main__":
    main()
