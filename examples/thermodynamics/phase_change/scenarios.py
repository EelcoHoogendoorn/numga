"""Heat taken in at a steady rate by a fixed amount of fluid under a loaded piston: the liquid warms, boils at fixed
temperature and pressure, and the vapour warms."""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.thermodynamics.phase_change import core

CAPACITY, ATTRACTION, EXCLUSION = 1.5, 9 / 8, 1 / 3
N_NEWTON = 8


# --- math -----------------------------------------------------------------------------
def pressure_at(volume: core.Scalar, temperature: np.ndarray) -> core.Scalar:
    return core.pressure(volume, temperature, ATTRACTION, EXCLUSION)


def enthalpy_at(volume: core.Scalar, temperature: np.ndarray) -> core.Scalar:
    return core.enthalpy(volume, temperature, CAPACITY, ATTRACTION, EXCLUSION)


def coexistence_at(temperature: np.ndarray) -> core.State:
    return core.coexistence(temperature, CAPACITY, ATTRACTION, EXCLUSION, N_NEWTON)


def isobar_at(volume: core.Scalar, level: float) -> core.Scalar:
    return core.isobar_temperature(volume, level, ATTRACTION, EXCLUSION)


def volume_at(temperature: np.ndarray, level: float, guess: core.Scalar) -> core.Scalar:
    return core.volume_at(temperature, level, guess, ATTRACTION, EXCLUSION, N_NEWTON)


def heating(temperature: float, coldest: float, hottest: float, n_frames: int
            ) -> Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Scalar]]:
    """The fluid's volume, temperature, enthalpy and the volume its liquid takes up, in equal steps of heat at the
    saturation pressure of the boiling temperature: from liquid at the coldest temperature, through boiling, to
    vapour at the hottest."""
    coexisting = coexistence_at(np.array(temperature))
    liquid, vapour = core.mv.l & coexisting, core.mv.g & coexisting
    level = float(pressure_at(liquid, temperature).to_array())
    # The enthalpies where each stage starts and ends: the steps of heat each stage takes are in their proportion.
    starts = [enthalpy_at(volume_at(coldest, level, liquid), coldest), enthalpy_at(liquid, temperature),
              enthalpy_at(vapour, temperature), enthalpy_at(volume_at(hottest, level, vapour), hottest)]
    spans = np.diff([float(start.to_array()) for start in starts])
    counts = np.round(n_frames * spans / spans.sum()).astype(int)
    for warmth in np.linspace(coldest, temperature, counts[0], endpoint=False):
        volume = volume_at(warmth, level, liquid)
        yield volume, core.mv.scalar([warmth]), enthalpy_at(volume, warmth), volume
    for fraction in np.linspace(0, 1, counts[1], endpoint=False):
        volume, heat_content, liquid_volume = core.mixture(coexisting, fraction, temperature, CAPACITY, ATTRACTION,
                                                           EXCLUSION)
        yield volume, core.mv.scalar([temperature]), heat_content, liquid_volume
    for warmth in np.linspace(temperature, hottest, counts[2]):
        volume = volume_at(warmth, level, vapour)
        yield volume, core.mv.scalar([warmth]), enthalpy_at(volume, warmth), volume * 0


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.thermodynamics.phase_change import render

    temperature = 0.85
    save_figure(render.draw_charts(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature),
                "phase_change_charts")
    save_animation(render.heating(pressure_at, enthalpy_at, coexistence_at, isobar_at, temperature,
                                  heating(temperature, 0.7, 0.98, 60)), "phase_change", 70)
