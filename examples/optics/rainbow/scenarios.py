"""Scene setup and verification run for the primary and secondary rainbows."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import stack
from examples.optics.rainbow import core

forward, up = core.forward, core.up                                           # [] Direction-, Direction-
# Cauchy dispersion model across the visible spectrum in nanometres, and yellow index for the drop sweep.
n_colours = 31
wavelengths = np.linspace(400, 700, n_colours)
index_constant = 1.3238
index_dispersion = 3140.0
indices = (index_constant + index_dispersion / wavelengths**2)[:, None]       # [n_colours, 1]
drawn_index = np.array(1.333)
# Sample counts across the drop radius for sky integration, animated sweep, and background ray fan.
n_rays = 4000
n_frames = 144
n_bundle = 48
frame_duration_ms = 40
n_reflections = 2
# Solar elevation and angular diameter in degrees.
sun_elevation = 12.0
sun_width = 0.53


# --- plumbing -------------------------------------------------------------------------
def save_states(
    states: Iterable,
    animate: Callable[..., Iterable[np.ndarray]],
    *arguments,
    name: str,
    duration_ms: int,
) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns
    the collected states."""
    from examples.util.animation import save_animation

    recorded = []

    def passing() -> Iterator:
        for state in states:
            recorded.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return recorded


def main() -> None:
    import matplotlib.pyplot as plt

    from examples.optics.rainbow import render
    from examples.util.animation import save_figure

    offsets = up * ((np.arange(n_rays) + 0.5) / n_rays)                       # [n_rays] Direction-
    sight_cosines, flux = core.sky_bows(offsets, indices, n_reflections)      # [n_reflections, n_colours, n_rays] Scalar+, Scalar+
    sky = render.sky_figure(sight_cosines, flux, wavelengths, sun_elevation, sun_width)
    save_figure(sky, "rainbow_sky")
    plt.close(sky)

    sweep_offsets = up * ((np.arange(n_frames) + 0.5) / n_frames)             # [n_frames] Direction-
    bundle_offsets = up * ((np.arange(n_bundle) + 0.5) / n_bundle)            # [n_bundle] Direction-
    bundle = core.ray_bundle(bundle_offsets, drawn_index, n_reflections)
    swept_states = save_states(
        core.swept_ray_states(sweep_offsets, drawn_index, n_reflections),
        render.sweep_animation,
        bundle,
        offsets,
        sight_cosines,
        wavelengths,
        name="rainbow_sweep",
        duration_ms=frame_duration_ms,
    )

    # checks
    # Stationary deflection occurs at Descartes's incidence angle, largest at one bounce and smallest at two.
    index = indices[:, 0]
    reflections = np.arange(1, n_reflections + 1)[:, None]
    incidence = np.arccos(np.sqrt((index**2 - 1) / (reflections * (reflections + 2))))
    refraction = np.arcsin(np.sin(incidence) / index)
    deflection = 2 * (incidence - refraction) + reflections * (np.pi - 2 * refraction)
    assert (sight_cosines[0].min(axis=-1) + np.cos(deflection[0])).abs().max() < 1e-6
    assert (sight_cosines[1].max(axis=-1) + np.cos(deflection[1])).abs().max() < 1e-6
    # Every exited ray has unit spatial length, and secondary flux stays below primary flux at every offset.
    exits = stack([state[2] for state in swept_states])                       # [n_frames, 3] Direction-
    assert (exits.antinorm() - 1).abs().max() < 1e-6
    assert (flux[1] < flux[0]).all()


if __name__ == "__main__":
    main()
