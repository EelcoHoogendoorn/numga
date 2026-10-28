"""Scenes for the rainbow: light swept across a raindrop, leaving it after one and two reflections, and
the double bow it paints on the sky opposite the sun."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.optics.rainbow import core

mv = core.mv
# The direction sunlight travels, and the direction up, square to it.
forward, up = mv.x.dual(), mv.y.dual()                                        # [] Direction-, Direction-
# The wavelengths of the sunlight, in nanometres, and the refractive index of water for each: Cauchy's
# fit, a constant and a term falling with the wavelength squared, its coefficient in square nanometres.
wavelengths = np.linspace(400, 700, 31)
index_constant = 1.3238
index_dispersion = 3140.0
indices = (index_constant + index_dispersion / wavelengths**2)[:, None]       # [colours, 1]
# The refractive index the swept light is drawn for, that of yellow light.
drawn_index = 1.333
# Offsets of the light from the drop's centre, as fractions of its radius: for the sky, for the light
# swept across the drop, and for the bundle drawn behind it; the duration of a frame of the sweep.
n_rays = 4000
n_frames = 72
n_bundle = 48
duration_ms = 80
# Reflections inside the drop: once for the primary bow, twice for the secondary.
n_reflections = 2
# The sun's height above the horizon, and the width of its disc, in degrees.
sun_elevation = 12.0
sun_width = 0.53


# --- math -----------------------------------------------------------------------------
def sky(offsets: core.Direction) -> tuple[core.Scalar, core.Scalar]:
    """Per number of reflections, from one, and per colour, for light at each offset `[orders, colours,
    rays]`: the cosine of the angle from the point opposite the sun at which it is seen, and the sunlight
    it carries there, from the ring of offsets of its size."""
    _, *bows = core.paths(forward, offsets, indices, n_reflections)
    # Light leaving along `exit` is seen looking back against it; the point opposite the sun lies along
    # the sunlight.
    sight_cosines = stack([(-exit).dot(forward) for _, exit, _ in bows])      # [orders, colours, rays] Scalar+
    # The rings of offsets grow with their size.
    ring_sizes = offsets.norm()                                               # [rays] Scalar+
    return sight_cosines, stack([passed * ring_sizes for _, _, passed in bows])   # [orders, colours, rays] Scalar+


def sweep(offsets: core.Direction) -> tuple[core.Direction, core.Direction, core.Direction, core.Scalar]:
    """For light at each offset `[...]`: where it enters the drop; and after each number of
    reflections `[orders + 1, ...]`, the point where it leaves, the direction it leaves in, and the
    cosine of the angle from the point opposite the sun at which it is seen."""
    vertices, exits, _ = zip(*core.paths(forward, offsets, drawn_index, n_reflections))
    exits = stack(list(exits))                                                # [orders + 1, ...] Direction-
    return core.entry(forward, offsets), stack(list(vertices)), exits, (-exits).dot(forward)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.rainbow import render

    # Offsets at the middle of equal steps across the drop's radius.
    offsets = up * ((np.arange(n_rays) + 0.5) / n_rays)                       # [rays] Direction-
    sight_cosines, flux = sky(offsets)                                        # [orders, colours, rays] Scalar+, Scalar+
    save_figure(render.draw_sky(sight_cosines, flux, wavelengths, sun_elevation, sun_width), "rainbow_sky")
    swept_light = sweep(up * ((np.arange(n_frames) + 0.5) / n_frames))
    bundle = sweep(up * ((np.arange(n_bundle) + 0.5) / n_bundle))
    images = render.animate_sweep(swept_light, bundle, offsets, sight_cosines, wavelengths)
    save_animation(images, "rainbow_sweep", duration_ms)

    # checks
    # Each bow lies where the angle of the light leaving is stationary, Descartes's angle: after one
    # reflection the largest from the point opposite the sun, after two the smallest.
    index = indices[:, 0]
    reflections = np.arange(1, n_reflections + 1)[:, None]                    # [orders, 1]
    incidence = np.arccos(np.sqrt((index**2 - 1) / (reflections * (reflections + 2))))
    refraction = np.arcsin(np.sin(incidence) / index)
    deflection = 2 * (incidence - refraction) + reflections * (np.pi - 2 * refraction)
    cosines = sight_cosines.to_array()                                        # [orders, colours, rays]
    np.testing.assert_allclose([cosines[0].min(axis=-1), cosines[1].max(axis=-1)], -np.cos(deflection), atol=1e-6)
    # The light leaving is a unit direction, and each fraction of sunlight is less after more reflections.
    _, _, exits, _ = swept_light
    np.testing.assert_allclose(exits.scalar_norm_squared().to_array(), 1, atol=1e-6)
    np.testing.assert_array_less(flux.to_array()[1], flux.to_array()[0])


if __name__ == "__main__":
    main()
