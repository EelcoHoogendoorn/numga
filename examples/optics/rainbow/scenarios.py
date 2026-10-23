"""Scenes for the rainbow: light swept across a raindrop, leaving it after one and two reflections, and
the double bow it paints on the sky opposite the sun."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.optics.rainbow import core

mv = core.mv
# The direction sunlight travels, and the direction up, square to it.
forward, up, _ = stack([mv.x, mv.y, mv.z]).dual()
# The wavelengths of the sunlight, in nanometres, and the refractive index of water for each: Cauchy's
# fit, a constant and a term falling with the wavelength squared.
WAVELENGTHS = np.linspace(400, 700, 31)
INDEX_CONSTANT = 1.3238
INDEX_DISPERSION = 3140.0                                     # square nanometres
# The refractive index the swept light is drawn for, that of yellow light.
DRAWN_INDEX = 1.333
# Offsets of the light from the drop's centre, as fractions of its radius: for the sky, for the light
# swept across the drop, and for the bundle drawn behind it.
N_RAYS = 4000
N_FRAMES = 72
N_BUNDLE = 48
DURATION_MS = 80
# Reflections inside the drop: once for the primary bow, twice for the secondary.
N_REFLECTIONS = 2
# The sun's height above the horizon, and the width of its disc, in degrees.
SUN_ELEVATION = 12.0
SUN_WIDTH = 0.53


# --- math -----------------------------------------------------------------------------
def indices() -> np.ndarray:
    """The refractive index of water for each wavelength `[colours, 1]`, broadcasting over the light."""
    return (INDEX_CONSTANT + INDEX_DISPERSION / WAVELENGTHS**2)[:, None]


def sky(offsets: core.Direction) -> tuple[core.Scalar, core.Scalar]:
    """Per number of reflections, from one, and per colour, for light at each offset `[orders, colours,
    rays]`: the cosine of the angle from the point opposite the sun at which it is seen, and the sunlight
    it carries there, from the ring of offsets of its size."""
    _, *bows = core.paths(forward, offsets, indices(), N_REFLECTIONS)
    # Light leaving along `exit` is seen looking back against it; the point opposite the sun lies along
    # the sunlight.
    apart = stack([(-exit).dot(forward) for _, exit, _ in bows])              # [orders, colours, rays] Scalar
    # The rings of offsets grow with their size.
    rings = offsets.norm()                                                    # [rays] Scalar
    return apart, stack([passed * rings for _, _, passed in bows])            # [orders, colours, rays] Scalar


def sweep(offsets: core.Direction) -> tuple[core.Direction, core.Direction, core.Direction, core.Scalar]:
    """For light at each offset `[...]`: where it enters the drop; and after each number of
    reflections `[orders + 1, ...]`, the point where it leaves, the direction it leaves in, and the
    cosine of the angle from the point opposite the sun at which it is seen."""
    vertices, exits, _ = zip(*core.paths(forward, offsets, DRAWN_INDEX, N_REFLECTIONS))
    exits = stack(list(exits))                                                # [orders + 1, ...] Direction
    return core.entry(forward, offsets), stack(list(vertices)), exits, (-exits).dot(forward)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.rainbow import render

    # Offsets at the middle of equal steps across the drop's radius.
    offsets = up * ((np.arange(N_RAYS) + 0.5) / N_RAYS)                       # [rays] Direction
    apart, flux = sky(offsets)                                                # [orders, colours, rays] Scalar
    save_figure(render.draw_sky(apart, flux, WAVELENGTHS, SUN_ELEVATION, SUN_WIDTH), "rainbow_sky")
    swept = sweep(up * ((np.arange(N_FRAMES) + 0.5) / N_FRAMES))
    bundle = sweep(up * ((np.arange(N_BUNDLE) + 0.5) / N_BUNDLE))
    images = render.animate_sweep(swept, bundle, offsets, apart, WAVELENGTHS)
    save_animation(images, "rainbow_sweep", DURATION_MS)

    # checks
    # Each bow lies where the angle of the light leaving is stationary, Descartes's angle: after one
    # reflection the largest from the point opposite the sun, after two the smallest.
    n = indices()[:, 0]
    reflections = np.arange(1, N_REFLECTIONS + 1)[:, None]                    # [orders, 1]
    incidence = np.arccos(np.sqrt((n**2 - 1) / (reflections * (reflections + 2))))
    refraction = np.arcsin(np.sin(incidence) / n)
    deflection = 2 * (incidence - refraction) + reflections * (np.pi - 2 * refraction)
    cosines = apart.to_array()                                                # [orders, colours, rays]
    np.testing.assert_allclose([cosines[0].min(axis=-1), cosines[1].max(axis=-1)], -np.cos(deflection), atol=1e-6)
    # The light leaving is a unit direction, and each fraction of sunlight is less after more reflections.
    _, _, exits, _ = swept
    np.testing.assert_allclose(exits.scalar_norm_squared().to_array(), 1, atol=1e-6)
    np.testing.assert_array_less(flux.to_array()[1], flux.to_array()[0])


if __name__ == "__main__":
    main()
