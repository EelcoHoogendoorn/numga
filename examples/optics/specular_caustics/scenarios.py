"""Moving-water caustics, their fold curves and three paths to one receiver point."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import numpy as np

from numga.backend.jax import derivative
from examples.optics.specular_caustics.core import Direction, Horizontal, Scalar, Wavevector, mv, rays, receiver, up

x, y = mv.x.dual(), mv.y.dual()

# Four travelling waves: their headings, wavenumbers, amplitudes, phases and phase rates.
wave_angles = np.array([0, 0.7, 1.5, 2.2])
wavenumbers = np.array([6, 8, 10, 12])
amplitudes = np.array([0.045, 0.035, 0.025, 0.018])
phases = np.array([0, 1.2, -0.7, 0.6])
phase_rates = np.array([1, -1, 2, -2])
# The ratio of the refractive indices of air and water, and the depth of the receiver below the surface.
index_ratio = 0.75
depth = 2.5
# The square of water the rays enter through, and the square of the receiver drawn.
source_halfwidth = 1.6
receiver_halfwidth = 1
# One ray in each cell of a grid over the water, and a coarser grid for the map's derivative.
n_rays = 640
n_derivative = 192
n_frames = 72
seed = 17
# A single ripple across x, for a side view of three paths to one receiver point.
ripple_wavenumber = np.array([4])
ripple_amplitude = np.array([0.12])
ripple_phase = np.array([0])
seed_offsets = np.array([-0.3, 0, 0.3])
section_halfwidth = 0.8
n_section = 300
n_fan = 45
n_iterations = 3


# --- math -----------------------------------------------------------------------------
def water() -> tuple[Horizontal, float, Wavevector]:
    """One jittered ray per cell over the water, the area each ray carries, and the wavevectors."""
    spacing = 2 * source_halfwidth / n_rays
    samples = (np.arange(n_rays) + 0.5) * spacing - source_halfwidth
    rng = np.random.default_rng(seed)
    # A fixed jitter within each cell hides the sampling lattice.
    x_offsets = rng.uniform(-0.5, 0.5, (n_rays, n_rays)) * spacing
    y_offsets = rng.uniform(-0.5, 0.5, (n_rays, n_rays)) * spacing
    positions = x * (samples[None, :] + x_offsets) + y * (samples[:, None] + y_offsets)   # [rows, columns] Horizontal-
    wavevectors = ((-mv.xy * wave_angles / 2).exp() >> mv.x) * wavenumbers   # [waves] Wavevector+
    return positions, spacing**2, wavevectors


def caustics(positions: Horizontal, wavevectors: Wavevector) -> Iterator[Horizontal]:
    """The landings of every ray on the receiver as the waves travel."""
    for time in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        yield receiver(positions, wavevectors, amplitudes, phases + phase_rates * time, depth, index_ratio)


def folds(wavevectors: Wavevector) -> tuple[Horizontal, Scalar]:
    """The landings of a grid over the water, and the area the ray map gives each patch of it."""
    chart_samples = np.linspace(-source_halfwidth, source_halfwidth, n_derivative)
    chart = x * chart_samples[None, :] + y * chart_samples[:, None]           # [rows, columns] Horizontal-
    landing = partial(receiver, wavevectors=wavevectors, amplitudes=amplitudes, phases=phases,
                      depth=depth, index_ratio=index_ratio)
    return landing(chart), derivative(landing)(chart).det()                  # [rows, columns] Horizontal-, Scalar+


def three_paths() -> tuple[Direction, Direction, Direction, Direction, Direction]:
    """Under a single ripple, the paths from three surface points to one receiver point, found by
    Newton's method on the ray map, with a fan of rays and the surface along the section."""
    wavevector = mv.x * ripple_wavenumber                                     # [waves] Wavevector+
    ripple = partial(rays, wavevectors=wavevector, amplitudes=ripple_amplitude, phases=ripple_phase,
                     index_ratio=index_ratio)
    landing = partial(receiver, wavevectors=wavevector, amplitudes=ripple_amplitude, phases=ripple_phase,
                      depth=depth, index_ratio=index_ratio)
    target = 0 * x                                                            # [] Horizontal-
    # Trial points start on the section, but may move anywhere on the water.
    trial = (x * seed_offsets).cast(Horizontal)                               # [paths] Horizontal-
    for _ in range(n_iterations):
        # The local ray map turns the miss at the receiver into a correction on the water.
        trial = trial - derivative(landing)(trial).solve(landing(trial) - target)   # [paths] Horizontal-

    section = x * np.linspace(-section_halfwidth, section_halfwidth, n_section)   # [samples] Horizontal-
    fan = x * np.linspace(-section_halfwidth, section_halfwidth, n_fan)           # [rays] Horizontal-
    surface, _, _ = ripple(section)                                           # [samples] Direction-
    fan_surface, _, _ = ripple(fan)                                           # [rays] Direction-
    path_surface, _, _ = ripple(trial)                                        # [paths] Direction-
    fan_landing = landing(fan) - up * depth                                   # [rays] Direction-
    receiver_target = target - up * depth                                     # [] Direction-
    return surface, fan_surface, fan_landing, path_surface, receiver_target


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.specular_caustics import render

    n_pixels = 300
    blur_pixels = 0.95
    exposure = 0.3
    incident_height = 0.6
    duration_ms = 70
    n_colours = 16

    positions, ray_area, wavevectors = water()
    save_animation(render.animate(caustics(positions, wavevectors), receiver_halfwidth, ray_area, n_pixels,
                                  blur_pixels, exposure), "specular_caustics", duration_ms, n_colours)
    landings = receiver(positions, wavevectors, amplitudes, phases, depth, index_ratio)   # [rows, columns] Horizontal-
    chart_landings, area = folds(wavevectors)
    save_figure(render.draw_critical(landings, chart_landings, area, receiver_halfwidth, ray_area, n_pixels,
                                     blur_pixels, exposure), "specular_caustics_folds")
    surface, fan_surface, fan_landing, path_surface, receiver_target = three_paths()
    save_figure(render.draw_paths(surface, fan_surface, fan_landing, path_surface, receiver_target,
                                  incident_height), "specular_caustics_paths")

    # checks
    # The ray map folds over part of the water, and three distinct surface points send their rays to the
    # one receiver point, symmetric about the ripple's crest.
    areas = area.to_array()
    assert areas.min() < 0 < areas.max()
    crossings = (mv.x & path_surface).to_array()                              # [paths]
    assert crossings[0] < -0.2 and abs(crossings[1]) < 1e-9 and crossings[2] > 0.2


if __name__ == "__main__":
    main()
