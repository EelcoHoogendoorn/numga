"""Travelling-wave caustics, Jacobian fold curves, and multi-branch specular path solves."""

from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import numpy as np

from numga.backend.jax import derivative
from examples.optics.specular_caustics.core import (
    Direction, Horizontal, Scalar, Wavevector, mv, receiver_landing, refracted_rays, up,
)

x, y = mv.x.dual(), mv.y.dual()                                                   # [] Direction-, Direction-

# Four travelling ripple components: propagation angles, wavenumbers, amplitudes, initial phases, and angular frequencies.
wave_angles = np.array([0, 0.7, 1.5, 2.2])
wavenumbers = np.array([6, 8, 10, 12])
amplitudes = np.array([0.045, 0.035, 0.025, 0.018])
phases = np.array([0, 1.2, -0.7, 0.6])
phase_rates = np.array([1, -1, 2, -2])
# Air-to-water refractive index ratio and receiver depth below mean water level.
index_ratio = 0.75
depth = 2.5
# Half-widths of the illuminated water patch and of the rendered receiver window.
source_halfwidth = 1.6
receiver_halfwidth = 1.0
# Default grid resolutions and animation frame count.
n_rays = 640
n_derivative = 192
n_frames = 126
seed = 17
# Single sinusoidal ripple parameters for the vertical cross-section and Newton path solver.
ripple_wavenumber = np.array([4])
ripple_amplitude = np.array([0.12])
ripple_phase = np.array([0])
seed_offsets = np.array([-0.3, 0.0, 0.3])
section_halfwidth = 0.8
n_section = 300
n_fan = 45
n_iterations = 3
# Rendering parameters shared between `scenarios` and the notebook.
n_pixels = 300
blur_pixels = 0.95
exposure = 0.3
incident_height = 0.6
frame_duration_ms = 40
n_colours = 16


# --- math -----------------------------------------------------------------------------
def stratified_water_rays(n_grid: int) -> tuple[Horizontal, float, Wavevector]:
    """Stratified jittered sampling on an `n_grid` by `n_grid` lattice over `[-source_halfwidth, source_halfwidth]^2`
    gives ray entry positions (`[n_grid, n_grid] Horizontal-`), per-ray area, and `[4] Wavevector+` ripple wavevectors."""
    spacing = 2 * source_halfwidth / n_grid
    centres = (np.arange(n_grid) + 0.5) * spacing - source_halfwidth
    rng = np.random.default_rng(seed)
    x_offsets = rng.uniform(-0.5, 0.5, (n_grid, n_grid)) * spacing
    y_offsets = rng.uniform(-0.5, 0.5, (n_grid, n_grid)) * spacing
    positions = x * (centres[None, :] + x_offsets) + y * (centres[:, None] + y_offsets)  # [n_grid, n_grid] Horizontal-
    wave_rotors = (-mv.xy * wave_angles / 2).exp()                                # [4] Rotor+
    wavevectors = (wave_rotors >> mv.x) * wavenumbers                             # [4] Wavevector+
    return positions, spacing ** 2, wavevectors


def caustic_states(positions: Horizontal, wavevectors: Wavevector, n_steps: int) -> Iterator[Horizontal]:
    """Yields `[n_grid, n_grid] Horizontal-` receiver landings across `n_steps` frames of one wave period."""
    for time in np.linspace(0, 2 * np.pi, n_steps, endpoint=False):
        yield receiver_landing(positions, wavevectors, amplitudes, phases + phase_rates * time, depth, index_ratio)


def fold_chart(wavevectors: Wavevector, n_chart: int) -> tuple[Horizontal, Scalar]:
    """Evaluating `receiver_landing` and its Jacobian determinant on an `n_chart` by `n_chart` source chart
    gives `[n_chart, n_chart] Horizontal-` landings and `[n_chart, n_chart] Scalar+` signed area ratios."""
    chart_samples = np.linspace(-source_halfwidth, source_halfwidth, n_chart)
    chart = x * chart_samples[None, :] + y * chart_samples[:, None]               # [n_chart, n_chart] Horizontal-
    landing = partial(receiver_landing, wavevectors=wavevectors, amplitudes=amplitudes, phases=phases,
                      depth=depth, index_ratio=index_ratio)
    return landing(chart), derivative(landing)(chart).det()                       # [n_chart, n_chart] Horizontal-, Scalar+


def specular_paths() -> tuple[Direction, Direction, Direction, Direction, Direction, Horizontal]:
    """Newton iteration on a single-ripple `receiver_landing` locates `[3] Horizontal-` entry points reaching
    the origin target, paired with `[n_section] Direction-` surface samples and `[n_fan] Direction-` ray-fan points."""
    wavevector = mv.x * ripple_wavenumber                                         # [1] Wavevector+
    ripple = partial(refracted_rays, wavevectors=wavevector, amplitudes=ripple_amplitude, phases=ripple_phase,
                     index_ratio=index_ratio)
    landing = partial(receiver_landing, wavevectors=wavevector, amplitudes=ripple_amplitude, phases=ripple_phase,
                      depth=depth, index_ratio=index_ratio)
    target = 0 * x                                                                # [] Horizontal-
    trial_positions = (x * seed_offsets).cast(Horizontal)                         # [3] Horizontal-
    for _ in range(n_iterations):
        # Invert the local horizontal Jacobian to pull receiver residuals back into water-surface steps.
        trial_positions = trial_positions - derivative(landing)(trial_positions).solve(
            landing(trial_positions) - target
        )                                                                         # [3] Horizontal-

    section = x * np.linspace(-section_halfwidth, section_halfwidth, n_section)   # [n_section] Horizontal-
    fan = x * np.linspace(-section_halfwidth, section_halfwidth, n_fan)           # [n_fan] Horizontal-
    surface_profile, _, _ = ripple(section)                                       # [n_section] Direction-
    fan_surface, _, _ = ripple(fan)                                               # [n_fan] Direction-
    path_surface, _, _ = ripple(trial_positions)                                  # [3] Direction-
    fan_landing = landing(fan) - up * depth                                       # [n_fan] Direction-
    receiver_target = target - up * depth                                         # [] Direction-
    return surface_profile, fan_surface, fan_landing, path_surface, receiver_target, landing(trial_positions)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.optics.specular_caustics import render

    positions, ray_area, wavevectors = stratified_water_rays(n_rays)
    first_landings: list[Horizontal] = []

    def recording_first() -> Iterator[Horizontal]:
        for index, state in enumerate(caustic_states(positions, wavevectors, n_frames)):
            if index == 0:
                first_landings.append(state)
            yield state

    save_animation(
        render.caustics_animation(receiver_halfwidth, ray_area, n_pixels, blur_pixels, exposure, recording_first()),
        "specular_caustics", frame_duration_ms, n_colours,
    )
    chart_landings, area_ratios = fold_chart(wavevectors, n_derivative)
    save_figure(
        render.folds_figure(first_landings[0], chart_landings, area_ratios, receiver_halfwidth, ray_area,
                            n_pixels, blur_pixels, exposure),
        "specular_caustics_folds",
    )
    (surface_profile, fan_surface, fan_landing,
     path_surface, receiver_target, path_residuals) = specular_paths()
    save_figure(
        render.paths_figure(surface_profile, fan_surface, fan_landing, path_surface, receiver_target, incident_height),
        "specular_caustics_paths",
    )

    # checks
    # The ray map reverses orientation across folded regions, and three distinct entry points focus onto the target.
    assert bool(area_ratios.min() < 0) and bool(area_ratios.max() > 0)
    assert (path_residuals.scalar_norm_squared().abs() < 1e-12).all()
    crossings = mv.x & path_surface                                               # [3] Scalar+
    assert bool(crossings[0] < -0.2) and bool(crossings[1].abs() < 1e-9) and bool(crossings[2] > 0.2)


if __name__ == "__main__":
    main()
