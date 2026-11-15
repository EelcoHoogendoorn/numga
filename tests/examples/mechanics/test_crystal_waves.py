"""Coverage tests for elastic waves in cubic crystals: stiffness readouts, isotropic vs anisotropic energy transport, and figure rendering."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.crystal_waves import core, render, scenarios


def test_stiffness_reads_back_the_elastic_constants():
    """Stretch and shear tractions on the cube faces reproduce the elastic constants c11, c12 and c44."""
    x_plane, y_plane = core.axes[0], core.axes[1]                   # Plane+, Plane+
    x_direction, y_direction = x_plane.dual(), y_plane.dual()       # Direction-, Direction-
    crystal_stiffnesses, _ = scenarios.crystals()                   # [3] Plane+ <- (Plane+, Direction-, Plane+), [3]
    sampled_moduli = [
        crystal_stiffnesses(x_plane, x_direction, x_plane) & x_direction,
        crystal_stiffnesses(y_plane, x_direction, x_plane) & y_direction,
        crystal_stiffnesses(y_plane, x_direction, y_plane) & x_direction,
    ]                                                               # [3] Scalar+
    for modulus, expected_constant in zip(sampled_moduli, (scenarios.c11, scenarios.c12, scenarios.c44)):
        assert ((modulus - expected_constant).abs() < 1e-8 * expected_constant).all()


def test_energy_follows_the_wave_only_when_isotropic():
    """Fused silica keeps energy transport aligned with the heading and degenerate shear speeds, whereas silicon deflects shear energy sideways."""
    n_headings = 32
    headings = core.mv.vector(np.random.default_rng(1).normal(size=(n_headings, 3))).normalized()   # [n_headings] Plane+
    crystal_stiffnesses, material_densities = scenarios.crystals()   # [3] Plane+ <- (Plane+, Direction-, Plane+), [3]
    wave_moduli, polarizations = core.waves(crystal_stiffnesses[:, None], headings)   # [3, n_headings, 3] Scalar+, [3, n_headings, 3] Direction-
    velocities = core.energy_flow(crystal_stiffnesses[:, None], headings, polarizations, material_densities[:, None])   # [3, n_headings, 3] Direction-
    phase_projections = headings[:, None] & velocities              # [3, n_headings, 3] Scalar+
    group_speeds = velocities.antinorm()                            # [3, n_headings, 3] Scalar+
    cosine_alignment = phase_projections / group_speeds             # [3, n_headings, 3] Scalar+
    silica_index = scenarios.names.index("fused silica")
    silicon_index = scenarios.names.index("silicon")
    assert ((1 - cosine_alignment[silica_index]).abs() < 1e-8).all()
    assert ((1 - cosine_alignment[silicon_index]) > 0.02).any()
    assert ((wave_moduli[silica_index, :, 0] - wave_moduli[silica_index, :, 1]).abs() < 1e-7 * wave_moduli[silica_index, :, 0]).all()


def test_figures_render():
    """Both focusing and wave-front figures build cleanly on a small heading sample."""
    velocities, _, _ = scenarios.focusing(8, scenarios.seed)        # [3, 8, 3] Direction-, [8] Plane+, [3, 8, 3] Scalar+
    focusing_fig = render.focusing_figure(velocities, scenarios.names)
    fronts_fig = render.wave_fronts_figure(scenarios.wave_fronts(8), scenarios.names)
    plt.close(focusing_fig)
    plt.close(fronts_fig)
