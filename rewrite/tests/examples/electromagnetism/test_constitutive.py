"""Unit tests for the constitutive extensor: projectors, media, birefringence, Fresnel drag."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
import numpy as np

from examples.electromagnetism.constitutive import core, scenarios, render, main

B = core.B
V = core.V
Spatial = core.Spatial
mv = core.mv
minimum_speeds = core.minimum_speeds
t = core.t
x = core.x
y = core.y
z = core.z


def phase_speeds(chi, direction, speeds):
    samples = mv.scalar(speeds[:, None])
    k = samples * t + mv(Spatial, direction)
    wave = k.commutator(chi(k.wedge(Spatial)))
    return minimum_speeds(samples.kernel[..., 0], wave.svdvals()[..., -1].kernel[..., 0])


def polarisation(chi, direction, speed):
    k = t * speed + mv(Spatial, direction)
    return k.commutator(chi(k.wedge(Spatial))).svd()[2][-1]


SPEEDS = np.linspace(0.05, 1.5, 6001)
Z = np.array([0.0, 0.0, 1.0])


def test_mathematics_does_not_import_plotting():
    """The math layer must stay free of the plotting stack, transitively."""
    probe = (
        "import examples.electromagnetism.constitutive.core as c, sys; "
        "bad = [m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')]; "
        "print(bad)"
    )
    repo_rewrite = Path(__file__).resolve().parents[3]
    env = {**os.environ, "PYTHONPATH": f"{repo_rewrite / 'src'}:{repo_rewrite}"}
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert out.stdout.strip() == "[]", f"plotting reached the math layer: {out.stdout}"


def test_observer_projectors_are_complementary_idempotents():
    electric, magnetic = core.observer_projectors()
    np.testing.assert_allclose(electric(electric).kernel, electric.kernel, atol=1e-14)
    np.testing.assert_allclose(magnetic(magnetic).kernel, magnetic.kernel, atol=1e-14)
    np.testing.assert_allclose(electric(magnetic).kernel, 0.0, atol=1e-14)
    F = mv.tx * 1.0 + mv.ty * 2.0 + mv.tz * 5.0 + mv.yz * 3.0 + mv.xy * 4.0
    np.testing.assert_allclose((electric(F) + magnetic(F) - F).kernel, 0.0, atol=1e-14)


def test_isotropic_medium_speed_is_one_over_n():
    for eps, mu in ((2.25, 1.0), (4.0, 1.5)):
        chi = core.isotropic_medium(eps, mu)
        np.testing.assert_allclose(phase_speeds(chi, Z, SPEEDS), [1.0 / np.sqrt(eps * mu)], atol=1e-3)


def test_crystal_is_birefringent_and_reduces_to_glass_when_isotropic():
    iso = core.crystal_medium(2.25, 2.25, 2.25, 1.0)
    glass = core.isotropic_medium(2.25, 1.0)
    np.testing.assert_allclose(iso.kernel, glass.kernel, atol=1e-14)

    crystal = core.crystal_medium(2.25, 1.5, 1.5, 1.0)
    np.testing.assert_allclose(phase_speeds(crystal, Z, SPEEDS), [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)], atol=1e-3)
    np.testing.assert_allclose(phase_speeds(crystal, np.array([1.0, 0.0, 0.0]), SPEEDS), [1.0 / np.sqrt(1.5)], atol=1e-3)


def test_crystal_polarisations_lie_along_its_axes():
    crystal = core.crystal_medium(2.25, 1.5, 1.5, 1.0)
    slow, fast = phase_speeds(crystal, Z, SPEEDS)
    np.testing.assert_allclose(polarisation(crystal, Z, slow).wedge(x).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose(polarisation(crystal, Z, fast).wedge(y).kernel, 0.0, atol=1e-6)


def test_ferrite_lifts_permeability_through_the_dual_field():
    ferrite = core.ferrite_medium(eps=2.25, mu_inv_x=1.0, mu_inv_y=0.5, mu_inv_z=1.0)
    np.testing.assert_allclose(phase_speeds(ferrite, Z, SPEEDS), [1.0 / np.sqrt(4.5), 1.0 / np.sqrt(2.25)], atol=1e-3)


def test_axion_term_is_invisible_to_bulk_waves():
    glass = core.isotropic_medium(2.25, 1.0)
    for alpha in (0.4, -1.3):
        axion = core.axion_medium(glass, alpha)
        np.testing.assert_allclose(phase_speeds(axion, Z, SPEEDS), phase_speeds(glass, Z, SPEEDS), atol=1e-12)


def test_moving_glass_shows_exact_fresnel_drag():
    eps, mu = 2.25, 1.0
    refractive_index = np.sqrt(eps * mu)
    glass = core.isotropic_medium(eps, mu)
    for beta in (0.0, 0.3, -0.5):
        moving = core.boosted_medium(glass, beta, direction=z)
        np.testing.assert_allclose(phase_speeds(moving, Z, SPEEDS), [(1.0 / refractive_index + beta) / (1.0 + beta / refractive_index)], atol=1e-3)


def test_fresnel_surface_and_drag_scenarios():
    angles, surfaces = scenarios.fresnel_surface_scenario(n_angles=12)
    assert len(angles) == 12
    assert "glass (rest)" in surfaces
    assert "crystal" in surfaces

    betas, v_down, v_up = scenarios.fresnel_drag_scenario(n_betas=5)
    assert len(betas) == 5
    assert not np.isnan(v_down).any()
    assert not np.isnan(v_up).any()


def test_tutorial_runs_and_saves(tmp_path):
    out = tmp_path / "constitutive.png"
    main(plot_path=str(out))
    assert out.exists()


def test_wave_and_polarization_rendering(tmp_path):
    import matplotlib.pyplot as plt

    fig2, ax2 = plt.subplots()
    modes = [("Slow Wave", 0.667, x, "crimson")]
    render.draw_polarizations(ax2, modes)
    assert len(ax2.collections) > 0

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111, projection="3d")
    crystal_modes = [(0.667, x), (0.816, y)]
    render.draw_wave_propagation(ax3, crystal_modes)
    assert len(ax3.collections) > 0 or len(ax3.lines) > 0

    gif_out = tmp_path / "test_wave.gif"
    render.animate_wave_propagation(crystal_modes, plot_path=gif_out, n_frames=4)
    assert gif_out.exists()

    plt.close("all")
