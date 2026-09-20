"""Unit tests for the constitutive extensor: projectors, media, birefringence, Fresnel drag."""

from __future__ import annotations

import numpy as np

from examples.electromagnetism.constitutive import main
from examples.electromagnetism.constitutive_plumbing import B, V, Spatial, mv, minimum_speeds, t, x, y, z

def phase_speeds(chi, direction, speeds):
    samples = mv.scalar(speeds[:, None])
    k = samples * t + mv(Spatial, direction)
    wave = k.commutator(chi(k.wedge(Spatial)))
    return minimum_speeds(samples, wave.svdvals()[..., -1]).kernel[..., 0]


def polarisation(chi, direction, speed):
    k = t * speed + mv(Spatial, direction)
    return k.commutator(chi(k.wedge(Spatial))).svd()[2][-1]


SPEEDS = np.linspace(0.05, 1.5, 6001)
Z = np.array([0.0, 0.0, 1.0])


def projectors():
    electric = B.commutator(t).wedge(t)
    return electric, B - electric


def test_observer_projectors_are_complementary_idempotents():
    electric, magnetic = projectors()
    np.testing.assert_allclose(electric(electric).kernel, electric.kernel, atol=1e-14)
    np.testing.assert_allclose(magnetic(magnetic).kernel, magnetic.kernel, atol=1e-14)
    np.testing.assert_allclose(electric(magnetic).kernel, 0.0, atol=1e-14)
    F = mv.tx * 1.0 + mv.ty * 2.0 + mv.tz * 5.0 + mv.yz * 3.0 + mv.xy * 4.0
    np.testing.assert_allclose((electric(F) + magnetic(F) - F).kernel, 0.0, atol=1e-14)


def test_isotropic_medium_speed_is_one_over_n():
    electric, magnetic = projectors()
    for eps, mu in ((2.25, 1.0), (4.0, 1.5)):
        chi = eps * electric + (1.0 / mu) * magnetic
        np.testing.assert_allclose(phase_speeds(chi, Z, SPEEDS), [1.0 / np.sqrt(eps * mu)], atol=1e-3)


def test_crystal_is_birefringent_and_reduces_to_glass_when_isotropic():
    electric, magnetic = projectors()
    def crystal(ex, ey, ez):
        permittivity = -(ex * x * (x | V) + ey * y * (y | V) + ez * z * (z | V))
        return permittivity(B.commutator(t)).wedge(t) + magnetic
    np.testing.assert_allclose(crystal(2.25, 2.25, 2.25).kernel, (2.25 * electric + magnetic).kernel, atol=1e-14)
    np.testing.assert_allclose(phase_speeds(crystal(2.25, 1.5, 1.5), Z, SPEEDS), [1.0 / np.sqrt(2.25), 1.0 / np.sqrt(1.5)], atol=1e-3)
    np.testing.assert_allclose(phase_speeds(crystal(2.25, 1.5, 1.5), np.array([1.0, 0.0, 0.0]), SPEEDS), [1.0 / np.sqrt(1.5)], atol=1e-3)


def test_crystal_polarisations_lie_along_its_axes():
    electric, magnetic = projectors()
    permittivity = -(2.25 * x * (x | V) + 1.5 * y * (y | V) + 1.5 * z * (z | V))
    crystal = permittivity(B.commutator(t)).wedge(t) + magnetic
    slow, fast = phase_speeds(crystal, Z, SPEEDS)
    np.testing.assert_allclose(polarisation(crystal, Z, slow).wedge(x).kernel, 0.0, atol=1e-6)
    np.testing.assert_allclose(polarisation(crystal, Z, fast).wedge(y).kernel, 0.0, atol=1e-6)


def test_ferrite_lifts_permeability_through_the_dual_field():
    electric, magnetic = projectors()
    isotropic_inv = -(x * (x | V) + y * (y | V) + z * (z | V))
    np.testing.assert_allclose(isotropic_inv(B.dual().commutator(t)).wedge(t).dual_inverse().kernel, magnetic.kernel, atol=1e-14)
    permeability_inv = -(1.0 * x * (x | V) + 0.5 * y * (y | V) + 1.0 * z * (z | V))
    ferrite = 2.25 * electric + permeability_inv(B.dual().commutator(t)).wedge(t).dual_inverse()
    np.testing.assert_allclose(phase_speeds(ferrite, Z, SPEEDS), [1.0 / np.sqrt(4.5), 1.0 / np.sqrt(2.25)], atol=1e-3)


def test_axion_term_is_invisible_to_bulk_waves():
    electric, magnetic = projectors()
    glass = 2.25 * electric + magnetic
    for alpha in (0.4, -1.3):
        axion = glass + mv.scalar([alpha]) * B.dual()
        np.testing.assert_allclose(phase_speeds(axion, Z, SPEEDS), phase_speeds(glass, Z, SPEEDS), atol=1e-12)


def test_moving_glass_shows_exact_fresnel_drag():
    electric, magnetic = projectors()
    eps, mu = 2.25, 1.0
    glass = eps * electric + (1.0 / mu) * magnetic
    n = np.sqrt(eps * mu)
    for beta in (0.0, 0.3, -0.5):
        boost = (mv.zt * (np.arctanh(beta) / 2.0)).exp()
        moving = boost >> glass(boost << B)
        np.testing.assert_allclose(phase_speeds(moving, Z, SPEEDS), [(1.0 / n + beta) / (1.0 + beta / n)], atol=1e-3)


def test_tutorial_runs_and_saves(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    out = tmp_path / "constitutive.png"
    main(plot_path=str(out))
    assert out.exists()
