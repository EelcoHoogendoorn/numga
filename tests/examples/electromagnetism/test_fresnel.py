"""The Fresnel adjugate detects physical electromagnetic waves beyond the gauge null direction."""

import numpy as np

from examples.electromagnetism.fresnel import core


def test_vacuum_quartic_and_observer_independence():
    permittivity = np.ones(3)
    reluctivity = np.array(1.0)
    medium = core.dielectric(permittivity, reluctivity)
    wavevectors = core.mv.vector([[2, 0.3, 0.4, -0.2], [1, 0, 0, 1], [0.8, 1.1, -0.2, 0.3]])
    moving = (core.mv.tx * 0.23).exp() >> core.mv.xyz
    expected = wavevectors.squared().squared()
    for rest_space in (core.mv.xyz, moving):
        actual = core.polynomial(medium, rest_space, wavevectors)
        np.testing.assert_allclose((actual - expected).kernel, 0, atol=1e-10)
    wave = core.potential_map(medium, wavevectors)
    np.testing.assert_allclose(wave(wavevectors).kernel, 0, atol=1e-12)


def test_principal_wave_radii_match_transverse_material_weights():
    permittivity = np.array([2.0, 3.0, 4.0])
    reluctivity = np.array(0.7)
    frequency = 1.2
    medium = core.dielectric(permittivity, reluctivity)
    sheets = core.radial_sheets(medium, core.axes, frequency)
    radii_squared = sheets.squared()
    expected = frequency**2 * np.array([[3, 4], [2, 4], [2, 3]]) / reluctivity
    np.testing.assert_allclose(radii_squared.kernel[..., 0], expected, atol=1e-10)


def test_both_sheets_support_maxwell_fields_and_transform_with_the_medium():
    permittivity = np.array([[1.0, 1.0, 1.0], [2.25, 2.25, 3.24], [1.44, 2.25, 3.24]])
    reluctivity = np.ones(len(permittivity))
    frequency = 1.0
    media = core.dielectric(permittivity, reluctivity)[:, None, None, None]
    directions = core.sphere(7, 13)
    sheets = core.radial_sheets(media[..., 0], directions, frequency)
    wavevectors = core.mv.t * frequency + sheets
    wave = core.potential_map(media, wavevectors)
    _, _, potentials = wave.svd()
    fields = wavevectors[..., None] ^ potentials[..., -2:]
    source = wavevectors[..., None] ^ media[..., None](fields)
    residual = core.polynomial(media, core.mv.xyz, wavevectors)
    adjugate = wave.outermorphism(core.Trivector).adjugate()

    rotor = (core.mv.tx * 0.19 + core.mv.yz * 0.13).exp()
    moved_media = rotor >> media(rotor << core.Bivector)
    moved_wavevectors = rotor >> wavevectors
    moved_residual = core.polynomial(moved_media, rotor >> core.mv.xyz, moved_wavevectors)

    np.testing.assert_allclose(residual.kernel, 0, atol=1e-9)
    np.testing.assert_allclose(adjugate.kernel, 0, atol=1e-9)
    # Repeated radial roots lose half the significant digits to the square root.
    np.testing.assert_allclose(source.kernel, 0, atol=1e-5)
    np.testing.assert_allclose((moved_residual - residual).kernel, 0, atol=1e-9)
    assert np.all(np.sum(fields.kernel**2, axis=(-2, -1)) > 0.1)


def test_the_axion_part_leaves_light_alone_and_reverses_in_a_mirror():
    permittivity = np.array([1.44, 2.25, 3.24])
    medium = core.dielectric(permittivity, np.array(1.0))
    axion = core.axion(core.mv(core.Axion, [0.3]))
    wavevectors = core.mv.t + core.radial_sheets(medium, core.axes, 1.0)
    mirror = core.mv.x

    np.testing.assert_allclose(core.potential_map(axion, wavevectors).kernel, 0, atol=1e-12)
    np.testing.assert_allclose(core.polynomial(medium + axion, core.mv.xyz, wavevectors).kernel, 0, atol=1e-9)
    # The mirror image of the material: the reflected map has the opposite coupling.
    reflected = mirror >> axion(mirror << core.Bivector)
    np.testing.assert_allclose((reflected + axion).kernel, 0, atol=1e-12)
