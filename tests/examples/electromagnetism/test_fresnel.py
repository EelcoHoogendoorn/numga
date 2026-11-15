"""The potential map's action on volumes detects physical electromagnetic waves beyond the gauge null direction."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack

from examples.electromagnetism.fresnel import core, render

potentials = stack((core.mv.x, core.mv.y, core.mv.z, core.mv.t))                       # [4] Vector+
volumes_basis = stack((core.mv.xyz, core.mv.xyt, core.mv.yzt, core.mv.zxt))            # [4] Trivector+
field_planes = stack((core.mv.xt, core.mv.yt, core.mv.zt, core.mv.yz, core.mv.zx, core.mv.xy))  # [6] Bivector+


def test_vacuum_quartic_and_observer_independence():
    permittivity = np.ones(3)
    reluctivity = np.array(1.0)
    medium = core.dielectric(permittivity, reluctivity)                                # [] Antibivector- <- Bivector+
    wavevectors = core.mv.vector([[2, 0.3, 0.4, -0.2], [1, 0, 0, 1], [0.8, 1.1, -0.2, 0.3]])  # [3] Vector+
    boost = (core.mv.tx * 0.23).exp()                                                  # [] Rotor+
    expected = wavevectors.squared().squared()                                         # [3] Scalar+
    boosted_time = boost >> core.time                                                  # [] Antivector-
    boosted_space = boost >> core.rest_space                                           # [] Trivector+
    for observer_time, observer_space in ((core.time, core.rest_space), (boosted_time, boosted_space)):
        actual = core.fresnel_quartic(medium, observer_time, observer_space, wavevectors)  # [3] Scalar+
        assert ((actual - expected).abs() < 1e-10).all()
    wave = core.potential_map(medium, wavevectors)                                     # [3] Antivector- <- Vector+
    assert ((potentials & wave(wavevectors)[..., None]).abs() < 1e-12).all()


def test_principal_wave_radii_match_transverse_material_weights():
    permittivity = np.array([2.0, 3.0, 4.0])
    reluctivity = np.array(0.7)
    frequency = 1.2
    medium = core.dielectric(permittivity, reluctivity)                                # [] Antibivector- <- Bivector+
    principal_wavefronts = stack((core.mv.x, core.mv.y, core.mv.z))                    # [3] Vector+
    sheets = core.radial_sheets(medium, principal_wavefronts, frequency)               # [3, 2] Vector+
    radii_squared = sheets.squared()                                                   # [3, 2] Scalar+
    expected = core.mv.scalar((frequency**2 * np.array([[3, 4], [2, 4], [2, 3]]) / reluctivity)[..., None])  # [3, 2] Scalar+
    assert ((radii_squared - expected).abs() < 1e-10).all()


def test_both_sheets_support_maxwell_fields_and_transform_with_the_medium():
    permittivity = np.array([[1.0, 1.0, 1.0], [2.25, 2.25, 3.24], [1.44, 2.25, 3.24]])
    n_materials = len(permittivity)
    reluctivity = np.ones(n_materials)
    frequency = 1.0
    n_latitudes, n_longitudes = 5, 9
    media = core.dielectric(permittivity, reluctivity)[:, None, None, None]            # [n_materials, 1, 1, 1] Antibivector- <- Bivector+
    wavefronts = core.sphere(n_latitudes, n_longitudes)                                # [n_latitudes, n_longitudes] Vector+
    sheets = core.radial_sheets(media[..., 0], wavefronts, frequency)                  # [n_materials, n_latitudes, n_longitudes, 2] Vector+
    wavevectors = core.mv.t * frequency + sheets                                       # [n_materials, n_latitudes, n_longitudes, 2] Vector+
    wave = core.potential_map(media, wavevectors)                                      # [n_materials, n_latitudes, n_longitudes, 2] Antivector- <- Vector+
    singular_left, singular_values, null_potentials = wave.svd()
    fields = wavevectors[..., None] ^ null_potentials[..., -2:]                        # [n_materials, n_latitudes, n_longitudes, 2, 2] Bivector+
    source = wavevectors[..., None] ^ media[..., None](fields)                         # [n_materials, n_latitudes, n_longitudes, 2, 2] Antivector-
    residual = core.fresnel_quartic(media, core.time, core.rest_space, wavevectors)    # [n_materials, n_latitudes, n_longitudes, 2] Scalar+
    volumes = wave.outermorphism(core.Trivector)                                       # [n_materials, n_latitudes, n_longitudes, 2] Vector- <- Trivector+

    rotor = (core.mv.tx * 0.19 + core.mv.yz * 0.13).exp()                              # [] Rotor+
    lorentz_media = rotor >> media(rotor << core.Bivector)                             # [n_materials, 1, 1, 1] Antibivector- <- Bivector+
    lorentz_wavevectors = rotor >> wavevectors                                         # [n_materials, n_latitudes, n_longitudes, 2] Vector+
    lorentz_time = rotor >> core.time                                                  # [] Antivector-
    lorentz_space = rotor >> core.rest_space                                           # [] Trivector+
    lorentz_residual = core.fresnel_quartic(lorentz_media, lorentz_time, lorentz_space, lorentz_wavevectors)  # [n_materials, n_latitudes, n_longitudes, 2] Scalar+

    assert (residual.abs() < 1e-9).all()
    assert ((volumes_basis[:, None] & volumes[..., None, None](volumes_basis[None, :])).abs() < 1e-9).all()
    # Repeated radial roots lose half the significant digits to the square root.
    assert ((potentials & source[..., None]).abs() < 1e-5).all()
    assert ((lorentz_residual - residual).abs() < 1e-9).all()
    field_strength = (fields[..., None] & field_planes.dual()).squared().sum(axis=(-2, -1))  # [n_materials, n_latitudes, n_longitudes, 2] Scalar+
    assert (field_strength > 0.1).all()


def test_the_axion_part_leaves_light_alone_and_reverses_in_a_mirror():
    permittivity = np.array([1.44, 2.25, 3.24])
    medium = core.dielectric(permittivity, np.array(1.0))                              # [] Antibivector- <- Bivector+
    axion = core.mv(core.AxionCoupling, [0.3]) * core.Bivector                         # [] Antibivector- <- Bivector+
    principal_wavefronts = stack((core.mv.x, core.mv.y, core.mv.z))                    # [3] Vector+
    wavevectors = core.mv.t + core.radial_sheets(medium, principal_wavefronts, 1.0)    # [3, 2] Vector+
    mirror = core.mv.x                                                                 # [] Vector+

    axion_waves = core.potential_map(axion, wavevectors)                               # [3, 2] Antivector- <- Vector+
    assert ((potentials[:, None] & axion_waves[..., None, None](potentials[None, :])).abs() < 1e-12).all()
    coupled_quartic = core.fresnel_quartic(medium + axion, core.time, core.rest_space, wavevectors)  # [3, 2] Scalar+
    assert (coupled_quartic.abs() < 1e-9).all()
    reflected_axion = mirror >> axion(mirror << core.Bivector)                         # [] Antibivector- <- Bivector+
    assert ((field_planes[:, None] & (reflected_axion + axion)(field_planes[None, :])).abs() < 1e-12).all()


def test_comparison_figure_and_single_frame_animation():
    permittivity = np.array([[1.0, 1.0, 1.0], [1.44, 2.25, 3.24]])
    n_materials = len(permittivity)
    media = core.dielectric(permittivity, np.ones(n_materials))                        # [n_materials] Antibivector- <- Bivector+
    n_latitudes, n_longitudes, n_section_samples = 4, 7, 8
    sheets = core.radial_sheets(media[:, None, None], core.sphere(n_latitudes, n_longitudes), 1.0)  # [n_materials, n_latitudes, n_longitudes, 2] Vector+
    sections = core.radial_sheets(media[:, None], core.circle(n_section_samples), 1.0)  # [n_materials, n_section_samples, 2] Vector+
    figure = render.comparison_figure(sheets, sections)
    plt.close(figure)
    frames = list(render.sheet_animation([(sheets[-1], sections[-1])]))
    assert len(frames) == 1
    assert frames[0].ndim == 3 and frames[0].shape[2] == 3
