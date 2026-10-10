"""The strip agrees with all six free laminate strain and curvature components."""

import numpy as np

from examples.mechanics.composite_strip import core


def laminate_response(angles, thickness, longitudinal, transverse, poisson, shear, tension):
    """Integrate the transformed ply stiffness and solve all membrane and bending modes."""
    denominator = 1 - poisson**2 * transverse / longitudinal
    along, across = longitudinal / denominator, transverse / denominator
    coupling = poisson * across
    cosine, sine = np.cos(angles), np.sin(angles)
    mixed = cosine**2 * sine**2
    stiffness = np.empty(angles.shape + (3, 3))
    stiffness[..., 0, 0] = along * cosine**4 + across * sine**4 + 2 * (coupling + 2 * shear) * mixed
    stiffness[..., 1, 1] = along * sine**4 + across * cosine**4 + 2 * (coupling + 2 * shear) * mixed
    stiffness[..., 0, 1] = stiffness[..., 1, 0] = (
        (along + across - 4 * shear) * mixed + coupling * (cosine**4 + sine**4)
    )
    stiffness[..., 0, 2] = stiffness[..., 2, 0] = (
        (along - coupling - 2 * shear) * cosine**3 * sine
        - (across - coupling - 2 * shear) * cosine * sine**3
    )
    stiffness[..., 1, 2] = stiffness[..., 2, 1] = (
        (along - coupling - 2 * shear) * cosine * sine**3
        - (across - coupling - 2 * shear) * cosine**3 * sine
    )
    stiffness[..., 2, 2] = (
        (along + across - 2 * coupling - 2 * shear) * mixed
        + shear * (cosine**4 + sine**4)
    )
    heights = np.linspace(-thickness / 2, thickness / 2, angles.shape[-1] + 1)
    moments = (
        np.sum(stiffness * (np.diff(heights**power) / power)[..., None, None], axis=-3)
        for power in (1, 2, 3)
    )
    membrane, coupling, bending = moments
    system = np.concatenate((
        np.concatenate((membrane, coupling), axis=-1),
        np.concatenate((coupling, bending), axis=-1),
    ), axis=-2)
    load = np.zeros(system.shape[:-1])
    load[..., 0] = tension
    return np.linalg.solve(system, load[..., None])[..., 0]


def test_tension_response_matches_full_laminate_and_reverses_with_stack():
    longitudinal, transverse, poisson, shear = 130000., 10000., 0.3, 5000.
    thickness, tension = 1., 30.
    angles = np.deg2rad([[-45, -45, 45, 45], [-45, 45, 45, -45]])
    angles = np.concatenate((angles, angles[:, ::-1]), axis=0)
    ply_count, quadrature_count = angles.shape[-1], 2
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_count)
    edges = np.linspace(-thickness / 2, thickness / 2, ply_count + 1)
    half_height = thickness / (2 * ply_count)
    heights = core.mv.z * ((edges[:-1] + edges[1:])[:, None] / 2 + nodes * half_height)
    weights = np.broadcast_to(weights * half_height, (ply_count, quadrature_count))
    turns = -angles[..., None] / 2
    fibres = (core.mv.xy * turns).exp() >> core.mv.x

    state, transverse_strain = core.response(
        fibres, heights, weights, thickness,
        longitudinal, transverse, poisson, shear, tension,
    )
    reference = laminate_response(angles, thickness, longitudinal, transverse, poisson, shear, tension)
    extension = core.extension(state).to_array()
    twist = core.twist(state).to_array() / thickness

    np.testing.assert_allclose(extension, reference[:, 0], atol=1e-12)
    np.testing.assert_allclose(transverse_strain.kernel[..., 0], reference[:, 1], atol=1e-12)
    np.testing.assert_allclose(twist, -reference[:, 5] / 2, atol=1e-12)
    np.testing.assert_allclose(reference[:, 2:5], 0, atol=1e-12)
    np.testing.assert_allclose(extension[:2], extension[2:], atol=1e-12)
    np.testing.assert_allclose(twist[:2], -twist[2:], atol=1e-12)
    np.testing.assert_allclose(twist[[1, 3]], 0, atol=1e-12)
    assert twist[0] > 1e-3
