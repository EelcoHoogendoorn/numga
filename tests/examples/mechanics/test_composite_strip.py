"""The strip response agrees with full six-mode classical lamination theory and renders cleanly."""

import numpy as np

from examples.mechanics.composite_strip import core, render


def laminate_response(angles: np.ndarray, thickness: float, longitudinal: float, transverse: float,
                      poisson: float, shear: float, tension: float) -> np.ndarray:
    """Integrating transformed ply stiffnesses solves all six membrane and bending modes."""
    denominator = 1 - poisson**2 * transverse / longitudinal
    along, across = longitudinal / denominator, transverse / denominator
    coupling_modulus = poisson * across
    cosine, sine = np.cos(angles), np.sin(angles)
    mixed = cosine**2 * sine**2
    stiffness = np.empty(angles.shape + (3, 3))
    stiffness[..., 0, 0] = along * cosine**4 + across * sine**4 + 2 * (coupling_modulus + 2 * shear) * mixed
    stiffness[..., 1, 1] = along * sine**4 + across * cosine**4 + 2 * (coupling_modulus + 2 * shear) * mixed
    stiffness[..., 0, 1] = stiffness[..., 1, 0] = (
        (along + across - 4 * shear) * mixed + coupling_modulus * (cosine**4 + sine**4)
    )
    stiffness[..., 0, 2] = stiffness[..., 2, 0] = (
        (along - coupling_modulus - 2 * shear) * cosine**3 * sine
        - (across - coupling_modulus - 2 * shear) * cosine * sine**3
    )
    stiffness[..., 1, 2] = stiffness[..., 2, 1] = (
        (along - coupling_modulus - 2 * shear) * cosine * sine**3
        - (across - coupling_modulus - 2 * shear) * cosine**3 * sine
    )
    stiffness[..., 2, 2] = (
        (along + across - 2 * coupling_modulus - 2 * shear) * mixed
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


def test_tension_response_and_strip_frame():
    longitudinal, transverse, poisson, shear = 130000.0, 10000.0, 0.3, 5000.0
    thickness, tension = 1.0, 30.0
    angles = np.deg2rad([[-45, -45, 45, 45], [-45, 45, 45, -45]])
    angles = np.concatenate((angles, angles[:, ::-1]), axis=0)                  # [4, 4]
    n_plies, n_nodes = angles.shape[-1], 2
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    edges = np.linspace(-thickness / 2, thickness / 2, n_plies + 1)
    half_height = thickness / (2 * n_plies)
    heights = core.unit_z * ((edges[:-1] + edges[1:])[:, None] / 2 + nodes * half_height)  # [n_plies, n_nodes] Direction-
    weights = np.broadcast_to(weights * half_height, (n_plies, n_nodes))        # [n_plies, n_nodes]
    ply_rotors = (core.mv.xy * (-angles[..., None] / 2)).exp()                  # [4, n_plies, 1] Rotor+
    fibres = ply_rotors >> core.unit_x                                          # [4, n_plies, 1] Direction-

    state, width_strain = core.tension_response(
        fibres, heights, weights, thickness,
        longitudinal, transverse, poisson, shear, tension,
    )                                                                           # [4] State+, [4] Scalar+
    reference = laminate_response(angles, thickness, longitudinal, transverse, poisson, shear, tension)
    extension = core.extension(state).to_array()
    twist = core.twist(state).to_array() / thickness

    np.testing.assert_allclose(extension, reference[:, 0], atol=1e-12)
    np.testing.assert_allclose(width_strain.to_array(), reference[:, 1], atol=1e-12)
    np.testing.assert_allclose(twist, -reference[:, 5] / 2, atol=1e-12)
    np.testing.assert_allclose(reference[:, 2:5], 0, atol=1e-12)
    assert ((core.extension(state[:2]) - core.extension(state[2:])).abs() < 1e-12).all()
    assert ((core.twist(state[:2]) + core.twist(state[2:])).abs() < 1e-12).all()
    assert (core.twist(state[[1, 3]]).abs() < 1e-12).all()
    assert core.twist(state[0]) > 1e-3

    # Exercise midsurface deformation and a single rendered frame on a tiny 4x3 grid.
    grid = (
        core.unit_x * np.linspace(0.0, 20.0, 4)[:, None]
        + core.unit_y * np.linspace(-5.0, 5.0, 3)[None, :]
    )                                                                           # [4, 3] Direction-
    placement = core.unit_y * np.array([-8.0, 8.0])[:, None, None]              # [2, 1, 1] Direction-
    rest = grid + placement                                                     # [2, 4, 3] Direction-
    peak = core.midsurface_points(grid, state[:2], width_strain[:2], thickness) + placement  # [2, 4, 3] Direction-
    top_normal = ply_rotors[:2, -1, 0] >> core.mv.y                             # [2] Plane+
    across_fibres = top_normal[:, None, None] & grid                            # [2, 4, 3] Scalar+
    frames = core.pull_frames(grid, placement, state[:2], width_strain[:2], thickness, np.array([1.0]))
    frame = next(render.animate_strips(rest, peak, across_fibres, 4.0, frames))
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert max(frame.shape[:2]) == render.n_pixels
    assert frame.dtype == np.uint8
