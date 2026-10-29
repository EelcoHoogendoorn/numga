"""Reference tetrahedra and solid toroidal cavities for 3D Whitney forms and Beltrami fields."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.electromagnetism.beltrami_fields.core import (
    Operators, Point, Scalar, as_scalar,
    edge_heads, edge_tails, edge_whitney_forms, face_first, face_second,
    face_third, face_whitney_forms, mv, reconstruct, selective_decay, tetrahedral_hats, tetrahedral_helicity,
)

from examples.electromagnetism.beltrami_fields.plumbing import Embedding, Mesh, flux_preserving, solid_torus
from examples.electromagnetism.beltrami_fields.numerical import coexact_helicity_eigh

n_tet_subdivisions = 6
major_radius = 1.0
minor_radius = 0.42
n_major = 16
n_minor = 8
shown_indices = np.array([-1, -4])
selected_indices = np.array([-1, -2, -4, -8, 3])
initial_weights = np.array([0.45, 0.70, 0.85, 0.85, 0.45])                    # [selected]
n_frames = 46
relaxation_time = 0.025
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def tetrahedron_samples(corners: Point, n_subdivisions: int) -> Point:
    """Interior barycentric lattice points inside a tetrahedron."""
    steps = np.arange(1, n_subdivisions)
    first, second, third = np.meshgrid(steps, steps, steps, indexing="ij")
    inside = first + second + third < n_subdivisions
    weights = np.stack((
        n_subdivisions - first[inside] - second[inside] - third[inside],
        first[inside],
        second[inside],
        third[inside],
    ), axis=-1) / n_subdivisions                                             # [samples, corners]
    return (corners * weights).sum(axis=-1)                                  # [samples] Point-


def single_tetrahedron(n_subdivisions: int) -> tuple[Point, Point]:
    """A positively oriented scalene tetrahedron and its interior sample points."""
    ex, ey, ez = mv.x.dual(), mv.y.dual(), mv.z.dual()                       # [] Point-, Point-, Point-
    corners = stack((
        -ex * 0.56 - ey * 0.38 - ez * 0.34,
        ex * 0.62 - ey * 0.28 - ez * 0.34,
        -ex * 0.05 + ey * 0.68 - ez * 0.34,
        ex * 0.03 + ey * 0.02 + ez * 0.66,
    ))                                                                       # [corners] Point-
    return corners, tetrahedron_samples(corners, n_subdivisions)


def deformed_tetrahedron() -> Point:
    """A strongly sheared, stretched, and translated tetrahedron with the same orientation."""
    ex, ey, ez = mv.x.dual(), mv.y.dual(), mv.z.dual()                       # [] Point-, Point-, Point-
    return stack((
        ex * 0.40 - ey * 1.10 + ez * 0.25,
        ex * 2.30 - ey * 0.35 - ez * 0.15,
        ex * 0.85 + ey * 1.65 + ez * 0.55,
        ex * 1.15 + ey * 0.20 + ez * 2.45,
    ))                                                                       # [corners] Point-


def torus_cavity() -> tuple[Mesh, Point, Embedding]:
    """A reflection-symmetric solid tetrahedral torus around the z axis and its flux-preserving edge subspace."""
    mesh, vertices = solid_torus(major_radius, minor_radius, n_major, n_minor)
    return mesh, vertices, flux_preserving(mesh, n_major, n_minor)


def main() -> None:
    from examples.electromagnetism.beltrami_fields import render
    from examples.util.animation import save_animation, save_figure

    # 1. Tetrahedral Whitney 0-, 1-, and 2-forms and topological opposite-edge helicity.
    corners, points = single_tetrahedron(n_tet_subdivisions)
    hats, gradients, face_patches, volume = tetrahedral_hats(corners, points)
    edge_forms, edge_curls = edge_whitney_forms(corners, points)
    face_forms = face_whitney_forms(corners, points)
    save_figure(
        render.draw_tet_whitney(corners, points, edge_forms, face_forms, volume),
        "beltrami_tet_whitney",
    )

    # 2. Curl-curl energy, metric-free helicity, and chiral Beltrami eigenmodes on a torus.
    mesh, vertices, free = torus_cavity()
    operators = Operators(mesh, vertices)
    free_energy = free.adjugate()(operators.stiffness(free))
    free_helicity = free.adjugate()(operators.helicity(free))
    inverse_eigenvalues, free_modes, n_null = coexact_helicity_eigh(free_energy, free_helicity)
    modes = free(free_modes)
    shown_modes = modes[shown_indices]                                       # [shown] Scalar+[edges]
    shown_magnetic = operators.curl(shown_modes)                             # [shown] Field+[tets]
    save_figure(
        render.draw_beltrami_eigenmodes(
            mesh, vertices, operators.centres, operators.volumes, inverse_eigenvalues, shown_magnetic, shown_indices,
        ),
        "beltrami_eigenmodes",
    )

    # 3. Selective modal decay with the total helicity held fixed.
    selected_modes = modes[selected_indices]                                 # [selected] Scalar+[edges]
    selected_eigenvalues = 1 / inverse_eigenvalues[selected_indices]         # [selected] Scalar+
    initial_amplitudes = as_scalar(initial_weights)                          # [selected] Scalar+
    times = np.linspace(0.0, relaxation_time, n_frames)
    states = selective_decay(initial_amplitudes, selected_eigenvalues, times)
    magnetic_frames = reconstruct(states, selected_modes, operators.curl)
    save_animation(
        render.animate_relaxation(mesh, vertices, operators.centres, operators.volumes, magnetic_frames),
        "beltrami_relaxation", frame_duration_ms,
    )

    # checks
    amplitude_frames = stack(list(selective_decay(initial_amplitudes, selected_eigenvalues, times)))
    energy_frames = amplitude_frames.squared().sum(axis=1) / 2
    helicity_frames = (amplitude_frames.squared() / selected_eigenvalues).sum(axis=1)
    corner_hats, _, _, _ = tetrahedral_hats(corners, corners)
    np.testing.assert_allclose(corner_hats.kernel, np.eye(4)[..., None], atol=1e-13)
    np.testing.assert_allclose(hats.sum(axis=0).kernel, 1.0, atol=1e-13)
    np.testing.assert_allclose(gradients.sum(axis=0).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(face_patches.sum(axis=0).kernel, 0.0, atol=1e-14)

    classic_1forms = hats[edge_tails] * gradients[edge_heads, None] - hats[edge_heads] * gradients[edge_tails, None]
    classic_curls = 2 * (gradients[edge_tails] ^ gradients[edge_heads])
    np.testing.assert_allclose((edge_forms - classic_1forms).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose((edge_curls - classic_curls).kernel, 0.0, atol=1e-13)

    edge_vectors = corners[edge_heads] - corners[edge_tails]
    edge_midpoints = (corners[edge_tails] + corners[edge_heads]) / 2
    midpoint_1forms, _ = edge_whitney_forms(corners, edge_midpoints)
    np.testing.assert_allclose((midpoint_1forms & edge_vectors[None, :]).kernel, np.eye(6)[..., None], atol=1e-13)

    li, lj, ll = hats[face_first], hats[face_second], hats[face_third]
    gi, gj, gl = gradients[face_first, None], gradients[face_second, None], gradients[face_third, None]
    classic_2forms = 2 * (li * (gj ^ gl) + lj * (gl ^ gi) + ll * (gi ^ gj))
    np.testing.assert_allclose((face_forms - classic_2forms).kernel, 0.0, atol=1e-13)
    face_centres = (corners[face_first] + corners[face_second] + corners[face_third]) / 3
    np.testing.assert_allclose((face_whitney_forms(corners, face_centres) & face_patches[None, :]).kernel,
                               np.eye(4)[..., None], atol=1e-14)

    tet_helicity, _ = tetrahedral_helicity(corners)
    deformed_helicity, _ = tetrahedral_helicity(deformed_tetrahedron())
    np.testing.assert_allclose((tet_helicity - deformed_helicity).kernel, 0.0, atol=1e-14)
    expected_linking = np.zeros((6, 6))
    expected_linking[np.arange(6), 5 - np.arange(6)] = np.array([1, -1, 1, 1, -1, 1]) / 6
    np.testing.assert_allclose(tet_helicity.kernel[..., 0], expected_linking, atol=1e-14)

    # Whitney reconstruction commutes with the exterior derivative: the edge forms of a potential's differences are
    # its gradient, and the curl of an edge field is the face forms of its circulations.
    d0, d1 = mesh.d[0] * Scalar, mesh.d[1] * Scalar
    test_phi = (mv.x & vertices) * 0.6 - (mv.y & vertices) * 0.9 + (mv.z & vertices) * 0.4
    np.testing.assert_allclose((operators.whitney1(d0(test_phi)) - operators.gradient(test_phi)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((operators.curl(shown_modes[0]) - operators.whitney2(d1(shown_modes[0]))).kernel,
                               0.0, atol=1e-12)

    assert n_null == n_major
    shown_eigenvalues = 1 / inverse_eigenvalues[shown_indices]
    shown_energies = (operators.stiffness(shown_modes) & shown_modes).sites.sum() / 2
    shown_helicities = (operators.helicity(shown_modes) & shown_modes).sites.sum()
    shown_residuals = free.adjugate()(operators.stiffness(shown_modes) - shown_eigenvalues * operators.helicity(shown_modes))
    np.testing.assert_allclose((inverse_eigenvalues + inverse_eigenvalues[::-1]).kernel, 0.0, atol=1e-13)
    np.testing.assert_allclose(shown_energies.kernel[..., 0], 0.5, atol=1e-12)
    np.testing.assert_allclose((shown_helicities - inverse_eigenvalues[shown_indices]).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(shown_residuals.kernel, 0.0, atol=1e-11)

    np.testing.assert_allclose((helicity_frames - helicity_frames[0]).kernel, 0.0, atol=1e-12)
    assert np.all(np.diff(energy_frames.kernel[:, 0]) < 0)
    taylor_bound = (helicity_frames[0] * selected_eigenvalues[0] / 2).kernel[0]
    assert energy_frames[-1].kernel[0] > taylor_bound
    assert abs(energy_frames[-1].kernel[0] - taylor_bound) / taylor_bound < 0.02


if __name__ == "__main__":
    main()
