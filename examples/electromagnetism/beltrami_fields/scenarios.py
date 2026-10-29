"""Solid toroidal cavities and the Beltrami fields they hold."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.electromagnetism.beltrami_fields.core import Operators, Point, Scalar, as_scalar, mv, reconstruct, selective_decay

from examples.electromagnetism.beltrami_fields.plumbing import Embedding, Mesh, flux_preserving, solid_torus
from examples.electromagnetism.beltrami_fields.numerical import coexact_helicity_eigh

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
# Gallery picture: this makes plots/beltrami.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
# The gallery's palette: the colours its frames share.
n_gallery_colours = 64


# --- plumbing -------------------------------------------------------------------------
def torus_cavity() -> tuple[Mesh, Point, Embedding]:
    """A reflection-symmetric solid tetrahedral torus around the z axis and its flux-preserving edge subspace."""
    mesh, vertices = solid_torus(major_radius, minor_radius, n_major, n_minor)
    return mesh, vertices, flux_preserving(mesh, n_major, n_minor)


def main() -> None:
    from examples.electromagnetism.beltrami_fields import render
    from examples.util.animation import save_animation, save_figure

    # Curl-curl energy, metric-free helicity, and chiral Beltrami eigenmodes on a torus.
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

    # Selective modal decay with the total helicity held fixed.
    selected_modes = modes[selected_indices]                                 # [selected] Scalar+[edges]
    selected_eigenvalues = 1 / inverse_eigenvalues[selected_indices]         # [selected] Scalar+
    initial_amplitudes = as_scalar(initial_weights)                          # [selected] Scalar+
    times = np.linspace(0.0, relaxation_time, n_frames)
    states = selective_decay(initial_amplitudes, selected_eigenvalues, times)
    magnetic_frames = reconstruct(states, selected_modes, operators.curl)
    relaxation = list(render.animate_relaxation(mesh, vertices, operators.centres, operators.volumes, magnetic_frames))
    save_animation(relaxation, "beltrami_relaxation", frame_duration_ms)
    # Gallery picture: this makes plots/beltrami.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
    save_animation(relaxation, "beltrami", frame_duration_ms, colors=n_gallery_colours)

    # checks
    amplitude_frames = stack(list(selective_decay(initial_amplitudes, selected_eigenvalues, times)))
    energy_frames = amplitude_frames.squared().sum(axis=1) / 2
    helicity_frames = (amplitude_frames.squared() / selected_eigenvalues).sum(axis=1)

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
