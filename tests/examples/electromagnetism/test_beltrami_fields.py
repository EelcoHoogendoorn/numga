"""Tests verify Whitney operators, chiral Beltrami modes, selective decay, and toroidal field-line rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.electromagnetism.beltrami_fields import core, numerical, plumbing, render

major_radius = 1.0
minor_radius = 0.42
n_major = 6
n_minor = 4
shown_indices = np.array([-1, -3])                                              # [2]
selected_indices = np.array([-1, -2, -3, 2])                                    # [4]
n_selected = len(selected_indices)
initial_weights = np.array([0.45, 0.75, 0.80, 0.45])                            # [n_selected]
decay_times = np.linspace(0.0, 0.004, 4)                                        # [4]


def test_whitney_operators_flux_preserving_subspace_and_beltrami_spectrum():
    """Whitney 1-forms, curls, and energy-helicity forms produce chiral Beltrami eigenmodes on a solid torus."""
    topology, vertices = plumbing.solid_torus(major_radius, minor_radius, n_major, n_minor)
    free = plumbing.flux_preserving(topology, n_minor)
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes = core.simplex_volumes(topology, vertices)                          # Volume-[T]
    gradients = core.simplex_gradients(topology, vertices, volumes)             # [4] Plane+[T]
    potential_map = core.whitney_1form(topology, gradients)                     # (Plane+ <- Scalar+)[T <- E]
    curl = core.whitney_curl(topology, gradients)                               # (Field+ <- Scalar+)[T <- E]
    stiffness = core.magnetic_energy(volumes, curl)                             # (Antiscalar- <- Scalar+)[E <- E]
    helicity = core.magnetic_helicity(volumes, potential_map, curl)             # (Antiscalar- <- Scalar+)[E <- E]

    assert potential_map.gatype == core.OneForm
    assert curl.gatype == core.Curl
    assert stiffness.gatype == core.Form
    assert helicity.gatype == core.Form
    unit_scalar = core.mv.scalar([1])
    assert ((stiffness - stiffness.adjugate()).cells(unit_scalar).dual_inverse().norm() < 1e-11).all()
    assert ((helicity - helicity.adjugate()).cells(unit_scalar).dual_inverse().norm() < 1e-12).all()

    # Whitney 1-forms reproduce exact simplex gradients and vanish under the Whitney curl.
    potential = (core.mv.x & vertices) * 0.7 - (core.mv.y & vertices) * 1.1 + (core.mv.z & vertices) * 0.5
    exact_gradient = (gradients * (topology.incidence[3, 0] * potential)).sum(axis=0)
    assert ((potential_map(topology.d[0] * potential) - exact_gradient).norm() < 1e-11).all()
    assert (curl(topology.d[0] * potential).norm() < 1e-11).all()

    n_free = free.shape[1]
    free_test = free * core.mv.scalar(np.sin(np.arange(n_free))[:, None]).field()
    assert ((topology.reading(topology.exterior[2], 2) * (topology.d[1] * free_test)).norm() < 1e-12).all()

    # An interior vertex gradient lies in the nullspace of both magnetic energy and free-subspace helicity.
    V, E, F, T = topology.n_elements
    interior_potential = core.mv.scalar(np.isin(np.arange(V), topology.interior[0]).astype(float)[:, None]).field()
    gauge_edge = topology.d[0] * interior_potential
    assert (stiffness(gauge_edge).dual_inverse().norm() < 1e-11).all()
    assert ((free.adjugate() * helicity(gauge_edge)).dual_inverse().norm() < 1e-12).all()

    inverse_eigenvalues, free_modes, n_null = numerical.coexact_helicity_eigh(
        free.adjugate() * stiffness * free,
        free.adjugate() * helicity * free,
    )
    modes = free * free_modes                                                   # [...] Scalar+[E]
    assert n_null == n_major
    assert ((inverse_eigenvalues + inverse_eigenvalues[::-1]).norm() < 1e-12).all()

    check_indices = np.array([0, 1, -2, -1])                                    # [4]
    check_modes = modes[check_indices]                                          # [4] Scalar+[E]
    check_eigenvalues = 1 / inverse_eigenvalues[check_indices]                  # [4] Scalar+
    magnetic = curl(check_modes)                                                # [4] Field+[T]
    energies = (stiffness(check_modes) & check_modes).sites.sum() / 2
    helicities = (helicity(check_modes) & check_modes).sites.sum()
    residuals = free.adjugate() * (stiffness(check_modes) - check_eigenvalues * helicity(check_modes))
    assert magnetic.batch().gatype == core.Field
    assert ((energies - 0.5).norm() < 1e-11).all()
    assert ((helicities - inverse_eigenvalues[check_indices]).norm() < 1e-11).all()
    assert (residuals.dual_inverse().norm().sites.sum() < 1e-10).all()

    shown_fields = curl(modes[shown_indices])                                   # [2] Field+[T]
    figure = render.eigenmodes_figure(
        topology, vertices, centres, volumes, inverse_eigenvalues, shown_fields, shown_indices,
    )
    assert len(figure.axes) == 3
    plt.close(figure)


def test_selective_decay_and_relaxation_animation():
    """Selective modal decay conserves helicity, lowers energy toward the Taylor bound, and renders a frame."""
    topology, vertices = plumbing.solid_torus(major_radius, minor_radius, n_major, n_minor)
    free = plumbing.flux_preserving(topology, n_minor)
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes = core.simplex_volumes(topology, vertices)                          # Volume-[T]
    gradients = core.simplex_gradients(topology, vertices, volumes)             # [4] Plane+[T]
    potential_map = core.whitney_1form(topology, gradients)                     # (Plane+ <- Scalar+)[T <- E]
    curl = core.whitney_curl(topology, gradients)                               # (Field+ <- Scalar+)[T <- E]
    stiffness = core.magnetic_energy(volumes, curl)                             # (Antiscalar- <- Scalar+)[E <- E]
    helicity = core.magnetic_helicity(volumes, potential_map, curl)             # (Antiscalar- <- Scalar+)[E <- E]

    inverse_eigenvalues, free_modes, _ = numerical.coexact_helicity_eigh(
        free.adjugate() * stiffness * free,
        free.adjugate() * helicity * free,
    )
    modes = free * free_modes                                                   # [...] Scalar+[E]
    selected_modes = modes[selected_indices]                                    # [n_selected] Scalar+[E]
    selected_eigenvalues = 1 / inverse_eigenvalues[selected_indices]            # [n_selected] Scalar+
    initial_amplitudes = core.mv.scalar(initial_weights[:, None])               # [n_selected] Scalar+

    states = list(core.relaxation_states(initial_amplitudes, selected_eigenvalues, selected_modes, curl, decay_times))
    for amplitudes, field in states:
        direct_field = curl((amplitudes * selected_modes).sum(axis=0))
        assert ((field - direct_field).norm() < 1e-11).all()

    amplitude_frames = stack([amplitudes for amplitudes, _ in states])
    energy_frames = amplitude_frames.squared().sum(axis=1) / 2
    helicity_frames = (amplitude_frames.squared() / selected_eigenvalues).sum(axis=1)
    assert ((helicity_frames - helicity_frames[0]).norm() < 1e-11).all()
    assert (energy_frames[1:] < energy_frames[:-1]).all()

    taylor_bound = helicity_frames[0] * selected_eigenvalues[0] / 2
    assert energy_frames[0] > 1.2 * taylor_bound
    assert energy_frames[-1] > taylor_bound
    assert (energy_frames[-1] - taylor_bound) / taylor_bound < 0.05

    final_cochain = (amplitude_frames[-1] * selected_modes).sum(axis=0)
    direct_energy = (stiffness(final_cochain) & final_cochain).sites.sum() / 2
    direct_helicity = (helicity(final_cochain) & final_cochain).sites.sum()
    assert ((direct_energy - energy_frames[-1]).norm() < 1e-11).all()
    assert ((direct_helicity - helicity_frames[0]).norm() < 1e-11).all()

    frame = next(render.relaxation_animation(topology, vertices, centres, volumes, iter(states[:1])))
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
