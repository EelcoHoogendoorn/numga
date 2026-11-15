"""Clifford bivector action, Kitaev spectra, localized end modes, and scene rendering."""

import matplotlib.pyplot as plt
import numpy as np

from numga import Algebra, NumpyContext, stack
from examples.quantum.kitaev_wire import core, render, scenarios

ROUND_OFF = 1e-10


def test_sparse_generator_matches_clifford_bivector_action():
    full = NumpyContext(Algebra("a+b+c+d+")).multivector
    potentials = np.array([0.6, 1.1])                           # [2]
    hopping, pairing = 1.0, 0.7
    couplings = core.chain_couplings(len(potentials), hopping, pairing)
    generator = core.wire_generator(couplings, core.mv.scalar(potentials[:, None]).field())
    planes = stack([full.ab, full.cd, full.bc, full.ad])
    strengths = np.array([-potentials[0], -potentials[1], hopping + pairing, pairing - hopping])
    bivector = (strengths * planes).sum(axis=0)
    fields = core.mv.vector(np.eye(4).reshape(4, 2, 2)).field()
    vectors = full.vector(np.eye(4))
    expected = bivector.commutator(vectors)
    np.testing.assert_allclose(generator(fields).kernel.reshape(4, 4), expected.kernel,
                               atol=ROUND_OFF, rtol=0)
    assert ((generator + generator.adjoint())(fields).scalar_norm_squared().sites.sum() < ROUND_OFF**2).all()


def test_decoupled_majoranas_and_normal_chain_spectrum():
    n_sites = 12
    hopping = 1.0
    n_modes = 8
    index = np.arange(n_sites)                                  # [n_sites]
    sweet_couplings = core.chain_couplings(n_sites, hopping, hopping)
    sweet_generator = core.wire_generator(sweet_couplings, core.mv.scalar(np.zeros((n_sites, 1))).field())
    left = (core.mv.x * (index == 0)).field()
    right = (core.mv.y * (index == n_sites - 1)).field()
    assert (sweet_generator(stack([left, right])).scalar_norm_squared().sites.sum() < ROUND_OFF**2).all()
    energies, _ = core.excitation_modes(sweet_generator, n_modes)
    expected = core.mv.scalar(np.concatenate([np.zeros(2), np.full(n_modes - 2, 2 * hopping)])[:, None])
    assert ((energies - expected).abs() < ROUND_OFF).all()

    chemical_potential = 0.6
    normal_couplings = core.chain_couplings(n_sites, hopping, 0.0)
    normal_generator = core.wire_generator(normal_couplings, core.mv.scalar(np.full((n_sites, 1), chemical_potential)).field())
    energies, _ = core.excitation_modes(normal_generator, n_modes)
    excitation = np.abs(-chemical_potential - 2 * hopping * np.cos(np.pi * (index + 1) / (n_sites + 1)))
    expected_normal = core.mv.scalar(np.repeat(np.sort(excitation), 2)[:n_modes, None])
    assert ((energies - expected_normal).abs() < ROUND_OFF).all()


def test_localization_is_independent_of_eigensolver_basis():
    potential = core.gate_potential(
        scenarios.sites, np.array(scenarios.left_end), np.array(scenarios.right_end),
        np.array(scenarios.inside_potential), scenarios.outside_potential, scenarios.edge_width,
    )
    _, modes = core.excitation_modes(core.wire_generator(scenarios.couplings, potential), 2)
    localized = core.localized_modes(modes, scenarios.site_positions)
    # An orthogonal mixing of the two modes spans the same space.
    mixing = core.mv.scalar([[[0.6], [-0.8]], [[0.8], [0.6]]])   # [2, 2] Scalar+
    mixed = (mixing * modes[None, :]).sum(axis=-1)
    recovered = core.localized_modes(mixed, scenarios.site_positions)
    assert ((localized.scalar_norm_squared() - recovered.scalar_norm_squared()).abs() < ROUND_OFF).all()
    centres = (localized.scalar_norm_squared() * scenarios.site_positions).sites.sum()
    assert centres[1] - centres[0] > 50


def test_scenes_render_at_minimal_samples():
    fig_formation = render.formation_figure(*scenarios.formation_sweep(2), scenarios.hopping)
    plt.close(fig_formation)

    progress, potentials, _, target, states = scenarios.transport_schedule(scenarios.steps_per_frame)
    recorded = list(states)
    frames_transport = render.transport_animation(progress, potentials, target, scenarios.hopping, recorded)
    next(frames_transport)
    frames_transport.close()

    history = stack([state for state, _ in recorded])
    fig_transport = render.transport_figure(progress, history, potentials, scenarios.hopping)
    plt.close(fig_transport)

    separations = scenarios.separations[:2]
    frames_overlap = render.overlap_animation(separations, *scenarios.overlap_sweep(separations), scenarios.hopping)
    next(frames_overlap)
    frames_overlap.close()
