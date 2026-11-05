"""Clifford action, Kitaev spectra and the localized end modes, and the scenes at a few samples."""

import numpy as np

from numga import Algebra, NumpyContext, stack
from examples.quantum.kitaev_wire import core, render, scenarios

ROUND_OFF = 1e-10


def test_sparse_generator_is_the_clifford_bivector_action():
    full = NumpyContext(Algebra("a+b+c+d+")).multivector
    potentials = np.array([0.6, 1.1])
    hopping, pairing = 1.0, 0.7
    wire = core.Wire.chain(len(potentials), hopping, pairing)
    generator = wire.generator(core.mv.scalar(potentials[:, None]).field())
    planes = stack([full.ab, full.cd, full.bc, full.ad])
    strengths = np.array([-potentials[0], -potentials[1], hopping + pairing, pairing - hopping])
    bivector = (strengths * planes).sum(axis=0)
    fields = core.mv.vector(np.eye(4).reshape(4, 2, 2)).field()
    vectors = full.vector(np.eye(4))
    expected = bivector.commutator(vectors)
    np.testing.assert_allclose(generator(fields).kernel.reshape(4, 4), expected.kernel,
                               atol=ROUND_OFF, rtol=0)
    np.testing.assert_allclose((generator + generator.adjoint())(fields).kernel,
                               0, atol=ROUND_OFF, rtol=0)


def test_decoupled_majoranas_and_normal_chain_spectrum():
    n_sites = 12
    hopping = 1.0
    n_modes = 8
    index = np.arange(n_sites)
    sweet = core.Wire.chain(n_sites, hopping, hopping)
    generator = sweet.generator(core.mv.scalar(np.zeros((n_sites, 1))).field())
    left, right = (core.mv.x * (index == 0)).field(), (core.mv.y * (index == n_sites - 1)).field()
    np.testing.assert_allclose(generator(stack([left, right])).kernel, 0, atol=ROUND_OFF, rtol=0)
    energies, _ = sweet.modes(generator, n_modes)
    expected = np.concatenate([np.zeros(2), np.full(n_modes - 2, 2 * hopping)])
    np.testing.assert_allclose(energies.kernel[:, 0], expected, atol=ROUND_OFF, rtol=0)

    chemical_potential = 0.6
    normal = core.Wire.chain(n_sites, hopping, 0.0)
    generator = normal.generator(core.mv.scalar(np.full((n_sites, 1), chemical_potential)).field())
    energies, _ = normal.modes(generator, n_modes)
    excitation = np.abs(-chemical_potential - 2 * hopping * np.cos(np.pi * (index + 1) / (n_sites + 1)))
    expected = np.repeat(np.sort(excitation), 2)[:n_modes]
    np.testing.assert_allclose(energies.kernel[:, 0], expected, atol=ROUND_OFF, rtol=0)


def test_localization_does_not_depend_on_the_eigensolver_basis():
    wire = scenarios.wire
    potential = core.gate(scenarios.SITES, scenarios.LEFT, scenarios.RIGHT,
                          scenarios.INSIDE, scenarios.OUTSIDE, scenarios.WIDTH)
    _, modes = wire.modes(wire.generator(potential), 2)
    localized = wire.localized(modes)
    # An orthogonal mixing of the two modes spans the same space.
    mixing = core.mv.scalar([[[0.6], [-0.8]], [[0.8], [0.6]]])        # [modes, modes] Scalar+
    mixed = (mixing * modes[None, :]).sum(axis=-1)
    recovered = wire.localized(mixed)
    np.testing.assert_allclose((localized.scalar_norm_squared() - recovered.scalar_norm_squared()).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    index = core.mv.scalar(scenarios.SITES[:, None]).field()
    centres = (localized.scalar_norm_squared() * index).sites.sum()
    assert centres.kernel[1, 0] - centres.kernel[0, 0] > 50


def test_the_scenes_draw_at_a_few_samples():
    render.draw_formation(*scenarios.formation(n_samples=2), scenarios.HOPPING)
    progress, potentials, history, target, share = scenarios.moving(n_steps=scenarios.STEPS_PER_FRAME)
    next(render.animate_transport(progress, potentials, history, target, share, scenarios.HOPPING))
    render.draw_transport(progress, history, potentials, scenarios.HOPPING)
    separations = scenarios.SEPARATIONS[:2]
    next(render.animate_overlap(separations, *scenarios.overlap(separations), scenarios.HOPPING))
