"""Clifford action, Kitaev spectra, orthogonal transport and boundary overlap."""

import numpy as np
import pytest

from numga import Algebra, NumpyContext, stack
from examples.quantum.kitaev_wire import core, scenarios

ROUND_OFF = 1e-10


@pytest.fixture(scope="module")
def moving():
    return scenarios.moving()


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
    sites = 12
    hopping = 1.0
    count = 8
    index = np.arange(sites)
    sweet = core.Wire.chain(sites, hopping, hopping)
    generator = sweet.generator(core.mv.scalar(np.zeros((sites, 1))).field())
    left, right = (core.mv.x * (index == 0)).field(), (core.mv.y * (index == sites - 1)).field()
    np.testing.assert_allclose(generator(stack([left, right])).kernel, 0, atol=ROUND_OFF, rtol=0)
    energies, _ = sweet.modes(generator, count)
    expected = np.concatenate([np.zeros(2), np.full(count - 2, 2 * hopping)])
    np.testing.assert_allclose(energies.kernel[:, 0], expected, atol=ROUND_OFF, rtol=0)

    chemical_potential = 0.6
    normal = core.Wire.chain(sites, hopping, 0.0)
    generator = normal.generator(core.mv.scalar(np.full((sites, 1), chemical_potential)).field())
    energies, _ = normal.modes(generator, count)
    excitation = np.abs(-chemical_potential - 2 * hopping * np.cos(np.pi * (index + 1) / (sites + 1)))
    expected = np.repeat(np.sort(excitation), 2)[:count]
    np.testing.assert_allclose(energies.kernel[:, 0], expected, atol=ROUND_OFF, rtol=0)


def test_cayley_evolution_converges_to_the_rotor():
    full = NumpyContext(Algebra("a+b+c+d+")).multivector
    wire = core.Wire.chain(2, 1.0, 0.7)
    potential = core.mv.scalar([[0.6], [1.1]]).field()
    bivector = -0.6 * full.ab - 1.1 * full.cd + 1.7 * full.bc - 0.3 * full.ad
    initial = core.mv.vector([[1.0, 2.0], [3.0, 4.0]]).field() / np.sqrt(30)
    duration = 0.8
    exact = (bivector * (duration / 2)).exp() >> full.vector([1.0, 2.0, 3.0, 4.0]) / np.sqrt(30)
    errors = []
    for steps in (40, 80):
        profiles = potential.broadcast_to((steps,))
        final = tuple(core.transport(wire, initial, profiles, np.array([duration]), steps))[-1]
        np.testing.assert_allclose(final.scalar_norm_squared().sites.sum().kernel, 1,
                                   atol=ROUND_OFF, rtol=0)
        errors.append(np.linalg.norm(final.kernel.reshape(4) - exact.kernel))
    assert errors[1] < 1e-4
    assert 3.9 < errors[0] / errors[1] < 4.1


def test_localization_does_not_depend_on_the_eigensolver_basis():
    wire = core.Wire.chain(scenarios.SITES, scenarios.HOPPING, scenarios.PAIRING)
    potential = core.gate(np.arange(scenarios.SITES), scenarios.LEFT, scenarios.RIGHT,
                          scenarios.INSIDE, scenarios.OUTSIDE, scenarios.WIDTH)
    _, modes = wire.modes(wire.generator(potential), 2)
    localized = wire.localized(modes)
    # An orthogonal mixing of the two modes spans the same space.
    mixing = core.mv.scalar([[[0.6], [-0.8]], [[0.8], [0.6]]])        # [modes, modes] Scalar
    mixed = (mixing * modes[None, :]).sum(axis=-1)
    recovered = wire.localized(mixed)
    np.testing.assert_allclose((localized.scalar_norm_squared() - recovered.scalar_norm_squared()).kernel,
                               0, atol=ROUND_OFF, rtol=0)
    index = core.mv.scalar(np.arange(scenarios.SITES)[:, None]).field()
    centres = (localized.scalar_norm_squared() * index).sites.sum()
    assert centres.kernel[1, 0] - centres.kernel[0, 0] > 50


def test_slow_transport_follows_the_boundary_and_fast_transport_leaks(moving):
    _, _, history, target, share = moving
    np.testing.assert_allclose(history.scalar_norm_squared().sites.sum().kernel,
                               1, atol=ROUND_OFF, rtol=0)
    assert share.kernel[-1, 0, 0] > 0.999
    assert share.kernel[-1, 1, 0] < 0.01
    overlap = history[-1, 0].scalar_product(target[-1]).sites.sum().squared()
    assert overlap.kernel[0] > 0.999


def test_overlap_lifts_the_lowest_excitation():
    energies, _, _ = scenarios.overlap()
    assert energies.kernel[0, 0] > 0.1
    assert energies.kernel[-1, 0] < 2e-7
    assert np.all(np.diff(energies.kernel[:, 0]) < 0)


def test_scenes_draw(moving):
    import matplotlib.pyplot as plt
    from examples.quantum.kitaev_wire import render

    progress, potentials, history, target, share = moving
    figures = [render.draw_formation(*scenarios.formation(), scenarios.HOPPING),
               render.draw_transport(progress, history, potentials, scenarios.HOPPING),
               render.draw_overlap(scenarios.SEPARATIONS, scenarios.PROFILE_SEPARATIONS,
                                    *scenarios.overlap(), scenarios.HOPPING)]
    for figure in figures:
        figure.canvas.draw()
        plt.close(figure)
    frames = render.animate_transport(progress[:2], potentials[:2], history[:2], target[:2], share[:2],
                                       scenarios.HOPPING)
    assert len(frames) == 2
    assert np.any(frames[0] != frames[1])
