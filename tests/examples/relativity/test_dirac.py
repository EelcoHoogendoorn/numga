"""The Dirac electron: a spinor's map on spacetime and on bivectors, and the scenarios with their
figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.dirac import core, render, scenarios


def test_a_spinor_is_rho_times_a_lorentz_map_and_beta_acts_only_on_bivectors():
    """`psi >> v == density * (rotor >> v)` for vectors; on bivectors the spinor's sandwich is its
    frame's extension times `(core.pseudoscalar * beta).exp() / density`."""
    mv = core.mv
    rotor = ((mv.tx * 0.4 + mv.xy * 0.9 + mv.yz * -0.3) * 0.5).exp()
    psi = core.spinor(2.0, 0.7, rotor)
    v = mv.vector(np.array([0.3, -1.2, 0.5, 2.0]))
    np.testing.assert_allclose(core.frame(psi)(v).kernel, (2.0 * (rotor >> v)).kernel, atol=1e-12)
    B = mv.bivector(np.array([0.2, -0.4, 1.1, 0.3, -0.7, 0.5]))
    np.testing.assert_allclose(core.duality(psi)(B).kernel, ((core.pseudoscalar * 0.7).exp() / 2.0 * B).kernel, atol=1e-9)


def test_scenarios_pass_their_checks_and_draw():
    times, spinors, paths, _ = scenarios.trembling(core.mv(core.Spatial, np.array([0.0, 0.0, 0.3])), 3.0, 60)
    for figure in (render.draw_paths(times, spinors, paths, scenarios.mixtures),):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)


def test_plane_wave_energies_are_fourfold_with_beta_telling_their_sign():
    """Each momentum has the energies minus and plus E, each fourfold; the positive-energy states have
    beta zero and the negative-energy states beta a half turn, where the invariants' scalar part is negative."""
    along = np.linspace(-1.0, 1.0, 5)
    px, py = np.meshgrid(along, along)
    momenta = core.mv(core.Spatial, np.stack([px, py, np.zeros_like(px)], axis=-1))
    values, states = core.hamiltonian(momenta, scenarios.mass).eigh()
    energy = core.energy(momenta, scenarios.mass).to_array()[..., None]
    np.testing.assert_allclose(values.to_array(), np.concatenate([-energy.repeat(4, -1), energy.repeat(4, -1)], -1), atol=1e-12)
    scalar = core.invariants(states).select[0].to_array()
    np.testing.assert_allclose(np.sign(scalar), np.concatenate([-np.ones(4), np.ones(4)]) * np.ones_like(scalar))
