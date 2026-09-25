"""The Dirac electron: a spinor's map on spacetime and on bivectors, and the scenarios with their
figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.relativity.dirac import core, render, scenarios


def test_a_spinor_is_rho_times_a_lorentz_map_and_beta_acts_only_on_bivectors():
    """`psi >> v == density * (rotor >> v)` for vectors; on bivectors the spinor's sandwich is its
    frame's extension times `(core.PSEUDOSCALAR * beta).exp() / density`."""
    mv = core.mv
    rotor = ((mv.tx * 0.4 + mv.xy * 0.9 + mv.yz * -0.3) * 0.5).exp()
    psi = core.spinor(2.0, 0.7, rotor)
    v = mv.vector(np.array([0.3, -1.2, 0.5, 2.0]))
    np.testing.assert_allclose(core.frame(psi)(v).kernel, (2.0 * (rotor >> v)).kernel, atol=1e-12)
    B = mv.bivector(np.array([0.2, -0.4, 1.1, 0.3, -0.7, 0.5]))
    np.testing.assert_allclose(core.duality(psi)(B).kernel, ((core.PSEUDOSCALAR * 0.7).exp() / 2.0 * B).kernel, atol=1e-9)


def test_scenarios_pass_their_checks_and_draw():
    momenta, values = scenarios.mass_shell(1.0, 5)
    times, spinors, paths = scenarios.trembling(core.mv(core.Spatial, np.array([0.0, 0.0, 0.3])), 3.0, 60)
    for figure in (render.draw_mass_shell(momenta, values, scenarios.MASS),
                   render.draw_paths(times, spinors, paths, scenarios.MIXTURES)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
