"""Magnetic resonance: the generator is the Bloch equations, and the scenarios pass their checks
and draw."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quantum.magnetic_resonance import core, render, scenarios


def test_generator_is_the_bloch_equations():
    """The Bloch vector's rate of change is np.cross(w, r) less the relaxation, with w = (drive, 0, detuning), the
    transverse part decaying at 1 / T2 and the part along the field relaxing to one at 1 / T1."""
    detuning, drive, t1, t2 = 0.7, 1.3, 5.0, 2.0
    rates = core.generator(detuning, drive, t1, t2)                            # [] State- <- State-
    r = np.array([0.4, -0.2, 0.6])
    # A spin's Bloch vector components are its pairings with the planes it turns in, yz, zx and xy.
    planes = stack((core.mv.yz, core.mv.zx, core.mv.xy))                       # [3] Bivector+
    spin = (planes.dual() * r).sum(axis=0)                                     # [] Spin-
    change = (planes & (2 * rates(core.state(spin))).select[2]).to_array()
    expected = np.cross([drive, 0.0, detuning], r) - np.array([r[0] / t2, r[1] / t2, (r[2] - 1) / t1])
    np.testing.assert_allclose(change, expected, atol=1e-12)


def test_scenarios_pass_their_checks_and_draw():
    detunings, drives = np.linspace(-2.0, 2.0, 9), np.array([0.1, 1.0])
    history, settled, _ = scenarios.nutation(np.array([0.0, 0.5]), 1.0, 60.0, 0.05, 20)
    detuned, ensemble = scenarios.echo(40, 2.0, 2.5, 6.0, 0.01, 20, 1)
    for figure in (render.draw_nutation(history, settled),
                   render.draw_lines(detunings, drives, scenarios.lines(detunings, drives)[0]),
                   render.draw_echo(ensemble, 0.2, 2.5, scenarios.T2),
                   render.draw_decays(*scenarios.echo_decay(40, 2.0, 0.01, 8, 1), scenarios.T2)):
        assert isinstance(figure, plt.Figure)
        plt.close(figure)
