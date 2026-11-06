"""Magnetic resonance: the generator is the Bloch equations."""

from __future__ import annotations

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


def test_the_scenes_draw_at_a_few_steps():
    dt = scenarios.ECHO_DT
    history, settled, _ = scenarios.nutation(scenarios.NUTATION_DETUNINGS[:1], scenarios.NUTATION_DRIVE,
                                             scenarios.NUTATION_DT, scenarios.NUTATION_DT, 1)
    render.draw_nutation(history, settled)
    detunings, drives = scenarios.LINE_DETUNINGS[:2], scenarios.LINE_DRIVES[:1]
    render.draw_lines(detunings, drives, scenarios.lines(detunings, drives)[0])
    _, ensemble = scenarios.echo(2, scenarios.SPREAD, dt, 2 * dt, dt, 1, scenarios.SEED)
    render.draw_echo(ensemble, dt, dt, scenarios.T2)
    render.draw_decays(*scenarios.echo_decay(2, scenarios.SPREAD, dt, 2, scenarios.SEED), scenarios.T2)
