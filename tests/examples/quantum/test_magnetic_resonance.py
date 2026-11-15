"""Coverage tests for magnetic resonance verify Bloch-equation rates and exercise figure and frame
renderers without writing files."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from numga import stack
from examples.quantum.magnetic_resonance import core, render, scenarios


def test_rate_map_matches_bloch_equations() -> None:
    """The generator's rate of change matches the rotating-frame Bloch cross product minus `t1` and
    `t2` relaxation."""
    detuning, drive, t1, t2 = 0.7, 1.3, 5.0, 2.0
    rates = core.rate_map(detuning, drive, core.magnet_plane, core.drive_plane, core.observer, t1, t2)  # [] State- <- State-
    readings = np.array([0.4, -0.2, 0.6])
    planes = stack((core.mv.yz, core.mv.zx, core.mv.xy))                       # [3] Bivector+
    spin = (planes.dual() * readings).sum(axis=0)                              # [] Spin-
    change = 2 * (planes & rates(core.spin_state(spin)).select.antibivector())
    expected = np.cross([drive, 0.0, detuning], readings) - np.array(
        [readings[0] / t2, readings[1] / t2, (readings[2] - 1) / t1]
    )
    assert ((change - expected).abs() < 1e-12).all()


def test_scene_renderers_produce_figures_and_frames() -> None:
    """Every figure and animation renderer runs on a minimal 2-step trajectory without saving."""
    dt = scenarios.echo_dt
    history, steady_states, _ = scenarios.nutation_run(
        scenarios.nutation_detunings[:1], scenarios.nutation_drive, scenarios.nutation_dt, scenarios.nutation_dt, 1
    )
    plt.close(render.nutation_figure(history, steady_states))
    detunings, drives = scenarios.line_detunings[:2], scenarios.line_drives[:1]
    plt.close(render.lines_figure(detunings, drives, scenarios.spectral_lines(detunings, drives)[0]))
    echo_detunings, echo_iter = scenarios.echo_stream(2, scenarios.spread, dt, 2 * dt, dt, 1, scenarios.seed)
    ensemble = stack(list(echo_iter))                                          # [2, 2] State-
    plt.close(render.echo_figure(ensemble, dt, dt, scenarios.t2))
    frame = next(render.echo_frames(echo_detunings, dt, dt, scenarios.t2, 1, ensemble))
    assert frame.ndim == 3
    plt.close(render.decays_figure(*scenarios.echo_decay_curves(2, scenarios.spread, dt, 2, scenarios.seed), scenarios.t2))
