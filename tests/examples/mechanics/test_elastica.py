"""The elastica: a strip curled by whole turns closes on its clamp, Euler's elastica reach their far pins at
the chords of the closed form, Kirchhoff's helices end straight above their starts, the moving cord keeps its
energy with no weight and no damping, and the figures draw."""

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.elastica import core, render, scenarios


def test_curled_strip_closes_on_its_clamp():
    curled = scenarios.curling()                                             # [curls, pieces + 1] Motor
    whole = [scenarios.CURLS.index(turns) for turns in (1.0, 2.0)]
    assert np.abs(scenarios.positions(curled[whole, -1] >> core.centre)).max() < 1e-11


def test_euler_elastica_reach_their_far_pins():
    angles = np.array(scenarios.END_ANGLES)
    far = scenarios.positions(scenarios.euler(angles)[:, -1] >> core.centre)   # [angles, 3]
    # The tolerances test the steps along the rod.
    assert np.abs(far[:, :2]).max() < 1e-3
    assert np.abs(far[:, 2] - scenarios.chords(np.radians(angles))).max() < 1e-3


def test_helices_end_above_their_starts():
    tips = scenarios.positions(scenarios.kirchhoff()[:, -1] >> core.centre)   # [tilts, 3]
    rise = scenarios.LENGTH * np.cos(np.radians(scenarios.TILTS))
    # The tolerance tests the steps along the rod.
    assert np.abs(tips - rise[:, None] * [0, 0, 1]).max() < 3e-3


def test_moving_cord_keeps_its_energy():
    """Kicked, with no weight and no damping, the cord's kinetic and elastic energy together stay put: the joint
    loads are exactly the derivative of the joints' energy."""
    pieces = scenarios.N_CORD_COILS * scenarios.N_PER_COIL
    step = scenarios.CORD_LENGTH / pieces
    rest, frames = scenarios.coil(pieces)
    joint = (rest * (step / 2)).exp()
    stiff = core.stiffness(scenarios.CORD_TURNING, scenarios.CORD_SHIFTING)
    inertia = core.segment_inertia(scenarios.CORD_MASS * step, step, scenarios.CORD_SPREAD * step)
    inverse = inertia.inverse()
    rates = core.mv.bivector(np.random.default_rng(0).normal(size=(pieces, 6)) * [0, 0, 0, 0.3, 0.3, 0.3])

    def energy(frames, rates):
        deviations = (joint.reverse() * frames[:-1].reverse() * frames[1:]).log() * 2 / step
        return ((inertia(rates) & rates).sum() + (stiff(deviations) & deviations).sum() * step).to_array() / 2

    start = energy(frames, rates)
    for _ in range(200):
        frames, rates = core.moving(frames, rates, joint, stiff, step, inertia, inverse, core.mv.y.dual() * 0.0, 0.0,
                                    scenarios.TIME_STEP / 4)
    # The energy drifts in proportion to the time step; the tolerance tests the time steps.
    assert abs(energy(frames, rates) / start - 1) < 3e-2


def test_figures_draw():
    coiled = scenarios.kirchhoff()
    edges = scenarios.ribbon(coiled, scenarios.RIBBON)
    for figure in (render.draw_curls(scenarios.curling() >> core.centre),
                   render.draw_plate(scenarios.euler(np.array(scenarios.END_ANGLES)) >> core.centre),
                   render.draw_helices(coiled >> core.centre, edges)):
        plt.close(figure)
    assert len(list(render.animate_sweep(scenarios.euler(scenarios.SWEEP[:1]) >> core.centre))) == 1
    first = next(scenarios.pulled())[None]
    assert len(list(render.animate_cord(first >> core.centre, scenarios.ribbon(first, scenarios.CORD_RIBBON)))) == 1
    uncoiled = scenarios.uncoiling()[:1]
    assert len(list(render.animate_uncoiling(uncoiled >> core.centre, scenarios.ribbon(uncoiled, scenarios.UNCOIL_RIBBON)))) == 1
