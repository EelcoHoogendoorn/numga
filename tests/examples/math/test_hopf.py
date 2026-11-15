"""Coverage and algebraic property tests for the Hopf fibration example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.math.hopf import core, render, scenarios


def test_fibre_eigenspace_and_right_turns():
    """The paired form `spin.dot(core.hopf)` has spectrum `-1, -1, 1, 1` and its `+1` eigenspace forms the fibre over `spin`."""
    spin = core.mv.vector([3, -5, 8]).dual().normalized()                      # [] Antivector-
    values, spinors = spin.dot(core.hopf).eigh()                               # [4] Scalar+, [4] Even+
    assert ((values - np.array([-1, -1, 1, 1])).abs() < 1e-12).all()
    top_pair = spinors[-2:]                                                    # [2] Even+
    assert ((core.hopf(top_pair, top_pair) - spin).antinorm() < 1e-12).all()
    right_turns = (core.mv.xy * np.array([0.4, 2.1])).exp()                    # [2] Rotor+
    turned_spinors = spinors[-1] * right_turns                                 # [2] Even+
    assert ((core.hopf(turned_spinors, turned_spinors) - spin).antinorm() < 1e-9).all()


def test_torus_fibres_and_linking_number():
    """Sampled torus fibres map back to their spins and link once under stereographic projection."""
    spins, fibres = scenarios.torus_fibres(np.pi * np.array([0.8, 0.5]), 5, 120)
    assert ((core.hopf(fibres, fibres) - spins[..., None]).antinorm() < 1e-9).all()
    projected_fibres = core.stereographic_projection(fibres)                   # [2, 5, 121] Antivector-
    linking_numbers = core.linking_number(
        projected_fibres[[0, 0, 1], [0, 1, 2]],
        projected_fibres[[1, 0, 1], [3, 4, 0]],
    )                                                                          # [3] Scalar-
    assert ((linking_numbers - core.handedness).abs() < 2e-2).all()


def test_parallel_transport_holonomy():
    """Minimal-angle rotor transport around a closed spin loop shifts the spinor by half the enclosed solid angle."""
    polar = 2.2
    _, carried_spinors, start_fibre = scenarios.lifted_loop(polar, 200)
    half_solid_angle = np.pi * (1 - np.cos(polar))
    expected_end = start_fibre[0] * (core.mv.xy * (-half_solid_angle)).exp()   # [] Even+
    assert ((carried_spinors[-1] - expected_end).norm() < 1e-3).all()


def test_render_figures_and_sweep_frame():
    """Both static figures and one frame of `sweep_animation` render without writing files."""
    spins, fibres = scenarios.torus_fibres(scenarios.torus_polars[:1], 2, 8)
    tori_fig = render.tori_figure(spins, core.stereographic_projection(fibres))
    plt.close(tori_fig)

    loop, carried_spinors, start_fibre = scenarios.lifted_loop(scenarios.loop_polar, 8)
    lift_fig = render.lift_figure(
        loop,
        core.stereographic_projection(carried_spinors),
        core.stereographic_projection(start_fibre),
    )
    plt.close(lift_fig)

    frame = next(render.sweep_animation(core.fibre_sweep(scenarios.spiral_polars, 1, 1, 8)))
    assert frame.ndim == 3 and frame.shape[-1] == 3
