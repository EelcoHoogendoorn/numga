"""Coverage tests for spinor wave propagation and standing Dirac modes on a triangulated sphere."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.surfaces.dirac_waves import core, render, scenarios


def test_the_scenes_run_and_draw():
    """The icosahedron builds its Dirac operators, steps one pulse frame, and renders static and animated views."""
    topology, vertices, dirac, face_areas, vertex_areas, initial_spinor = scenarios.sphere_bump_setup(0)
    assert ((vertex_areas.diagonal().sites.sum() - face_areas.diagonal().sites.sum()).abs() < 1e-12)

    fig = render.spinor_figure(topology, vertices, initial_spinor, dirac * initial_spinor)
    plt.close(fig)

    pulse_frames = list(
        render.pulse_animation(
            topology,
            vertices,
            core.pulse_states(dirac, face_areas, vertex_areas, initial_spinor, scenarios.dt, 1),
        )
    )
    assert len(pulse_frames) == 1

    squared, vertex_modes, face_modes, _, _ = scenarios.sphere_standing_setup(
        topology, vertices, dirac, face_areas, vertex_areas, 1,
    )
    assert (squared[:scenarios.n_constant].abs() < 1e-9).all()
    mode_frames = list(
        render.modes_animation(
            topology,
            vertices,
            1,
            core.mode_states(vertex_modes, face_modes, np.array([0.0])),
        )
    )
    assert len(mode_frames) == 1
