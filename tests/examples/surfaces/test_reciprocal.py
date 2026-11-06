"""A lifted planar mesh and its reciprocal: one frame of the breathing grid's stresses and slopes runs and draws."""

from examples.surfaces.reciprocal import core, render, scenarios


def test_one_breath_runs_and_draws():
    topology, points, breaths = scenarios.breathing(n_frames=1)
    frames = [(breath, core.stresses(topology, points, breath), (core.slopes(topology, points, breath) ^ core.mv.z) * core.mv.z)
              for breath in breaths]
    assert len(list(render.animate(topology, points, frames))) == 1
