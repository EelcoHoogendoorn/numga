"""Spinor waves on the sphere: the pulse and the standing waves run on the icosahedron for one step and one
phase, and one frame of each draws."""

from __future__ import annotations

from examples.surfaces.dirac_waves import render, scenarios


def test_the_scenes_run_and_draw():
    topology, points, states = scenarios.pulse(subdivisions=0, n_steps=1)
    vertices, faces = states[0]
    assert len(list(render.animate_pulse(topology, points, vertices[None], faces[None]))) == 1
    topology, points, _, vertices, faces = scenarios.modes(subdivisions=0, n_levels=1, n_phases=1)
    assert len(list(render.animate_modes(topology, points, vertices, faces))) == 1
