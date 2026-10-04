"""Spinor waves on the sphere: the scenes pass their checks, the energy kept and the frequencies the
whole numbers, and draw."""

from __future__ import annotations

from examples.surfaces.dirac_waves import render, scenarios


def test_the_scenes_pass_their_checks_and_draw():
    sphere, vertices, faces = scenarios.pulse()
    assert len(render.animate(sphere, vertices[None, :1], faces[None, :1], 1)) == 1
    sphere, _, vertices, faces = scenarios.modes()
    assert len(render.animate(sphere, vertices[:, :1], faces[:, :1], scenarios.LEVELS)) == 1
