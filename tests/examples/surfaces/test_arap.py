"""Turning the rest bar and its handles turns the deformed bar, and the bending scene satisfies its energy and angle bounds."""

from __future__ import annotations

import numpy as np

from examples.surfaces.arap import core, render, scenarios


def test_turning_the_bar_and_its_handles_turns_the_answer():
    """Rotating the rest mesh and handle poses by a rigid rotor rotates the deformed mesh by the same rotor."""
    bar_mesh, rest_vertices = core.bar(4, 4.0)
    axial_position = core.mv.x & rest_vertices
    handles = (axial_position < -1.999) | (axial_position > 1.999)
    pose = rest_vertices + core.as_scalar(axial_position > 1.999).field() * core.mv.z.dual() * 0.8
    turn = (core.mv.xy * -0.3).exp() * (core.mv.yz * 0.2).exp()
    (deformed_vertices, _, _), = core.deform(bar_mesh, rest_vertices, handles, [pose], 1e3, 4)
    (rotated_vertices, _, _), = core.deform(bar_mesh, turn >> rest_vertices, handles, [turn >> pose], 1e3, 4)
    np.testing.assert_allclose(rotated_vertices.kernel, (turn >> deformed_vertices).kernel, atol=1e-9)


def test_the_bending_scene_passes_its_checks_and_draws():
    """The bending scene decreases energy at every iteration, pins the handles, preserves corner angles, and renders."""
    bar_mesh, rest_vertices, rigid_frames, laplacian_frames, handles, target_pose, last_energies = scenarios.bending()
    # At the final frame the energy decreases at every iteration and the handles match their target pose.
    assert np.all((last_energies[1:] - last_energies[:-1]) <= 1e-9 * last_energies[0])
    assert np.all((rigid_frames[-1] - target_pose).norm().batch()[handles] < 1e-3)
    # Corner angle cosines change less than half as much as under Laplacian editing.
    angle_changes = [
        (core.corner_cosines(bar_mesh, shape) - core.corner_cosines(bar_mesh, rest_vertices)).abs().sites.mean().mean()
        for shape in (rigid_frames[-1], laplacian_frames[-1])
    ]
    assert angle_changes[0] < 0.5 * angle_changes[1]
    images = list(render.animate(bar_mesh, rest_vertices, rigid_frames[-1:], np.array([scenarios.LENGTH, 2.0, 2.0]) / scenarios.N_DIVISIONS))
    assert len(images) == 1
