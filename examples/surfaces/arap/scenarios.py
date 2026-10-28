"""The bending scene holds a square bar at one end and bends the other a quarter turn up, comparing as-rigid-as-possible deformation against Laplacian editing."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.surfaces.arap import core

mv = core.mv
# Grid divisions per side of the cube the bar is stretched from, and the bar's length along x.
N_DIVISIONS = 10
LENGTH = 6.0
# Handle penalty stiffness, total bend angle, tolerance for selecting end vertices, frame count, and iterations per frame.
STIFFNESS = 1e3
BEND_ANGLE = np.pi / 2
MARGIN = 1e-6
N_FRAMES = 48
N_ITERATIONS = 6
# Checkerboard cell dimensions matching one square of the bar at rest, and frame duration in milliseconds.
CELL = np.array([LENGTH, 2.0, 2.0]) / N_DIVISIONS
DURATION_MS = 40


# --- math -----------------------------------------------------------------------------
def bending():
    """The bar bent across the handle poses by as-rigid-as-possible iteration and by Laplacian editing."""
    bar_mesh, rest_vertices = core.bar(N_DIVISIONS, LENGTH)
    axial_position = mv.x & rest_vertices                                           # Scalar+[vertices]
    near_end = axial_position < -LENGTH / 2 + MARGIN                                # [vertices]
    far_end = axial_position > LENGTH / 2 - MARGIN                                  # [vertices]
    handles = near_end | far_end                                                    # [vertices]
    far_weight = core.as_scalar(far_end).field()                                    # Scalar+[vertices]
    poses = [
        rest_vertices + far_weight * (far_end_pose(rest_vertices, angle) - rest_vertices)  # Point-[vertices]
        for angle in np.linspace(0.0, BEND_ANGLE, N_FRAMES + 1)[1:]
    ]
    frames = list(core.deform(bar_mesh, rest_vertices, handles, poses, STIFFNESS, N_ITERATIONS))
    rigid_frames = stack([rigid_vertices for rigid_vertices, _, _ in frames])       # [frames] Point-[vertices]
    laplacian_frames = stack([laplacian_vertices for _, laplacian_vertices, _ in frames])  # [frames] Point-[vertices]
    return bar_mesh, rest_vertices, rigid_frames, laplacian_frames, handles, poses[-1], frames[-1][2]


# --- plumbing -------------------------------------------------------------------------
def far_end_pose(vertices: core.Point, angle: float) -> core.Point:
    """The vertices carried to the far-end pose on a circular arc of the bar's length turned by `angle` in the xz plane."""
    radius = LENGTH / angle
    turn = (mv.xz * (-angle / 2)).exp()                                             # [] Rotor+
    # The arc bends about a centre above the near end; the far end's centre is the near end's turned about it.
    bend_centre = (mv.x * (-LENGTH / 2) + mv.z * radius).dual()                     # [] Point-
    far_centre = bend_centre + (turn >> (mv.z.dual() * -radius))                    # [] Point-
    return far_centre + (turn >> (vertices - mv.x.dual() * (LENGTH / 2)))           # Point-[vertices]


def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.arap import render

    bar_mesh, rest_vertices, rigid_frames, laplacian_frames, handles, target_pose, last_energies = bending()
    save_animation(render.animate(bar_mesh, rest_vertices, rigid_frames, CELL), "arap_bending", DURATION_MS)

    # checks
    # At the final frame the energy decreases at every iteration and the handles match their target pose.
    assert np.all((last_energies[1:] - last_energies[:-1]) <= 1e-9 * last_energies[0])
    assert np.all((rigid_frames[-1] - target_pose).norm().batch()[handles] < 1e-3)
    # Corner angle cosines change less than half as much as under Laplacian editing.
    angle_changes = [
        (core.corner_cosines(bar_mesh, shape) - core.corner_cosines(bar_mesh, rest_vertices)).abs().sites.mean().mean()
        for shape in (rigid_frames[-1], laplacian_frames[-1])
    ]
    assert angle_changes[0] < 0.5 * angle_changes[1]
    # Left at rest, the bar remains unchanged.
    (rest_result, _, _), = core.deform(bar_mesh, rest_vertices, handles, [rest_vertices], STIFFNESS, 2)
    np.testing.assert_allclose((rest_result - rest_vertices).kernel, 0.0, atol=1e-9)


if __name__ == "__main__":
    main()
