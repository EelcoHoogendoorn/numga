"""Bending scenario holding one end of a square bar and curving the other end a quarter turn upward, comparing as-rigid-as-possible deformation against Laplacian editing."""

from __future__ import annotations

import numpy as np

from examples.surfaces.arap import core
from numga.topology import Topology

mv = core.mv
n_divisions = 10
bar_length = 6.0
stiffness = 1e3
bend_angle = np.pi / 2
end_margin = 1e-6
n_frames = 48
n_iterations = 6
tile_size = np.array([bar_length, 2.0, 2.0]) / n_divisions
frame_duration_ms = 40


# --- plumbing -------------------------------------------------------------------------
def subdivided_cube(divisions: int) -> tuple[Topology, core.Point]:
    """Subdividing each face of `[-1, 1]^3` into `divisions` by `divisions` squares builds a closed triangulated cube surface (`Topology`, `Point-[V]`)."""
    grid = np.linspace(-1.0, 1.0, divisions + 1)
    first_coord, second_coord = (axis.ravel() for axis in np.meshgrid(grid, grid, indexing="ij"))
    square = np.arange(divisions)[:, None] * (divisions + 1) + np.arange(divisions)[None, :]
    c0, c1, c2, c3 = ((square + offset).ravel() for offset in (0, divisions + 1, divisions + 2, 1))
    side_triangles = np.concatenate([np.stack([c0, c1, c2], -1), np.stack([c0, c2, c3], -1)])
    face_coords, face_triangles = [], []
    for axis in range(3):
        for sign in (1.0, -1.0):
            points = np.zeros((len(first_coord), 3))
            points[:, axis], points[:, (axis + 1) % 3], points[:, (axis + 2) % 3] = sign, first_coord * sign, second_coord
            face_triangles.append(side_triangles + len(face_coords) * len(first_coord))
            face_coords.append(points)
    unique, inverse = np.unique(np.round(np.concatenate(face_coords), 9), axis=0, return_inverse=True)
    return Topology(len(unique), inverse.reshape(-1)[np.concatenate(face_triangles)]), mv.vector(unique).dual().field()


def square_bar(divisions: int, length: float) -> tuple[Topology, core.Point]:
    """Stretching `subdivided_cube(divisions)` along `x` forms a closed square bar of cross-section width `2` and length `length`."""
    topology, corners = subdivided_cube(divisions)
    return topology, corners + (mv.x & corners) * mv.x.dual() * (length / 2 - 1)    # Point-[V]


def far_end_pose(vertices: core.Point, angle: float, length: float) -> core.Point:
    """Carrying `vertices` (`Point-[V]`) along a circular arc of arc-length `length` turned by `angle` in the xz plane places the far-end handle pose."""
    radius = length / angle
    turn = (mv.xz * (-angle / 2)).exp()                                             # [] Rotor+
    bend_centre = (mv.x * (-length / 2) + mv.z * radius).dual()                     # [] Point-
    far_centre = bend_centre + (turn >> (mv.z.dual() * -radius))                    # [] Point-
    return far_centre + (turn >> (vertices - mv.x.dual() * (length / 2)))           # Point-[V]


def bending_setup(divisions: int, n_poses: int):
    """Rest bar (`Topology`, `Point-[V]`), handle mask (`[V]`), and per-frame target handle poses (`list[Point]`) across `n_poses` bend steps."""
    topology, rest_vertices = square_bar(divisions, bar_length)
    axial_position = mv.x & rest_vertices                                           # Scalar+[V]
    near_end = axial_position < -bar_length / 2 + end_margin                        # [V]
    far_end = axial_position > bar_length / 2 - end_margin                          # [V]
    handles = near_end | far_end                                                    # [V]
    far_weight = mv.scalar(far_end[..., None]).field()                              # Scalar+[V]
    angles = np.linspace(0.0, bend_angle, n_poses + 1)[1:]                          # [n_poses]
    poses = [
        rest_vertices + far_weight * (far_end_pose(rest_vertices, angle, bar_length) - rest_vertices)  # Point-[V]
        for angle in angles
    ]
    return topology, rest_vertices, handles, poses


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.surfaces.arap import render

    topology, rest_vertices, handles, poses = bending_setup(n_divisions, n_frames)
    kept = []

    def streaming():
        for state in core.deformation_states(topology, rest_vertices, handles, poses, stiffness, n_iterations):
            kept.append(state)
            yield state

    save_animation(render.bar_animation(topology, rest_vertices, tile_size, streaming()), "arap_bending", frame_duration_ms)
    final_rigid, final_laplacian, last_energies = kept[-1]
    save_figure(render.bar_figure(topology, rest_vertices, [final_rigid, final_laplacian], tile_size), "arap_comparison")

    # checks
    # At the final frame the energy decreases monotonically across iterations and the handles match their target pose.
    assert ((last_energies[1:] - last_energies[:-1]) <= 1e-9 * last_energies[0]).all()
    assert ((final_rigid - poses[-1]).antinorm().batch()[handles] < 1e-3).all()
    # Corner angle cosines distort less than half as much as under Laplacian editing.
    rest_cosines = core.corner_cosines(topology, rest_vertices)                     # [3] Scalar+[F]
    rigid_distortion = (core.corner_cosines(topology, final_rigid) - rest_cosines).abs().sites.mean().mean()
    laplacian_distortion = (core.corner_cosines(topology, final_laplacian) - rest_cosines).abs().sites.mean().mean()
    assert (rigid_distortion < 0.5 * laplacian_distortion).all()


def pinned() -> None:
    """Regenerates the gallery GIF `arap_bending.gif` for the bent square bar."""
    from examples.util.animation import pin_animation
    from examples.surfaces.arap import render

    topology, rest_vertices, handles, poses = bending_setup(n_divisions, n_frames)
    states = core.deformation_states(topology, rest_vertices, handles, poses, stiffness, n_iterations)
    pin_animation(render.bar_animation(topology, rest_vertices, tile_size, states), "arap_bending", frame_duration_ms)


if __name__ == "__main__":
    main()
