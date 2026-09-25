"""Scenes for the scenegraph, robot kinematics, and compound optics example.

`scenegraph` returns one posed frame for the figures; `robot_sweep` yields the frames of
the kinematic animation. Both return geometry for `render`.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.geometry.scenegraph.core import (
    canonical_unit_box, lens_camera, lens_train, look_at, mv, origin, project_vertices,
    robot_arm, trace_rays, viewport,
)


def camera_rig():
    """The compound camera: pose, objective and relay lens, pupil, sensor, and the world-to-pixel map."""
    pose = look_at(np.array([0.0, -3.6, 1.3]), np.array([0.0, 0.0, 1.18]))

    # Objective lens L1 at the origin and relay lens L2 0.3 behind it:
    front_lens, rear_lens, rear_plane = lens_train(1.0, 0.8, -0.3)
    # Pupil point on the entrance pupil aperture (offset from optical center to induce bending):
    pupil = (mv.xw * 0.02).exp() >> origin
    # The sensor plane, where z == -1.25:
    sensor_plane = mv.z + 1.25 * mv.w
    camera = lens_camera(pose, front_lens, rear_lens, pupil, sensor_plane)

    # Viewport: the physical 1.6 x 1.2 sensor chip onto 640 x 480 pixels.
    world_to_pixel = viewport(640, 480, 1.6, 1.2)(camera)
    return pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel


def scenegraph():
    """The robot arm posed once, photographed through the compound camera."""
    unit_box = canonical_unit_box()
    pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel = camera_rig()

    # Articulated robot arm forward kinematics:
    bodies_to_world, _ = robot_arm((0.35, -0.45, 0.85, -0.40))

    # Collapse the entire visual pipeline, kinematics to pixels, into a single extensor
    # per body, and project all canonical vertices in one pass.
    local_to_pixel = world_to_pixel(bodies_to_world)                 # [bodies] Point <- Point
    projected_pixels = project_vertices(local_to_pixel, unit_box)     # [bodies, vertices] Point
    world_vertices = bodies_to_world[:, None](unit_box[None, :])      # [bodies, vertices] Point

    # Three gripper corners traced through the lenses to the sensor:
    gripper = world_vertices[-1]
    rays = trace_rays(stack([gripper[7], gripper[6], gripper[2]]), pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane)

    # --- checks
    # The collapsed extensor agrees with applying kinematics, camera and viewport in turn.
    sequential = world_to_pixel(world_vertices)
    sequential = sequential / (mv.w & sequential)
    mismatch = (sequential - projected_pixels).dual().norm_squared()
    assert mismatch.to_array().max() < 1e-16

    return world_vertices, projected_pixels, pose, rays


def robot_sweep(num_frames: int):
    """Frames of a looping joint trajectory, each photographed through the compound camera."""
    unit_box = canonical_unit_box()
    pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel = camera_rig()
    for t in np.linspace(0, 2 * np.pi, num_frames, endpoint=False):
        joint_angles = (
            0.45 * np.sin(t),
            -0.40 + 0.25 * np.cos(t),
            0.85 + 0.35 * np.sin(t),
            -0.45 + 0.25 * np.cos(2 * t),
        )
        bodies_to_world, _ = robot_arm(joint_angles)
        projected = project_vertices(world_to_pixel(bodies_to_world), unit_box)
        world_vertices = bodies_to_world[:, None](unit_box[None, :])
        # The ray from the gripper tip:
        rays = trace_rays(world_vertices[-1, 7:8], pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane)
        yield world_vertices, projected, pose, rays


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.geometry.scenegraph import render

    world_vertices, projected_pixels, camera_pose, rays = scenegraph()
    save_figure(render.draw_scenegraph(world_vertices, projected_pixels, camera_pose, rays), "scenegraph")
    save_figure(render.draw_scene_3d(world_vertices, camera_pose, rays), "scenegraph_scene_3d")
    save_figure(render.draw_camera_image(projected_pixels), "scenegraph_camera_image")
    save_animation(render.animate_scenegraph(robot_sweep(36)), "scenegraph", 60)
