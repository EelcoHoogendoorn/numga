"""Scenes for the scenegraph, robot kinematics, and compound optics example.

`scenegraph` returns one posed frame for the figures; `robot_sweep` yields the frames of the kinematic
animation. Both return geometry for `render`.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.geometry.scenegraph.core import (
    canonical_unit_box, lens_camera, lens_train, look_at, mv, origin, project_vertices,
    robot_arm, trace_rays, viewport,
)

# The camera: where it stands and what it looks at; the objective's and relay's focal lengths and the
# relay's place behind the objective; the pupil point's offset from the optical centre, which bends the
# rays; the sensor's distance behind the objective; and the sensor's size and pixels.
EYE, TARGET = np.array([0.0, -3.6, 1.3]), np.array([0.0, 0.0, 1.18])
FRONT_FOCAL, REAR_FOCAL, REAR_GAP = 1.0, 0.8, -0.3
PUPIL_OFFSET = 0.02
SENSOR_DISTANCE = 1.25
SENSOR_SIZE, PIXELS = (1.6, 1.2), (640, 480)
# The arm's pose in the still: base yaw, shoulder, elbow and wrist pitch; and the gripper corners traced.
POSE = (0.35, -0.45, 0.85, -0.40)
TRACED_CORNERS = [7, 6, 2]
# The sweep: each joint's middle and swing, and how many times it swings per loop; its frames, and their
# duration in milliseconds.
SWEEP_MIDDLES = np.array([0.0, -0.40, 0.85, -0.45])
SWEEP_SWINGS = np.array([0.45, 0.25, 0.35, 0.25])
N_FRAMES = 36
FRAME_DURATION = 60


# --- math -----------------------------------------------------------------------------
def camera_rig():
    """The compound camera: pose, objective and relay lens, pupil, sensor, and the world-to-pixel map."""
    pose = look_at(EYE, TARGET)
    front_lens, rear_lens, rear_plane = lens_train(FRONT_FOCAL, REAR_FOCAL, REAR_GAP)
    pupil = (mv.xw * PUPIL_OFFSET).exp() >> origin
    sensor_plane = mv.z + SENSOR_DISTANCE * mv.w
    camera = lens_camera(pose, front_lens, rear_lens, pupil, sensor_plane)
    # The viewport: the physical sensor chip onto its pixels.
    world_to_pixel = viewport(*PIXELS, *SENSOR_SIZE)(camera)                  # [] Point <- Point
    return pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel


def photograph(joint_angles, corners: list[int]):
    """The arm posed, its box vertices in the world and on the sensor's pixels, the camera's pose, and rays
    from the given gripper corners traced through the lenses."""
    unit_box = canonical_unit_box()
    pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel = camera_rig()
    bodies_to_world, _ = robot_arm(joint_angles)
    # The whole pipeline, kinematics to pixels, as one map per body, applied to every vertex in one pass.
    projected = project_vertices(world_to_pixel(bodies_to_world), unit_box)    # [bodies, vertices] Point
    world_vertices = bodies_to_world[:, None](unit_box[None, :])               # [bodies, vertices] Point
    rays = trace_rays(world_vertices[-1, corners], pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane)
    return world_vertices, projected, pose, rays


def scenegraph():
    """The robot arm posed once, photographed through the compound camera."""
    return photograph(POSE, TRACED_CORNERS)


def robot_sweep(n_frames: int) -> Iterator[tuple]:
    """Frames of a looping joint trajectory, each photographed through the compound camera, with the ray
    from the gripper's tip."""
    for t in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        # The base and elbow swing with the sine, the shoulder with the cosine, the wrist twice as fast.
        phases = np.array([np.sin(t), np.cos(t), np.sin(t), np.cos(2 * t)])
        yield photograph(tuple(SWEEP_MIDDLES + SWEEP_SWINGS * phases), TRACED_CORNERS[:1])


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.geometry.scenegraph import render

    world_vertices, projected_pixels, camera_pose, rays = scenegraph()
    save_figure(render.draw_scenegraph(world_vertices, projected_pixels, camera_pose, rays), "scenegraph")
    save_figure(render.draw_scene_3d(world_vertices, camera_pose, rays), "scenegraph_scene_3d")
    save_figure(render.draw_camera_image(world_vertices, projected_pixels, camera_pose), "scenegraph_camera_image")
    save_animation(render.animate_scenegraph(robot_sweep(N_FRAMES)), "scenegraph", FRAME_DURATION)

    # checks
    # The collapsed map agrees with applying kinematics, camera and viewport in turn.
    *_, world_to_pixel = camera_rig()
    sequential = world_to_pixel(world_vertices)
    sequential = sequential / (mv.w & sequential)
    assert (sequential - projected_pixels).dual().norm_squared().to_array().max() < 1e-16


if __name__ == "__main__":
    main()
