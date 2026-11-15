"""Scene configurations pose the five-part robot arm and photograph it through the compound two-lens camera."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.geometry.scenegraph.core import (
    aim_motor, canonical_unit_box, lens_camera, lens_train, mv, origin,
    ray_waypoints, robot_arm, viewport_map,
)

eye, target = mv.point([0.0, -3.6, 1.3]), mv.point([0.0, 0.0, 1.18])
front_focal, rear_focal, rear_gap = 1.0, 0.8, -0.3
pupil_offset = 0.02
sensor_distance = 1.25
sensor_size, sensor_pixels = (1.6, 1.2), (640, 480)
still_angles = (0.35, -0.45, 0.85, -0.40)
traced_corners = [7, 6, 2]
sweep_middles = np.array([0.0, -0.40, 0.85, -0.45])
sweep_swings = np.array([0.45, 0.25, 0.35, 0.25])
n_frames = 54
frame_duration_ms = 40


# --- math -----------------------------------------------------------------------------
def camera_rig():
    """Camera pose, objective and relay lenses, rear lens plane, off-axis pupil, sensor plane, and world-to-pixel map."""
    pose = aim_motor(eye, target)                                               # [] Motor+
    front_lens, rear_lens, rear_plane = lens_train(front_focal, rear_focal, rear_gap)
    pupil_shift = (mv.xw * pupil_offset).exp()                                  # [] Motor+
    pupil = pupil_shift >> origin                                               # [] Point-
    sensor_plane = mv.z + sensor_distance * mv.w                                # [] Plane+
    camera = lens_camera(pose, front_lens, rear_lens, pupil, sensor_plane)      # [] Point- <- Point-
    world_to_pixel = viewport_map(*sensor_pixels, *sensor_size)(camera)         # [] Point- <- Point-
    return pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel


def posed_photograph(joint_angles: tuple, corners: list[int]):
    """World link vertices (`[5, 8] Point-`), projected sensor pixels (`[5, 8] Point-`), camera pose, and traced ray waypoints for `joint_angles`."""
    unit_box = canonical_unit_box()                                             # [8] Point-
    pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane, world_to_pixel = camera_rig()
    bodies_to_world, _ = robot_arm(joint_angles)                                # [5] Point- <- Point-, [4] Motor+
    # Collapse world-to-pixel and body placement into one map per body before evaluating vertices.
    local_to_pixel = world_to_pixel(bodies_to_world)                            # [5] Point- <- Point-
    raw_pixels = local_to_pixel[:, None](unit_box[None, :])                     # [5, 8] Point-
    projected = raw_pixels / (mv.w & raw_pixels)                                # [5, 8] Point-
    world_vertices = bodies_to_world[:, None](unit_box[None, :])                # [5, 8] Point-
    rays = ray_waypoints(world_vertices[-1, corners], pose, front_lens, rear_lens, rear_plane, pupil, sensor_plane)
    return world_vertices, projected, pose, rays


def still_scenegraph():
    """Posed robot arm, projected pixels, camera pose, and three gripper rays for the static figure."""
    return posed_photograph(still_angles, traced_corners)


def arm_sweep_states(n_sweep_frames: int) -> Iterator[tuple]:
    """Yields world vertices, projected pixels, camera pose, and gripper-tip ray waypoints across `n_sweep_frames` phases of a looping joint trajectory."""
    for phase in np.linspace(0.0, 2 * np.pi, n_sweep_frames, endpoint=False):
        modulations = np.array([np.sin(phase), np.cos(phase), np.sin(phase), np.cos(2 * phase)])
        yield posed_photograph(tuple(sweep_middles + sweep_swings * modulations), traced_corners[:1])


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.geometry.scenegraph import render

    world_vertices, projected_pixels, camera_pose, rays = still_scenegraph()
    save_figure(render.scenegraph_figure(world_vertices, projected_pixels, camera_pose, rays), "scenegraph")
    save_figure(render.world_scene_figure(world_vertices, camera_pose, rays), "scenegraph_scene_3d")
    save_figure(render.sensor_photograph_figure(world_vertices, projected_pixels, camera_pose), "scenegraph_camera_image")

    sweep_states = list(arm_sweep_states(n_frames))
    save_animation(render.scenegraph_animation(sweep_states), "scenegraph", frame_duration_ms)

    # checks
    # Collapsed single-map projection matches sequential kinematics, camera, and viewport evaluation,
    # and every swept frame preserves unit-weight projected pixels.
    *_, sensor_plane, world_to_pixel = camera_rig()
    sequential = world_to_pixel(world_vertices)                                 # [5, 8] Point-
    sequential = sequential / (mv.w & sequential)                               # [5, 8] Point-
    assert ((sequential - projected_pixels).antinorm() < 1e-9).all()
    assert ((sensor_plane & (camera_pose << rays[3])).abs() < 1e-9).all()
    assert (((mv.w & sweep_states[-1][1]) - 1).abs() < 1e-9).all()


def pinned() -> None:
    """Gallery animation of the five-part robot arm swinging across its joint trajectory."""
    from examples.util.animation import pin_animation
    from examples.geometry.scenegraph import render

    pin_animation(render.scenegraph_animation(arm_sweep_states(n_frames)), "scenegraph", frame_duration_ms)


if __name__ == "__main__":
    main()
