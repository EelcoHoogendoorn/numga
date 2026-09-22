"""Scenarios and CLI runner for the scenegraph, robot kinematics, and compound optics example."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt

from numga import stack
from examples import PLOT_DIR
from . import core, render


def main(
    plot_path: str | Path = str(PLOT_DIR / "scenegraph.png"),
    scene_3d_path: str | Path = str(PLOT_DIR / "scenegraph_scene_3d.png"),
    camera_image_path: str | Path = str(PLOT_DIR / "scenegraph_camera_image.png"),
    animation_path: str | Path = str(PLOT_DIR / "scenegraph.gif"),
    generate_animation: bool = True,
) -> plt.Figure:
    """Execute the full scenegraph demonstration, verify algebraic checks, and export deliverables."""
    print("Running kernel-level algebraic assertions...")
    core.run_checks()
    print("All assertions passed.")

    # 1. Construct canonical unit box geometry:
    unit_box = core.canonical_unit_box()

    # 2. Articulated robot arm forward kinematics:
    joint_angles = (0.35, -0.45, 0.85, -0.40)
    links_to_world, joint_pivots = core.make_robot_arm(joint_angles)

    # 3. Compound multi-lens camera (objective lens L1 + relay lens L2):
    camera_pose = core.make_camera_pose(
        position=(0.0, -3.6, 1.3),
        target=(0.0, 0.0, 1.18),
    )
    camera = core.make_multi_lens_camera(
        camera_pose=camera_pose,
        focal_front=1.0,
        focal_rear=0.8,
        rear_gap=-0.3,
        pupil_radius=0.02,
        sensor_distance=1.25,
    )

    # 4. Viewport extensor (640 x 480 px, physical 1.6 x 1.2 sensor chip):
    viewport = core.make_viewport(width=640, height=480, sensor_width=1.6, sensor_height=1.2)

    # 5. Core punchline: collapse entire visual pipeline into a single extensor:
    local_to_pixel = core.collapse_scenegraph(links_to_world, camera, viewport)
    print(f"Collapsed extensor shape: {local_to_pixel.shape}")

    # 6. Batch project all canonical vertices in one pass:
    projected_pixels = core.project_vertices(local_to_pixel, unit_box)
    world_vertices = links_to_world[:, None](unit_box[None, :])

    # 7. Render split deliverables:
    # (a) 3D scene only:
    fig_3d = plt.figure(figsize=(7.5, 6.0), dpi=130)
    ax_3d = fig_3d.add_subplot(1, 1, 1, projection="3d")
    render.draw_scenegraph_scene_3d(ax_3d, world_vertices=world_vertices, camera_pose=camera_pose)
    fig_3d.tight_layout()
    Path(scene_3d_path).parent.mkdir(parents=True, exist_ok=True)
    fig_3d.savefig(scene_3d_path, bbox_inches="tight", dpi=140)
    plt.close(fig_3d)
    print(f"3D scene saved to {scene_3d_path}")

    # (b) 2D camera photograph only:
    fig_2d, ax_2d = plt.subplots(figsize=(6.4, 4.8), dpi=130)
    render.draw_camera_image(ax_2d, projected_pixels, width=640, height=480)
    fig_2d.tight_layout()
    Path(camera_image_path).parent.mkdir(parents=True, exist_ok=True)
    fig_2d.savefig(camera_image_path, bbox_inches="tight", dpi=140)
    plt.close(fig_2d)
    print(f"2D camera photograph saved to {camera_image_path}")

    # (c) Combined 2-panel figure:
    fig = render.draw_scenegraph_figure(
        world_vertices=world_vertices,
        projected_pixels=projected_pixels,
        camera_pose=camera_pose,
        plot_path=plot_path,
    )

    # (d) Animated kinematic sweep:
    if generate_animation and animation_path:
        print("Generating forward kinematics animation...")
        render.animate_robot_kinematics(output_gif_path=animation_path, num_frames=36, duration_ms=60)

    return fig


if __name__ == "__main__":
    main()
