"""Constructive Solid Geometry scene with ruled hyperboloid, hollow bowl, mirror sphere, and shadows."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quadrics.csg_raytracer.core import (
    Plane, Point, Quadric, ellipsoid, evaluate, hyperboloid, mv, place, ruling_lines,
)

# Geometry constants
A_HYP, C_HYP = 0.55, 0.85
HYP_POSITION = np.array([-1.2, 0.4, 0.85])
HYP_TILT = np.radians(12.0)

R_OUT = np.array([0.75, 0.75, 0.75])
WALL_FRACTION = 0.80
BOWL_POSITION = np.array([1.1, -0.2, 0.75])
BOWL_TILT_YZ = np.radians(-35.0)
BOWL_TILT_XZ = np.radians(15.0)

R_MIRROR = 0.55
MIRROR_POSITION = np.array([0.0, 0.1, 0.55])

# Illumination and framing
LAMP_DIRECTION = np.array([0.5, -0.5, 0.70, 0.0])
LAMP_DIRECTION[:3] /= np.linalg.norm(LAMP_DIRECTION[:3])

TARGET = np.array([0.0, 0.2, 0.7], dtype=np.float32)
PRIMARY_CAM = np.array([0.0, -4.2, 2.5, 1.0], dtype=np.float32)

N_FRAMES = 48
ORBIT_RADIUS = 4.2
ORBIT_HEIGHT = 2.5
IMAGE_SIZE = (640, 640)
ANIM_SIZE = (480, 480)
FRAME_DURATION = 80


# --- math -----------------------------------------------------------------------------
def make_scene():
    """Build the geometric scene in pure PGA algebraic elements."""
    # 1. Ruled hyperboloid of one sheet:
    hyp_raw = hyperboloid([A_HYP, A_HYP, C_HYP])
    hyp_placed = place(hyp_raw, HYP_POSITION, HYP_TILT, turn_axis="xz")

    # Bounding planes for finite height:
    motor_hyp = (((stack((mv.xw, mv.yw, mv.zw)) * (HYP_POSITION / 2)).sum(axis=-1)).exp()
                 * (mv.xz * (HYP_TILT / 2)).exp())
    hyp_top = motor_hyp >> mv.vector([0.0, 0.0, 1.0, -C_HYP])
    hyp_bot = motor_hyp >> mv.vector([0.0, 0.0, -1.0, -C_HYP])

    # 2. Hollow clipped bowl (CSG):
    r_in = R_OUT * WALL_FRACTION
    bowl_out_raw = ellipsoid(R_OUT)
    bowl_in_raw = ellipsoid(r_in)

    motor_bowl = (((stack((mv.xw, mv.yw, mv.zw)) * (BOWL_POSITION / 2)).sum(axis=-1)).exp()
                  * (mv.yz * (BOWL_TILT_YZ / 2)).exp()
                  * (mv.xz * (BOWL_TILT_XZ / 2)).exp())
    bowl_out = motor_bowl >> bowl_out_raw(motor_bowl << Point)
    bowl_in = motor_bowl >> bowl_in_raw(motor_bowl << Point)

    bowl_clip_dir = np.array([0.0, 0.15, 0.98, -0.1], dtype=np.float32)
    bowl_clip_dir[:3] /= np.linalg.norm(bowl_clip_dir[:3])
    bowl_clip = motor_bowl >> mv.vector(bowl_clip_dir)

    # 3. Mirror sphere:
    mirror_raw = ellipsoid([R_MIRROR, R_MIRROR, R_MIRROR])
    mirror_placed = place(mirror_raw, MIRROR_POSITION)

    # 4. Floor plane & lamp:
    floor_plane = mv.vector([0.0, 0.0, 1.0, 0.0])
    lamp = mv.vector(LAMP_DIRECTION)

    scene_objects = {
        "hyp": hyp_placed,
        "hyp_top": hyp_top,
        "hyp_bot": hyp_bot,
        "bowl_out": bowl_out,
        "bowl_in": bowl_in,
        "bowl_clip": bowl_clip,
        "mirror": mirror_placed,
        "floor": floor_plane,
        "lamp": lamp,
    }
    return scene_objects


def orbit_cameras(n_frames: int, radius: float = ORBIT_RADIUS, height: float = ORBIT_HEIGHT) -> np.ndarray:
    """Generate camera positions orbiting circularly around the scene center."""
    angles = np.linspace(-np.pi / 2, 3 * np.pi / 2, n_frames, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)
    zs = np.full_like(angles, height)
    ws = np.ones_like(angles)
    return np.stack([xs, ys, zs, ws], axis=-1).astype(np.float32)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.quadrics.csg_raytracer import render

    scene = make_scene()

    # 1. Render primary showcase figure:
    fig = render.draw_scene(scene, PRIMARY_CAM, TARGET, *IMAGE_SIZE)
    save_figure(fig, "csg_raytracer")

    # 2. Render orbit turntable animation:
    cam_orbit = orbit_cameras(N_FRAMES)
    anim_frames = render.frames(scene, cam_orbit, TARGET, *ANIM_SIZE)
    save_animation(anim_frames, "csg_raytracer", FRAME_DURATION)

    # --- checks -----------------------------------------------------------------------
    # 1. Verify ruling lines on the hyperboloid:
    # Any point along the generator line p(theta) + t * v(theta) must lie exactly on the surface:
    throat, v_plus, v_minus = ruling_lines([A_HYP, A_HYP, C_HYP], 12)
    hyp_raw = hyperboloid([A_HYP, A_HYP, C_HYP])
    for t_step in [-1.5, -0.5, 0.0, 0.8, 2.0]:
        test_pt = throat + v_plus * t_step
        eval_vals = evaluate(hyp_raw, test_pt).to_array()
        np.testing.assert_allclose(eval_vals, 0.0, atol=1e-7)

    # 2. Verify mirror sphere boundary:
    mirror_quad = ellipsoid([R_MIRROR, R_MIRROR, R_MIRROR])
    sample_pt = mv.point([R_MIRROR, 0.0, 0.0])
    np.testing.assert_allclose(evaluate(mirror_quad, sample_pt).to_array(), 0.0, atol=1e-7)

    # 3. Verify bowl inner and outer boundary radii:
    r_in = R_OUT * WALL_FRACTION
    out_pt = mv.point([R_OUT[0], 0.0, 0.0])
    in_pt = mv.point([r_in[0], 0.0, 0.0])
    np.testing.assert_allclose(evaluate(ellipsoid(R_OUT), out_pt).to_array(), 0.0, atol=1e-7)
    np.testing.assert_allclose(evaluate(ellipsoid(r_in), in_pt).to_array(), 0.0, atol=1e-7)


if __name__ == "__main__":
    main()
