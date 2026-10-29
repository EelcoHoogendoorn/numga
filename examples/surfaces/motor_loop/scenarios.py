"""Scenes for Loop subdivision of motors compare point and motor subdivision on a twisted ribbon and on a flat ribbon whose vertex frames turn in place."""

from __future__ import annotations


import numpy as np

from numga import stack
from examples.surfaces.motor_loop import core

# The ribbon's control mesh has vertices along z and across x, with its four corners pinned.
N_ALONG, N_ACROSS = 5, 3
LENGTH, WIDTH = 4.0, 1.2
CORNERS = [0, N_ACROSS - 1, N_ALONG * N_ACROSS - N_ACROSS, N_ALONG * N_ACROSS - 1]
# The twist angle of the far end at the peak of the wind-up while the near end is held fixed,
# together with the frame count and frame duration of the animation.
TWIST_ANGLE = 3 * np.pi
N_FRAMES = 72
DURATION_MS = 40
# The amplitude and random seed for turning each vertex frame in place about its own axis.
TURN_AMPLITUDE = np.pi / 3
TURN_SEED = 2
# The number of Loop subdivision steps and the length of the rendered frame axes.
N_LOOP_STEPS = 3
AXIS_LENGTH = 0.3


# --- math -----------------------------------------------------------------------------
def winding() -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """The control frames and the point- and motor-subdivided surfaces across one wind-up of the ribbon."""
    z_coords, x_coords = grid()
    twist_angles = TWIST_ANGLE * (1 - np.cos(np.linspace(0, np.pi, N_FRAMES // 2 + 1))) / 2  # [frames]
    twist_profile = ((z_coords + LENGTH / 2) / LENGTH) ** 2                    # [vertices]
    # Place each frame on the flat ribbon along z and across x, then twist around the z axis.
    rest_motors = (core.mv.zw * (z_coords / 2)).exp() * (core.mv.xw * (x_coords / 2)).exp()  # [vertices] Motor+
    twist_motors = (core.mv.xy * (-twist_angles[:, None] * twist_profile / 2)).exp()  # [frames, vertices] Motor+
    motors = (twist_motors * rest_motors).field()                              # [frames] Motor+[vertices]
    return subdivide_ribbon(motors)


def turning() -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """The control frames and the point- and motor-subdivided surfaces as each frame turns in place on the flat ribbon."""
    z_coords, x_coords = grid()
    rng = np.random.default_rng(TURN_SEED)
    axes = core.mv(core.Line.derive.nondegenerate, rng.normal(size=(len(z_coords), 3))).normalized()  # [vertices] Line+
    phases = rng.uniform(0, 2 * np.pi, len(z_coords))
    angles = TURN_AMPLITUDE * np.sin(np.linspace(0, 2 * np.pi, N_FRAMES, endpoint=False)[:, None] + phases)  # [frames, vertices]
    rest_motors = (core.mv.zw * (z_coords / 2)).exp() * (core.mv.xw * (x_coords / 2)).exp()  # [vertices] Motor+
    local_turns = (axes * (-angles / 2)).exp()                                 # [frames, vertices] Motor+
    motors = (rest_motors * local_turns).field()                               # [frames] Motor+[vertices]
    return subdivide_ribbon(motors)


def subdivide_ribbon(motors: core.Motor) -> tuple[core.Point, core.Point, np.ndarray, np.ndarray]:
    """The control frames, point- and motor-subdivided surfaces, and coarse and refined triangle faces of the ribbon."""
    faces = core.grid_faces(N_ALONG, N_ACROSS)
    pinned = np.isin(np.arange(N_ALONG * N_ACROSS), CORNERS)
    markers = stack([
        core.mv.w.dual(),
        core.mv.point([AXIS_LENGTH, 0.0, 0.0]),
        core.mv.point([0.0, AXIS_LENGTH, 0.0]),
        core.mv.point([0.0, 0.0, AXIS_LENGTH]),
    ])                                                                         # [axes + 1] Point-
    origin = markers[0]                                                        # [] Point-
    control_mesh = core.Mesh(N_ALONG * N_ACROSS, faces)
    refined_mesh = control_mesh.loop_subdivided(pinned, N_LOOP_STEPS)
    point_surface = refined_mesh.subdivision * (motors >> origin)              # [frames] Point-[refined]
    motor_surface = core.subdivide_motors(refined_mesh, motors) >> origin      # [frames] Point-[refined]
    surfaces = stack([point_surface, motor_surface], axis=1)                   # [frames, schemes] Point-[refined]
    control_frames = motors[:, None] >> markers                                # [frames, axes + 1] Point-[vertices]
    return control_frames, surfaces, control_mesh, refined_mesh


# --- plumbing -------------------------------------------------------------------------
def grid() -> tuple[np.ndarray, np.ndarray]:
    """The flattened coordinates along z and across x for the vertices of the ribbon's control mesh."""
    z_coords, x_coords = np.meshgrid(
        np.linspace(-LENGTH / 2, LENGTH / 2, N_ALONG),
        np.linspace(-WIDTH / 2, WIDTH / 2, N_ACROSS),
        indexing="ij",
    )
    return z_coords.reshape(-1), x_coords.reshape(-1)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.surfaces.motor_loop import render

    winding_scene = winding()
    turning_scene = turning()
    save_animation(render.animate(*winding_scene), "motor_loop_ribbon", DURATION_MS, forward_then_back=True)
    save_animation(render.animate(*turning_scene), "motor_loop_turning", DURATION_MS)

    # checks
    # Untwisted, the two schemes agree; wound up, point subdivision wrings the ribbon toward its axis
    # while motor subdivision preserves its width. Joining a unit point with the z axis line forms a
    # plane whose norm is the point's radial distance from the axis.
    centre_line = core.mv.w.dual() & core.mv.point([0.0, 0.0, 1.0])            # [] Line-
    distance = (centre_line & winding_scene[1][[0, N_FRAMES // 2]].batch()).norm().mean(axis=-1)  # [cases, schemes] Scalar+
    (points_rest, motors_rest), (points_peak, motors_peak) = distance
    assert (points_rest - motors_rest).abs() < 1e-12
    assert points_peak < 0.7 * points_rest and motors_peak >= motors_rest
    # Turning the frames about their origins leaves the control points fixed while bending the motor surface.
    point_surfaces, motor_surfaces = turning_scene[1][:, 0].batch(), turning_scene[1][:, 1].batch()  # [frames, refined] Point-
    assert np.all((point_surfaces & point_surfaces[:1]).norm() < 1e-12)
    assert np.any((motor_surfaces & point_surfaces).norm() > 0.05)
    # A rigid turn shared by every control frame leaves the subdivided surface unchanged.
    z_coords, x_coords = grid()
    rest_motors = (core.mv.zw * (z_coords / 2)).exp() * (core.mv.xw * (x_coords / 2)).exp()  # [vertices] Motor+
    shared_turn = (core.mv.yz * (-0.7 / 2)).exp()                              # [] Motor+
    cases = subdivide_ribbon(stack([rest_motors, rest_motors * shared_turn]).field())[1][:, 1].batch()  # [cases, refined] Point-
    assert np.all((cases[0] & cases[1]).norm() < 1e-12)


if __name__ == "__main__":
    main()
