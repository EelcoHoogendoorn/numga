"""Parameters, closed-form boundary schedules, and figure entry points for the static elastica and dynamic telephone cord."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.mechanics.elastica import core

mv = core.mv
length = 1.0
n_pieces = 200
turning_compliance = np.array([1.0, 1.0, 1.4])
shifting_compliance = np.zeros(3)
curl_turns = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]
n_curls = len(curl_turns)
n_sweep_frames = 90
sweep_angles = np.linspace(2.0, 176.0, n_sweep_frames)
frame_duration_ms = 40
tilt_angles = [80.0, 60.0, 40.0, 20.0]
n_tilts = len(tilt_angles)
n_helix_turns = 3
helix_ribbon_width = 0.015
uncoil_turns = 1.5
n_uncoil_frames = 135
uncoil_ribbon_width = 0.035

cord_length, n_cord_coils, n_segments_per_coil, cord_pitch = 1.0, 5, 6, 10.0
cord_turning_rigidity = np.array([1e-3, 1e-3, 7e-4])
cord_shifting_rigidity = np.array([20.0, 20.0, 20.0])
cord_mass, cord_spread, gravity, cord_damping = 0.05, 2.0, 9.81, 1.0
hang_time, pull_time, handset_reach, hold_time, release_time = 0.3, 0.9, 0.6, 0.3, 0.4
time_step, n_steps_per_frame, n_cord_frames = 1e-3, 40, 70
n_subdivisions, cord_ribbon_width = 4, 0.025
n_gallery_pixels = 600


# --- math -----------------------------------------------------------------------------
def curled_strips(n_rod_pieces: int) -> core.Motor:
    """End couples in multiples of a full bend curl the strip into circular arcs (`[n_curls, n_rod_pieces + 1] Motor+`)."""
    couples = mv.yz.dual() * (2 * np.pi / (turning_compliance[0] * length) * np.array(curl_turns))  # [n_curls] Forque-
    rod_compliance = core.compliance(turning_compliance, shifting_compliance)                       # [] Twist+ <- Forque-
    return core.rod_frames(couples, rod_compliance, mv.rotor(), length, n_rod_pieces)               # [n_curls, n_rod_pieces + 1] Motor+


def euler_rods(angles_deg: np.ndarray, n_rod_pieces: int) -> core.Motor:
    """Critical axial thrusts bend pinned rods departing at `angles_deg` back onto the line of thrust (`[..., n_rod_pieces + 1] Motor+`)."""
    pin_thrusts = (core.centre & mv.z.dual()) * thrusts(np.radians(angles_deg))                     # [...] Forque-
    pin_starts = (mv.yz * (-np.radians(angles_deg) / 2)).exp()                                      # [...] Motor+
    rod_compliance = core.compliance(turning_compliance, shifting_compliance)                       # [] Twist+ <- Forque-
    return core.rod_frames(pin_thrusts, rod_compliance, pin_starts, length, n_rod_pieces)           # [..., n_rod_pieces + 1] Motor+


def kirchhoff_helices(n_rod_pieces: int) -> core.Motor:
    """An upward couple winds tilted wires through `n_helix_turns` helical turns (`[n_tilts, n_rod_pieces + 1] Motor+`)."""
    helix_couple = mv.xy.dual() * (2 * np.pi * n_helix_turns / (turning_compliance[0] * length))    # [] Forque-
    helix_starts = (mv.yz * (-np.radians(tilt_angles) / 2)).exp()                                   # [n_tilts] Motor+
    rod_compliance = core.compliance(turning_compliance, shifting_compliance)                       # [] Twist+ <- Forque-
    return core.rod_frames(helix_couple, rod_compliance, helix_starts, length, n_rod_pieces)        # [n_tilts, n_rod_pieces + 1] Motor+


def uncoiling_wire(n_frames: int, n_rod_pieces: int) -> core.Motor:
    """Sweeping the initial tilt between perpendicular and parallel carries a coupled wire from ring to straight twist (`[n_frames, n_rod_pieces + 1] Motor+`)."""
    uncoil_tilts = 45 * (1 + np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False)))           # [n_frames]
    uncoil_couple = mv.xy.dual() * (2 * np.pi * uncoil_turns / (turning_compliance[0] * length))    # [] Forque-
    uncoil_starts = (mv.yz * (-np.radians(uncoil_tilts) / 2)).exp()                                 # [n_frames] Motor+
    rod_compliance = core.compliance(turning_compliance, shifting_compliance)                       # [] Twist+ <- Forque-
    return core.rod_frames(uncoil_couple, rod_compliance, uncoil_starts, length, n_rod_pieces)      # [n_frames, n_rod_pieces + 1] Motor+


def cord_coil(n_segments: int) -> tuple[core.Motor, core.Motor, core.Twist]:
    """Pitch-tilted helical coiling places the cord's rest segments, single-joint step, and axial shift (`[n_segments + 2] Motor+`)."""
    piece_length = cord_length / n_segments
    pitch_tilt = (mv.xz * (-np.radians(cord_pitch) / 2)).exp()                                      # [] Motor+
    coiling_rate = 2 * np.pi * n_cord_coils / cord_length
    coiling_twist = core.shift_along_rod + (pitch_tilt >> mv.yz) * coiling_rate                     # [] Twist+
    rest_frames = (coiling_twist * (np.arange(-1, n_segments + 1) * piece_length / 2)).exp()        # [n_segments + 2] Motor+
    rest_step = (coiling_twist * (piece_length / 2)).exp()                                          # [] Motor+
    coil_shift = pitch_tilt >> mv.xw                                                                # [] Twist+
    return rest_frames, rest_step, coil_shift


def cord_frames(n_frames: int, n_steps: int) -> Iterator[core.Motor]:
    """Yields the telephone cord's segment placements across `n_frames` frames of `n_steps` integration steps."""
    n_segments = n_cord_coils * n_segments_per_coil
    piece_length = cord_length / n_segments
    rest_frames, rest_step, coil_shift = cord_coil(n_segments)
    joint_stiffness = core.stiffness(cord_turning_rigidity, cord_shifting_rigidity)                 # [] Forque- <- Twist+
    segment_mass = cord_mass * piece_length
    inertia = core.segment_inertia(segment_mass, piece_length, cord_spread * piece_length)          # [] Forque- <- Twist+
    weight = mv.y.dual() * (-gravity * segment_mass)                                                # [] Point-
    reaches = handset_reaches(
        n_frames, n_steps, time_step, hang_time, pull_time, handset_reach, hold_time, release_time,
    )                                                                                               # [n_frames, n_steps]
    return core.cord_trajectory(
        rest_frames, rest_step, coil_shift, joint_stiffness, piece_length,
        inertia, weight, cord_damping, time_step, reaches,
    )


# --- plumbing -------------------------------------------------------------------------
def handset_reaches(
    n_frames: int,
    n_steps: int,
    step_dt: float,
    hang: float,
    pull: float,
    reach: float,
    hold: float,
    release: float,
) -> np.ndarray:
    """Cubic Hermite easing moves the handset out after `hang`, holds it at `reach`, and eases it back (`[n_frames, n_steps]`)."""
    times = np.arange(n_frames * n_steps) * step_dt
    pull_phase = np.clip((times - hang) / pull, 0, 1)
    release_phase = np.clip((times - hang - pull - hold) / release, 0, 1)
    pull_ease = 3 * pull_phase ** 2 - 2 * pull_phase ** 3
    release_ease = 3 * release_phase ** 2 - 2 * release_phase ** 3
    return (reach * pull_ease * (1 - release_ease)).reshape(n_frames, n_steps)                      # [n_frames, n_steps]


def ribbon_edges(frames: core.Motor, width: float) -> core.Point:
    """Each cross-section motor places the two lateral edge points of a ribbon of `width` (`[..., 2] Point-`)."""
    offsets = mv.point([[-width / 2, 0, 0], [width / 2, 0, 0]])                                     # [2] Point-
    return frames[..., None] >> offsets                                                             # [..., 2] Point-


def thrusts(angles: np.ndarray) -> np.ndarray:
    """Arithmetic–geometric mean iteration evaluates the elliptic-integral critical thrust for each pin angle."""
    modulus = np.sin(angles / 2)
    mean, geometric = np.ones_like(modulus), np.sqrt(1 - modulus ** 2)
    for _ in range(8):
        mean, geometric = (mean + geometric) / 2, np.sqrt(mean * geometric)
    return (np.pi / mean / length) ** 2 / turning_compliance[0]


def chords(angles: np.ndarray) -> np.ndarray:
    """Arithmetic–geometric mean iteration evaluates the elliptic-integral chord length between the two pins."""
    modulus = np.sin(angles / 2)
    mean, geometric, weight, deficit = np.ones_like(modulus), np.sqrt(1 - modulus ** 2), 0.5, modulus ** 2 / 2
    for _ in range(8):
        mean, geometric, difference = (mean + geometric) / 2, np.sqrt(mean * geometric), (mean - geometric) / 2
        weight, deficit = 2 * weight, deficit + 2 * weight * difference ** 2
    return length * (2 * (1 - deficit) - 1)


def main() -> None:
    """Generates every elastica figure and animation and verifies geometric and physical invariants."""
    from examples.mechanics.elastica import render
    from examples.util.animation import save_animation, save_figure

    n_segments = n_cord_coils * n_segments_per_coil
    n_drawn = (n_segments + 1) * n_subdivisions + 1
    curled_rods = curled_strips(n_pieces)                                                           # [n_curls, n_pieces + 1] Motor+
    save_figure(render.draw_curls(curled_rods >> core.centre), "elastica_curls")
    sweep_rods = euler_rods(sweep_angles, n_pieces)                                                 # [n_sweep_frames, n_pieces + 1] Motor+
    save_animation(render.animate_sweep(sweep_rods >> core.centre), "elastica_sweep", frame_duration_ms)
    coiled_rods = kirchhoff_helices(n_pieces)                                                       # [n_tilts, n_pieces + 1] Motor+
    save_figure(render.draw_helices(coiled_rods >> core.centre, ribbon_edges(coiled_rods, helix_ribbon_width)), "elastica_helices")
    uncoiled_rods = uncoiling_wire(n_uncoil_frames, n_pieces)                                       # [n_uncoil_frames, n_pieces + 1] Motor+
    save_animation(
        render.animate_uncoiling(uncoiled_rods >> core.centre, ribbon_edges(uncoiled_rods, uncoil_ribbon_width)),
        "elastica_uncoiling",
        frame_duration_ms,
    )
    moved_cord = stack(list(cord_frames(n_cord_frames, n_steps_per_frame)))                         # [n_cord_frames, n_segments + 2] Motor+
    drawn_cord = core.dense_frames(moved_cord, n_subdivisions)                                      # [n_cord_frames, n_drawn] Motor+
    save_animation(
        render.animate_cord(drawn_cord >> core.centre, ribbon_edges(drawn_cord, cord_ribbon_width)),
        "elastica_cord",
        frame_duration_ms,
    )

    # checks
    # Whole-turn couples return the curled strip's free end to the clamp, while zero couple leaves it straight.
    curl_ends = curled_rods[:, -1] >> core.centre                                                   # [n_curls] Point-
    assert (curl_ends[0] & (core.centre + mv.z.dual() * length)).norm() < 1e-10
    assert ((curl_ends[[4, 5]] & core.centre).norm() < 1e-10).all()
    # Every Euler elastica returns to the line of thrust at the elliptic-integral chord distance from the first pin.
    expected_pins = core.centre + mv.z.dual() * chords(np.radians(sweep_angles))                    # [n_sweep_frames] Point-
    assert (((sweep_rods[:, -1] >> core.centre) & expected_pins).norm() < 1e-3).all()
    # Three full helical turns bring each wire's far end directly above the start at height L cos(tilt).
    expected_tips = core.centre + mv.z.dual() * (length * np.cos(np.radians(tilt_angles)))          # [n_tilts] Point-
    assert (((coiled_rods[:, -1] >> core.centre) & expected_tips).norm() < 3e-3).all()
    # High shear and stretch rigidities keep relative joint shifts small compared with segment length.
    piece_length = cord_length / n_segments
    _, rest_step, _ = cord_coil(n_segments)
    deviations = (rest_step.reverse() * moved_cord[:, :-1].reverse() * moved_cord[:, 1:]).log() * 2  # [n_cord_frames, n_segments + 1] Twist+
    assert (((deviations[..., None] & core.force_axes) / piece_length).abs() < 0.1).all()


def pinned() -> None:
    """Renders the gallery animation of the telephone cord pulled out and released."""
    from examples.mechanics.elastica import render
    from examples.util.animation import pin_animation

    n_segments = n_cord_coils * n_segments_per_coil
    n_drawn = (n_segments + 1) * n_subdivisions + 1
    moved_cord = stack(list(cord_frames(n_cord_frames, n_steps_per_frame)))                         # [n_cord_frames, n_segments + 2] Motor+
    drawn_cord = core.dense_frames(moved_cord, n_subdivisions)                                      # [n_cord_frames, n_drawn] Motor+
    pin_animation(
        render.sized_cord(drawn_cord >> core.centre, ribbon_edges(drawn_cord, cord_ribbon_width), n_gallery_pixels),
        "telephone_cord",
        frame_duration_ms,
    )


if __name__ == "__main__":
    main()
