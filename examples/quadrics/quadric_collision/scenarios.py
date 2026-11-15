"""Scene parameters and sliding-contact setup for two planar ellipsoids classified through blends of their dual quadrics."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quadrics.quadric_collision.core import (
    Motor, Plane, Polarity, Point, Quadric, Scalar,
    cubic_peak, infinity, mv, placement_motor, quadric_blend, rest_ellipsoid, tangent_plane,
)

# First ellipsoid semi-axes, centre, and orientation angle.
first_semi = (2.0, 1.0)
first_centre = (-1.2, 0.0)
first_turn = np.radians(25)
# Second ellipsoid semi-axes, orientation angle, contact normal angle on the first ellipsoid,
# and static test offsets (apart, touching, overlapping) along that normal.
second_semi = (1.6, 0.9)
second_turn = np.radians(-35)
contact_angle = np.radians(22)
pose_offsets = np.array([0.8, 0.0, -0.6])
n_poses = len(pose_offsets)
# Offsets swept during the animation and per-frame duration in milliseconds.
n_frames = 85
slide_offsets = np.linspace(0.8, -0.6, n_frames)
frame_duration_ms = 40
# Four shares fixing the cubic determinant, and dense shares for plotting the determinant curve.
cubic_samples = [[0], [1], [2], [-1]]
n_shares = 500


# --- math -----------------------------------------------------------------------------
def touching_pair() -> tuple[Quadric, Quadric, Plane, Motor]:
    """The first ellipsoid (`[] Point- <- Plane+`), the unplaced second ellipsoid, the contact normal (`[] Plane+`), and the motor (`[] Motor+`) bringing them into tangency meet at one shared contact point."""
    placement = placement_motor(*first_centre, first_turn)                     # [] Motor+
    first_shape = rest_ellipsoid(*first_semi)                                  # [] Point- <- Plane+
    first_ellipsoid = placement >> first_shape(placement << Plane)             # [] Point- <- Plane+

    rotor = (mv.xy * (-contact_angle / 2)).exp()                               # [] Motor+
    normal = rotor >> mv.x                                                     # [] Plane+
    second_ellipsoid = rest_ellipsoid(*second_semi)                            # [] Point- <- Plane+
    turn = placement_motor(0.0, 0.0, second_turn)                              # [] Motor+
    turned_second = turn >> second_ellipsoid(turn << Plane)                    # [] Point- <- Plane+
    # Translate the turned second ellipsoid so its opposite-normal contact lands on the first's contact.
    target = first_ellipsoid(tangent_plane(first_ellipsoid, normal))           # [] Point-
    start = turned_second(tangent_plane(turned_second, -normal))               # [] Point-
    displacement = target.normalized() - start.normalized()                    # [] Point-
    translator = ((infinity ^ displacement.dual()) * -0.5).exp()               # [] Motor+
    touch_motor = translator * turn                                            # [] Motor+
    return first_ellipsoid, second_ellipsoid, normal, touch_motor


def posed_pair() -> tuple[Quadric, Quadric, Scalar, Scalar, Plane, Point]:
    """Apart, touching, and overlapping poses (`[n_poses]`) along `normal` yield positive, zero, and negative peak blend determinants, while the singular blend at contact locates the shared tangent plane and contact point."""
    first_ellipsoid, second_ellipsoid, normal, touch_motor = touching_pair()
    shifts = ((infinity ^ (normal * pose_offsets)) * -0.5).exp()               # [n_poses] Motor+
    poses = shifts * touch_motor                                               # [n_poses] Motor+
    placed_second = poses >> second_ellipsoid(poses << Plane)                  # [n_poses] Point- <- Plane+

    sample_shares = mv.scalar(cubic_samples)[:, None]                          # [4, 1] Scalar+
    peak_shares, peak_dets = cubic_peak(quadric_blend(first_ellipsoid, placed_second, sample_shares).dual().det())   # [n_poses] Scalar+, Scalar+

    # At contact (`pose_offsets[1] == 0`), the singular peak blend's null plane is the shared tangent.
    _, _, planes = quadric_blend(first_ellipsoid, placed_second[1], peak_shares[1]).dual().svd()
    contact_plane = planes[-1].normalized()                                    # [] Plane+
    contact_point = first_ellipsoid(contact_plane).normalized()                # [] Point-
    return first_ellipsoid, placed_second, peak_shares, peak_dets, contact_plane, contact_point


def slide_setup(offsets: np.ndarray) -> tuple[Quadric, Plane, Scalar, Iterator[tuple[Quadric, Plane, Plane, Polarity, Scalar, Scalar, Scalar]]]:
    """Pairs the stationary first ellipsoid, its outward tangent, and the share axis (`[n_shares] Scalar+`) with a generator yielding the sliding second ellipsoid and its blend diagnostics at each offset."""
    first_ellipsoid, second_ellipsoid, normal, touch_motor = touching_pair()
    first_tangent = tangent_plane(first_ellipsoid, normal)                     # [] Plane+
    shares = mv.scalar(np.linspace(0.001, 0.999, n_shares)[:, None])           # [n_shares] Scalar+
    sample_shares = mv.scalar(cubic_samples)                                   # [4] Scalar+

    def states() -> Iterator[tuple[Quadric, Plane, Plane, Polarity, Scalar, Scalar, Scalar]]:
        for offset in offsets:
            shift = ((infinity ^ (normal * offset)) * -0.5).exp()              # [] Motor+
            pose = shift * touch_motor                                         # [] Motor+
            placed_second = pose >> second_ellipsoid(pose << Plane)            # [] Point- <- Plane+
            peak_share, peak_det = cubic_peak(quadric_blend(first_ellipsoid, placed_second, sample_shares).dual().det())   # [] Scalar+, Scalar+
            second_tangent = tangent_plane(placed_second, -normal)             # [] Plane+
            midplane = ((first_tangent - second_tangent) * 0.5).normalized()   # [] Plane+
            yield (
                placed_second,
                second_tangent,
                midplane,
                quadric_blend(first_ellipsoid, placed_second, peak_share).inverse(),
                quadric_blend(first_ellipsoid, placed_second, shares).dual().det(),
                peak_share,
                peak_det,
            )

    return first_ellipsoid, first_tangent, shares, states()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.quadrics.quadric_collision import render
    from examples.util.animation import save_animation

    first_ellipsoid, first_tangent, shares, states = slide_setup(slide_offsets)
    recorded = []

    def streamed():
        for state in states:
            recorded.append(state)
            yield state

    save_animation(
        render.collision_animation(first_ellipsoid, first_tangent, shares, streamed()),
        "quadric_collision",
        frame_duration_ms,
        forward_then_back=True,
    )

    # checks
    _, placed_second, _, peak_dets, contact_plane, contact_point = posed_pair()
    assert peak_dets[0] > 0 and peak_dets[2] < 0
    assert peak_dets[1].abs() < 1e-12
    assert (contact_plane & first_ellipsoid(contact_plane)).abs() < 1e-12
    assert (contact_plane & placed_second[1](contact_plane)).abs() < 1e-12
    assert (placed_second[1](contact_plane).normalized() & contact_point).norm() < 1e-12
    assert recorded[0][6] > 0 and recorded[-1][6] < 0


if __name__ == "__main__":
    main()
