"""Three relativistic impulse scenarios assemble worldline geometry for rod reversal, the ladder paradox, and Bell's spaceships."""

from __future__ import annotations

import numpy as np

from examples.relativity.impulse.core import (
    Point, boost_map, coasting_events, impulse_train, mv, ringing_lengths,
    space, time, velocity_step, worldline_events, worldline_polylines,
)
from numga import stack

reversal_speed = 0.5
n_train_steps = 10
proper_step = 0.1
ladder_speed = 0.8
barn_length = 0.8
damping_ratio = 0.25
natural_frequency = 5.0


# --- math -----------------------------------------------------------------------------
def impulse_scene() -> tuple[Point, Point, Point, Point, Point, Point]:
    """A symmetric velocity reversal, its boosted rest-to-motion view, and a ten-step impulse train form the rod-impulse scene."""
    half_rapidity = np.arctanh(reversal_speed)
    kinks, directions = velocity_step(-half_rapidity, half_rapidity)            # [2] Point-, [2] Point-
    views = boost_map(np.array([0.0, half_rapidity]))                           # [2] Point- <- Point-
    view_kinks = views[:, None](kinks)                                          # [2, 2] Point-
    view_directions = views[:, None](directions)                                # [2, 2] Point-
    outer_times = np.array([-0.60, 1.45])
    slice_times = np.array([-0.30, 1.05])
    view_worldlines = stack([
        worldline_polylines(view_kinks[view, None], view_directions[view], outer_times)
        for view in range(2)
    ])                                                                          # [2, 3, 2] Point-
    view_slices = stack([
        worldline_events(slice_times, view_kinks[view, None], view_directions[view])
        for view in range(2)
    ])                                                                          # [2, 2, 2] Point-

    steps, train_directions = impulse_train(2 * half_rapidity, n_train_steps, proper_step)  # [n_train_steps, 2] Point-, [n_train_steps + 1] Point-
    finish = mv.t & steps[-1, 1]                                                # [] Scalar+
    train_bounds = stack((0 * finish - 0.30, finish + 0.40))                    # [2] Scalar+
    train = worldline_polylines(steps, train_directions, train_bounds)          # [n_train_steps + 2, 2] Point-
    train_slice = worldline_events(finish + 0.20, steps, train_directions)      # [2] Point-
    return view_worldlines, view_kinks, view_slices, train, steps, train_slice


def ladder_scene() -> tuple[tuple[Point, ...], tuple[Point, ...], tuple[Point, ...], tuple[Point, ...]]:
    """Four ladder-paradox panels contrast simultaneous door closure across frames with strain-preserving and compressive stops."""
    rapidity = np.arctanh(ladder_speed)
    moving_length = 1 / np.cosh(rapidity)
    rear_at_closure = (barn_length - moving_length) / 2
    closure = space * np.array([0.0, barn_length])                              # [2] Point-
    incoming = space * np.array([rear_at_closure, rear_at_closure + moving_length])  # [2] Point-
    incoming_direction = boost_map(np.asarray(rapidity))(time)                  # [] Point-
    at_rest = time                                                              # [] Point-

    barn_span = np.array([[-0.6], [0.65]])
    barn_frame = (
        coasting_events(closure, at_rest, barn_span), closure,
        coasting_events(incoming, incoming_direction, barn_span), incoming,
    )

    to_ladder = boost_map(np.asarray(-rapidity))                                # [] Point- <- Point-
    moving_closure, moving_events = to_ladder(closure), to_ladder(incoming)     # [2] Point-, [2] Point-
    moving_direction, moving_doors = to_ladder(incoming_direction), to_ladder(at_rest)  # [] Point-, [] Point-
    ladder_span = np.array([[-1.62], [0.30]])
    ladder_frame = (
        coasting_events(moving_closure, moving_doors, ladder_span), moving_closure,
        coasting_events(moving_events, moving_direction, ladder_span),
        coasting_events(moving_events, moving_direction, np.array([-1.40])),
        coasting_events(moving_closure, moving_doors, np.array([-0.72])),
    )

    stop_limits = np.array([-0.48, 2.12])
    kinks, directions = velocity_step(rapidity, 0.0)                            # [2] Point-, [2] Point-
    steps = (kinks + space * rear_at_closure)[None]                             # [1, 2] Point-
    contact_time = (barn_length - rear_at_closure - moving_length) / ladder_speed
    stopped = worldline_events(np.asarray(1.43), steps, directions)             # [2] Point-
    contact = worldline_events(np.asarray(contact_time), steps, directions)[1]  # [] Point-
    strain_preserving = (
        coasting_events(closure, at_rest, stop_limits[:, None]), closure,
        worldline_polylines(steps, directions, stop_limits), steps[0],
        incoming, stopped, contact,
    )

    n_ring_samples = 1000
    ring_times = np.linspace(0.0, 2.12, n_ring_samples)
    lengths = ringing_lengths(ring_times, damping_ratio, natural_frequency)
    centre = barn_length / 2
    ring = time * ring_times[:, None] + space * (centre + np.array([-0.5, 0.5]) * lengths[:, None])  # [n_ring_samples, 2] Point-
    relaxed = space * (centre + np.array([-0.5, 0.5]))                          # [2] Point-
    first_peak = ring_times <= np.pi / (natural_frequency * np.sqrt(1.0 - damping_ratio**2))
    ringing_contact = np.interp(barn_length, lengths[first_peak], ring_times[first_peak])
    ringing = (
        coasting_events(closure, at_rest, stop_limits[:, None]), closure,
        coasting_events(incoming, incoming_direction, np.array([[-0.48], [0.0]])),
        ring, coasting_events(relaxed, at_rest, np.array([[0.0], [2.12]])), incoming,
        coasting_events(closure, at_rest, np.array([ringing_contact])),
        coasting_events(relaxed, at_rest, np.array([1.91])),
    )
    return barn_frame, ladder_frame, strain_preserving, ringing


def spaceships_scene() -> tuple[Point, Point, Point, Point]:
    """Strain-preserving bisector timing and synchronized onboard-clock timing form the Bell's-spaceships comparison."""
    rapidity = np.arctanh(ladder_speed)
    preserved, directions = impulse_train(rapidity, n_train_steps, proper_step)  # [n_train_steps, 2] Point-, [n_train_steps + 1] Point-
    bell = preserved[:, :1] + space * np.array([0, 1])                          # [n_train_steps, 2] Point-
    schedules = stack((preserved, bell))                                        # [2, n_train_steps, 2] Point-

    final_frame = boost_map(np.asarray(-rapidity))                              # [] Point- <- Point-
    final_events = final_frame(schedules[:, -1] - schedules[:, -1, :1])         # [2, 2] Point-

    end_time = (mv.t & preserved[-1, 1]) + 0.53                                 # [] Scalar+
    outer_bounds = stack((0 * end_time - 0.30, end_time))                       # [2] Scalar+
    slice_bounds = stack((0 * end_time - 0.18, end_time - 0.22))                # [2] Scalar+
    tracks = stack([worldline_polylines(schedule, directions, outer_bounds) for schedule in (preserved, bell)])  # [2, n_train_steps + 2, 2] Point-
    slices = stack([worldline_events(slice_bounds, schedule, directions) for schedule in (preserved, bell)])     # [2, 2, 2] Point-
    return tracks, schedules, slices, final_events


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Renders all three relativistic impulse figures and checks proper-length, simultaneity, and rope-extension invariants."""
    from examples.relativity.impulse import render
    from examples.util.animation import save_figure

    impulse_data = impulse_scene()
    ladder_data = ladder_scene()
    spaceships_data = spaceships_scene()

    save_figure(render.impulse_figure(*impulse_data), "relativistic_rod_impulse")
    save_figure(render.ladder_figure(*ladder_data), "ladder_paradox")
    save_figure(render.spaceships_figure(*spaceships_data), "bell_spaceships")

    # checks
    _, view_kinks, view_slices, _, _, train_slice = impulse_data
    assert ((mv.t & view_kinks[0]).abs() < 1e-8).all()
    assert (((mv.t & view_kinks[1]) - np.array([0.0, 0.5])).abs() < 1e-8).all()
    boosted_lengths = mv.x & (view_slices[1, :, 1] - view_slices[1, :, 0])      # [2] Scalar+
    assert ((boosted_lengths - np.array([1.0, 0.6])).abs() < 1e-8).all()
    assert (((mv.x & (train_slice[1] - train_slice[0])) - 0.6).abs() < 1e-8).all()

    _, ladder_frame, strain_preserving, _ = ladder_data
    moving_closure, ladder_cut = ladder_frame[1], ladder_frame[3]
    stop_kinks, stopped = strain_preserving[3], strain_preserving[5]
    rapidity = np.arctanh(ladder_speed)
    assert (((mv.x & (ladder_cut[1] - ladder_cut[0])) - 1.0).abs() < 1e-8).all()
    assert (((mv.t & moving_closure) - np.array([0.0, -np.sinh(rapidity) * barn_length])).abs() < 1e-8).all()
    assert (((mv.x & stop_kinks) - np.array([0.1, 1.1])).abs() < 1e-8).all()
    assert (((mv.x & (stopped[1] - stopped[0])) - 1.0).abs() < 1e-8).all()

    _, schedules, slices, final_events = spaceships_data
    ticks = (schedules[:, 1:] - schedules[:, :-1]).norm()                       # [2, n_train_steps - 1, 2] Scalar+
    assert ((ticks[0, :, 0] - proper_step).abs() < 1e-8).all()
    assert ((ticks[1] - proper_step).abs() < 1e-8).all()
    assert ((mv.t & schedules[1, 0]).abs() < 1e-8).all()
    assert (((mv.x & final_events[:, 1]) - np.array([1.0, np.cosh(rapidity)])).abs() < 1e-8).all()
    lab_gaps = mv.x & (slices[:, 1, 1] - slices[:, 1, 0])                       # [2] Scalar+
    assert ((lab_gaps - np.array([1.0 / np.cosh(rapidity), 1.0])).abs() < 1e-8).all()


if __name__ == "__main__":
    main()
