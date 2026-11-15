"""Launch scenarios for a wooden rattleback hull with inset steel balls rolling on a flat table.

Releasing the tilted hull from rest converts gravitational potential energy into pitching, rolling, and steady spin.
Spinning the hull in the unfavourable clockwise sense destabilizes into rocking, halts, and reverses into the
preferred counterclockwise sense. Both scenarios share one quadric model for the sampled inertia and the ray-traced
geometry.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import islice

import numpy as np

from numga import concatenate, stack
from examples.mechanics.rattleback import core

mv = core.mv

semi_axes = np.array([5, 17 / 16, 1])
mass = 4
gravity_strength = 9.8

deck_height = 0.5
chamfer_roll = np.deg2rad(35)
chamfer_pitch = np.deg2rad(30)
chamfer_point = [3.2, 0, 0.2]

ball_radius = 0.4
ball_fore_offset = 3
ball_sink = 0.6
wood_density = 0.7
steel_density = 7.85
sample_spacing = 0.15

rocking_tilt = np.deg2rad(15)
reversal_spin_rate = -1.5
reversal_pitch_rate = 0.1

dt = 0.04
rocking_duration = 8
n_rocking_steps = round(rocking_duration / dt)
n_reversal_steps = 250

n_gallery_pixels = 420
gallery_dt = 0.06
gallery_duration_ms = 40
gallery_span = (0.4, 7.2)


def rattleback_body() -> tuple[core.Shape, core.Direction, core.Inertia, core.Quadric, core.Quadric, core.Point]:
    """The quadric hull and inset steel balls define the support map, centre offset, sampled inertia, and centred body geometry."""
    directions = core.mirrors.dual()                                  # [3] Direction-
    outline = (directions * (core.mirrors & core.Direction) * semi_axes).sum(axis=0)  # [] Direction- <- Direction-
    shape = (directions * (core.Plane & directions) * semi_axes**2).sum(axis=0)       # [] Direction- <- Plane+
    shell_planes = outline.adjugate().solve(core.mirrors)             # [3] Plane+
    underside = (shell_planes * (shell_planes & core.Point)).sum(axis=0) - core.infinity * (core.infinity & core.Point)  # [] Plane+ <- Point-
    deck = core.half_space(mv.z - core.infinity * deck_height)        # [] Plane+ <- Point-

    side_turns = (mv.xy * np.array([0, np.pi]) / 2).exp()             # [2] Motor+
    chamfer_turn = (mv.yz * (-chamfer_roll / 2)).exp() * (mv.xz * (chamfer_pitch / 2)).exp()  # [] Motor+
    chamfer_normal = chamfer_turn >> mv.z                             # [] Plane+
    chamfer_anchor = core.origin + mv.direction(chamfer_point)        # [] Point-
    chamfer_planes = side_turns >> (chamfer_normal - core.infinity * (chamfer_normal & chamfer_anchor))  # [2] Plane+
    chamfers = core.half_space(chamfer_planes)                        # [2] Plane+ <- Point-
    hull_half_width = semi_axes[1] * np.sqrt(1 - (ball_fore_offset / semi_axes[0])**2)
    ball_seat = core.origin + mv.direction([ball_fore_offset, hull_half_width + ball_radius * (1 - 2 * ball_sink), 0])  # [] Point-
    balls = core.ball(side_turns >> ball_seat, ball_radius)           # [2] Plane+ <- Point-
    wood = concatenate([underside[None], deck[None], chamfers, -balls])  # [6] Plane+ <- Point-

    low = -semi_axes - [0, 2 * ball_radius, 0]
    high = np.array([semi_axes[0], semi_axes[1] + 2 * ball_radius, deck_height])
    grid_axes = [np.arange(start, stop, sample_spacing) + sample_spacing / 2 for start, stop in zip(low, high)]
    grid_coords = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    n_grid = len(grid_coords)
    grid = mv.point(grid_coords)                                      # [n_grid] Point-
    samples, mass_centre, inertia = core.mass_distribution(
        wood, balls, grid, wood_density, steel_density, mass
    )
    n_samples = samples.shape[0]

    centre_shift = ((core.origin - mass_centre) * core.origin.inverse() / 2).exp()  # [] Motor+
    centred_wood = centre_shift >> wood(centre_shift << core.Point)   # [6] Plane+ <- Point-
    centred_balls = centre_shift >> balls(centre_shift << core.Point)  # [2] Plane+ <- Point-
    centred_samples = centre_shift >> samples                         # [n_samples] Point-
    return shape, core.origin - mass_centre, inertia, centred_wood, centred_balls, centred_samples


def launch_trajectory(tilt: float, spin_rate: float, pitch_rate: float, step_dt: float,
                      shape: core.Shape, centre: core.Direction,
                      inertia: core.Inertia) -> Iterator[tuple[core.Motor, core.Spin]]:
    """Yields placements (`[] Motor+`) and body-frame spins (`[] Spin+`) for one table-resting launch."""
    gravity = -core.up * gravity_strength                             # [] Direction-
    spin = mv.xy * spin_rate - mv.xz * pitch_rate                     # [] Spin+
    tilt_motor = (-mv.xz * (tilt / 2)).exp()                          # [] Motor+
    normal = (tilt_motor << core.up).dual_inverse()                   # [] Plane+
    height = (normal & shape(normal)).square_root() - (normal & centre)  # [] Scalar+
    motor = (core.up * core.origin.inverse() * (height / 2)).exp() * tilt_motor  # [] Motor+
    return core.trajectory(motor, spin, step_dt, shape, centre, inertia, mass, gravity, core.up)


def recorded_states(states: Iterator[tuple[core.Motor, core.Spin]],
                    history: list[tuple[core.Motor, core.Spin]]) -> Iterator[tuple[core.Motor, core.Spin]]:
    """Yields each `(motor, spin)` pair from `states` while appending it to `history`."""
    for state in states:
        history.append(state)
        yield state


def main() -> None:
    """The scenario runner writes the setup still and both launch animations to `plots/` and checks spin reversal and energy conservation."""
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.rattleback import render

    duration_ms = round(dt * 1000)
    shape, centre, inertia, wood, balls, samples = rattleback_body()
    gravity = -core.up * gravity_strength                             # [] Direction-

    table_plane = core.up.dual_inverse()                              # [] Plane+
    rest_height = (table_plane & shape(table_plane)).square_root() - (table_plane & centre)  # [] Scalar+
    rest_placement = (core.up * core.origin.inverse() * (rest_height / 2)).exp()  # [] Motor+
    save_figure(render.setup_figure(wood, balls, samples, rest_placement), "rattleback_body")

    rocking_states: list[tuple[core.Motor, core.Spin]] = []
    rocking_stream = islice(launch_trajectory(rocking_tilt, 0.0, 0.0, dt, shape, centre, inertia), n_rocking_steps + 1)
    save_animation(render.frames(wood, balls, samples, recorded_states(rocking_stream, rocking_states)),
                   "rattleback_rocking", duration_ms)

    reversal_states: list[tuple[core.Motor, core.Spin]] = []
    reversal_stream = islice(
        launch_trajectory(0.0, reversal_spin_rate, reversal_pitch_rate, dt, shape, centre, inertia),
        n_reversal_steps + 1,
    )
    save_animation(render.frames(wood, balls, samples, recorded_states(reversal_stream, reversal_states)),
                   "rattleback_reversal", duration_ms)

    # checks
    rocking_motors, rocking_spins = (stack(items) for items in zip(*rocking_states))
    reversal_motors, reversal_spins = (stack(items) for items in zip(*reversal_states))
    rocking_vertical_spin = -(rocking_motors >> rocking_spins).scalar_product(mv.xy)    # [n_rocking_steps + 1] Scalar+
    reversal_vertical_spin = -(reversal_motors >> reversal_spins).scalar_product(mv.xy)  # [n_reversal_steps + 1] Scalar+
    rocking_energies = core.energy(rocking_motors, rocking_spins, shape, centre, inertia, mass, gravity, core.up)      # [n_rocking_steps + 1] Scalar+
    reversal_energies = core.energy(reversal_motors, reversal_spins, shape, centre, inertia, mass, gravity, core.up)   # [n_reversal_steps + 1] Scalar+

    assert rocking_vertical_spin[0].abs() < 1e-11 and rocking_vertical_spin[-1] > 1.4
    assert reversal_vertical_spin[0] < -1.4 and reversal_vertical_spin[-1] > 1.4
    assert ((rocking_energies / rocking_energies[0] - 1).abs() < 1e-2).all()
    assert ((reversal_energies / reversal_energies[0] - 1).abs() < 1e-2).all()


def pinned() -> None:
    """The gallery runner writes the pinned spin-reversal animation."""
    from examples.util.animation import pin_animation
    from examples.mechanics.rattleback import render

    first_step, last_step = (round(time / gallery_dt) for time in gallery_span)
    shape, centre, inertia, wood, balls, samples = rattleback_body()
    states = islice(
        launch_trajectory(0.0, reversal_spin_rate, reversal_pitch_rate, gallery_dt, shape, centre, inertia),
        first_step, last_step + 1,
    )
    pin_animation(render.sized_frames(wood, balls, samples, states, n_gallery_pixels), "rattleback",
                  gallery_duration_ms)


if __name__ == "__main__":
    main()
