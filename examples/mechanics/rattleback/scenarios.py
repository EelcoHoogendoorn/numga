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

from numga import concatenate, dim, stack
from examples.mechanics.rattleback import core

mv = core.mv

semi_axes = mv.scalar([[5], [17 / 16], [1]], unit=dim.length)         # [3] Scalar+, length
mass = mv.scalar(4.0, unit=dim.mass)                                  # [] Scalar+, mass
gravity_strength = mv.scalar(9.8, unit=dim.length / dim.time**2)      # [] Scalar+, length/time^2

deck_height = mv.scalar(0.5, unit=dim.length)                         # [] Scalar+, length
chamfer_roll = np.deg2rad(35)
chamfer_pitch = np.deg2rad(30)
chamfer_point = [3.2, 0, 0.2]

ball_radius = mv.scalar(0.4, unit=dim.length)                         # [] Scalar+, length
ball_fore_offset = mv.scalar(3.0, unit=dim.length)                    # [] Scalar+, length
ball_sink = 0.6
wood_density = mv.scalar(0.7, unit=dim.mass / dim.length**3)          # [] Scalar+, mass/length^3
steel_density = mv.scalar(7.85, unit=dim.mass / dim.length**3)        # [] Scalar+, mass/length^3
sample_spacing = mv.scalar(0.15, unit=dim.length)                     # [] Scalar+, length

rocking_tilt = np.deg2rad(15)
reversal_spin_rate = mv.scalar(-1.5, unit=1 / dim.time)               # [] Scalar+, 1/time
reversal_pitch_rate = mv.scalar(0.1, unit=1 / dim.time)               # [] Scalar+, 1/time

dt = mv.scalar(0.04, unit=dim.time)                                   # [] Scalar+, time
rocking_duration = mv.scalar(8.0, unit=dim.time)                      # [] Scalar+, time
n_rocking_steps = round(float((rocking_duration / dt).to_array()))
n_reversal_steps = 250

n_gallery_pixels = 420
gallery_dt = mv.scalar(0.06, unit=dim.time)                           # [] Scalar+, time
gallery_duration_ms = 40
gallery_span = (mv.scalar(0.4, unit=dim.time), mv.scalar(7.2, unit=dim.time))


def sample_grid(axes: core.Scalar, radius: core.Scalar, top: core.Scalar, step: core.Scalar) -> core.Point:
    """Regular 3D grid of unit-weight sample points spanning the hull and inset balls."""
    zero = mv.scalar(0.0, unit=dim.length)
    low = ((-axes - stack((zero, 2 * radius, zero))) / dim.length).to_array()
    high = (stack((axes[0], axes[1] + 2 * radius, top)) / dim.length).to_array()
    spacing = float((step / dim.length).to_array())
    grid_axes = [np.arange(start, stop, spacing) + spacing / 2 for start, stop in zip(low, high)]
    grid_coords = np.stack(np.meshgrid(*grid_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return mv.point(grid_coords, unit=dim.length)                     # [n_grid] Point-, 1 | length


def rattleback_body() -> tuple[core.Shape, core.Direction, core.Inertia, core.Quadric, core.Quadric, core.Point]:
    """The quadric hull and inset steel balls define the support map, centre offset, sampled inertia, and centred body geometry."""
    directions = core.mirrors.dual()                                  # [3] Direction-
    outline = (directions * (core.mirrors & core.Direction) * semi_axes).sum(axis=0)  # [] Direction- <- Direction-, length
    shape = (directions * (core.Plane & directions) * semi_axes**2).sum(axis=0)       # [] Direction- <- Plane+, length^2
    shell_planes = outline.adjugate().solve(core.mirrors)             # [3] Plane+, 1/length
    underside = (shell_planes * (shell_planes & core.Point)).sum(axis=0) - core.infinity * (core.infinity & core.Point)  # [] Plane+ <- Point-
    # Dividing plane half-spaces by the hull length scale puts every quadric pairing on one dimensionless footing.
    unit_length = semi_axes[2]
    deck = core.half_space((mv.z - core.infinity * deck_height) / unit_length)        # [] Plane+ <- Point-

    side_turns = (mv.xy * np.array([0, np.pi]) / 2).exp()             # [2] Motor+
    chamfer_turn = (mv.yz * (-chamfer_roll / 2)).exp() * (mv.xz * (chamfer_pitch / 2)).exp()  # [] Motor+
    chamfer_normal = chamfer_turn >> mv.z                             # [] Plane+
    chamfer_anchor = core.origin + mv.direction(chamfer_point, unit=dim.length)  # [] Point-, 1 | length
    chamfer_planes = side_turns >> (chamfer_normal - core.infinity * (chamfer_normal & chamfer_anchor))  # [2] Plane+
    chamfers = core.half_space(chamfer_planes / unit_length)          # [2] Plane+ <- Point-
    hull_half_width = semi_axes[1] * (1 - (ball_fore_offset / semi_axes[0])**2).square_root()
    ball_seat = (
        core.origin
        + ball_fore_offset * mv.x.dual()
        + (hull_half_width + ball_radius * (1 - 2 * ball_sink)) * mv.y.dual()
    )                                                                 # [] Point-, 1 | length
    balls = core.ball(side_turns >> ball_seat, ball_radius)           # [2] Plane+ <- Point-
    wood = concatenate([underside[None], deck[None], chamfers, -balls])  # [6] Plane+ <- Point-

    grid = sample_grid(semi_axes, ball_radius, deck_height, sample_spacing)
    samples, mass_centre, inertia = core.mass_distribution(
        wood, balls, grid, wood_density, steel_density, mass
    )
    n_samples = samples.shape[0]

    centre_shift = ((core.origin - mass_centre) * core.origin.inverse() / 2).exp()  # [] Motor+, 1 | length
    centred_wood = centre_shift >> wood(centre_shift << core.Point)   # [6] Plane+ <- Point-
    centred_balls = centre_shift >> balls(centre_shift << core.Point)  # [2] Plane+ <- Point-
    centred_samples = centre_shift >> samples                         # [n_samples] Point-, 1 | length
    return shape, core.origin - mass_centre, inertia, centred_wood, centred_balls, centred_samples


def launch_trajectory(tilt: float, spin_rate: core.Scalar, pitch_rate: core.Scalar, step_dt: core.Scalar,
                      shape: core.Shape, centre: core.Direction,
                      inertia: core.Inertia) -> Iterator[tuple[core.Motor, core.Spin]]:
    """Yields placements (`[] Motor+`) and body-frame spins (`[] Spin+`) for one table-resting launch."""
    gravity = -core.up * gravity_strength                             # [] Direction-, length/time^2
    spin = mv.xy * spin_rate - mv.xz * pitch_rate                     # [] Spin+, 1/time
    tilt_motor = (-mv.xz * (tilt / 2)).exp()                          # [] Motor+
    normal = (tilt_motor << core.up).dual_inverse()                   # [] Plane+
    height = (normal & shape(normal)).square_root() - (normal & centre)  # [] Scalar+, length
    motor = (core.up * core.origin.inverse() * (height / 2)).exp() * tilt_motor  # [] Motor+, 1 | length
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

    duration_ms = round(float((dt / dim.time).to_array()) * 1000)
    shape, centre, inertia, wood, balls, samples = rattleback_body()
    gravity = -core.up * gravity_strength                             # [] Direction-, length/time^2

    table_plane = core.up.dual_inverse()                              # [] Plane+
    rest_height = (table_plane & shape(table_plane)).square_root() - (table_plane & centre)  # [] Scalar+, length
    rest_placement = (core.up * core.origin.inverse() * (rest_height / 2)).exp()  # [] Motor+, 1 | length
    save_figure(render.setup_figure(wood, balls, samples, rest_placement), "rattleback_body")

    rocking_states: list[tuple[core.Motor, core.Spin]] = []
    still = mv.scalar(0.0, unit=1 / dim.time)
    rocking_stream = islice(launch_trajectory(rocking_tilt, still, still, dt, shape, centre, inertia), n_rocking_steps + 1)
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
    rocking_vertical_spin = -(rocking_motors >> rocking_spins).scalar_product(mv.xy)    # [n_rocking_steps + 1] Scalar+, 1/time
    reversal_vertical_spin = -(reversal_motors >> reversal_spins).scalar_product(mv.xy)  # [n_reversal_steps + 1] Scalar+, 1/time
    rocking_energies = core.energy(rocking_motors, rocking_spins, shape, centre, inertia, mass, gravity, core.up)      # [n_rocking_steps + 1] Scalar+, length^2*mass/time^2
    reversal_energies = core.energy(reversal_motors, reversal_spins, shape, centre, inertia, mass, gravity, core.up)   # [n_reversal_steps + 1] Scalar+, length^2*mass/time^2

    assert rocking_vertical_spin[0].abs() < mv.scalar(1e-11, unit=1 / dim.time) and rocking_vertical_spin[-1] > mv.scalar(1.4, unit=1 / dim.time)
    assert reversal_vertical_spin[0] < mv.scalar(-1.4, unit=1 / dim.time) and reversal_vertical_spin[-1] > mv.scalar(1.4, unit=1 / dim.time)
    assert ((rocking_energies / rocking_energies[0] - 1).abs() < 1e-2).all()
    assert ((reversal_energies / reversal_energies[0] - 1).abs() < 1e-2).all()


def pinned() -> None:
    """The gallery runner writes the pinned spin-reversal animation."""
    from examples.util.animation import pin_animation
    from examples.mechanics.rattleback import render

    first_step, last_step = (round(float((time / gallery_dt).to_array())) for time in gallery_span)
    shape, centre, inertia, wood, balls, samples = rattleback_body()
    states = islice(
        launch_trajectory(0.0, reversal_spin_rate, reversal_pitch_rate, gallery_dt, shape, centre, inertia),
        first_step, last_step + 1,
    )
    pin_animation(render.sized_frames(wood, balls, samples, states, n_gallery_pixels), "rattleback",
                  gallery_duration_ms)


if __name__ == "__main__":
    main()
