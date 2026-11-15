"""A wheel of three rim masses turns under a tangential push beside its mirror image in the plane and tilted in space.

Instantiating `core` on `VGA2D` and `VGA3D` evaluates the same expressions for the planar wheel and the tilted spatial
wheel.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.algebras import VGA2D, VGA3D
from examples import instantiate

CORE = "examples.orientation.spinning_wheel.core"

# Frame duration in milliseconds, frame count per revolution, rim masses, and angular offsets in radians.
frame_duration_ms = 40
n_frames = 81
masses = np.array([1, 1.5, 0.5])                                                # [3]
mass_angles = np.array([0, 2.1, 4.0])                                           # [3]
push_strength = 1.5
speed = 3
tilt_angle = 0.9


def mirrored_wheels(wheel_plane, spoke, mirror):
    """A wheel turning in `wheel_plane` and its reflection across `mirror` share positive kinetic energy and mechanical
    power while reversing angular momentum and torque."""
    core = instantiate(CORE, wheel_plane.algebra)
    seats = (wheel_plane * (-mass_angles / 2)).exp()                            # [3] Rotor+
    points = seats >> spoke                                                     # [3] Point-
    rate = wheel_plane * speed                                                  # [] Rate+
    handle = points[0]                                                          # [] Point-
    # Tangential push along the first mass's direction of motion.
    push = handle.commutator(wheel_plane) * push_strength                       # [] Point-

    points = stack([points, mirror >> points])                                  # [2, 3] Point-
    handles = stack([handle, mirror >> handle])                                 # [2] Point-
    pushes = stack([push, mirror >> push])                                      # [2] Point-
    rates = stack([rate, mirror >> rate])                                       # [2] Rate+
    return points, pushes, rates, *core.readings(points, masses, handles, pushes, rates)


def wheel_in_the_plane():
    """The planar wheel in `VGA2D` turns beside its reflection across the plane `x = 0`."""
    mv = instantiate(CORE, VGA2D).mv
    return mirrored_wheels(mv.xy, mv.x.dual(), mv.x)


def wheel_in_space():
    """Tilting the wheel out of `xy` by `tilt_angle` in `yz` embeds the experiment and its `x = 0` reflection in `VGA3D`."""
    mv = instantiate(CORE, VGA3D).mv
    tilt = (mv.yz * (-tilt_angle / 2)).exp()                                    # [] Rotor+
    return mirrored_wheels(tilt >> mv.xy, mv.x.dual(), mv.x)


def main() -> None:
    from examples.util.animation import save_animation
    from examples.orientation.spinning_wheel import render

    times = np.linspace(0, 2 * np.pi / speed, n_frames, endpoint=False)
    scenes = {"spinning_wheel_plane": wheel_in_the_plane(), "spinning_wheel_space": wheel_in_space()}
    for name, (points, pushes, rates, momenta, energies, torques, powers) in scenes.items():
        core = instantiate(CORE, rates.algebra)
        for label, values in (("Angular momentum", momenta), ("Kinetic energy", energies), ("Torque", torques),
                              ("Power", powers)):
            for wheel, value in zip(("the wheel", "its mirror image"), values):
                print(f"{name}: {label} of {wheel}", value.coefficient_table(), sep="\n")
        save_animation(render.animate(points, rates, masses, core.turning(points, pushes, rates, times)), name,
                       frame_duration_ms)

    # checks
    for points, pushes, rates, momenta, energies, torques, powers in scenes.values():
        mirror = instantiate(CORE, rates.algebra).mv.x                          # [] Plane+
        velocities = points.commutator(rates[:, None])                          # [2, 3] Point-
        half_mass_speeds = (masses * velocities.dot(velocities)).sum(axis=-1) / 2   # [2] Scalar+
        assert ((energies - half_mass_speeds).abs() < 1e-9).all()
        assert ((powers - pushes.dot(velocities[:, 0])).abs() < 1e-9).all()
        assert (momenta[1] - (mirror >> momenta[0])).antinorm() < 1e-9
        assert (torques[1] - (mirror >> torques[0])).antinorm() < 1e-9
        assert (energies[1] - energies[0]).abs() < 1e-9
        assert (powers[1] - powers[0]).abs() < 1e-9
    (*_, plane_energies, _, plane_powers), (*_, space_energies, _, space_powers) = scenes.values()
    embed = space_energies.context.include_from_algebra
    assert ((space_energies - embed(plane_energies)).abs() < 1e-9).all()
    assert ((space_powers - embed(plane_powers)).abs() < 1e-9).all()


def pinned() -> None:
    """Saves the repository-tracked animation of the planar wheel and its mirror image."""
    from examples.orientation.spinning_wheel import render
    from examples.util.animation import pin_animation

    times = np.linspace(0, 2 * np.pi / speed, n_frames, endpoint=False)
    points, pushes, rates, *_ = wheel_in_the_plane()
    core = instantiate(CORE, rates.algebra)
    pin_animation(render.animate(points, rates, masses, core.turning(points, pushes, rates, times)),
                  "spinning_wheel_plane", frame_duration_ms)


if __name__ == "__main__":
    main()
