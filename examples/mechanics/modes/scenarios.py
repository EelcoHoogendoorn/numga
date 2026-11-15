"""Two spring arrangements support the same planar rectangular plate."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.mechanics.modes import core
from examples.mechanics.modes.core import Point, Twist


n_frames = 150
frame_duration_ms = 40

plate_coordinates = np.array([[0.0, 0.5], [2.0, 0.5], [2.0, 1.5], [0.0, 1.5]])  # [4, 2]
plate_masses = np.full(4, 0.25)                                                 # [4]

vertical_attachment_coordinates = np.array([[0.2, 1.5], [1.8, 1.5]])            # [2, 2]
vertical_anchor_coordinates = np.array([[0.2, 2.55], [1.8, 2.55]])              # [2, 2]
vertical_spring_constants = np.array([6.0, 6.0])                                # [2]

angled_attachment_coordinates = np.array([[2.0, 1.0]])                          # [1, 2]
angled_anchor_coordinates = np.array([[2.9, 1.85]])                             # [1, 2]
angled_spring_constants = np.array([6.0])                                       # [1]

coupled_attachment_coordinates = np.concatenate([vertical_attachment_coordinates, angled_attachment_coordinates])  # [3, 2]
coupled_anchor_coordinates = np.concatenate([vertical_anchor_coordinates, angled_anchor_coordinates])              # [3, 2]
coupled_spring_constants = np.concatenate([vertical_spring_constants, angled_spring_constants])                    # [3]


def vertical_suspension() -> tuple[Point, np.ndarray, Point, Point, np.ndarray]:
    """Two symmetric vertical springs (`[2] Point-`) support the uniform rectangular plate (`[4] Point-`)."""
    corners = core.mv.point(plate_coordinates)                                   # [4] Point-
    anchors = core.mv.point(vertical_anchor_coordinates)                         # [2] Point-
    attachments = core.mv.point(vertical_attachment_coordinates)                 # [2] Point-
    return corners, plate_masses, anchors, attachments, vertical_spring_constants


def coupled_suspension() -> tuple[Point, np.ndarray, Point, Point, np.ndarray]:
    """Two vertical springs and one off-centre angled spring (`[3] Point-`) restrain all three planar motions of the plate (`[4] Point-`)."""
    corners = core.mv.point(plate_coordinates)                                   # [4] Point-
    anchors = core.mv.point(coupled_anchor_coordinates)                          # [3] Point-
    attachments = core.mv.point(coupled_attachment_coordinates)                  # [3] Point-
    return corners, plate_masses, anchors, attachments, coupled_spring_constants


def main() -> None:
    """The scenario entry point saves the coupled mode figure and synchronized modal animation to `plots/`."""
    from examples.util.animation import save_animation, save_figure
    from examples.mechanics.modes import render

    corners, masses, anchors, attachments, spring_constants = coupled_suspension()
    mass_points = core.quadrature_points(corners)                                # [4] Point-
    inertia = core.body_inertia(mass_points, masses)                             # [] Forque- <- Twist+
    stiffness = core.spring_stiffness(anchors, attachments, spring_constants)    # [] Forque- <- Twist+
    frequencies, mode_twists = core.normal_modes(stiffness, inertia)             # [3] Scalar+, Twist+

    save_figure(render.mode_figure(corners, anchors, attachments, mode_twists), "modes")
    streamed_twists = []

    def recorded_states() -> Iterator[Twist]:
        for state in core.mode_oscillation(frequencies, mode_twists, n_frames):
            streamed_twists.append(state)
            yield state

    save_animation(
        render.mode_frames(corners, anchors, attachments, mode_twists, recorded_states()),
        "modes", frame_duration_ms,
    )

    # checks
    angular_rates_squared = (2 * np.pi * frequencies).squared()                  # [3] Scalar+
    modal_residuals = stiffness(mode_twists) - angular_rates_squared * inertia(mode_twists)  # [3] Forque-
    return_error = corners.commutator(streamed_twists[-1][0] - mode_twists[0]).antinorm()    # [4] Scalar+
    assert len(streamed_twists) == n_frames
    assert bool((frequencies > 0).all())
    assert bool(((mode_twists[:, None] & modal_residuals[None, :]).abs() < 1e-9).all())
    assert bool((return_error < 1e-9).all())


def pinned() -> None:
    """The pinned gallery entry point records the three coupled vibration modes."""
    from examples.util.animation import pin_animation
    from examples.mechanics.modes import render

    corners, masses, anchors, attachments, spring_constants = coupled_suspension()
    inertia = core.body_inertia(core.quadrature_points(corners), masses)         # [] Forque- <- Twist+
    stiffness = core.spring_stiffness(anchors, attachments, spring_constants)    # [] Forque- <- Twist+
    frequencies, mode_twists = core.normal_modes(stiffness, inertia)             # [3] Scalar+, Twist+
    states = core.mode_oscillation(frequencies, mode_twists, n_frames)
    pin_animation(
        render.mode_frames(corners, anchors, attachments, mode_twists, states),
        "modes", frame_duration_ms,
    )


if __name__ == "__main__":
    main()
