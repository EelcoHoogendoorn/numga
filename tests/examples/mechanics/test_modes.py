"""Mechanical and rendering checks verify the suspended-plate normal modes example."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.modes import core, render, scenarios
from examples.mechanics.modes.core import mv


def test_analytic_unrestrained_and_coupled_frequencies() -> None:
    """Two vertical springs match analytic slide, bounce, and rock frequencies while three springs restrain all modes."""
    corners, masses, vertical_anchors, vertical_attachments, vertical_constants = scenarios.vertical_suspension()
    inertia = core.body_inertia(core.quadrature_points(corners), masses)         # [] Forque- <- Twist+
    unrestrained_stiffness = core.spring_stiffness(vertical_anchors, vertical_attachments, vertical_constants)  # [] Forque- <- Twist+
    frequencies, _ = core.normal_modes(unrestrained_stiffness, inertia)          # [3] Scalar+, Twist+

    # A 2-by-1 plate of unit mass has polar inertia 5/12, with vertical springs of constant 6 at +/- 0.8 from centre.
    expected_squared = mv.scalar([[0.0], [12.0], [12.0 * 0.64 / (5.0 / 12.0)]])  # [3] Scalar+
    computed_squared = (2 * np.pi * frequencies).squared()                       # [3] Scalar+
    assert ((computed_squared - expected_squared).abs() < 1e-10).all()

    _, _, coupled_anchors, coupled_attachments, coupled_constants = scenarios.coupled_suspension()
    coupled_stiffness = core.spring_stiffness(coupled_anchors, coupled_attachments, coupled_constants)  # [] Forque- <- Twist+
    coupled_frequencies, _ = core.normal_modes(coupled_stiffness, inertia)       # [3] Scalar+, Twist+
    assert (coupled_frequencies > 0).all()


def test_mass_orthogonality_and_modal_balance() -> None:
    """Coupled mode twists are orthonormal under the kinetic form and diagonalize the potential form."""
    corners, masses, anchors, attachments, spring_constants = scenarios.coupled_suspension()
    inertia = core.body_inertia(core.quadrature_points(corners), masses)         # [] Forque- <- Twist+
    stiffness = core.spring_stiffness(anchors, attachments, spring_constants)    # [] Forque- <- Twist+
    frequencies, mode_twists = core.normal_modes(stiffness, inertia)             # [3] Scalar+, Twist+

    mass_pairings = mode_twists[:, None] & inertia(mode_twists[None, :])         # [3, 3] Scalar+
    elastic_pairings = mode_twists[:, None] & stiffness(mode_twists[None, :])    # [3, 3] Scalar+
    identity = mv.scalar(np.eye(3)[..., None])                                   # [3, 3] Scalar+
    diagonal_rates = mv.scalar(np.eye(3)[..., None]) * (2 * np.pi * frequencies).squared()  # [3, 3] Scalar+
    assert ((mass_pairings - identity).abs() < 1e-10).all()
    assert ((elastic_pairings - diagonal_rates).abs() < 1e-10).all()


def test_euclidean_invariance_under_world_motor() -> None:
    """Rigid world transformations of plate and springs preserve all three natural frequencies."""
    corners, masses, anchors, attachments, spring_constants = scenarios.coupled_suspension()
    inertia = core.body_inertia(core.quadrature_points(corners), masses)         # [] Forque- <- Twist+
    stiffness = core.spring_stiffness(anchors, attachments, spring_constants)    # [] Forque- <- Twist+
    before, _ = core.normal_modes(stiffness, inertia)                            # [3] Scalar+, Twist+

    motor = (mv.xw * -0.9 + mv.yw * 0.3).exp() * (mv.xy * 0.37).exp()            # [] Rotor+
    moved_corners = motor >> corners                                             # [4] Point-
    moved_anchors = motor >> anchors                                             # [3] Point-
    moved_attachments = motor >> attachments                                     # [3] Point-
    moved_inertia = core.body_inertia(core.quadrature_points(moved_corners), masses)  # [] Forque- <- Twist+
    moved_stiffness = core.spring_stiffness(moved_anchors, moved_attachments, spring_constants)  # [] Forque- <- Twist+
    after, _ = core.normal_modes(moved_stiffness, moved_inertia)                 # [3] Scalar+, Twist+
    assert ((after - before).abs() < 1e-9).all()


def test_mode_oscillation_and_views_render() -> None:
    """The oscillation generator and static and animated views execute without writing files."""
    corners, masses, anchors, attachments, spring_constants = scenarios.coupled_suspension()
    inertia = core.body_inertia(core.quadrature_points(corners), masses)         # [] Forque- <- Twist+
    stiffness = core.spring_stiffness(anchors, attachments, spring_constants)    # [] Forque- <- Twist+
    frequencies, mode_twists = core.normal_modes(stiffness, inertia)             # [3] Scalar+, Twist+

    render.setup_figure(corners, anchors, attachments)
    render.mode_figure(corners, anchors, attachments, mode_twists)
    frames = list(render.mode_frames(corners, anchors, attachments, mode_twists, core.mode_oscillation(frequencies, mode_twists, 1)))
    plt.close("all")
    assert len(frames) == 1
