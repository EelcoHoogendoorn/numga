"""Independent mechanical checks of the planar normal modes and stiffness example."""

from __future__ import annotations

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np

from examples.mechanics.modes import render, scenarios
from examples.mechanics.modes.core import (
    Inertia,
    Point,
    SpringExtension,
    Stiffness,
    Suspension,
    Twist,
    Wrench,
    mv,
    suspension as inputs,
)
from examples.mechanics.modes.render import coordinates


def spring_stiffness(
    lines: Wrench,
    spring_constants: np.ndarray,
) -> tuple[Stiffness, SpringExtension]:
    """Compute total stiffness extensor and extension linear form from spring lines.

    Parameters
    ----------
    lines : [n_springs] Wrench
        Normalized lines of action of the springs.
    spring_constants : [n_springs] float
        Spring elastic constants.

    Returns
    -------
    stiffness : [] Stiffness
        Total stiffness extensor (Wrench <- Twist).
    extension : [n_springs] SpringExtension
        Linear form measuring spring extension from twist displacement (Scalar <- Twist).
    """
    extension: SpringExtension = Twist & lines                                   # [n_springs] Scalar <- Twist
    spring_stiffness_dyads: Stiffness = lines * extension * spring_constants     # [n_springs] Wrench <- Twist
    stiffness: Stiffness = spring_stiffness_dyads.sum(axis=0)                   # [] Wrench <- Twist
    return stiffness, extension


def body_inertia(
    points: Point,
    masses: np.ndarray,
) -> Inertia:
    """Compute rigid body inertia extensor from lumped mass points.

    Parameters
    ----------
    points : [n_points] Point
        Sampling points on the rigid body.
    masses : [n_points] float
        Point masses.

    Returns
    -------
    Inertia
        [] Inertia extensor (Wrench <- Twist).
    """
    velocities: Point = points.commutator(Twist)                                # [n_points] Point <- Twist
    point_momenta: Inertia = (points & velocities) * masses                     # [n_points] Wrench <- Twist
    return point_momenta.sum(axis=0)                                            # [] Wrench <- Twist


def normal_modes(
    stiffness: Stiffness,
    inertia: Inertia,
) -> tuple[Twist, np.ndarray]:
    """Solve the generalized eigenvalue problem for normal vibration modes.

    Parameters
    ----------
    stiffness : [] Stiffness
        Stiffness extensor (Wrench <- Twist).
    inertia : [] Inertia
        Inertia extensor (Wrench <- Twist).

    Returns
    -------
    modes : [3] Twist
        Normal mode eigenvectors.
    frequencies : [3] np.ndarray
        Natural frequencies in Hz.
    """
    pe_form = Twist & stiffness                                                 # [] Scalar <- (Twist, Twist)
    ke_form = Twist & inertia                                                   # [] Scalar <- (Twist, Twist)
    values, modes = pe_form.eigh(ke_form)                                       # values: [3] Scalar, modes: [3] Twist
    frequencies = np.sqrt(np.maximum(values.kernel[..., 0], 0.0)) / (2 * np.pi) # [3] float
    return modes, frequencies


@dataclass
class System:
    geometry: Suspension
    stiffness: Stiffness            # [] Wrench <- Twist
    extension: SpringExtension      # [n_springs] Scalar <- Twist
    inertia: Inertia                # [] Wrench <- Twist

    def __getattr__(self, name: str) -> object:
        return getattr(self.geometry, name)


def suspension(springs: int) -> System:
    """Construct full suspension system including geometry, stiffness, and inertia."""
    geometry = inputs(springs)
    lines: Wrench = (geometry.anchors & geometry.attachments).normalized()       # [n_springs] Wrench
    stiffness, extension = spring_stiffness(lines, geometry.spring_constants)
    return System(geometry, stiffness, extension,
                  body_inertia(geometry.mass_points, geometry.masses))


def test_scenario_has_analytic_modes_and_renders():
    free, restrained = scenarios.suspensions()
    np.testing.assert_allclose((2 * np.pi * free.frequencies.to_array())**2,
                               [0, 12, 12 * .8**2 / (5 / 12)], atol=1e-12)
    assert np.all(restrained.frequencies.to_array() > 0)
    assert isinstance(render.draw_modes([free, restrained], "Normal Modes"), plt.Figure)
    frames = render.animate_modes([free, restrained], 4)
    assert frames and frames[0].ndim == 3 and all(f.shape == frames[0].shape for f in frames)


def test_mathematics_does_not_import_plotting():
    import subprocess
    import sys

    probe = (
        "import examples.mechanics.modes.core, sys; "
        "print([m for m in sys.modules if m.split('.')[0] in ('matplotlib', 'PIL')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_two_springs_have_analytic_slide_bounce_and_rock_frequencies():
    system = suspension(2)
    _, frequencies = normal_modes(system.stiffness, system.inertia)
    # Mass 1, polar inertia (width² + height²)/12, anchors at x = ±.8.
    expected_squared = [0, 2 * 6, 2 * 6 * .8**2 / (5 / 12)]
    np.testing.assert_allclose((2 * np.pi * frequencies)**2, expected_squared, atol=1e-12)
    _, restrained = normal_modes(suspension(3).stiffness, system.inertia)
    assert np.all(restrained > 0)


def test_spring_extension_and_energy_match_finite_rigid_displacements():
    system = suspension(3)
    q = mv.bivector([.3, -.4, .25])
    h = 1e-4
    # Independent exact geometry: move points with a motor and measure lengths.
    anchors = coordinates(system.anchors)
    rest = np.linalg.norm(coordinates(system.attachments) - anchors, axis=-1)
    plus = coordinates((q * (-h / 2)).exp().normalized() >> system.attachments)
    minus = coordinates((q * (h / 2)).exp().normalized() >> system.attachments)
    eplus = np.linalg.norm(plus - anchors, axis=-1) - rest
    eminus = np.linalg.norm(minus - anchors, axis=-1) - rest
    np.testing.assert_allclose(
        (eplus - eminus) / (2 * h), system.extension(q).kernel[:, 0], atol=2e-8,
    )
    actual = np.sum(system.spring_constants * (eplus**2 + eminus**2)) / (4 * h**2)
    predicted = .5 * q.regressive(system.stiffness(q)).kernel.item()
    np.testing.assert_allclose(actual, predicted, rtol=2e-7)


def test_mass_normalized_modes_and_spring_work():
    system = suspension(3)
    modes, frequencies = normal_modes(system.stiffness, system.inertia)
    mass = modes[:, None].regressive(system.inertia(modes[None, :])).kernel[..., 0]
    elastic = modes[:, None].regressive(system.stiffness(modes[None, :])).kernel[..., 0]
    np.testing.assert_allclose(mass, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(elastic, np.diag((2 * np.pi * frequencies)**2), atol=1e-12)
    extensions = system.extension(modes[:, None]).kernel[..., 0]
    work = (extensions * system.spring_constants) @ extensions.T
    np.testing.assert_allclose(work, elastic, atol=1e-12)


def test_modes_do_not_depend_on_world_pose():
    system = suspension(3)
    motor = (mv.xw * -.9 + mv.yw * .3).exp() * (mv.xy * .37).exp().normalized()
    lines = system.anchors.regressive(system.attachments).normalized()
    stiffness, _ = spring_stiffness(motor >> lines, system.spring_constants)
    inertia = body_inertia(motor >> system.mass_points, system.masses)
    _, before = normal_modes(system.stiffness, system.inertia)
    _, after = normal_modes(stiffness, inertia)
    np.testing.assert_allclose(after, before, atol=1e-10)
