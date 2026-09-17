"""Independent mechanical checks of the planar spring stiffness example."""

import numpy as np

from dataclasses import dataclass
from unittest.mock import patch

from examples.mechanics.stiffness import (
    Twist, coordinates, main, mv, point, suspension as inputs,
)


def spring_stiffness(lines, stiffnesses):
    extension = Twist & lines
    return (lines * extension * stiffnesses).sum(axis=0), extension


def body_inertia(points, masses):
    return (points & points.commutator(Twist) * masses).sum(axis=0)


def normal_modes(stiffness, inertia):
    values, modes = (Twist & stiffness).eigh(Twist & inertia)
    return modes, np.sqrt(np.maximum(values.kernel[..., 0], 0.0)) / (2 * np.pi)


@dataclass
class System:
    geometry: object
    stiffness: object
    extension: object
    inertia: object

    def __getattr__(self, name):
        return getattr(self.geometry, name)


def suspension(angled=False):
    geometry = inputs(angled)
    stiffness, extension = spring_stiffness(
        (geometry.anchors & geometry.attachments).normalized(), geometry.stiffnesses,
    )
    return System(geometry, stiffness, extension,
                  body_inertia(geometry.mass_points, geometry.masses))


def test_tutorial_passes_analytic_modes_to_renderer():
    with patch("examples.mechanics.stiffness_plumbing.draw_modes") as draw:
        main(plot_path="")
    free, restrained = draw.call_args.args[0]
    np.testing.assert_allclose((2 * np.pi * free.frequencies)**2,
                               [0, 12, 12 * .8**2 / (5 / 12)], atol=1e-12)
    assert np.all(restrained.frequencies > 0)


def test_two_springs_have_analytic_slide_bounce_and_rock_frequencies():
    system = suspension()
    _, frequencies = normal_modes(system.stiffness, system.inertia)
    # Mass 1, polar inertia (width² + height²)/12, anchors at x = ±.8.
    expected_squared = [0, 2 * 6, 2 * 6 * .8**2 / (5 / 12)]
    np.testing.assert_allclose((2 * np.pi * frequencies)**2, expected_squared, atol=1e-12)
    _, restrained = normal_modes(suspension(True).stiffness, system.inertia)
    assert np.all(restrained > 0)


def test_spring_extension_and_energy_match_finite_rigid_displacements():
    system = suspension(True)
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
    actual = np.sum(system.stiffnesses.kernel[:, 0] * (eplus**2 + eminus**2)) / (4 * h**2)
    predicted = .5 * q.regressive(system.stiffness(q)).kernel.item()
    np.testing.assert_allclose(actual, predicted, rtol=2e-7)


def test_mass_normalized_modes_and_spring_work():
    system = suspension(True)
    modes, frequencies = normal_modes(system.stiffness, system.inertia)
    mass = modes[:, None].regressive(system.inertia(modes[None, :])).kernel[..., 0]
    elastic = modes[:, None].regressive(system.stiffness(modes[None, :])).kernel[..., 0]
    np.testing.assert_allclose(mass, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(elastic, np.diag((2 * np.pi * frequencies)**2), atol=1e-12)
    extensions = system.extension(modes[:, None]).kernel[..., 0]
    work = (extensions * system.stiffnesses.kernel[:, 0]) @ extensions.T
    np.testing.assert_allclose(work, elastic, atol=1e-12)


def test_modes_do_not_depend_on_world_pose():
    system = suspension(True)
    motor = (mv.xw * -.9 + mv.yw * .3).exp() * (mv.xy * .37).exp().normalized()
    lines = system.anchors.regressive(system.attachments).normalized()
    stiffness, _ = spring_stiffness(motor >> lines, system.stiffnesses)
    mass_points = point(coordinates(system.body) / np.sqrt(3))
    inertia = body_inertia(motor >> mass_points, mv.scalar(np.full((4, 1), .25)))
    _, before = normal_modes(system.stiffness, system.inertia)
    _, after = normal_modes(stiffness, inertia)
    np.testing.assert_allclose(after, before, atol=1e-10)
