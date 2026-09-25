"""Point-mass lenses: a single lens matches its analytic images and magnification, the local map is
the derivative of the deflection and turns with the lens, a finite source's light matches the
radial integral of the point-source magnification, and the binary scenes pass their checks and draw."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from numga import stack

from examples.relativity.gravitational_lensing import core, render, scenarios

mv = core.mv
CENTRE = mv.vector(np.array([[0.0, 0.0]]))                                     # [masses] Vector
ONE = mv.scalar(np.array([[1.0]]))                                             # [masses] Scalar


def test_single_lens_images_have_the_analytic_stretches_orientation_and_magnification():
    source = 0.3 * mv.x + 0.2 * mv.y
    radius = (source | source).square_root()
    direction = source / radius
    image_radii = stack(((radius + (radius**2 + 4).square_root()) / 2,
                         (radius - (radius**2 + 4).square_root()) / 2))
    images = direction * image_radii
    local = core.local_map(images, CENTRE, ONE)
    area = local.outermorphism(core.Area)(mv.xy) / mv.xy

    # Both images reach the source; the radial stretch is 1 + 1/r², the tangential one 1 - 1/r².
    np.testing.assert_allclose((core.deflected(images, CENTRE, ONE) - source).kernel, 0, atol=1e-13)
    np.testing.assert_allclose((local(direction) - direction * (1 + 1 / image_radii**2)).kernel, 0, atol=1e-13)
    tangent = direction * mv.xy
    np.testing.assert_allclose((local(tangent) - tangent * (1 - 1 / image_radii**2)).kernel, 0, atol=1e-13)
    assert area.to_array()[0] > 0 > area.to_array()[1]
    magnification = (1 / area).abs().sum()
    expected = (radius**2 + 2) / (radius * (radius**2 + 4).square_root())
    np.testing.assert_allclose((magnification - expected).kernel, 0, atol=1e-13)


def test_local_map_is_the_derivative_and_turns_with_the_lens():
    positions = mv.vector(np.array([[-0.4, 0.1], [0.6, -0.2], [-0.1, 0.7]]))
    masses = mv.scalar(np.array([[0.2], [0.5], [0.3]]))
    observed = mv.vector(np.array([[-1.2, 0.8], [0.3, -0.9], [1.5, 1.2], [-0.8, -1.4]]))
    small = mv.vector(np.array([[0.3, 0.8], [-0.6, 0.1], [0.4, -0.7], [0.9, 0.2]]))
    step = 1e-4
    lens = lambda directions: core.deflected(directions, positions, masses)
    numerical = (-lens(observed + 2 * step * small) + 8 * lens(observed + step * small)
                 - 8 * lens(observed - step * small) + lens(observed - 2 * step * small)) / (12 * step)
    np.testing.assert_allclose((numerical - core.local_map(observed, positions, masses)(small)).kernel, 0, atol=1e-9)

    turn = (mv.xy * 0.37).exp()
    turned = core.local_map(turn >> observed, turn >> positions, masses)(turn >> small)
    np.testing.assert_allclose((turned - (turn >> core.local_map(observed, positions, masses)(small))).kernel, 0, atol=1e-9)


def test_finite_source_light_integrates_to_the_radial_magnification():
    width = 0.15
    directions = core.sky(2.0, 128)
    flux = core.brightness(core.deflected(directions, CENTRE, ONE), 0 * mv.x, width).sum() * (4 / 128)**2
    # The analytic point-source magnification, integrated over rings of the source.
    expected = 2 * np.pi * quad(
        lambda distance: (distance**2 + 2) / np.sqrt(distance**2 + 4) * np.exp(-distance**2 / (2 * width**2)),
        0, np.inf, epsabs=1e-12,
    )[0]
    np.testing.assert_allclose(flux.to_array(), expected, rtol=1e-11, atol=1e-11)


def test_scenes_pass_their_checks_and_draw():
    directions, reached, area = scenarios.lens(97)
    steps = [scenarios.light(directions, reached, centre, 0.035) for centre in mv.x * np.array([-0.3, 0.0, 0.3])]
    figure = render.draw(directions, area, scenarios.deflected, scenarios.POSITIONS, steps[0])
    assert isinstance(figure, plt.Figure)
    plt.close(figure)
    assert len(render.animate(directions, area, scenarios.deflected, scenarios.POSITIONS, steps)) == 3
    figure = render.draw_tissot(directions, area, scenarios.POSITIONS, *scenarios.tissot(5, 0.025))
    plt.close(figure)
