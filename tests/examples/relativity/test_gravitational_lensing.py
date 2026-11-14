"""Coverage tests for gravitational lensing by point masses in the plane of the sky."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from numga import stack
from examples.relativity.gravitational_lensing import core, render, scenarios

mv = core.mv
single_position = mv.vector(np.array([[0.0, 0.0]])).dual()                     # [1] Point-
unit_mass = np.array([1.0])                                                    # [1]


def test_single_lens_images_have_analytic_stretches_orientation_and_magnification():
    """A single point mass produces two collinear images with analytic stretches `1 +- 1/r^2` and point magnification."""
    source = (0.3 * mv.x + 0.2 * mv.y).dual()                                  # Point-
    radius = (source | source).square_root()                                   # Scalar+
    direction = source / radius                                                # Point-
    image_radii = stack((
        (radius + (radius**2 + 4).square_root()) / 2,
        (radius - (radius**2 + 4).square_root()) / 2,
    ))                                                                         # [2] Scalar+
    images = direction * image_radii                                           # [2] Point-
    local = core.local_map(images, single_position, unit_mass)                 # [2] Point- <- Point-
    area = local.det()                                                         # [2] Scalar+
    tangent = direction * mv.xy                                                # Point-

    assert ((core.source_direction(images, single_position, unit_mass) - source).scalar_norm_squared() < 1e-24).all()
    assert ((local(direction) - direction * (1 + 1 / image_radii**2)).scalar_norm_squared() < 1e-24).all()
    assert ((local(tangent) - tangent * (1 - 1 / image_radii**2)).scalar_norm_squared() < 1e-24).all()
    assert bool(area[0] > 0) and bool(area[1] < 0)
    magnification = (1 / area).abs().sum()                                     # Scalar+
    expected = (radius**2 + 2) / (radius * (radius**2 + 4).square_root())      # Scalar+
    assert (magnification - expected).abs() < 1e-12


def test_local_map_turns_covariantly_with_the_lens():
    """Rotating observed directions, lens positions, and displacements rotates the local Jacobian's output identically."""
    positions = mv.vector(np.array([[-0.4, 0.1], [0.6, -0.2], [-0.1, 0.7]])).dual()            # [3] Point-
    masses = np.array([0.2, 0.5, 0.3])                                                         # [3]
    observed = mv.vector(np.array([[-1.2, 0.8], [0.3, -0.9], [1.5, 1.2], [-0.8, -1.4]])).dual()   # [4] Point-
    step_direction = mv.vector(np.array([[0.3, 0.8], [-0.6, 0.1], [0.4, -0.7], [0.9, 0.2]])).dual()   # [4] Point-
    rotor = (mv.xy * 0.37).exp()                                                               # Rotor+
    turned_output = core.local_map(rotor >> observed, rotor >> positions, masses)(rotor >> step_direction)   # [4] Point-
    expected_output = rotor >> core.local_map(observed, positions, masses)(step_direction)     # [4] Point-
    assert ((turned_output - expected_output).scalar_norm_squared() < 1e-18).all()


def test_finite_source_light_integrates_to_the_radial_magnification():
    """Integrating lensed surface brightness over the sky matches the radially integrated point-source magnification."""
    width = 0.15
    n_grid = 128
    directions = core.sky_grid(2.0, n_grid)                                    # [n_grid, n_grid] Point-
    flux = core.source_brightness(
        core.source_direction(directions, single_position, unit_mass), 0 * mv.x.dual(), width,
    ).sum() * (4 / n_grid) ** 2                                                # Scalar+
    expected = 2 * np.pi * quad(
        lambda distance: (distance**2 + 2) / np.sqrt(distance**2 + 4) * np.exp(-distance**2 / (2 * width**2)),
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    assert (flux - expected).abs() < 1e-10


def test_local_map_is_undone_by_its_trace_two_inverse():
    """A planar map with trace two is inverted by `(2 * Point - local) / area`."""
    n_grid = 5
    spread = np.linspace(-1.25, 1.25, n_grid)
    centres = (mv.x * spread[None, :] + mv.y * spread[:, None]).dual()         # [n_grid, n_grid] Point-
    local = core.local_map(centres, scenarios.positions, scenarios.masses)     # [n_grid, n_grid] Point- <- Point-
    area = local.det()                                                         # [n_grid, n_grid] Scalar+
    offsets = mv.vector([[0.3, -0.2], [-0.1, 0.4]]).dual()                     # [2] Point-
    solved = local[..., None].solve(offsets)                                   # [n_grid, n_grid, 2] Point-
    analytic_inverse = ((2 * core.Point - local) / area)[..., None](offsets)   # [n_grid, n_grid, 2] Point-
    assert ((solved - analytic_inverse).scalar_norm_squared() < 1e-20).all()


def test_the_binary_scene_runs_and_draws():
    """The binary scene builds a coarse grid, draws a static figure, and yields one animation frame."""
    directions, area, _ = scenarios.binary_lens_grid(8)
    exposure = next(core.path_exposures(scenarios.path[:1], scenarios.positions, scenarios.masses, scenarios.source_width))
    fig = render.lensing_figure(directions, area, scenarios.binary_source_direction, scenarios.positions, exposure)
    plt.close(fig)
    frames = list(render.lensing_animation(directions, area, scenarios.binary_source_direction, scenarios.positions, [exposure]))
    assert len(frames) == 1
