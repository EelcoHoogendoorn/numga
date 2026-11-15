"""Euclidean and three-sphere registration scenes fit noisy point correspondences with the one-sided motor eigenproblem."""

from __future__ import annotations

from types import ModuleType

import numpy as np

from numga import Algebra, Extensor, stack
from numga.algebras import PGA3D
from examples import instantiate

# Source cloud point count, coordinate spreads along x, y, and z, RNG seed, true shift and turn vectors,
# and per-point perturbation scale.
n_points = 60
cloud_spread = np.array([2.0, 1.0, 0.5])
seed = 0
shift_vector = np.array([1.5, -0.5, 2.0])
turn_vector = np.array([0.8, -0.6, 1.4])
noise_sigma = 0.02
# Instantiated registration cores for three-dimensional Euclidean space and for the three-sphere.
scenes = {
    name: instantiate("examples.geometry.registration.core", ga)
    for name, ga in (("euclidean", PGA3D), ("spherical", Algebra("w+x+y+z+")))
}


# --- math -----------------------------------------------------------------------------
def ground_truth_motor(core: ModuleType) -> Extensor:
    """A translation motor composed with a rotation motor forms the ground-truth rigid motion (`[] Motor+`)."""
    mv = core.mv
    shift_generator = (stack((mv.xw, mv.yw, mv.zw)) * shift_vector).sum(axis=0) / 2  # [] Bivector+
    turn_generator = (stack((mv.xy, mv.yz, mv.zx)) * turn_vector).sum(axis=0) / 2    # [] Bivector+
    return shift_generator.exp() * turn_generator.exp()                              # [] Motor+


def correspondences(core: ModuleType) -> tuple[Extensor, Extensor]:
    """Moving the anisotropic source cloud by `ground_truth_motor` and perturbing each point produces noisy correspondences (`[n_points] Point-`)."""
    rng = np.random.default_rng(seed)
    source = core.point_cloud(n_points, cloud_spread, rng)                     # [n_points] Point-
    moved_source = ground_truth_motor(core) >> source                          # [n_points] Point-
    target = core.noisy_cloud(moved_source, noise_sigma, rng)                  # [n_points] Point-
    return source, target


def registration(core: ModuleType) -> tuple[Extensor, Extensor, Extensor]:
    """Fitting the noisy correspondences aligns the source cloud onto the target (`[n_points] Point-`)."""
    source, target = correspondences(core)                                     # [n_points] Point-, [n_points] Point-
    alignment = core.fitted_motor(source, target)                              # [] Motor+
    aligned = alignment >> source                                              # [n_points] Point-
    return source, target, aligned


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Renders the Euclidean and three-sphere registration figures and verifies the alignment residuals."""
    from examples.util.animation import save_figure
    from examples.geometry.registration import render

    euclidean_source, euclidean_target, euclidean_aligned = registration(scenes["euclidean"])
    spherical_source, spherical_target, spherical_aligned = registration(scenes["spherical"])
    save_figure(render.registration_figure(euclidean_source, euclidean_target, euclidean_aligned), "registration_one_sided_residual")
    save_figure(render.registration_figure(spherical_source, spherical_target, spherical_aligned), "registration_three_sphere")

    # checks
    for name, (_, scene_target, scene_aligned) in (
        ("euclidean", (euclidean_source, euclidean_target, euclidean_aligned)),
        ("spherical", (spherical_source, spherical_target, spherical_aligned)),
    ):
        rms = (scene_aligned.normalized() - scene_target.normalized()).dual().norm_squared().mean(axis=0).square_root()
        assert rms < 0.15, name


if __name__ == "__main__":
    main()
