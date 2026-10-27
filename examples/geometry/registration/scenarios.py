"""Scenes for point-cloud registration fit noisy point correspondences by the one-sided motor residual, in flat space and on the three-sphere."""

from __future__ import annotations

from types import ModuleType

import numpy as np

from numga import Algebra, stack
from numga.algebras import PGA3D
from examples import instantiate

# The source cloud: its points, its spreads along x, y and z, and the seed of its noise; the true motion, a
# shift after a turn; and the noise that perturbs each target point.
N_POINTS = 60
CLOUD_SPREAD = np.array([2.0, 1.0, 0.5])
SEED = 0
SHIFT = np.array([1.5, -0.5, 2.0])
TURN = np.array([0.8, -0.6, 1.4])
NOISE = 0.02
# The same module, instantiated for flat space and for the three-sphere.
SCENES = {
    name: instantiate("examples.geometry.registration.core", ga)
    for name, ga in (("euclidean", PGA3D), ("spherical", Algebra("w+x+y+z+")))
}


# --- math -----------------------------------------------------------------------------
def true_motor(core: ModuleType):
    """The true motion is a shift after a turn, each the exponential of half its generator."""
    mv = core.mv
    shift = (stack((mv.xw, mv.yw, mv.zw)) * SHIFT).sum(axis=0) / 2
    turn = (stack((mv.xy, mv.yz, mv.zx)) * TURN).sum(axis=0) / 2
    return shift.exp() * turn.exp()


def correspondences(core: ModuleType):
    """The noisy correspondences are the source cloud and its image under the true motion, each point perturbed."""
    rng = np.random.default_rng(SEED)
    source = core.cloud(N_POINTS, CLOUD_SPREAD, rng)                           # [points] Point-
    target = core.jitter(true_motor(core) >> source, NOISE, rng)               # [points] Point-
    return source, target


def registration(core: ModuleType):
    """The one-sided motor residual fits the noisy correspondences: the source, the target, and the aligned source."""
    source, target = correspondences(core)                                     # [points] Point-, Point-
    return source, target, core.fit_motor(source, target) >> source


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_figure
    from examples.geometry.registration import render

    save_figure(render.draw_registration(*registration(SCENES["euclidean"])), "registration_one_sided_residual")
    save_figure(render.draw_registration(*registration(SCENES["spherical"])), "registration_three_sphere")

    # checks
    # In both flat space and on the three-sphere, the fitted motor brings the cloud within the noise of the target.
    for name, scene in SCENES.items():
        _, scene_target, scene_aligned = registration(scene)
        rms = ((scene_aligned.normalized() - scene_target.normalized()).dual().norm_squared().sum(axis=0) / N_POINTS).square_root()
        assert rms < 0.15, name


if __name__ == "__main__":
    main()
