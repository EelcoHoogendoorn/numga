"""Camera pose recovery by gradient descent differentiates the reprojection misfit through the motor exponential and projective camera map.

A pinhole camera is modelled as the projective map `(origin & Point) ^ screen`, joining each world point with the optical centre into a sight ray and meeting that ray with the image plane, so pulling world points into the rig's frame `rig << world_points` projects them onto the rig's screen. Two unit-weight image points on the screen differ by an ideal displacement whose antinorm equals their Euclidean separation, and `derivative` differentiates that mean squared misfit into a linear slope form on bivectors that pulls back across `&` to the descent generator.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from numga.algebras import PGA3D
from numga.backend.jax import JaxContext, derivative


ga = PGA3D
ctx = JaxContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Bivector = ga.gatype.bivector()                                               # Bivector+


# --- math -----------------------------------------------------------------------------
def screen_images(world_points: Point, generator: Bivector) -> Point:
    """Projecting `world_points` (`[n_points] Point-`) into the rig's frame at `generator.exp()` places unit-weight image points (`[n_points] Point-`) on its screen."""
    origin = mv.w.dual()                                                      # [] Point-
    screen = mv.z - mv.w                                                      # [] Plane+
    camera = (origin & Point) ^ screen                                        # [] Point- <- Point-
    rig = generator.exp()                                                     # [] Motor+
    local_points = rig << world_points                                        # [n_points] Point-
    return camera(local_points).normalized()                                  # [n_points] Point-


def reprojection_misfit(world_points: Point, target_images: Point, generator: Bivector) -> Scalar:
    """Mean squared Euclidean separation (`[] Scalar+`) compares `screen_images(world_points, generator)` against `target_images` (`[n_points] Point-`)."""
    # Two unit-weight screen points differ by an ideal point whose antinorm squared is their squared screen distance.
    displacements = screen_images(world_points, generator) - target_images    # [n_points] Point-
    return displacements.dual().norm_squared().mean(axis=0)                   # [] Scalar+


def fitted_generator(
    world_points: Point,
    target_images: Point,
    initial_generator: Bivector,
    n_steps: int,
    step_size: float,
) -> Bivector:
    """`n_steps` gradient-descent updates of size `step_size` on `reprojection_misfit` refine `initial_generator` (`[] Bivector+`) toward the camera pose."""
    unit_antiscalar = mv.scalar([1]).dual()                                   # [] Scalar-

    def misfit(generator: Bivector) -> Scalar:
        return reprojection_misfit(world_points, target_images, generator)

    @jax.jit
    def descent_step(generator: Bivector) -> Bivector:
        slope = derivative(misfit)(generator)                                 # [] Scalar+ <- Bivector+
        # Pulling the slope form back across & yields the antibivector gradient, whose dual is the Euclidean bivector step.
        gradient = slope.adjugate()(unit_antiscalar).dual()                   # [] Bivector+
        return generator - step_size * gradient                               # [] Bivector+

    generator, _ = jax.lax.scan(
        lambda state, _: (descent_step(state).retyped(Bivector), None),
        initial_generator.retyped(Bivector),
        None,
        length=n_steps,
    )
    return generator


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    """Fits a synthetic six-degree-of-freedom camera pose from 30 world points and checks turn and shift recovery."""
    n_points = 30
    n_steps = 600
    step_size = 0.25

    rng = np.random.default_rng(0)
    world_points = mv.point(jnp.asarray(rng.normal(size=(n_points, 3))))      # [n_points] Point-
    true_generator = mv.yz * 0.1 - mv.zx * 0.2 + mv.xy * 0.15 + mv.xw * 0.3 - mv.yw * 0.1 + mv.zw * 2.5  # [] Bivector+
    target_images = screen_images(world_points, true_generator)               # [n_points] Point-
    initial_generator = true_generator + (mv.yz + mv.zx + mv.xy + mv.xw + mv.yw + mv.zw) * 0.3  # [] Bivector+
    generator = fitted_generator(world_points, target_images, initial_generator, n_steps, step_size)  # [] Bivector+

    initial_misfit = reprojection_misfit(world_points, target_images, initial_generator)  # [] Scalar+
    final_misfit = reprojection_misfit(world_points, target_images, generator)  # [] Scalar+

    # Comparing motors directly through the logarithm of their quotient separates Euclidean rotation error from ideal translation error.
    pose_error = (true_generator.exp().inverse() * generator.exp()).log()     # [] Bivector+
    turn_error = pose_error.scalar_norm_squared().square_root()               # [] Scalar+
    shift_error = pose_error.dual().scalar_norm_squared().square_root()       # [] Scalar+

    print(f"mean squared image misfit: {float(initial_misfit.to_array()):.2e} -> {float(final_misfit.to_array()):.2e}")
    print(f"pose error: turn {float(turn_error.to_array()):.4f}, shift {float(shift_error.to_array()):.4f}")

    # checks
    assert (final_misfit < 1e-4 * initial_misfit).all()
    assert (turn_error < 0.02).all()
    assert (shift_error < 0.02).all()


if __name__ == "__main__":
    main()
