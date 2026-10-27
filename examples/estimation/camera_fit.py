"""Camera pose recovery by gradient descent differentiates the reprojection misfit directly through the motor exponential and projective camera map. A pinhole camera joins each world point with the optical centre into a sight ray and meets that ray with the image screen; moving the camera rig carries the projection map with it.

The pinhole camera is the map `(origin & Point) ^ screen`, carried to the rig's pose `generator.exp()` as `rig >> camera(rig << Point)`. Two unit-weight image points on the screen differ by an ideal point whose dual has the Euclidean length of their separation, and because every extensor is a JAX pytree, `jax.grad` differentiates the mean squared displacement directly with respect to the bivector generator.

In the matrix notation of projective vision a point in space projects through $P = K [R \\mid \\mathbf t]$ with $(R, \\mathbf t) = \\exp(\\hat{\\boldsymbol\\xi})\\in\\mathrm{SE}(3)$, and gradient descent differentiates the sum of squared pixel residuals after homogeneous division.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from numga.algebras import PGA3D
from numga.backend.jax import JaxContext


ga = PGA3D
ctx = JaxContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()                      # Antivector-
Bivector = ga.gatype.bivector()                     # Bivector+


# --- math -----------------------------------------------------------------------------
def main() -> None:
    n_points = 30
    n_steps = 600
    step_size = 0.25

    rng = np.random.default_rng(0)
    world_points = mv.point(jnp.asarray(rng.normal(size=(n_points, 3))))                # [points] Point-
    origin = mv.w.dual()                                                                # [] Point-
    screen = mv.z - mv.w                                                                # [] Plane+
    camera = (origin & Point) ^ screen                                                  # [] Point- <- Point-

    def image(generator: Bivector) -> Point:
        """The images of the world points seen from the rig are points on its screen in the rig's frame."""
        rig = generator.exp()                                                           # [] Motor+
        posed_camera = rig >> camera(rig << Point)                                      # [] Point- <- Point-
        return (rig << posed_camera(world_points)).normalized()                         # [points] Point-

    true_generator = mv.yz * 0.1 - mv.zx * 0.2 + mv.xy * 0.15 + mv.xw * 0.3 - mv.yw * 0.1 + mv.zw * 2.5  # [] Bivector+
    target_image = image(true_generator)                                                # [points] Point-

    @jax.jit
    def loss(generator: Bivector) -> jax.Array:
        """The reprojection loss is the mean squared Euclidean image displacement on the screen."""
        # Two images of unit weight differ by an ideal point, whose dual is the Euclidean
        # displacement between them on the screen.
        misfit = (image(generator) - target_image).dual().norm_squared().mean(axis=0)   # [] Scalar+
        return misfit.to_array()

    # jax.grad needs a plain number to differentiate; the generator is an Extensor pytree,
    # so the gradient comes back as a bivector.
    descend = jax.jit(lambda generator: generator - jax.grad(loss)(generator) * step_size)

    initial_generator = true_generator + (mv.yz + mv.zx + mv.xy + mv.xw + mv.yw + mv.zw) * 0.3  # [] Bivector+
    generator = initial_generator                                                       # [] Bivector+
    for _ in range(n_steps):
        generator = descend(generator)

    # Compare motors, not coordinates: the log of the relative motor is the pose error.
    # Its Euclidean part turns and its ideal part shifts; the dual swaps the two.
    pose_error = (true_generator.exp().inverse() * generator.exp()).log()               # [] Bivector+
    turn_error = pose_error.scalar_norm_squared().square_root()                         # [] Scalar+
    shift_error = pose_error.dual().scalar_norm_squared().square_root()                 # [] Scalar+

    print(f"mean squared image misfit: {float(loss(initial_generator)):.2e} -> {float(loss(generator)):.2e}")
    print(f"pose error: turn {float(turn_error.to_array()):.4f}, shift {float(shift_error.to_array()):.4f}")

    # --- checks ---------------------------------------------------------------------------
    # Descent on the image misfit alone recovers the pose.
    assert turn_error < 0.02
    assert shift_error < 0.02


if __name__ == "__main__":
    main()
