"""Recover a camera pose by gradient descent through the extensor pytree, in PGA3D with JAX.

The camera is the join-then-meet expression from the projection example with the point left
open; the moved camera is that map conjugated by the rig motor. The motor is the exponential
of a bivector, and jax.grad differentiates the image misfit with respect to it through exp,
the sandwich, the bind and the normalization. Nothing is linearised by hand.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from numga.algebras import PGA3D
from numga.backend.jax import JaxContext

jax.config.update("jax_enable_x64", True)

ga = PGA3D
ctx = JaxContext(ga)
mv = ctx.multivector
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Scalar = ga.gatype.scalar()


def point(coords: np.ndarray) -> Point:
    """Finite points at (..., 3) coordinates: the dual of the homogeneous vector."""
    return (mv("x y z", coords) + mv.w).dual()


# --- plumbing -------------------------------------------------------------------------


# --- math -----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    world = point(jnp.asarray(rng.normal(size=(30, 3))))
    origin = mv.zyx
    screen = mv.z - mv.w
    camera = (origin & Point) ^ screen

    def image(generator: Bivector) -> Point:
        """Images of the world points, seen from the rig: points on its screen z == 1."""
        rig = generator.exp()
        moved = rig >> camera(rig << Point)
        return (rig << moved(world)).normalized()

    truth = mv.yz * 0.1 - mv.zx * 0.2 + mv.xy * 0.15 + mv.xw * 0.3 - mv.yw * 0.1 + mv.zw * 2.5
    observed = image(truth)

    def misfit(generator: Bivector) -> Scalar:
        # Two images of unit weight differ by an ideal point, whose dual is the Euclidean
        # displacement between them on the screen.
        return (image(generator) - observed).dual().norm_squared().mean(axis=0)

    # jax.grad needs a plain number to differentiate; the generator is an Extensor pytree,
    # so the gradient comes back as a bivector.
    loss = jax.jit(lambda generator: misfit(generator).to_array())
    descend = jax.jit(lambda generator: generator - jax.grad(loss)(generator) * 0.25)

    start = truth + (mv.yz + mv.zx + mv.xy + mv.xw + mv.yw + mv.zw) * 0.3
    generator = start
    for _ in range(600):
        generator = descend(generator)

    # Compare motors, not coordinates: the log of the relative motor is the pose error.
    # Its Euclidean part turns and its ideal part shifts; the dual swaps the two.
    relative = (truth.exp().inverse() * generator.exp()).log()
    turn = relative.scalar_norm_squared().square_root()
    shift = relative.dual().scalar_norm_squared().square_root()

    print(f"mean squared image misfit: {float(loss(start)):.2e} -> {float(loss(generator)):.2e}")
    print(f"pose error: turn {float(turn.to_array()):.4f}, shift {float(shift.to_array()):.4f}")

    # --- checks ---------------------------------------------------------------------------
    # Descent on the image misfit alone recovers the pose.
    assert float(turn.to_array()) < 0.02
    assert float(shift.to_array()) < 0.02


if __name__ == "__main__":
    main()
