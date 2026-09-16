"""Recover a camera pose by gradient descent through the extensor pytree, in PGA3D with JAX.

The camera is the join-then-meet expression from the projection example with the point left
open; the moved camera is that map conjugated by the rig motor. The motor is the exponential
of six bivector coordinates, and jax.grad differentiates the screen residual with respect to
them through exp, the sandwich, the bind and the dehomogenisation. Nothing is linearised by
hand.
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
P = ga.subspace.antivector()
Point = ga.gatype.antivector()
Bivector = ga.gatype.bivector()
Motor = ga.gatype.rotor()


# --- plumbing -------------------------------------------------------------------------
def point(coords: jnp.ndarray) -> Point:
    return mv.antivector(jnp.concatenate([coords, jnp.ones_like(coords[..., :1])], axis=-1))


def screen_xy(rig: Motor, image: Point) -> jnp.ndarray:
    """(x, y) screen coordinates of image points, read in the rig frame."""
    k = (rig << image).cast(P).kernel
    return k[..., :2] / k[..., 3:]


# --- math -----------------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(0)
    world = point(jnp.asarray(rng.normal(size=(30, 3))))
    origin = point(jnp.zeros(3))
    screen = mv.z - mv.w
    camera = origin.regressive(Point).wedge(screen)

    def project(generator: Bivector) -> jnp.ndarray:
        rig = generator.exp()
        moved = rig >> camera(rig << Point)
        return screen_xy(rig, moved(world))

    truth = mv.bivector(jnp.array([0.1, -0.2, 0.15, 0.3, -0.1, 2.5]))
    observed = project(truth)
    loss = jax.jit(lambda generator: jnp.mean((project(generator) - observed) ** 2))
    # The generator is an Extensor pytree, so jax.grad returns the gradient as a bivector.
    descend = jax.jit(lambda generator: generator - jax.grad(loss)(generator) * 0.5)

    generator = truth + mv.bivector(jnp.full(6, 0.3))
    for iteration in range(600):
        generator = descend(generator)
        if iteration % 100 == 0:
            print(f"iteration {iteration}: loss {float(loss(generator)):.2e}")

    # Compare motors, not coordinates: the log of the relative motor is the pose error.
    relative = (truth.exp().inverse() * generator.exp()).log()
    print("pose error bivector:", np.round(np.asarray(relative.kernel), 4))

    # --- checks: kernel-level assertions, deliberately outside the demonstration ----------
    assert float(jnp.linalg.norm(relative.kernel)) < 0.02


if __name__ == "__main__":
    main()
