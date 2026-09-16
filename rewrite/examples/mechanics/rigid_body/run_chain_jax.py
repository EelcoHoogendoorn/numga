"""Simulate a swinging rigid body chain in JAX using JIT compilation."""

from __future__ import annotations

import time
import jax
import numpy as np

from numga.backend.jax import JaxContext
from numga.algebras import PGA3D
from examples.mechanics.rigid_body.setup_chain import setup_bodies
from examples.mechanics.rigid_body.core import Body



def run_jax_simulation(
    n_bodies: int = 6,
    n_steps: int = 50,
    substeps: int = 5,
    dt: float = 0.02,
) -> list[Body]:
    """Execute swinging chain simulation using JAX with jax.jit."""
    context = JaxContext(PGA3D)
    bodies, constraints = setup_bodies(context, n_bodies=n_bodies)
    dt_sub = dt / substeps

    # Warmup unjitted step to stabilize dynamic trait sets
    bodies = bodies.integrate(dt=dt_sub, constraint_sets=constraints)

    @jax.jit
    def step_block(b: Body) -> Body:
        for _ in range(substeps):
            b = b.integrate(dt=dt_sub, constraint_sets=constraints)
        return b

    # Warmup / compile
    t_compile = time.perf_counter()
    bodies = step_block(bodies)
    print(f"JAX JIT compile time: {time.perf_counter() - t_compile:.3f}s")

    states = [bodies]
    t0 = time.perf_counter()
    for step in range(n_steps):
        bodies = step_block(bodies)
        states.append(bodies)

    elapsed = time.perf_counter() - t0
    print(f"JAX simulation: {n_steps} blocks ({substeps} substeps each) in {elapsed:.3f}s")
    return states


if __name__ == "__main__":
    states = run_jax_simulation()
