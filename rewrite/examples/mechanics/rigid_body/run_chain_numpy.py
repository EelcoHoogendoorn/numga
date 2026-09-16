"""Simulate a swinging rigid body chain in NumPy using XPBD."""

from __future__ import annotations

import time
import numpy as np

from numga import NumpyContext
from numga.algebras import PGA3D
from examples.mechanics.rigid_body.setup_chain import setup_bodies
from examples.mechanics.rigid_body.core import Body



def run_numpy_simulation(
    n_bodies: int = 6,
    n_steps: int = 50,
    substeps: int = 5,
    dt: float = 0.02,
) -> list[Body]:
    """Execute swinging chain simulation using NumPy backend."""
    context = NumpyContext(PGA3D)
    bodies, constraints = setup_bodies(context, n_bodies=n_bodies)

    states = [bodies]
    t0 = time.perf_counter()
    dt_sub = dt / substeps

    for step in range(n_steps):
        for _ in range(substeps):
            bodies = bodies.integrate(dt=dt_sub, constraint_sets=constraints)
        states.append(bodies)

    elapsed = time.perf_counter() - t0
    print(f"NumPy simulation: {n_steps} steps ({substeps} substeps each) in {elapsed:.3f}s")
    return states


if __name__ == "__main__":
    states = run_numpy_simulation()
