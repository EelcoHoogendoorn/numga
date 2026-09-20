"""Simulate a swinging rigid body chain in NumPy using XPBD."""

from __future__ import annotations

import time

from numga import NumpyContext
from numga.algebras import PGA3D
from examples.mechanics.xpbd_plumbing import setup_chain, simulate_chain, ChainState



def run_numpy_simulation(
    n_bodies: int = 6,
    n_steps: int = 50,
    substeps: int = 5,
    dt: float = 0.02,
) -> list[ChainState]:
    """Execute swinging chain simulation using NumPy backend."""
    context = NumpyContext(PGA3D)
    state, constraints = setup_chain(context, n_bodies=n_bodies)

    t0 = time.perf_counter()
    states, _ = simulate_chain(
        state,
        constraints,
        n_steps=n_steps,
        substeps=substeps,
        dt=dt,
    )
    elapsed = time.perf_counter() - t0
    print(f"NumPy simulation: {n_steps} steps ({substeps} substeps each) in {elapsed:.3f}s")
    return states


if __name__ == "__main__":
    states = run_numpy_simulation()
