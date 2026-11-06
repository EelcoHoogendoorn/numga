"""Electrostriction's loading at the smallest lattice: one ring of particles, a single load and one Newton step."""

import jax

from examples.electromagnetism.electrostriction import core, scenarios


def test_the_loading_runs():
    model = scenarios.lattice(n_rings=1)

    # One compiled call: run eagerly, every small operation would compile on its own.
    @jax.jit
    def loaded():
        _, _, positions, _ = scenarios.loading(model, n_load_samples=1, n_newton_steps=1)
        return core.strain(positions, model.rest, core.mv.x)

    loaded()
