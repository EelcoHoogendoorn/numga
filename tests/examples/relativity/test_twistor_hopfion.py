"""The electromagnetic hopfion is null and transports energy along the Robinson congruence."""

import numpy as np

from examples.relativity.twistors import core


def test_hopfion_is_null_and_its_energy_follows_the_robinson_congruence():
    rng = np.random.default_rng(41)
    events = core.mv(core.Event.derive.dual, rng.uniform(-2, 2, (16, 4))).dual()
    field = core.hopfion_field(events)
    flux = field >> core.time
    energy = core.mv.t & flux
    direction = core.robinson_direction(events)

    assert (field.squared().scalar_norm_squared() < 1e-20).all()
    assert ((flux / energy - direction).scalar_norm_squared() < 1e-18).all()
    assert (field.commutator(direction).scalar_norm_squared() < 1e-20).all()
    assert (energy > 0).all()
