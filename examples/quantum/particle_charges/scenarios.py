"""Particle charges and gauge connections in one chiral Spin(10) spinor."""

from __future__ import annotations

from functools import reduce
from itertools import product
from operator import mul

import numpy as np

from numga import stack
from examples.quantum.particle_charges.core import (
    Scalar, State, StateMap, action, mv, phase, sector_value, transition_strength,
)


# --- math
def sectors() -> tuple[np.ndarray, StateMap, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """The sixteen sectors and their electric, hypercharge, weak, and colour weights."""
    # Four signs choose the fifth: this chirality has an even occupation count.
    occupations = np.array([pattern + (sum(pattern) % 2,) for pattern in product((0, 1), repeat=4)])
    hypercharge_weights = np.array([-1 / 3, -1 / 3, -1 / 3, 1 / 2, 1 / 2])
    planes = stack((mv.ab, mv.cd, mv.ef, mv.gh, mv.ij))                       # [planes] Bivector
    signs = -phase * planes                                                # [planes] Even
    occupation = (State - action(signs)) / 2                               # [planes] State <- State

    # Each factor halves the spinor; four leave one sector with two phase directions.
    factors = (1 + signs[None, :4] * (1 - 2 * occupations[:, :4])) / 2
    selectors = reduce(mul, (factors[:, i] for i in range(4)))               # [sectors] Even
    projectors = action(selectors)                                         # [sectors] State <- State

    hypercharge = (hypercharge_weights * occupation).sum(axis=0)            # [] State <- State
    left_isospin = (occupation[3] - occupation[4]) / 2                       # [] State <- State
    right_isospin = (occupation[3] + occupation[4] - State) / 2              # [] State <- State
    charge = hypercharge + left_isospin                                    # [] State <- State
    baryon_minus_lepton = State - 2 * occupation[:3].sum(axis=0) / 3         # [] State <- State
    colour = stack((
        (occupation[0] - occupation[1]) / 2,
        (occupation[0] + occupation[1] - 2 * occupation[2]) / (2 * np.sqrt(3)),
    ))                                                                    # [colour weights] State <- State
    return (
        occupations, projectors,
        sector_value(charge, projectors),
        sector_value(hypercharge, projectors),
        sector_value(left_isospin, projectors),
        sector_value(right_isospin, projectors),
        sector_value(baryon_minus_lepton, projectors),
        sector_value(colour[None, :], projectors[:, None]),
    )


def interactions(projectors: StateMap) -> tuple[StateMap, StateMap, StateMap, Scalar, Scalar, Scalar]:
    """Colour, weak, and an additional unified generator acting between sectors."""
    # Paired turns exchange two occupations while preserving their sum.
    colour = stack((
        mv.ac + mv.bd, mv.ad - mv.bc,
        mv.ae + mv.bf, mv.af - mv.be,
        mv.ce + mv.df, mv.cf - mv.de,
        mv.ab - mv.cd,
        (mv.ab + mv.cd - 2 * mv.ef) / np.sqrt(3),
    )) / 4                                                                # [colour generators] Bivector
    weak = stack((mv.gi + mv.hj, mv.gj - mv.hi, mv.gh - mv.ij)) / 4           # [weak generators] Bivector
    colour_generators = action(colour)                                     # [colour generators] State <- State
    weak_generators = action(weak)                                         # [weak generators] State <- State
    # This additional Spin(10) turn exchanges a colour occupation with a weak one.
    mixing_generator = action((mv.ag + mv.bh) / 4)[None]                    # [mixing generators] State <- State
    return (
        colour_generators, weak_generators, mixing_generator,
        transition_strength(colour_generators, projectors),
        transition_strength(weak_generators, projectors),
        transition_strength(mixing_generator, projectors),
    )


# --- plumbing
def main() -> None:
    from examples.animation import save_figure
    from examples.quantum.particle_charges import render

    occupations, projectors, charge, hypercharge, left_isospin, right_isospin, _, colour_weights = sectors()
    _, _, _, colour_strength, weak_strength, mixing_strength = interactions(projectors)
    print(render.charge_table(occupations, charge, hypercharge, left_isospin))
    gauge_figure = render.draw_connections(
        occupations, left_isospin, right_isospin, colour_weights,
        stack((colour_strength, weak_strength)),
    )
    save_figure(gauge_figure, "particle_charges_gauge")
    unified_figure = render.draw_connections(
        occupations, left_isospin, right_isospin, colour_weights, mixing_strength[None],
    )
    save_figure(unified_figure, "particle_charges_unified")

    # checks
    # Electric charge is weak isospin plus hypercharge, and colour, weak and mixing actions connect 12, 4
    # and 4 pairs of sectors.
    np.testing.assert_allclose((charge - hypercharge - left_isospin).to_array(), 0, atol=1e-12)
    connected = [np.count_nonzero(np.triu(strength.to_array(), 1) > 1e-12)
                 for strength in (colour_strength, weak_strength, mixing_strength)]
    assert connected == [12, 4, 4]


if __name__ == "__main__":
    main()
