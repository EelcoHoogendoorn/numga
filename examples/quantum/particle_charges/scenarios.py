"""Particle charges and gauge connections in one chiral Spin(10) spinor."""

from __future__ import annotations

from itertools import product

import numpy as np

from numga import stack
from examples.quantum.particle_charges.core import (
    Scalar, State, StateMap, action, mv, sector_occupations, sector_projectors, sector_value, transition_strength,
)

# Four sign choices fix the fifth so all sixteen patterns share even occupation parity.
occupations = np.array([pattern + (sum(pattern) % 2,) for pattern in product((0, 1), repeat=4)])  # [16, 5]
hypercharge_weights = np.array([-1 / 3, -1 / 3, -1 / 3, 1 / 2, 1 / 2])     # [5]


# --- math
def charge_sectors() -> tuple[np.ndarray, StateMap, Scalar, Scalar, Scalar, Scalar, Scalar, Scalar]:
    """Five commuting planes resolve the chiral spinor into sixteen sectors and their Standard Model weights."""
    planes = stack((mv.ab, mv.cd, mv.ef, mv.gh, mv.ij))                    # [5] Bivector+
    occupation = sector_occupations(planes)                                # [5] State+ <- State+
    projectors = sector_projectors(planes, occupations)                    # [16] State+ <- State+

    hypercharge = (hypercharge_weights * occupation).sum(axis=0)           # [] State+ <- State+
    left_isospin = (occupation[3] - occupation[4]) / 2                     # [] State+ <- State+
    right_isospin = (occupation[3] + occupation[4] - State) / 2            # [] State+ <- State+
    electric_charge = hypercharge + left_isospin                           # [] State+ <- State+
    baryon_minus_lepton = State - 2 * occupation[:3].sum(axis=0) / 3       # [] State+ <- State+
    colour_observables = stack((
        (occupation[0] - occupation[1]) / 2,
        (occupation[0] + occupation[1] - 2 * occupation[2]) / (2 * np.sqrt(3)),
    ))                                                                     # [2] State+ <- State+
    return (
        occupations, projectors,
        sector_value(electric_charge, projectors),
        sector_value(hypercharge, projectors),
        sector_value(left_isospin, projectors),
        sector_value(right_isospin, projectors),
        sector_value(baryon_minus_lepton, projectors),
        sector_value(colour_observables[None, :], projectors[:, None]),
    )


def gauge_interactions(projectors: StateMap) -> tuple[StateMap, StateMap, StateMap, Scalar, Scalar, Scalar]:
    """Colour, weak, and quark-lepton mixing generators act between the sixteen particle sectors."""
    # Paired turns exchange two occupations while preserving their sum.
    colour_bivectors = stack((
        mv.ac + mv.bd, mv.ad - mv.bc,
        mv.ae + mv.bf, mv.af - mv.be,
        mv.ce + mv.df, mv.cf - mv.de,
        mv.ab - mv.cd,
        (mv.ab + mv.cd - 2 * mv.ef) / np.sqrt(3),
    )) / 4                                                                 # [8] Bivector+
    weak_bivectors = stack((mv.gi + mv.hj, mv.gj - mv.hi, mv.gh - mv.ij)) / 4  # [3] Bivector+
    colour_generators = action(colour_bivectors)                           # [8] State+ <- State+
    weak_generators = action(weak_bivectors)                               # [3] State+ <- State+
    # A mixed Spin(10) turn exchanges one colour occupation with one weak occupation.
    mixing_generators = action((mv.ag + mv.bh) / 4)[None]                  # [1] State+ <- State+
    return (
        colour_generators, weak_generators, mixing_generators,
        transition_strength(colour_generators, projectors),
        transition_strength(weak_generators, projectors),
        transition_strength(mixing_generators, projectors),
    )


# --- plumbing
def main() -> None:
    from examples.util.animation import save_figure
    from examples.quantum.particle_charges import render

    (
        sector_patterns, projectors, charges, hypercharges,
        left_isospins, right_isospins, baryon_minus_leptons, colour_weights,
    ) = charge_sectors()
    _, _, _, colour_strength, weak_strength, mixing_strength = gauge_interactions(projectors)
    print("charges:", charges.to_array())
    print("hypercharges:", hypercharges.to_array())
    print("left isospins:", left_isospins.to_array())
    gauge_figure = render.connections_figure(
        sector_patterns, left_isospins, right_isospins, colour_weights,
        stack((colour_strength, weak_strength)),
    )
    save_figure(gauge_figure, "particle_charges_gauge")
    unified_figure = render.connections_figure(
        sector_patterns, left_isospins, right_isospins, colour_weights, mixing_strength[None],
    )
    save_figure(unified_figure, "particle_charges_unified")

    # checks
    # Electric charge equals weak isospin plus hypercharge, hypercharge splits into right isospin and half baryon-minus-lepton number,
    # and colour, weak, and mixing generators connect 12, 4, and 4 unordered sector pairs.
    assert ((charges - hypercharges - left_isospins).abs() < 1e-10).all()
    assert ((hypercharges - right_isospins - baryon_minus_leptons / 2).abs() < 1e-10).all()
    edge_counts = [
        int(np.triu(strength > 1e-10, k=1).sum())
        for strength in (colour_strength, weak_strength, mixing_strength)
    ]
    assert edge_counts == [12, 4, 4]


if __name__ == "__main__":
    main()
