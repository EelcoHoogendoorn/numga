"""The Hubbard dimer, from electron hopping to localized spins: levels, charge fluctuations and exchange."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.hubbard_dimer import core

HOPPING = 1.0
NAMES = ("Delocalized", "Crossover", "Localized spins")
REPULSIONS = HOPPING * np.array([0, 4, 20])
N_SAMPLES = 241
SWEEP = np.linspace(0.0, REPULSIONS[-1], N_SAMPLES)
# The large-repulsion approximation of the gap, away from zero repulsion where it is singular.
STRONG_REPULSION = SWEEP[1:]
PERTURBATIVE = 4 * HOPPING**2 / STRONG_REPULSION
N_STEPS = 1200
N_CYCLES = 1
PHASE = np.linspace(0.0, N_CYCLES * 2 * np.pi, N_STEPS + 1)
SPIN_ONLY = np.sin(PHASE / 2)**2
# The dials: every few steps of the cycle, and each frame's duration.
DIAL_STRIDE = 6
DIAL_MS = 40
# Field differences between the sites, in units of the exchange at the strongest repulsion.
FIELD_RATIOS = np.array([0, 1, 5])
# Asymmetries between the sites, up to twice the strongest repulsion.
N_ASYMMETRIES = 401
ASYMMETRIES = HOPPING * np.linspace(0.0, 2 * REPULSIONS[-1], N_ASYMMETRIES)


# --- math -----------------------------------------------------------------------------
def spectrum(repulsion: np.ndarray) -> tuple[core.Scalar, np.ndarray]:
    """The six energy levels, and the singlet-triplet gap from the singlet block, its rate less half
    the repulsion."""
    energies = core.hamiltonian(HOPPING, repulsion).eigvalsh()  # [samples, modes] Scalar+
    return energies, core.block_rate(HOPPING, repulsion) - repulsion / 2


def ground_states(repulsion: np.ndarray) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    """Each ground state has these configuration probabilities, double occupancy and spin correlation."""
    ground = core.ground(HOPPING, repulsion)                    # [cases] Pair+
    return (
        core.probabilities(ground),
        core.expectation(ground, core.double_occupancy),
        core.expectation(ground, core.spin_correlation),
    )


def exchange(repulsion: np.ndarray, phase: np.ndarray) -> tuple[core.Pair, core.Pair]:
    """Opposite spins on different sites, over one cycle set by the gap, give a state in time with these
    cosine and sine parts."""
    gap = core.block_rate(HOPPING, repulsion) - repulsion / 2   # [cases]
    times = phase / gap[..., None]                              # [cases, times]
    return core.exchange(HOPPING, repulsion[..., None], times)


def field_exchange(field_ratios: np.ndarray, phase: np.ndarray) -> tuple[core.Pair, core.Pair]:
    """At the strongest repulsion, field differences in units of the exchange: spin up on the left and
    spin down on the right over one exchange cycle, as its cosine and sine parts."""
    strong = REPULSIONS[-1]
    gap = core.block_rate(HOPPING, strong) - strong / 2
    hamiltonians = core.hamiltonian(HOPPING, strong) + field_ratios * gap * core.field_difference   # [cases] Pair+ <- Pair+
    energies, modes = hamiltonians.eigh()                       # [cases, modes] Scalar+, Pair+
    return core.turn(energies, modes, core.mv.ad, phase / gap)


def charge_transfer(repulsion: np.ndarray, asymmetry: np.ndarray) -> core.Scalar:
    """The charge the ground state moves to the lower site is its occupation of the left site less the
    right, minus twice the lifted site difference."""
    hamiltonians = core.hamiltonian(HOPPING, repulsion[:, None]) + asymmetry * core.site_difference   # [cases, samples] Pair+ <- Pair+
    _, modes = hamiltonians.eigh()                              # [cases, samples, modes] Pair+
    ground = modes[..., 0]                                      # [cases, samples] Pair+
    return -2 * core.expectation(ground, core.site_difference)  # [cases, samples] Scalar+


def dials(parts: list[core.Pair]) -> list[core.Scalar]:
    """The dials read each configuration's amplitude in the cosine and sine parts, and in the share of
    the triplet of zero spin in each."""
    triplet = core.zero_triplet
    shares = [triplet * triplet.reverse().scalar_product(part) for part in parts]
    return [core.configurations.reverse().scalar_product(state[..., None]) for state in (*parts, *shares)]


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.hubbard_dimer import render

    energies, gap = spectrum(SWEEP)
    ground_probabilities, _, _ = ground_states(REPULSIONS)
    cosine_part, sine_part = exchange(REPULSIONS, PHASE)
    probabilities = core.probabilities(cosine_part) + core.probabilities(sine_part)
    save_figure(render.draw_spectrum(SWEEP, HOPPING, energies, gap, PERTURBATIVE, STRONG_REPULSION), "hubbard_dimer_spectrum")
    save_figure(render.draw_ground_states(NAMES, REPULSIONS / HOPPING, ground_probabilities),
                "hubbard_dimer_ground_states")
    save_figure(render.draw_exchange(PHASE, probabilities, SPIN_ONLY), "hubbard_dimer_exchange")
    parts = [part[:, ::DIAL_STRIDE] for part in (cosine_part, sine_part)]
    rows = [f"{name}\n$U/t = {ratio:g}$" for name, ratio in zip(NAMES, REPULSIONS / HOPPING)]
    save_animation(render.animate_dials(rows, *dials(parts)), "hubbard_dimer_dials", DIAL_MS)
    save_figure(render.draw_transfer(ASYMMETRIES / HOPPING, charge_transfer(REPULSIONS, ASYMMETRIES), NAMES, REPULSIONS / HOPPING),
                "hubbard_dimer_transfer")
    field_parts = field_exchange(FIELD_RATIOS, PHASE[::DIAL_STRIDE])
    field_rows = [f"field difference\n{ratio:g} exchange" for ratio in FIELD_RATIOS]
    save_animation(render.animate_dials(field_rows, *dials(list(field_parts))), "hubbard_dimer_field_dials", DIAL_MS)

    # --- checks
    # On the singlet block the shifted Hamiltonian applied twice is the rate squared; the gap from the
    # block matches the full spectrum; the state in time stays normalized.
    hamiltonians = core.hamiltonian(HOPPING, SWEEP[:, None])    # [samples, 1] Pair+ <- Pair+
    shift = SWEEP[:, None] / 2                                  # [samples, 1]
    rate = core.block_rate(HOPPING, SWEEP[:, None])             # [samples, 1]
    block = stack([core.singlet, core.symmetric_double])        # [block] Pair+
    shifted_block = hamiltonians(block) - shift * block         # [samples, block] Pair+
    np.testing.assert_allclose((hamiltonians(shifted_block) - shift * shifted_block - rate**2 * block).kernel, 0.0, atol=1e-10)
    np.testing.assert_allclose(gap, energies.to_array()[:, 1] - energies.to_array()[:, 0], atol=1e-10)
    np.testing.assert_allclose(probabilities.to_array().sum(axis=-1), 1.0, atol=1e-11)
    print("Hubbard dimer checks passed.")


if __name__ == "__main__":
    main()
