"""The Hubbard dimer across repulsion regimes: energy levels, charge fluctuations, and spin exchange."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from examples.quantum.hubbard_dimer import core

hopping = 1.0
names = ("Delocalized", "Crossover", "Localized spins")
repulsions = hopping * np.array([0, 4, 20])
n_samples = 241
sweep = np.linspace(0.0, repulsions[-1], n_samples)
# Strong-repulsion asymptote of the singlet-triplet gap, omitting the singularity at zero repulsion.
strong_repulsion = sweep[1:]
perturbative = 4 * hopping**2 / strong_repulsion
n_steps = 1200
n_cycles = 1
phase = np.linspace(0.0, n_cycles * 2 * np.pi, n_steps + 1)
spin_only = np.sin(phase / 2)**2
dial_stride = 6
frame_duration_ms = 40
# Site-dependent magnetic field differences measured in units of the strong-repulsion exchange gap.
field_ratios = np.array([0, 1, 5])
# Site energy offsets swept up to twice the strongest repulsion.
n_asymmetries = 401
asymmetries = hopping * np.linspace(0.0, 2 * repulsions[-1], n_asymmetries)


# --- math -----------------------------------------------------------------------------
def spectrum(repulsion: np.ndarray) -> tuple[core.Scalar, np.ndarray]:
    """Six two-electron energy levels (`[..., 6] Scalar+`) pair with the singlet-triplet gap (`[...]`)
    from `block_rate` minus half `repulsion`."""
    energies = core.hamiltonian(hopping, repulsion).eigvalsh()  # [..., 6] Scalar+
    return energies, core.block_rate(hopping, repulsion) - repulsion / 2


def ground_states(repulsion: np.ndarray) -> tuple[core.Scalar, core.Scalar, core.Scalar]:
    """Ground states at `repulsion` carry configuration probabilities (`[..., 6] Scalar+`), double
    occupancy (`[...] Scalar+`), and spin alignment (`[...] Scalar+`)."""
    ground = core.ground_state(hopping, repulsion)              # [...] Bivector+
    return (
        core.probabilities(ground),
        core.expectation(ground, core.double_occupancy),
        core.expectation(ground, core.spin_alignment),
    )


def exchange(repulsion: np.ndarray, exchange_phase: np.ndarray) -> tuple[core.Pair, core.Pair]:
    """Rescaling `exchange_phase` by each regime's singlet-triplet gap yields cosine and sine state
    parts (`[..., ...] Bivector+`, `[..., ...] Bivector+`) over a common exchange cycle."""
    gap = core.block_rate(hopping, repulsion) - repulsion / 2   # [...]
    times = exchange_phase / gap[..., None]                     # [..., ...]
    return core.exchange_parts(hopping, repulsion[..., None], times)


def exchange_frames(repulsion: np.ndarray, exchange_phase: np.ndarray) -> Iterator[tuple[core.Pair, core.Pair]]:
    """Yields `(cosine_part, sine_part)` (`[...] Bivector+`, `[...] Bivector+`) step by step along
    `exchange_phase`, with each regime scaled to its own singlet-triplet gap."""
    return core.exchange_states(hopping, repulsion, exchange_phase)


def field_exchange(ratios: np.ndarray, exchange_phase: np.ndarray) -> Iterator[tuple[core.Pair, core.Pair]]:
    """Yields `(cosine_part, sine_part)` (`[...] Bivector+`, `[...] Bivector+`) at the strongest
    repulsion under magnetic field differences `ratios` times the exchange gap."""
    strong = repulsions[-1]
    gap = core.block_rate(hopping, strong) - strong / 2
    hamiltonians = core.hamiltonian(hopping, strong) + ratios * gap * core.field_difference   # [...] Bivector+ <- Bivector+
    energies, modes = hamiltonians.eigh()                       # [..., 6] Scalar+, Bivector+
    return core.mode_evolution(energies, modes, core.mv.ad, exchange_phase / gap)


def charge_transfer(repulsion: np.ndarray, asymmetry: np.ndarray) -> core.Scalar:
    """Minus twice the expectation of `site_difference` measures ground-state occupation moved from the
    right site to the lower left site (`[..., ...] Scalar+`)."""
    hamiltonians = core.hamiltonian(hopping, repulsion[:, None]) + asymmetry * core.site_difference   # [..., ...] Bivector+ <- Bivector+
    _, modes = hamiltonians.eigh()                              # [..., ..., 6] Bivector+
    ground = modes[..., 0]                                      # [..., ...] Bivector+
    return -2 * core.expectation(ground, core.site_difference)  # [..., ...] Scalar+


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.quantum.hubbard_dimer import render
    from examples.util.animation import save_animation, save_figure
    from numga import stack

    energies, gap = spectrum(sweep)
    ground_probabilities, _, _ = ground_states(repulsions)
    cosine_part, sine_part = exchange(repulsions, phase)
    probabilities = core.probabilities(cosine_part) + core.probabilities(sine_part)
    save_figure(render.spectrum_figure(sweep, hopping, energies, gap, perturbative, strong_repulsion), "hubbard_dimer_spectrum")
    save_figure(render.ground_states_figure(names, repulsions / hopping, ground_probabilities),
                "hubbard_dimer_ground_states")
    save_figure(render.exchange_figure(phase, probabilities, spin_only), "hubbard_dimer_exchange")

    dial_rows = [f"{name}\n$U/t = {ratio:g}$" for name, ratio in zip(names, repulsions / hopping)]
    dial_history = []

    def dial_stream() -> Iterator[tuple[core.Pair, core.Pair]]:
        for state in exchange_frames(repulsions, phase[::dial_stride]):
            dial_history.append(state)
            yield state

    save_animation(
        render.dials_animation(dial_rows, core.configurations, core.zero_triplet, dial_stream()),
        "hubbard_dimer_dials", frame_duration_ms,
    )

    charge_differences = charge_transfer(repulsions, asymmetries)
    save_figure(render.transfer_figure(asymmetries / hopping, charge_differences, names, repulsions / hopping),
                "hubbard_dimer_transfer")

    field_rows = [f"field difference\n{ratio:g} exchange" for ratio in field_ratios]
    field_history = []

    def field_stream() -> Iterator[tuple[core.Pair, core.Pair]]:
        for state in field_exchange(field_ratios, phase[::dial_stride]):
            field_history.append(state)
            yield state

    save_animation(
        render.dials_animation(field_rows, core.configurations, core.zero_triplet, field_stream()),
        "hubbard_dimer_field_dials", frame_duration_ms,
    )

    # checks
    # Applying the shifted Hamiltonian twice on the singlet block multiplies by the block rate squared,
    # the analytic gap matches the full spectrum, and single-pass time evolution preserves normalization.
    hamiltonians = core.hamiltonian(hopping, sweep[:, None])    # [n_samples, 1] Bivector+ <- Bivector+
    shift = sweep[:, None] / 2                                  # [n_samples, 1]
    rate = core.block_rate(hopping, sweep[:, None])             # [n_samples, 1]
    block = stack([core.singlet, core.symmetric_double])        # [2] Bivector+
    block_step = hamiltonians(block) - shift * block            # [n_samples, 2] Bivector+
    assert ((hamiltonians(block_step) - shift * block_step - rate**2 * block).scalar_norm_squared().square_root() < 1e-10).all()
    assert (((energies[..., 1] - energies[..., 0]) - gap).abs() < 1e-10).all()
    assert ((probabilities.sum(axis=-1) - 1).abs() < 1e-10).all()
    field_cosine, field_sine = stack([s[0] for s in field_history], axis=1), stack([s[1] for s in field_history], axis=1)
    field_probabilities = core.probabilities(field_cosine) + core.probabilities(field_sine)
    assert ((field_probabilities.sum(axis=-1) - 1).abs() < 1e-10).all()
    assert (field_probabilities[-1, :, 3] < 0.05).all()
    print("Hubbard dimer checks passed.")


if __name__ == "__main__":
    main()
