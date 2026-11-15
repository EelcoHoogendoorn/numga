"""A gate forms a topological segment on a Kitaev wire, transports one end mode slowly and fast, and brings both
ends together until their modes split in energy."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import stack
from examples.quantum.kitaev_wire import core

n_sites = 96
sites = np.arange(n_sites)                                      # [n_sites]
site_positions = core.mv.scalar(sites[:, None]).field()         # [] Scalar+[n_sites]
hopping = 1.0
pairing = 0.8
inside_potential = 1.4
outside_potential = 3.5
edge_width = 1.2
left_end = 18.0
right_end = 82.0
left_destination = 48.0
durations = np.array([180.0, 12.0])                             # [2]
n_steps = 720
steps_per_frame = 3
frame_duration_ms = 40
n_potentials = 200
n_energies = 8
# Separations of the segment's two ends, closing from far apart to nearly touching.
n_separations = 200
separations = np.linspace(32.0, 4.0, n_separations)             # [n_separations]
couplings = core.chain_couplings(n_sites, hopping, pairing)     # (Majorana+ <- Majorana+)[n_sites <- n_sites]


# --- math -----------------------------------------------------------------------------
def formation_sweep(n_samples: int) -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    """Sweeping the gate's interior potential from `outside_potential` down to `inside_potential` gives the `n_energies` lowest excitation energies and lowest-pair mode density across `n_samples` samples."""
    chemical_potentials = np.linspace(outside_potential, inside_potential, n_samples)   # [n_samples]
    potential = core.gate_potential(
        sites, np.array(left_end), np.array(right_end), chemical_potentials[:, None], outside_potential, edge_width,
    )                                                           # [n_samples] Scalar+[n_sites]
    energies, states = core.excitation_modes(core.wire_generator(couplings, potential), n_energies)   # [n_samples, n_energies] Scalar+, Majorana+[n_sites]
    # Averaging the two quadratures removes the arbitrary eigenvector orientation within the pair.
    density = states[:, :2].scalar_norm_squared().mean(axis=-1)   # [n_samples] Scalar+[n_sites]
    return chemical_potentials, energies, density


def moving_gate_potential(progress: np.ndarray) -> core.Scalar:
    """Moving the gate's left boundary smoothly from `left_end` to `left_destination` gives the chemical-potential profile at each fraction `progress` (`[...] Scalar+[n_sites]`)."""
    moving_end = left_end + (left_destination - left_end) * core.smooth_schedule(progress)
    return core.gate_potential(sites, moving_end[:, None], np.array(right_end), np.array(inside_potential), outside_potential, edge_width)


def transport_schedule(n_move_steps: int) -> tuple[np.ndarray, core.Scalar, core.Majorana, core.Majorana, Iterator[tuple[core.Majorana, core.Scalar]]]:
    """Solving the instantaneous end modes along `n_move_steps` steps prepares the frame potentials, adiabatic targets, and streaming Cayley evolution."""
    n_frames = n_move_steps // steps_per_frame + 1
    progress = np.arange(n_frames) * steps_per_frame / n_move_steps   # [n_frames]
    potentials = moving_gate_potential(progress)                # [n_frames] Scalar+[n_sites]
    _, boundary_modes = core.excitation_modes(core.wire_generator(couplings, potentials), 2)   # [n_frames, 2] Majorana+[n_sites]
    localized = core.localized_modes(boundary_modes, site_positions)   # [n_frames, 2] Majorana+[n_sites]
    midpoint_potentials = moving_gate_potential((np.arange(n_move_steps) + 0.5) / n_move_steps)   # [n_move_steps] Scalar+[n_sites]
    states = core.transport_states(couplings, localized[0, 0], midpoint_potentials, boundary_modes, durations, steps_per_frame)
    return progress, potentials, boundary_modes, localized[:, 0], states


def overlap_sweep(gate_separations: np.ndarray) -> tuple[core.Scalar, core.Scalar, core.Majorana]:
    """Closing the separation between the gate's two boundaries yields the lowest excitation splitting, gate potentials, and localized end modes."""
    potentials = core.gate_potential(
        sites, right_end - gate_separations[:, None], np.array(right_end), np.array(inside_potential), outside_potential, edge_width,
    )                                                           # [...] Scalar+[n_sites]
    energies, states = core.excitation_modes(core.wire_generator(couplings, potentials), 2)   # [..., 2] Scalar+, Majorana+[n_sites]
    return energies.mean(axis=-1), potentials, core.localized_modes(states, site_positions)


# --- plumbing -------------------------------------------------------------------------
def save_states(states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments, name: str, duration_ms: int) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms)
    return kept


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.kitaev_wire import render

    chemical_potentials, formation_energies, formation_density = formation_sweep(n_potentials)
    save_figure(render.formation_figure(chemical_potentials, formation_energies, formation_density, hopping), "kitaev_formation")

    n_frames = n_steps // steps_per_frame + 1
    progress, potentials, _, target, states = transport_schedule(n_steps)
    recorded = save_states(
        states, render.transport_animation, progress, potentials, target, hopping,
        name="kitaev_transport", duration_ms=frame_duration_ms,
    )
    history = stack([state for state, _ in recorded])           # [n_frames, 2] Majorana+[n_sites]
    shares = stack([share for _, share in recorded])            # [n_frames, 2] Scalar+
    save_figure(render.transport_figure(progress, history, potentials, hopping), "kitaev_transport")

    splitting, closing_potentials, closing_modes = overlap_sweep(separations)
    save_animation(
        render.overlap_animation(separations, splitting, closing_potentials, closing_modes, hopping),
        "kitaev_overlap", frame_duration_ms, forward_then_back=True,
    )

    # checks
    # Topological ground pair sits near zero energy while outside the segment the gap stays open.
    assert formation_energies[-1, 0] < 1e-4 and formation_energies[0, 0] > 1.0
    # Midpoint Cayley stepping preserves the Majorana state norm across every frame and speed.
    norms = history.scalar_norm_squared().sites.sum()           # [n_frames, 2] Scalar+
    assert ((norms - 1).abs() < 1e-10).all()
    # Slow transport retains almost all weight in the boundary subspace, while fast transport leaks into the bulk.
    assert shares[-1, 0] > 0.99 and shares[-1, 1] < 0.5
    # Closing the separation lifts the hybridized splitting by orders of magnitude.
    assert splitting[-1] > 100 * splitting[0]


if __name__ == "__main__":
    main()
