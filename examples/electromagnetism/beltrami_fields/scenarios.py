"""Solid toroidal cavities host chiral Beltrami eigenmodes and helicity-preserving selective decay."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from numga.sparse import Sparse
from numga.topology import Topology
from examples.electromagnetism.beltrami_fields.core import (
    Field,
    Point,
    Scalar,
    magnetic_energy,
    magnetic_helicity,
    mv,
    relaxation_states,
    simplex_gradients,
    simplex_volumes,
    whitney_1form,
    whitney_curl,
)
from examples.electromagnetism.beltrami_fields.numerical import coexact_helicity_eigh
from examples.electromagnetism.beltrami_fields.plumbing import flux_preserving, solid_torus

major_radius = 1.0
minor_radius = 0.42
n_major = 16
n_minor = 8
shown_indices = np.array([-1, -4])                                              # [2]
n_shown = len(shown_indices)
selected_indices = np.array([-1, -2, -4, -8, 3])                                # [5]
n_selected = len(selected_indices)
initial_weights = np.array([0.45, 0.70, 0.85, 0.85, 0.45])                      # [n_selected]
n_frames = 46
relaxation_time = 0.025
decay_times = np.linspace(0.0, relaxation_time, n_frames)                       # [n_frames]
frame_duration_ms = 40
n_gallery_colours = 64


# --- plumbing -------------------------------------------------------------------------
def torus_cavity() -> tuple[Topology, Point, Sparse]:
    """The cavity parameters define a tetrahedral torus (`Point-[V]`) and its flux-preserving edge subspace (`Scalar+[E <- E_free]`)."""
    topology, vertices = solid_torus(major_radius, minor_radius, n_major, n_minor)
    return topology, vertices, flux_preserving(topology, n_minor)


def main() -> None:
    """The scenario entry point writes the Beltrami spectrum figure and relaxation animation and checks invariants."""
    from examples.electromagnetism.beltrami_fields import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, free = torus_cavity()
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes = simplex_volumes(topology, vertices)                               # Volume-[T]
    gradients = simplex_gradients(topology, vertices, volumes)                  # [4] Plane+[T]
    potential_map = whitney_1form(topology, gradients)                          # (Plane+ <- Scalar+)[T <- E]
    curl = whitney_curl(topology, gradients)                                    # (Field+ <- Scalar+)[T <- E]
    stiffness = magnetic_energy(volumes, curl)                                  # (Antiscalar- <- Scalar+)[E <- E]
    helicity = magnetic_helicity(volumes, potential_map, curl)                  # (Antiscalar- <- Scalar+)[E <- E]

    free_energy = free.adjugate() * stiffness * free                            # (Antiscalar- <- Scalar+)[E_free <- E_free]
    free_helicity = free.adjugate() * helicity * free                           # (Antiscalar- <- Scalar+)[E_free <- E_free]
    inverse_eigenvalues, free_modes, n_null = coexact_helicity_eigh(free_energy, free_helicity)
    modes = free * free_modes                                                   # [...] Scalar+[E]
    shown_modes = modes[shown_indices]                                          # [n_shown] Scalar+[E]
    shown_fields = curl(shown_modes)                                            # [n_shown] Field+[T]
    save_figure(
        render.eigenmodes_figure(
            topology, vertices, centres, volumes, inverse_eigenvalues, shown_fields, shown_indices,
        ),
        "beltrami_eigenmodes",
    )

    selected_modes = modes[selected_indices]                                    # [n_selected] Scalar+[E]
    selected_eigenvalues = 1 / inverse_eigenvalues[selected_indices]            # [n_selected] Scalar+
    initial_amplitudes = mv.scalar(initial_weights[:, None])                    # [n_selected] Scalar+
    streamed_states: list[tuple[Scalar, Field]] = []

    def recorded_states() -> Iterator[tuple[Scalar, Field]]:
        """Yields each relaxation state once while recording it for downstream checks."""
        for state in relaxation_states(initial_amplitudes, selected_eigenvalues, selected_modes, curl, decay_times):
            streamed_states.append(state)
            yield state

    save_animation(
        render.relaxation_animation(topology, vertices, centres, volumes, recorded_states()),
        "beltrami_relaxation",
        frame_duration_ms,
    )

    # checks
    linear_potential = (mv.x & vertices) * 0.6 - (mv.y & vertices) * 0.9 + (mv.z & vertices) * 0.4
    exact_gradient = (gradients * (topology.incidence[3, 0] * linear_potential)).sum(axis=0)
    assert ((potential_map(topology.d[0] * linear_potential) - exact_gradient).norm() < 1e-11).all()
    assert (curl(topology.d[0] * linear_potential).norm() < 1e-11).all()
    assert ((topology.reading(topology.exterior[2], 2) * (topology.d[1] * shown_modes[0])).norm() < 1e-12).all()

    assert n_null == n_major
    shown_eigenvalues = 1 / inverse_eigenvalues[shown_indices]
    shown_energies = (stiffness(shown_modes) & shown_modes).sites.sum() / 2
    shown_helicities = (helicity(shown_modes) & shown_modes).sites.sum()
    shown_residuals = free.adjugate() * (stiffness(shown_modes) - shown_eigenvalues * helicity(shown_modes))
    assert ((inverse_eigenvalues + inverse_eigenvalues[::-1]).norm() < 1e-11).all()
    assert ((shown_energies - 0.5).norm() < 1e-11).all()
    assert ((shown_helicities - inverse_eigenvalues[shown_indices]).norm() < 1e-11).all()
    assert (shown_residuals.dual_inverse().norm().sites.sum() < 1e-10).all()

    amplitude_frames = stack([amplitudes for amplitudes, _ in streamed_states])
    energy_frames = amplitude_frames.squared().sum(axis=1) / 2
    helicity_frames = (amplitude_frames.squared() / selected_eigenvalues).sum(axis=1)
    taylor_bound = helicity_frames[0] * selected_eigenvalues[0] / 2
    assert ((helicity_frames - helicity_frames[0]).norm() < 1e-11).all()
    assert (energy_frames[1:] < energy_frames[:-1]).all()
    assert energy_frames[-1] > taylor_bound
    assert (energy_frames[-1] - taylor_bound) / taylor_bound < 0.02


def pinned() -> None:
    """The gallery entry writes the animation of tangled magnetic field lines relaxing into the coherent Taylor state."""
    from examples.electromagnetism.beltrami_fields import render
    from examples.util.animation import pin_animation

    topology, vertices, free = torus_cavity()
    centres = topology.incidence[3, 0].mean(axis=0) * vertices                  # Point-[T]
    volumes = simplex_volumes(topology, vertices)                               # Volume-[T]
    gradients = simplex_gradients(topology, vertices, volumes)                  # [4] Plane+[T]
    potential_map = whitney_1form(topology, gradients)                          # (Plane+ <- Scalar+)[T <- E]
    curl = whitney_curl(topology, gradients)                                    # (Field+ <- Scalar+)[T <- E]
    stiffness = magnetic_energy(volumes, curl)                                  # (Antiscalar- <- Scalar+)[E <- E]
    helicity = magnetic_helicity(volumes, potential_map, curl)                  # (Antiscalar- <- Scalar+)[E <- E]

    free_energy = free.adjugate() * stiffness * free                            # (Antiscalar- <- Scalar+)[E_free <- E_free]
    free_helicity = free.adjugate() * helicity * free                           # (Antiscalar- <- Scalar+)[E_free <- E_free]
    inverse_eigenvalues, free_modes, _ = coexact_helicity_eigh(free_energy, free_helicity)
    modes = free * free_modes                                                   # [...] Scalar+[E]
    selected_modes = modes[selected_indices]                                    # [n_selected] Scalar+[E]
    selected_eigenvalues = 1 / inverse_eigenvalues[selected_indices]            # [n_selected] Scalar+
    initial_amplitudes = mv.scalar(initial_weights[:, None])                    # [n_selected] Scalar+
    states = relaxation_states(initial_amplitudes, selected_eigenvalues, selected_modes, curl, decay_times)
    pin_animation(
        render.relaxation_animation(topology, vertices, centres, volumes, states),
        "beltrami",
        frame_duration_ms,
        colors=n_gallery_colours,
    )


if __name__ == "__main__":
    main()
