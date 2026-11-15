"""Scenarios for the Hopf fibration, generating nested Hopf tori, parallel transport around a spin loop, and a spiralling fibre sweep."""

from __future__ import annotations

import numpy as np

from examples.math.hopf import core

mv = core.mv

torus_polars = np.pi * np.array([0.9, 0.7, 0.5])
n_per_circle = 18
n_samples = 240

loop_polar = 2.2
n_steps = 400

spiral_polars = np.pi * np.array([0.95, 0.45])
n_turns = 4
n_frames = 68
frame_duration_ms = 40


def torus_fibres(polars: np.ndarray, n_per_circle: int,
                 n_samples: int) -> tuple[core.Antivector, core.Even]:
    """Latitude circles at `polars` carry evenly spaced spins (`[..., n_per_circle] Antivector-`) and their sampled fibres (`[..., n_per_circle, n_samples + 1] Even+`)."""
    azimuths = np.linspace(0.0, 2 * np.pi, n_per_circle, endpoint=False)
    spins = core.sphere_spins(polars[:, None], azimuths[None, :])              # [..., n_per_circle] Antivector-
    fibre_turns = (mv.xy * np.linspace(0.0, 2 * np.pi, n_samples + 1)).exp()   # [n_samples + 1] Rotor+
    fibres = core.fibre_start(spins)[..., None] * fibre_turns                  # [..., n_per_circle, n_samples + 1] Even+
    return spins, fibres


def lifted_loop(polar: float, n_steps: int) -> tuple[core.Antivector, core.Even, core.Even]:
    """Parallel transport around the latitude circle at `polar` gives the spin loop (`[n_steps + 1] Antivector-`), transported spinors (`[n_steps] Even+`), and initial fibre (`[n_steps + 1] Even+`)."""
    angles = np.linspace(0.0, 2 * np.pi, n_steps + 1)
    loop = core.sphere_spins(polar, angles)                                    # [n_steps + 1] Antivector-
    loop_start = core.fibre_start(loop[0])                                     # [] Even+
    carried_spinors = core.transport_rotors(loop) * loop_start                 # [n_steps] Even+
    fibre_turns = (mv.xy * angles).exp()                                       # [n_steps + 1] Rotor+
    start_fibre = loop_start * fibre_turns                                     # [n_steps + 1] Even+
    return loop, carried_spinors, start_fibre


def main() -> None:
    from examples.math.hopf import render
    from examples.util.animation import save_animation, save_figure

    spins, fibres = torus_fibres(torus_polars, n_per_circle, n_samples)
    projected_fibres = core.stereographic_projection(fibres)                   # [3, n_per_circle, n_samples + 1] Antivector-
    save_figure(render.tori_figure(spins, projected_fibres), "hopf_tori")

    loop, carried_spinors, start_fibre = lifted_loop(loop_polar, n_steps)
    save_figure(
        render.lift_figure(
            loop,
            core.stereographic_projection(carried_spinors),
            core.stereographic_projection(start_fibre),
        ),
        "hopf_lift",
    )

    sweep_states = []

    def recorded_sweep():
        for state in core.fibre_sweep(spiral_polars, n_turns, n_frames, n_samples):
            sweep_states.append(state)
            yield state

    save_animation(render.sweep_animation(recorded_sweep()), "hopf_sweep", frame_duration_ms)

    # checks
    assert ((core.hopf(fibres, fibres) - spins[..., None]).antinorm() < 1e-9).all()
    pair_link = core.linking_number(projected_fibres[0, 0], projected_fibres[-1, n_per_circle // 3])
    assert ((pair_link - core.handedness).abs() < 1e-2).all()
    assert len(sweep_states) == n_frames
    assert ((sweep_states[-1][0].antinorm() - 1).abs() < 1e-9).all()
    half_solid_angle = np.pi * (1 - np.cos(loop_polar))
    expected_end = start_fibre[0] * (mv.xy * (-half_solid_angle)).exp()        # [] Even+
    assert ((carried_spinors[-1] - expected_end).norm() < 1e-4).all()


if __name__ == "__main__":
    main()
