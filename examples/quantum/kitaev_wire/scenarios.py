"""The scenes form a topological segment under a gate, move one of its end modes slowly and fast, and bring its
ends together until their modes split in energy."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.kitaev_wire import core

N_SITES = 96
SITES = np.arange(N_SITES)
HOPPING = 1.0
PAIRING = 0.8
INSIDE = 1.4
OUTSIDE = 3.5
WIDTH = 1.2
LEFT = 18.0
RIGHT = 82.0
DESTINATION = 48.0
DURATIONS = np.array([180.0, 12.0])
N_STEPS = 720
STEPS_PER_FRAME = 3
DURATION_MS = 40
N_SPECTRAL_SAMPLES = 200
N_ENERGIES = 8
# The separations of the segment's two ends, closing from far apart to nearly touching.
N_SEPARATIONS = 200
SEPARATIONS = np.linspace(32.0, 4.0, N_SEPARATIONS)
wire = core.Wire.chain(N_SITES, HOPPING, PAIRING)


# --- math -----------------------------------------------------------------------------
def formation() -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    chemical_potentials = np.linspace(OUTSIDE, INSIDE, N_SPECTRAL_SAMPLES)
    potential = core.gate(SITES, LEFT, RIGHT, chemical_potentials[:, None], OUTSIDE, WIDTH)   # [potentials] Scalar+[sites]
    energies, modes = wire.modes(wire.generator(potential), N_ENERGIES)   # [potentials, modes] Scalar+, Majorana+[sites]
    # Averaging the two quadratures removes the arbitrary eigenvector orientation.
    density = modes[:, :2].scalar_norm_squared().mean(axis=-1)    # [potentials] Scalar+[sites]
    return chemical_potentials, energies, density


def gate_at(progress: np.ndarray) -> core.Scalar:
    """The gate profile at fractions of the left boundary's way to its destination."""
    left = LEFT + (DESTINATION - LEFT) * core.smooth_progress(progress)
    return core.gate(SITES, left[:, None], RIGHT, INSIDE, OUTSIDE, WIDTH)


def moving() -> tuple[np.ndarray, core.Scalar, core.Majorana, core.Majorana, core.Scalar]:
    progress = np.arange(N_STEPS // STEPS_PER_FRAME + 1) * STEPS_PER_FRAME / N_STEPS
    potentials = gate_at(progress)                                  # [frames] Scalar+[sites]
    _, modes = wire.modes(wire.generator(potentials), 2)            # [frames, modes] Majorana+[sites]
    localized = wire.localized(modes)                               # [frames, ends] Majorana+[sites]

    midpoint_potentials = gate_at((np.arange(N_STEPS) + 0.5) / N_STEPS)   # [steps] Scalar+[sites]
    history = stack(core.transport(wire, localized[0, 0], midpoint_potentials, DURATIONS, STEPS_PER_FRAME))   # [frames, speeds] Majorana+[sites]
    share = core.boundary_share(modes[:, None], history)            # [frames, speeds] Scalar+
    return progress, potentials, history, localized[:, 0], share


def overlap() -> tuple[core.Scalar, core.Scalar, core.Majorana]:
    potentials = core.gate(SITES, RIGHT - SEPARATIONS[:, None], RIGHT, INSIDE, OUTSIDE, WIDTH)   # [separations] Scalar+[sites]
    energies, modes = wire.modes(wire.generator(potentials), 2)    # [separations, modes] Scalar+, Majorana+[sites]
    return energies.mean(axis=-1), potentials, wire.localized(modes)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.kitaev_wire import render

    save_figure(render.draw_formation(*formation(), HOPPING), "kitaev_formation")
    progress, potentials, history, target, share = moving()
    save_animation(render.animate_transport(progress, potentials, history, target, share, HOPPING),
                   "kitaev_transport", DURATION_MS)
    save_figure(render.draw_transport(progress, history, potentials, HOPPING), "kitaev_transport")
    save_animation(render.animate_overlap(SEPARATIONS, *overlap(), HOPPING), "kitaev_overlap", DURATION_MS,
                   forward_then_back=True)


if __name__ == "__main__":
    main()
