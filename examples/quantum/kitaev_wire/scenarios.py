"""A gated topological segment, slow and fast transport, and boundary-mode splitting."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.kitaev_wire import core

SITES = 96
HOPPING = 1.0
PAIRING = 0.8
INSIDE = 1.4
OUTSIDE = 3.5
WIDTH = 1.2
LEFT = 18.0
RIGHT = 82.0
DESTINATION = 48.0
DURATIONS = np.array([180.0, 12.0])
STEPS = 720
STRIDE = 6
DURATION_MS = 80
SPECTRAL_SAMPLES = 100
ENERGIES = 8
SEPARATIONS = np.linspace(4.0, 32.0, 100)
PROFILE_SEPARATIONS = np.array([24.0, 12.0, 5.0])


# --- math -----------------------------------------------------------------------------
def formation() -> tuple[np.ndarray, core.Scalar, core.Scalar]:
    wire = core.Wire.chain(SITES, HOPPING, PAIRING)
    chemical_potentials = np.linspace(OUTSIDE, INSIDE, SPECTRAL_SAMPLES)
    potential = core.gate(np.arange(SITES), LEFT, RIGHT, chemical_potentials, OUTSIDE, WIDTH)
    energies, modes = wire.modes(wire.generator(potential), ENERGIES)
    # Averaging the two quadratures removes the arbitrary eigenvector orientation.
    density = modes[:, :2].scalar_norm_squared().mean(axis=-1)
    return chemical_potentials, energies, density


def gate_at(progress: np.ndarray) -> core.Scalar:
    """The gate profile at fractions of the left boundary's way to its destination."""
    left = LEFT + (DESTINATION - LEFT) * core.smooth_progress(progress)
    return core.gate(np.arange(SITES), left, RIGHT, INSIDE, OUTSIDE, WIDTH)


def moving() -> tuple[np.ndarray, core.Scalar, core.Majorana, core.Majorana, core.Scalar]:
    wire = core.Wire.chain(SITES, HOPPING, PAIRING)
    progress = np.arange(STEPS // STRIDE + 1) * STRIDE / STEPS
    potentials = gate_at(progress)
    _, modes = wire.modes(wire.generator(potentials), 2)
    localized = wire.localized(modes)

    midpoint_potentials = gate_at((np.arange(STEPS) + 0.5) / STEPS)
    history = stack(core.transport(wire, localized[0, 0], midpoint_potentials, DURATIONS, STRIDE))
    share = core.boundary_share(modes[:, None], history)
    return progress, potentials, history, localized[:, 0], share


def overlap() -> tuple[core.Scalar, core.Scalar, core.Majorana]:
    wire = core.Wire.chain(SITES, HOPPING, PAIRING)
    potentials = core.gate(np.arange(SITES), RIGHT - SEPARATIONS, RIGHT, INSIDE, OUTSIDE, WIDTH)
    energies, _ = wire.modes(wire.generator(potentials), 2)
    profile_potentials = core.gate(np.arange(SITES), RIGHT - PROFILE_SEPARATIONS, RIGHT, INSIDE, OUTSIDE, WIDTH)
    _, modes = wire.modes(wire.generator(profile_potentials), 2)
    return energies.mean(axis=-1), profile_potentials, wire.localized(modes)


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation, save_figure
    from examples.quantum.kitaev_wire import render

    save_figure(render.draw_formation(*formation(), HOPPING), "kitaev_formation")
    progress, potentials, history, target, share = moving()
    save_animation(render.animate_transport(progress, potentials, history, target, share, HOPPING),
                   "kitaev_transport", DURATION_MS)
    save_figure(render.draw_transport(progress, history, potentials, HOPPING), "kitaev_transport")
    save_figure(render.draw_overlap(SEPARATIONS, PROFILE_SEPARATIONS, *overlap(), HOPPING),
                "kitaev_overlap")


if __name__ == "__main__":
    main()
