"""Scenes for two spins: the exchange interaction swapping a spin up and a spin down, entangling them
on the way, and the singlet, which reaches the largest Bell combination any state can."""

from __future__ import annotations

import numpy as np

from examples.quantum.two_spins import core

mv = core.mv
# The first spin up along z, the second down: a half turn in its ZX plane turns Z over.
UP_DOWN = -mv.ZX * core.CORRELATOR                                             # [] Spinor


# --- math -----------------------------------------------------------------------------
def swap(frames: int) -> tuple[np.ndarray, core.First, core.Second, core.Correlation, core.Scalar]:
    """Spin up and spin down under the exchange interaction, over the angle that swaps them. Returns
    the angles, the two Bloch vectors, the correlation maps and the largest Bell combinations."""
    angles = np.linspace(0.0, np.pi / 4, frames)
    states = core.exchange(UP_DOWN, angles)                                    # [frames] Spinor
    first, second = core.bloch(states)                                         # [frames] First, Second
    correlations = core.correlation(states)                                    # [frames] Correlation
    bells = core.bell(correlations)                                            # [frames] Scalar

    # --- checks
    # The singlet and triplet parts are idempotent and add up to one. The exchange keeps the state
    # normalized and both Bloch vectors equally long. Halfway the spins are fully entangled: no Bloch
    # vector, and the Bell combination at 2 * sqrt(2); at the end they have swapped.
    np.testing.assert_allclose((core.COUPLING * core.COUPLING - 3 - 2 * core.COUPLING).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((core.SINGLET * core.SINGLET - core.SINGLET).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((core.SINGLET + core.TRIPLET - core.ONE).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(core.expectation(core.ONE, states).to_array(), 1.0, atol=1e-7)
    np.testing.assert_allclose((first | first).to_array(), (second | second).to_array(), atol=1e-7)
    middle = frames // 2
    np.testing.assert_allclose(first[middle].kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose(bells.to_array()[[0, middle, -1]], [2.0, 2 * np.sqrt(2), 2.0], atol=1e-7)
    np.testing.assert_allclose((first[-1] + mv.z).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((second[-1] - mv.Z).kernel, 0.0, atol=1e-7)
    return angles, first, second, correlations, bells


def singlet(seed: int) -> tuple[core.Spinor, core.Scalar]:
    """The singlet, the singlet part of spin up and spin down normalized, and its Bell combination
    along directions at which it reaches `2 * sqrt(2)`."""
    state = np.sqrt(2) * core.SINGLET * UP_DOWN                                # [] Spinor
    # The first spin's directions a quarter turn apart; the second spin's halfway between them,
    # reversed, since the singlet's spins disagree along every direction.
    half = 1 / np.sqrt(2)
    element = core.bell_element(mv.x, mv.y, -half * (mv.X + mv.Y), -half * (mv.X - mv.Y))   # [] Spinor
    value = core.expectation(element, state)                                   # [] Scalar

    # --- checks
    # The singlet is normalized, the coupling acts on it as three, neither spin shows a Bloch vector,
    # and every direction of one spin is anticorrelated with the same direction of the other. For any
    # four unit directions the Bell element squares to 4 - 4 * (a ^ a') * (b ^ b'); the singlet
    # reaches the bound 2 * sqrt(2) that this sets.
    np.testing.assert_allclose(core.expectation(core.ONE, state).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((core.COUPLING * state - 3 * state).kernel, 0.0, atol=1e-12)
    first, _ = core.bloch(state)
    np.testing.assert_allclose(first.kernel, 0.0, atol=1e-12)
    correlations = core.correlation(state)
    for across, along in ((mv.X, mv.x), (mv.Y, mv.y), (mv.Z, mv.z)):
        np.testing.assert_allclose((correlations(across) + along).kernel, 0.0, atol=1e-12)
    rng = np.random.default_rng(seed)
    directions = [mv(space, rng.normal(size=3)).normalized() for space in (core.First, core.First, core.Second, core.Second)]
    first_plane = directions[0] ^ directions[1]                                # [] Bivector
    second_plane = directions[2] ^ directions[3]                               # [] Bivector
    square = core.bell_element(*directions) * core.bell_element(*directions)  # [] Spinor
    np.testing.assert_allclose((square - (4 - 4 * first_plane * second_plane)).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(value.to_array(), 2 * np.sqrt(2), atol=1e-12)
    return state, value


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.quantum.two_spins import render

    _, reached = singlet(0)
    print("Bell combination of the singlet:", reached.to_array())
    swapped = swap(121)
    save_figure(render.draw_pair(*swapped, 60), "two_spins")
    save_animation(render.animate_pair(*swapped), "two_spins", 60)
