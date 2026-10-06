"""Scenes for two spins: the exchange interaction swapping a spin up and a spin down, entangling them
on the way, and the singlet, which reaches the largest Bell combination any state can."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.two_spins import core

mv = core.mv
# The first spin up along z, the second down: a half turn in its ZX plane turns Z over.
UP_DOWN = -mv.ZX * core.CORRELATOR                                             # [] Spinor
# The four states of definite spin along z: the singlet, the triplet of zero spin, and both spins up or
# both down.
FOUR = stack([np.sqrt(2) * core.SINGLET * UP_DOWN, np.sqrt(2) * core.TRIPLET * UP_DOWN,
              core.CORRELATOR, mv.zx * mv.ZX * core.CORRELATOR])               # [4] Spinor


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


def steering(frames: int, around: int, down: int, seed: int) -> tuple[core.First, core.Scalar, core.Second]:
    """The first spin measured along directions spread over its sphere, at each stage of the exchange
    that swaps spin up and spin down. Returns the directions, how likely +1 is along each, and the
    second spin's Bloch vector after."""
    angles = np.linspace(0.0, np.pi / 4, frames)
    states = core.exchange(UP_DOWN, angles)                                    # [frames] Spinor
    directions = sphere(around, down)                                          # [down, around] First
    probability, steered = core.steer(states[:, None, None], directions)       # [frames, down, around] Scalar, Second

    # --- checks
    # The second spin's Bloch vector after the measurement is the correlation map read backwards; the
    # probability is set by the first spin's Bloch vector alone. Turning each spin by a rotor of its own
    # space acts on the correlation map from either side and leaves its singular values alone: one, and
    # twice the concurrence, with each Bloch vector as long as the square root of one less its square.
    first, second = core.bloch(states[:, None, None])                          # [frames, 1, 1] First, Second
    correlations = core.correlation(states[:, None, None])                     # [frames, 1, 1] Correlation
    np.testing.assert_allclose((probability - 0.5 * (1 + (directions | first))).kernel, 0.0, atol=1e-11)
    backwards = (second + correlations.adjoint()(directions)) / (1 + (directions | first))
    np.testing.assert_allclose((steered - backwards).kernel, 0.0, atol=1e-9)
    rng = np.random.default_rng(seed)
    first_turn = mv(core.FirstPlanes, rng.normal(size=3)).exp()                # [] Rotor
    second_turn = mv(core.SecondPlanes, rng.normal(size=3)).exp()              # [] Rotor
    turned = core.correlation(first_turn * second_turn * states)               # [frames] Correlation
    plain = core.correlation(states)                                           # [frames] Correlation
    np.testing.assert_allclose((turned - (first_turn >> plain(second_turn << core.Second))).kernel, 0.0, atol=1e-12)
    values = plain.svdvals().to_array()                                        # [frames, 3]
    np.testing.assert_allclose(turned.svdvals().to_array(), values, atol=1e-9)
    np.testing.assert_allclose(values[:, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(values[:, 1], values[:, 2], atol=1e-9)
    np.testing.assert_allclose((first | first).to_array()[:, 0, 0], 1 - values[:, 1] ** 2, atol=1e-9)
    return directions, probability, steered


def field_sweep(fields: np.ndarray, exchange_rate: float) -> core.Scalar:
    """The energies of the four states of definite spin along z as a field along z rises."""
    energies = core.energy(FOUR, exchange_rate, fields[:, None])               # [fields, 4] Scalar

    # --- checks
    # The singlet sits below the triplet of zero spin by twice the exchange rate, and the field moves the
    # triplets with both spins along it or against it by itself: the one along it crosses the singlet at
    # twice the exchange rate.
    values = energies.to_array()                                               # [fields, 4]
    expected = np.stack([-exchange_rate + 0 * fields, exchange_rate + 0 * fields,
                         exchange_rate - fields, exchange_rate + fields], axis=-1)
    np.testing.assert_allclose(values, expected, atol=1e-12)
    return energies


def singlet_return(times: np.ndarray, exchange_rates: np.ndarray, difference: float) -> tuple[core.Spinor, core.Scalar]:
    """The singlet under the exchange and a field difference, for each exchange rate. Returns the states
    and the probability of finding the pair back in the singlet."""
    states = core.qubit_turn(FOUR[0], exchange_rates[:, None], difference, times)   # [rates, times] Spinor
    probability = core.expectation(core.SINGLET, states)                       # [rates, times] Scalar

    # --- checks
    # The pair stays normalized; it returns to the singlet as one less the field difference's share of the
    # rate squared, times the sine of the rate times time squared; without a field difference the turn is
    # the exchange of spin up and spin down.
    np.testing.assert_allclose(core.expectation(core.ONE, states).to_array(), 1.0, atol=1e-12)
    rate = np.hypot(exchange_rates[:, None], difference)
    expected = 1 - (difference / rate) ** 2 * np.sin(rate * times) ** 2
    np.testing.assert_allclose(probability.to_array(), expected, atol=1e-12)
    across, turned, balance = core.qubit(states)                               # [rates, times] Scalar each
    np.testing.assert_allclose((across * across + turned * turned + balance * balance).to_array(), 1.0, atol=1e-12)
    first, _ = core.bloch(states)                                              # [rates, times] First
    alone = np.flatnonzero(exchange_rates == 0.0)
    np.testing.assert_allclose(first[alone].kernel, 0.0, atol=1e-12)
    angles = np.linspace(0.0, np.pi / 4, 7)
    exchanged, _ = core.bloch(core.exchange(UP_DOWN, angles))
    turned, _ = core.bloch(core.qubit_turn(UP_DOWN, 1.0, 0.0, 2 * angles))
    np.testing.assert_allclose((exchanged - turned).kernel, 0.0, atol=1e-11)
    return states, probability


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


# --- plumbing -------------------------------------------------------------------------
def sphere(around: int, down: int) -> core.First:
    """Unit directions spread over the sphere: z tipped down by the centres of equal steps of polar
    angle, then turned about z by equal steps of azimuth."""
    polar = (np.arange(down) + 0.5) / down * np.pi
    azimuth = np.arange(around) / around * 2 * np.pi
    tipped = (mv.zx * (-polar / 2)).exp() >> mv.z                             # [down] First
    return (mv.xy * (-azimuth / 2)).exp() >> tipped[:, None]                   # [down, around] First


if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.quantum.two_spins import render

    _, reached = singlet(0)
    print("Bell combination of the singlet:", reached.to_array())
    angles, first, _, _, bells = swap(121)
    save_figure(render.draw_exchange(angles, bells, first), "two_spins")
    save_animation(render.animate_steering(*steering(121, 24, 12, 0)), "two_spins_steering", 60)
    fields = np.linspace(0.0, 4.0, 201)
    first, second = core.bloch(FOUR)
    save_figure(render.draw_levels(fields, field_sweep(fields, 1.0), (first | mv.z) + (second | mv.Z)), "two_spins_levels")
    times = np.linspace(0.0, np.pi, 241)
    exchange_rates = np.array([0.0, 1.0, 3.0])
    states, probability = singlet_return(times, exchange_rates, 1.0)
    save_figure(render.draw_qubit(*core.qubit(states), times, probability, exchange_rates), "two_spins_qubit")
