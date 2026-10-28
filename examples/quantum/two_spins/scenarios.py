"""Scenes for two spins: the exchange interaction swapping a spin up and a spin down, entangling them
on the way, and the singlet, which reaches the largest Bell combination any state can."""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.two_spins import core

mv = core.mv
# The first spin up, turning in its xy plane, the second down: a half turn in its ZX plane turns XY over.
UP_DOWN = -mv.ZX * core.CORRELATOR                                             # [] Spinor+
# The four states of definite spin in the xy planes: the singlet, the triplet of zero spin, and both
# spins up or both down.
FOUR = stack([np.sqrt(2) * core.SINGLET * UP_DOWN, np.sqrt(2) * core.TRIPLET * UP_DOWN,
              core.CORRELATOR, mv.zx * mv.ZX * core.CORRELATOR])               # [states] Spinor+
# The stages of the exchange, and of the steering animation; the planes the first spin is measured in,
# around and down its sphere of unit planes, the duration of a frame of the steering animation in
# milliseconds, and the seed of the random planes and turns the checks run on.
N_FRAMES = 121
N_STEERING_FRAMES = 61
N_AROUND = 24
N_DOWN = 12
DURATION_MS = 120
SEED = 0
# The exchange rate while a field in the xy planes rises, the number of fields, and the largest field.
EXCHANGE_RATE = 1.0
N_FIELDS = 201
MAX_FIELD = 4.0
# The field difference the singlet returns under, the exchange rates compared, and the number of times.
DIFFERENCE = 1.0
EXCHANGE_RATES = np.array([0.0, 1.0, 3.0])
N_TIMES = 241


# --- math -----------------------------------------------------------------------------
def swap(frames: int) -> tuple[np.ndarray, core.FirstPlanes, core.SecondPlanes, core.Scalar, core.Spinor]:
    """Spin up and spin down under the exchange interaction, over the angle that swaps them: the angles,
    the plane each spin turns in, the largest Bell combinations and the states."""
    angles = np.linspace(0.0, np.pi / 4, frames)
    states = core.exchange(UP_DOWN, angles)                                    # [frames] Spinor+
    first, second = core.spin_planes(states)                                   # [frames] FirstPlanes+, SecondPlanes+
    correlations = core.correlation(states)                                    # [frames] FirstPlanes+ <- SecondPlanes+
    bells = core.bell(correlations)                                            # [frames] Scalar+
    return angles, first, second, bells, states


def steering(frames: int, around: int, down: int) -> tuple[core.FirstPlanes, core.Scalar, core.SecondPlanes, core.Spinor]:
    """The first spin measured in unit planes spread over its sphere of planes, at each stage of the
    exchange that swaps spin up and spin down: the planes, how likely +1 is in each, the second spin's
    plane after, and the states measured."""
    angles = np.linspace(0.0, np.pi / 4, frames)
    states = core.exchange(UP_DOWN, angles)                                    # [frames] Spinor+
    planes = sphere(around, down)                                              # [down, around] FirstPlanes+
    probability, steered = core.steer(states[:, None, None], planes)           # [frames, down, around] Scalar+, SecondPlanes+
    return planes, probability, steered, states


def field_sweep(fields: np.ndarray, exchange_rate: float) -> core.Scalar:
    """The energies of the four states of definite spin in the xy planes as a field in those planes
    rises."""
    return core.energy(FOUR, exchange_rate, fields[:, None])                   # [fields, states] Scalar+


def singlet_return(times: np.ndarray, exchange_rates: np.ndarray, difference: float) -> tuple[core.Spinor, core.Scalar]:
    """The singlet under the exchange and a field difference, for each exchange rate: the states and the
    probability of finding the pair back in the singlet."""
    states = core.qubit_turn(FOUR[0], exchange_rates[:, None], difference, times)   # [rates, times] Spinor+
    probability = core.expectation(core.SINGLET, states)                       # [rates, times] Scalar+
    return states, probability


def singlet() -> tuple[core.Spinor, core.Scalar]:
    """The singlet, the singlet part of spin up and spin down normalized, and its Bell combination in
    planes in which it reaches `2 * sqrt(2)`."""
    state = np.sqrt(2) * core.SINGLET * UP_DOWN                                # [] Spinor+
    # The first spin's planes a quarter turn apart; the second spin's halfway between them, reversed,
    # since the singlet's spins disagree in every plane.
    half = 1 / np.sqrt(2)
    element = core.bell_element(mv.yz, mv.zx, -half * (mv.YZ + mv.ZX), -half * (mv.YZ - mv.ZX))   # [] Spinor+
    value = core.expectation(element, state)                                   # [] Scalar+
    return state, value


# --- plumbing -------------------------------------------------------------------------
def sphere(around: int, down: int) -> core.FirstPlanes:
    """Unit planes of the first spin spread over its sphere of planes: xy tipped by the centres of equal
    steps of polar angle, then turned in xy by equal steps of azimuth."""
    polar = (np.arange(down) + 0.5) / down * np.pi
    azimuth = np.arange(around) / around * 2 * np.pi
    tips = (mv.zx * (-polar / 2)).exp()                                        # [down] Rotor+
    turns = (mv.xy * (-azimuth / 2)).exp()                                     # [around] Rotor+
    return turns >> (tips >> mv.xy)[:, None]                                   # [down, around] FirstPlanes+


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.two_spins import render

    state, reached = singlet()
    print("Bell combination of the singlet:", reached.to_array())
    angles, first, second, bells, states = swap(N_FRAMES)
    save_figure(render.draw_exchange(angles, bells, first, render.SPIN_PLANES), "two_spins")
    planes, probability, steered, measured_states = steering(N_STEERING_FRAMES, N_AROUND, N_DOWN)
    save_animation(render.animate_steering(planes, probability, steered, render.SPIN_PLANES), "two_spins_steering", DURATION_MS)
    fields = np.linspace(0.0, MAX_FIELD, N_FIELDS)
    four_first, four_second = core.spin_planes(FOUR)                           # [states] FirstPlanes+, SecondPlanes+
    energies = field_sweep(fields, EXCHANGE_RATE)                              # [fields, states] Scalar+
    save_figure(render.draw_levels(fields, energies, core.FIELD_PLANE.dot(four_first + four_second)), "two_spins_levels")
    times = np.linspace(0.0, np.pi, N_TIMES)
    qubit_states, back_in_singlet = singlet_return(times, EXCHANGE_RATES, DIFFERENCE)
    save_figure(render.draw_qubit(*core.qubit(qubit_states), times, back_in_singlet, EXCHANGE_RATES), "two_spins_qubit")

    # checks
    # The singlet and triplet parts are idempotent and add up to one. The exchange keeps the state
    # normalized and both spins' planes equally large. Halfway the spins are fully entangled: neither
    # turns in any plane, and the Bell combination is at 2 * sqrt(2); at the end they have swapped.
    np.testing.assert_allclose((core.COUPLING * core.COUPLING - 3 - 2 * core.COUPLING).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((core.SINGLET * core.SINGLET - core.SINGLET).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose((core.SINGLET + core.TRIPLET - core.ONE).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(core.expectation(core.ONE, states).to_array(), 1.0, atol=1e-7)
    np.testing.assert_allclose(first.scalar_norm_squared().to_array(), second.scalar_norm_squared().to_array(), atol=1e-7)
    middle = N_FRAMES // 2
    np.testing.assert_allclose(first[middle].kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose(bells.to_array()[[0, middle, -1]], [2.0, 2 * np.sqrt(2), 2.0], atol=1e-7)
    np.testing.assert_allclose((first[-1] + mv.xy).kernel, 0.0, atol=1e-7)
    np.testing.assert_allclose((second[-1] - mv.XY).kernel, 0.0, atol=1e-7)

    # The second spin's plane after the measurement is the correlation map read backwards; the
    # probability is set by the first spin's plane alone. Turning each spin by a rotor of its own space
    # acts on the correlation map from either side and leaves its singular values alone: one, and twice
    # the concurrence, with each spin's plane as large as the square root of one less its square.
    first, second = core.spin_planes(measured_states[:, None, None])           # [frames, 1, 1] FirstPlanes+, SecondPlanes+
    correlations = core.correlation(measured_states[:, None, None])            # [frames, 1, 1] FirstPlanes+ <- SecondPlanes+
    np.testing.assert_allclose((probability - 0.5 * (1 + planes.dot(first))).kernel, 0.0, atol=1e-11)
    backwards = (second + correlations.adjoint()(planes)) / (1 + planes.dot(first))   # [frames, down, around] SecondPlanes+
    np.testing.assert_allclose((steered - backwards).kernel, 0.0, atol=1e-9)
    rng = np.random.default_rng(SEED)
    first_turn = mv(core.FirstPlanes, rng.normal(size=3)).exp()                # [] Rotor+
    second_turn = mv(core.SecondPlanes, rng.normal(size=3)).exp()              # [] Rotor+
    turned = core.correlation(first_turn * second_turn * measured_states)      # [frames] FirstPlanes+ <- SecondPlanes+
    plain = core.correlation(measured_states)                                  # [frames] FirstPlanes+ <- SecondPlanes+
    np.testing.assert_allclose((turned - (first_turn >> plain(second_turn << core.SecondPlanes))).kernel, 0.0, atol=1e-12)
    values = plain.svdvals().to_array()                                        # [frames, 3]
    np.testing.assert_allclose(turned.svdvals().to_array(), values, atol=1e-9)
    np.testing.assert_allclose(values[:, 0], 1.0, atol=1e-9)
    np.testing.assert_allclose(values[:, 1], values[:, 2], atol=1e-9)
    np.testing.assert_allclose(first.scalar_norm_squared().to_array()[:, 0, 0], 1 - values[:, 1] ** 2, atol=1e-9)

    # The singlet sits below the triplet of zero spin by twice the exchange rate, and the field moves the
    # triplets with both spins in its plane or against it by itself: the one in its plane crosses the
    # singlet at twice the exchange rate.
    expected = np.stack([-EXCHANGE_RATE + 0 * fields, EXCHANGE_RATE + 0 * fields,
                         EXCHANGE_RATE - fields, EXCHANGE_RATE + fields], axis=-1)
    np.testing.assert_allclose(energies.to_array(), expected, atol=1e-12)

    # The pair stays normalized; it returns to the singlet as one less the field difference's share of the
    # rate squared, times the sine of the rate times time squared; without a field difference the turn is
    # the exchange of spin up and spin down.
    np.testing.assert_allclose(core.expectation(core.ONE, qubit_states).to_array(), 1.0, atol=1e-12)
    rate = np.hypot(EXCHANGE_RATES[:, None], DIFFERENCE)
    expected = 1 - (DIFFERENCE / rate) ** 2 * np.sin(rate * times) ** 2
    np.testing.assert_allclose(back_in_singlet.to_array(), expected, atol=1e-12)
    across, turned, balance = core.qubit(qubit_states)                         # [rates, times] Scalar+, Scalar+, Scalar+
    np.testing.assert_allclose((across * across + turned * turned + balance * balance).to_array(), 1.0, atol=1e-12)
    first, _ = core.spin_planes(qubit_states)                                  # [rates, times] FirstPlanes+
    alone = np.flatnonzero(EXCHANGE_RATES == 0.0)
    np.testing.assert_allclose(first[alone].kernel, 0.0, atol=1e-12)
    angles = np.linspace(0.0, np.pi / 4, 7)
    exchanged, _ = core.spin_planes(core.exchange(UP_DOWN, angles))
    turned, _ = core.spin_planes(core.qubit_turn(UP_DOWN, 1.0, 0.0, 2 * angles))
    np.testing.assert_allclose((exchanged - turned).kernel, 0.0, atol=1e-11)

    # The singlet is normalized, the coupling acts on it as three, neither spin turns in any plane, and
    # every plane of one spin is anticorrelated with the same plane of the other. For any four unit
    # planes the Bell element squares to four less four times the product of the commutator of the first
    # spin's planes and that of the second's; the singlet reaches the bound 2 * sqrt(2) that this sets.
    np.testing.assert_allclose(core.expectation(core.ONE, state).to_array(), 1.0, atol=1e-12)
    np.testing.assert_allclose((core.COUPLING * state - 3 * state).kernel, 0.0, atol=1e-12)
    first, _ = core.spin_planes(state)
    np.testing.assert_allclose(first.kernel, 0.0, atol=1e-12)
    correlations = core.correlation(state)
    for across, along in ((mv.YZ, mv.yz), (mv.ZX, mv.zx), (mv.XY, mv.xy)):
        np.testing.assert_allclose((correlations(across) + along).kernel, 0.0, atol=1e-12)
    rng = np.random.default_rng(SEED)
    a, a_other = mv(core.FirstPlanes, rng.normal(size=(2, 3))).normalized()    # [] FirstPlanes+, FirstPlanes+
    b, b_other = mv(core.SecondPlanes, rng.normal(size=(2, 3))).normalized()   # [] SecondPlanes+, SecondPlanes+
    element = core.bell_element(a, a_other, b, b_other)                        # [] Spinor+
    np.testing.assert_allclose((element * element - (4 - 4 * a.commutator(a_other) * b.commutator(b_other))).kernel, 0.0, atol=1e-12)
    np.testing.assert_allclose(reached.to_array(), 2 * np.sqrt(2), atol=1e-12)


if __name__ == "__main__":
    main()
