"""Scenarios for two spins under exchange, measurement steering, uniform magnetic fields, and differential driving.

Exchange evolution swaps an initial up-down pair through maximal entanglement, projective measurements steer the
second spin across its sphere, a uniform field drives a ground-state level crossing, and differential driving
precesses the pair on the singlet-triplet qubit sphere.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from examples.quantum.two_spins import core

mv = core.mv
# First spin up in xy and second spin down via a half turn in ZX.
up_down = -mv.ZX * core.correlator                                             # [] Spinor+
# Four states of definite spin in the xy planes: singlet, zero-spin triplet, both up, and both down.
basis_states = stack([
    np.sqrt(2) * core.singlet_part * up_down,
    np.sqrt(2) * core.triplet_part * up_down,
    core.correlator,
    mv.zx * mv.ZX * core.correlator,
])                                                                             # [4] Spinor+
n_frames = 181
n_around = 24
n_down = 12
frame_duration_ms = 40
seed = 0
exchange_rate = 1.0
n_fields = 201
max_field = 4.0
difference = 1.0
exchange_rates = np.array([0.0, 1.0, 3.0])
n_times = 241


# --- math -----------------------------------------------------------------------------
def exchange_swap(n_steps: int) -> tuple[np.ndarray, core.FirstPlanes, core.SecondPlanes, core.Scalar, core.Spinor]:
    """An exchange sweep from 0 to `pi / 4` across `n_steps` samples (`[n_steps] Spinor+`) swaps up-down into down-up."""
    angles = np.linspace(0.0, np.pi / 4, n_steps)
    states = core.exchange_state(up_down, angles)                              # [n_steps] Spinor+
    first_planes, second_planes = core.spin_planes(states)                     # [n_steps] FirstPlanes+, SecondPlanes+
    joints = core.joint_map(states)                                            # [n_steps] FirstPlanes+ <- SecondPlanes+
    bells = core.bell_maximum(joints)                                          # [n_steps] Scalar+
    return angles, first_planes, second_planes, bells, states


def steering_grid(
    states: core.Spinor,
    n_azimuth: int,
    n_polar: int,
) -> tuple[core.FirstPlanes, core.Scalar, core.SecondPlanes]:
    """Projective measurements across `n_polar` by `n_azimuth` unit planes steer the second spin (`[..., n_polar, n_azimuth]`)."""
    planes = plane_sphere(n_azimuth, n_polar)                                  # [n_polar, n_azimuth] FirstPlanes+
    probability, second_planes = core.steered_spin(states[:, None, None], planes)   # [..., n_polar, n_azimuth] Scalar+, SecondPlanes+
    return planes, probability, second_planes


def field_sweep(fields: np.ndarray, coupling_rate: float) -> core.Scalar:
    """The four definite-spin states trace linear Zeeman energies (`[..., 4] Scalar+`) across `fields`."""
    return core.energy(basis_states, np.asarray(coupling_rate), fields[:, None])   # [..., 4] Scalar+


def singlet_oscillation(
    times: np.ndarray,
    rates: np.ndarray,
    field_difference: float,
) -> tuple[core.Spinor, core.Scalar]:
    """Differential driving `field_difference` against `rates` precesses the singlet across `times` and tracks its return probability."""
    states = core.qubit_state(basis_states[0], rates[:, None], np.asarray(field_difference), times)   # [..., ...] Spinor+
    return_probability = core.expectation(core.singlet_part, states)           # [..., ...] Scalar+
    return states, return_probability


def singlet_bound() -> tuple[core.Spinor, core.Scalar]:
    """Quarter-turned measurement planes on the singlet (`[] Spinor+`) saturate the Tsirelson bound (`[] Scalar+`)."""
    state = basis_states[0]                                                    # [] Spinor+
    half = 1 / np.sqrt(2)
    element = core.bell_element(mv.yz, mv.zx, -half * (mv.YZ + mv.ZX), -half * (mv.YZ - mv.ZX))   # [] Spinor+
    bell_value = core.expectation(element, state)                              # [] Scalar+
    return state, bell_value


# --- plumbing -------------------------------------------------------------------------
def plane_sphere(n_azimuth: int, n_polar: int) -> core.FirstPlanes:
    """Polar tipping followed by azimuthal rotation spreads unit planes (`[n_polar, n_azimuth] FirstPlanes+`) over the first spin's sphere."""
    polar = (np.arange(n_polar) + 0.5) / n_polar * np.pi
    azimuth = np.arange(n_azimuth) / n_azimuth * 2 * np.pi
    polar_rotors = (mv.zx * (-polar / 2)).exp()                                # [n_polar] Rotor+
    azimuth_rotors = (mv.xy * (-azimuth / 2)).exp()                            # [n_azimuth] Rotor+
    polar_planes = polar_rotors >> mv.xy                                       # [n_polar] FirstPlanes+
    return azimuth_rotors >> polar_planes[:, None]                             # [n_polar, n_azimuth] FirstPlanes+


def main() -> None:
    from examples.quantum.two_spins import render
    from examples.util.animation import save_animation, save_figure

    singlet_state, singlet_bell = singlet_bound()                              # [] Spinor+, Scalar+
    angles, first_planes, second_planes, bells, states = exchange_swap(n_frames)
    save_figure(render.draw_exchange(angles, bells, first_planes, render.plane_reading), "two_spins")
    planes, probability, steered_planes = steering_grid(states, n_around, n_down)
    save_animation(
        render.animate_steering(planes, probability, steered_planes, render.plane_reading),
        "two_spins_steering",
        frame_duration_ms,
    )
    fields = np.linspace(0.0, max_field, n_fields)
    basis_first, basis_second = core.spin_planes(basis_states)                 # [4] FirstPlanes+, SecondPlanes+
    energies = field_sweep(fields, exchange_rate)                              # [n_fields, 4] Scalar+
    save_figure(
        render.draw_levels(fields, energies, core.field_plane.dot(basis_first + basis_second)),
        "two_spins_levels",
    )
    times = np.linspace(0.0, np.pi, n_times)
    qubit_states, singlet_return = singlet_oscillation(times, exchange_rates, difference)
    imbalance, phase_coordinate, singlet_balance = core.qubit_coordinates(qubit_states)   # [3, n_times] Scalar+, Scalar+, Scalar+
    save_figure(
        render.draw_qubit(imbalance, phase_coordinate, singlet_balance, times, singlet_return, exchange_rates),
        "two_spins_qubit",
    )

    # checks
    # Idempotents, normalization, midpoint maximal entanglement, and spin swap at pi / 4.
    assert ((core.coupling * core.coupling - 3 - 2 * core.coupling).scalar_norm_squared() < 1e-12).all()
    assert ((core.singlet_part * core.singlet_part - core.singlet_part).scalar_norm_squared() < 1e-12).all()
    assert ((core.singlet_part + core.triplet_part - 1).scalar_norm_squared() < 1e-12).all()
    assert ((2 * states.scalar_norm_squared() - 1).abs() < 1e-9).all()
    assert ((first_planes.scalar_norm_squared() - second_planes.scalar_norm_squared()).abs() < 1e-9).all()
    midpoint = n_frames // 2
    assert (first_planes[midpoint].scalar_norm_squared() < 1e-12).all()
    assert ((bells[0] - 2).abs() < 1e-9).all()
    assert ((bells[midpoint] - 2 * np.sqrt(2)).abs() < 1e-9).all()
    assert ((bells[-1] - 2).abs() < 1e-9).all()
    assert ((first_planes[-1] + mv.xy).scalar_norm_squared() < 1e-12).all()
    assert ((second_planes[-1] - mv.XY).scalar_norm_squared() < 1e-12).all()

    # Steering matches the adjoint of the joint spin map, and local turns preserve its singular values.
    joints = core.joint_map(states)                                            # [n_frames] FirstPlanes+ <- SecondPlanes+
    expected_probability = 0.5 * (1 + planes.dot(first_planes[:, None, None])) # [n_frames, n_down, n_around] Scalar+
    assert ((probability - expected_probability).abs() < 1e-11).all()
    adjoint_planes = (second_planes[:, None, None] + joints[:, None, None].adjoint()(planes)) / (
        1 + planes.dot(first_planes[:, None, None])
    )                                                                          # [n_frames, n_down, n_around] SecondPlanes+
    assert ((steered_planes - adjoint_planes).scalar_norm_squared() < 1e-12).all()
    rng = np.random.default_rng(seed)
    first_turn = mv(core.FirstPlanes, rng.normal(size=3)).exp()                # [] Rotor+
    second_turn = mv(core.SecondPlanes, rng.normal(size=3)).exp()              # [] Rotor+
    turned_joints = core.joint_map(first_turn * second_turn * states)          # [n_frames] FirstPlanes+ <- SecondPlanes+
    sandwich_joints = first_turn >> joints(second_turn << core.SecondPlanes)   # [n_frames] FirstPlanes+ <- SecondPlanes+
    assert ((turned_joints - sandwich_joints).svdvals().abs() < 1e-9).all()
    singular_values = joints.svdvals()                                         # [n_frames, 3] Scalar+
    assert ((turned_joints.svdvals() - singular_values).abs() < 1e-9).all()
    assert ((singular_values[:, 0] - 1).abs() < 1e-9).all()
    assert ((singular_values[:, 1] - singular_values[:, 2]).abs() < 1e-9).all()
    assert ((first_planes.scalar_norm_squared() + singular_values[:, 1] ** 2 - 1).abs() < 1e-9).all()

    # Field levels cross at twice the exchange rate, and differential driving stays on the singlet-triplet sphere.
    expected_energies = np.stack(
        [-exchange_rate + 0 * fields, exchange_rate + 0 * fields, exchange_rate - fields, exchange_rate + fields],
        axis=-1,
    )
    assert ((energies - expected_energies).abs() < 1e-12).all()
    assert ((2 * qubit_states.scalar_norm_squared() - 1).abs() < 1e-12).all()
    generalized_rate = np.hypot(exchange_rates[:, None], difference)
    expected_return = 1 - (difference / generalized_rate) ** 2 * np.sin(generalized_rate * times) ** 2
    assert ((singlet_return - expected_return).abs() < 1e-12).all()
    assert ((imbalance * imbalance + phase_coordinate * phase_coordinate + singlet_balance * singlet_balance - 1).abs() < 1e-12).all()
    qubit_first, _ = core.spin_planes(qubit_states)                            # [3, n_times] FirstPlanes+
    assert (qubit_first[0].scalar_norm_squared() < 1e-12).all()

    # Singlet anticorrelation across matching planes, Bell square identity, and spacetime time-commutation.
    assert ((2 * singlet_state.scalar_norm_squared() - 1).abs() < 1e-12).all()
    assert ((core.coupling * singlet_state - 3 * singlet_state).scalar_norm_squared() < 1e-12).all()
    singlet_joint = core.joint_map(singlet_state)                              # [] FirstPlanes+ <- SecondPlanes+
    for second_unit, first_unit in ((mv.YZ, mv.yz), (mv.ZX, mv.zx), (mv.XY, mv.xy)):
        assert ((singlet_joint(second_unit) + first_unit).scalar_norm_squared() < 1e-12).all()
    first_choice, first_alt = mv(core.FirstPlanes, rng.normal(size=(2, 3))).normalized()     # [] FirstPlanes+, FirstPlanes+
    second_choice, second_alt = mv(core.SecondPlanes, rng.normal(size=(2, 3))).normalized()  # [] SecondPlanes+, SecondPlanes+
    element = core.bell_element(first_choice, first_alt, second_choice, second_alt)          # [] Spinor+
    commutator_bound = 4 - 4 * first_choice.commutator(first_alt) * second_choice.commutator(second_alt)   # [] Spinor+
    assert ((element * element - commutator_bound).scalar_norm_squared() < 1e-12).all()
    assert ((singlet_bell - 2 * np.sqrt(2)).abs() < 1e-12).all()
    assert ((mv.t * states - states * mv.t).scalar_norm_squared().abs() < 1e-12).all()
    assert ((mv.T * states - states * mv.T).scalar_norm_squared().abs() < 1e-12).all()


if __name__ == "__main__":
    main()
