"""Scenes for pose-graph odometry recover the most likely trajectory around a closed lap by Gauss-Newton from dead reckoning in both planar and three-dimensional projective space.

Instantiating the core module for the algebra of the true step motors runs identical estimation functions on a planar lap in two dimensions and on a pitching, rolling lap in three dimensions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import partial
from types import ModuleType

import numpy as np

from examples import instantiate
from numga import concatenate
from numga.algebras import PGA2D, PGA3D
from numga.topology import Topology

core_module = "examples.estimation.odometry.core"
# Step counts around the planar lap and the spatial test lap, and step length in metres.
n_plane_steps = 36
n_space_steps = 6
step_length = 0.7
# Planar turn-rate modulation and amplitudes of lateral slip, out-of-plane sway, and vertical rise.
turn_variation = 0.4
slip_amplitude = 0.1
sway_amplitude = 0.1
rise_amplitude = 0.05
# Translation and rotation standard deviations for step readings, loop closure, the known start, and vague priors.
reading_std = (0.03, 0.01)
closing_std = (0.001, 0.0003)
known_std = (0.01, 0.01)
vague_std = (100.0, 30.0)
# Full and damped Gauss-Newton iteration counts, damping fraction, random seed, frame duration (ms), and sigma level.
n_full_steps = 3
damping = 0.3
n_damped_steps = 10
seed = 9
frame_duration_ms = 333
sigmas = 2.0


# --- math -----------------------------------------------------------------------------
def damped_closing_states(steps, damping: float, n_iterations: int, seed: int):
    """Damped Gauss-Newton from dead reckoning yields `(poses, uncertainty)` across `n_iterations` steps alongside the survey baselines."""
    core = instantiate(core_module, steps.algebra)
    truth, dead_reckoning, readings, noises, topology, anchors, priors = lap_survey(core, steps, seed)
    V, E = topology.n_elements
    weights, anchor_weights = noises.inverse(), priors.inverse()                # [E] Forque- <- Twist+, [V] Forque- <- Twist+
    reckoned_uncertainty = core.dead_reckoning_covariances(dead_reckoning, noises, priors)  # [V] Twist+ <- Forque-
    iterates = core.gauss_newton_iterates(
        dead_reckoning, readings, weights, topology, anchors, anchor_weights, damping, n_iterations,
    )
    states = core.relaxation_states(
        iterates, reckoned_uncertainty, damping, noises, topology, priors, weights, anchor_weights,
    )
    return truth, dead_reckoning, reckoned_uncertainty, states


def closed_lap_solution(steps, n_iterations: int, seed: int):
    """Full Gauss-Newton steps from dead reckoning recover the most likely poses and their marginal covariances."""
    core = instantiate(core_module, steps.algebra)
    truth, dead_reckoning, readings, noises, topology, anchors, priors = lap_survey(core, steps, seed)
    V, E = topology.n_elements
    weights, anchor_weights = noises.inverse(), priors.inverse()                # [E] Forque- <- Twist+, [V] Forque- <- Twist+
    reckoned_uncertainty = core.dead_reckoning_covariances(dead_reckoning, noises, priors)  # [V] Twist+ <- Forque-
    *_, poses = core.gauss_newton_iterates(
        dead_reckoning, readings, weights, topology, anchors, anchor_weights, 1.0, n_iterations,
    )                                                                           # [V] Rotor+
    uncertainty = core.marginal_covariances(poses, noises, topology, priors, weights, anchor_weights)  # [V] Twist+ <- Forque-
    return truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty


def endpoint_gradients(steps, poses, seed: int):
    """The pose-graph objective gradient at the dead-reckoned trajectory pairs with the gradient at `poses`."""
    core = instantiate(core_module, steps.algebra)
    _, dead_reckoning, readings, noises, topology, anchors, priors = lap_survey(core, steps, seed)
    V, E = topology.n_elements
    weights, anchor_weights = noises.inverse(), priors.inverse()                # [E] Forque- <- Twist+, [V] Forque- <- Twist+
    return (
        core.objective_gradient(dead_reckoning, readings, weights, topology, anchors, anchor_weights),
        core.objective_gradient(poses, readings, weights, topology, anchors, anchor_weights),
    )


# --- plumbing -------------------------------------------------------------------------
def lap_survey(core: ModuleType, steps, seed: int):
    """Simulating noisy readings along `steps` produces the true and dead-reckoned poses, the pose-graph `Topology`, and all reading and prior covariances."""
    rng = np.random.default_rng(seed)
    n_steps = len(steps)
    edges = np.stack([np.append(np.arange(n_steps), 0), np.append(np.arange(1, n_steps + 1), n_steps)], axis=-1)
    topology = Topology(n_steps + 1, edges)
    V, E = topology.n_elements
    tails, heads = topology.elements[1].T                                       # [E], [E]
    # Body-frame steps compose on the right; reversing the cumulative left product of reverses gives the cumulative right product.
    truth = concatenate([core.mv.rotor()[None], steps.reverse().cumprod(axis=0).reverse()])  # [V] Rotor+
    noises = concatenate([
        core.body_covariance(*reading_std) * np.ones(n_steps),
        core.body_covariance(*closing_std)[None],
    ])                                                                          # [E] Twist+ <- Forque-
    noise_modes = core.covariance_modes(noises)                                 # [E, ...] Twist+
    errors = (noise_modes * rng.normal(size=noise_modes.shape)).sum(axis=-1)    # [E] Twist+
    error_motors = (errors * 0.5).exp()                                         # [E] Rotor+
    readings = (truth[tails].inverse() * truth[heads]) * error_motors           # [E] Rotor+
    dead_reckoning = concatenate([
        truth[:1], truth[0] * readings[:n_steps].reverse().cumprod(axis=0).reverse(),
    ])                                                                          # [V] Rotor+
    anchors = concatenate([truth[:1], dead_reckoning[1:]])                      # [V] Rotor+
    priors = concatenate([
        core.body_covariance(*known_std)[None],
        core.body_covariance(*vague_std) * np.ones(n_steps),
    ])                                                                          # [V] Twist+ <- Forque-
    return truth, dead_reckoning, readings, noises, topology, anchors, priors


def planar_lap_steps(n_steps: int):
    """Sinusoidal turn modulation and lateral slip form `n_steps` true step motors (`[n_steps] Rotor+`) around a closed planar lap."""
    mv = instantiate(core_module, PGA2D).mv
    phase = 2 * np.pi * np.arange(n_steps) / n_steps
    turn = mv.xy * (1 + turn_variation * np.sin(2 * phase)) * 2 * np.pi / n_steps  # [n_steps] Twist+
    slip = mv.yw * slip_amplitude * np.cos(2 * phase)                           # [n_steps] Twist+
    return ((turn + slip + mv.xw * step_length) * 0.5).exp()                    # [n_steps] Rotor+


def spatial_lap_steps(n_steps: int):
    """Pitch, roll, lateral slip, and vertical rise form `n_steps` true step motors (`[n_steps] Rotor+`) around a closed spatial lap."""
    mv = instantiate(core_module, PGA3D).mv
    phase = 2 * np.pi * np.arange(n_steps) / n_steps
    turn = mv.xy * 2 * np.pi / n_steps                                          # [] Twist+
    sway = (mv.zx * np.cos(phase) + mv.yz * np.sin(phase)) * sway_amplitude     # [n_steps] Twist+
    slip = mv.yw * slip_amplitude * np.cos(2 * phase) + mv.zw * rise_amplitude * np.sin(2 * phase)  # [n_steps] Twist+
    return ((turn + sway + slip + mv.xw * step_length) * 0.5).exp()             # [n_steps] Rotor+


def save_states(
    states: Iterable,
    animate: Callable[..., Iterable[np.ndarray]],
    *arguments,
    name: str,
    duration_ms: int,
    stepped: bool = False,
) -> list:
    """Streams `states` once through `animate(*arguments, states)` into `save_animation` and returns the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms, stepped=stepped)
    return kept


def main() -> None:
    """Saves the static pose-graph recovery figure and damped loop-closing animation, then checks planar and spatial convergence."""
    from examples.estimation.odometry import render
    from examples.util.animation import save_figure

    plane = instantiate(core_module, PGA2D)
    plane_steps = planar_lap_steps(n_plane_steps)
    truth, dead_reckoning, reckoned_uncertainty, poses, uncertainty = closed_lap_solution(
        plane_steps, n_full_steps, seed,
    )
    reckoned_ellipsoids = plane.position_ellipsoids(dead_reckoning, reckoned_uncertainty, sigmas)  # [...] Plane+ <- Point-
    save_figure(
        render.trajectory_figure(
            truth, dead_reckoning, reckoned_ellipsoids, poses, plane.position_ellipsoids(poses, uncertainty, sigmas),
        ),
        "odometry",
    )

    _, _, _, closing_states = damped_closing_states(plane_steps, damping, n_damped_steps, seed)
    ellipsoids_of = partial(plane.position_ellipsoids, sigmas=sigmas)
    recorded = save_states(
        closing_states,
        render.closing_animation,
        truth,
        dead_reckoning,
        reckoned_ellipsoids,
        ellipsoids_of,
        name="odometry",
        duration_ms=frame_duration_ms,
        stepped=True,
    )

    # checks
    # Full Gauss-Newton drives the covariance-weighted squared gradient down more than twelve orders of magnitude
    # in both 2D and 3D, while damped relaxation contracts end-of-lap uncertainty far below dead reckoning.
    space_steps = spatial_lap_steps(n_space_steps)
    *_, poses_in_space, space_uncertainty = closed_lap_solution(space_steps, n_full_steps, seed)
    for steps, most_likely, posterior in (
        (plane_steps, poses, uncertainty),
        (space_steps, poses_in_space, space_uncertainty),
    ):
        dead_gradient, optimum_gradient = endpoint_gradients(steps, most_likely, seed)
        dead_scale = (posterior(dead_gradient) & dead_gradient).max()           # [] Scalar+
        optimum_scale = (posterior(optimum_gradient) & optimum_gradient).max()  # [] Scalar+
        assert bool(optimum_scale < 1e-12 * dead_scale)

    unit_info = plane.body_covariance(1.0, 1.0).inverse()                       # [] Forque- <- Twist+
    _, final_damped_uncertainty = recorded[-1]
    assert bool(final_damped_uncertainty[-1](unit_info).trace(0, 1) < 0.01 * reckoned_uncertainty[-1](unit_info).trace(0, 1))


def pinned() -> None:
    """Pins the gallery GIF of damped Gauss-Newton steps closing the planar lap from dead reckoning."""
    from examples.estimation.odometry import render
    from examples.util.animation import pin_animation

    plane = instantiate(core_module, PGA2D)
    truth, dead_reckoning, reckoned_uncertainty, states = damped_closing_states(
        planar_lap_steps(n_plane_steps), damping, n_damped_steps, seed,
    )
    reckoned_ellipsoids = plane.position_ellipsoids(dead_reckoning, reckoned_uncertainty, sigmas)  # [...] Plane+ <- Point-
    ellipsoids_of = partial(plane.position_ellipsoids, sigmas=sigmas)
    pin_animation(
        render.closing_animation(truth, dead_reckoning, reckoned_ellipsoids, ellipsoids_of, states),
        "odometry",
        frame_duration_ms,
        stepped=True,
    )


if __name__ == "__main__":
    main()
