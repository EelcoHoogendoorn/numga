"""Station-keeping scenes simulate fleets of vessels holding a set point alongside the predicted and steady-state covariances of their excursions.

Both scenes share the same proportional controller and the same body-frame gust map pushing much harder sideways than forward. In the first scene the vessel holds a fixed heading so excursions remain predominantly sideways; in the second it turns steadily on the spot, rotating sideways gusts into forward excursions and rounding the spread.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from functools import partial

import numpy as np

from examples.estimation.pose_diffusion.core import (
    Covariance,
    Gusts,
    Line,
    Motor,
    Twist,
    fleet_diffusion,
    gust_covariance,
    mv,
    position_ellipsoid,
    position_spread,
    steady_covariance,
)
from numga import stack

# Controller relaxation gain (1/s) and body turning rates in each scene.
relaxation_gain = 0.6
turning_rates = {"still": mv.xy * 0.0, "turning": mv.xy * 2.0}
# Body-frame push directions (sideways, forward, and yaw) and standard deviations per square-root second.
pushes = stack([mv.yw, mv.wx, mv.xy])                                           # [3] Twist+
gust_strengths = np.array([0.35, 0.08, 0.15])
gusts = (pushes * gust_strengths * (pushes & Line)).sum(axis=-1)                # [] Twist+ <- Line-
# Simulation duration and timestep in seconds, fleet size, striding per animation frame, and random seed.
seconds = 8.0
dt = 0.01
n_vessels = 3000
n_steps_per_frame = 4
seed = 0
# Half-width of each square cloud panel in metres, confidence level, animation frame duration in ms, and gain sweep.
extent = 1.6
sigmas = 2.0
frame_duration_ms = 40
n_gains = 60
gains = np.linspace(0.02, 0.5, n_gains)
tolerance = 2.0


# --- math -----------------------------------------------------------------------------
def scene_stream(
    rate: Twist,
    gusts: Gusts,
    dt: float,
    n_steps: int,
    n_stride: int,
    n_vessels: int,
    seed: int,
) -> Iterator[tuple[Motor, Covariance, Covariance]]:
    """Yields `(poses, predicted, limit)` every `n_stride` steps for a fleet turning at `rate` (`[] Twist+`)
    under `gusts` (`[] Twist+ <- Line-`)."""
    # Steady turning rotates body-frame errors by `Twist.commutator(rate)` while control damps them at `relaxation_gain`.
    dynamics = Twist.commutator(rate) - relaxation_gain * Twist                 # [] Twist+ <- Twist+
    noise = gust_covariance(gusts)                                              # [] Twist+ <- Line-
    limit = steady_covariance(dynamics, noise)                                  # [] Twist+ <- Line-
    return fleet_diffusion(dynamics, gusts, noise, limit, dt, n_steps, n_stride, n_vessels, seed)


# --- plumbing -------------------------------------------------------------------------
def save_states(states: Iterable, animate: Callable, *arguments, name: str, duration_ms: int) -> list:
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
    """Streams the side-by-side fleet diffusion animation and settled figure to `plots/` and checks convergence."""
    from examples.estimation.pose_diffusion import render
    from examples.util.animation import save_figure

    n_steps = int(seconds / dt)
    streams = [
        scene_stream(rate, gusts, dt, n_steps, n_steps_per_frame, n_vessels, seed + index)
        for index, rate in enumerate(turning_rates.values())
    ]
    ellipsoid_of = partial(position_ellipsoid, sigmas=sigmas)
    frames = save_states(
        zip(*streams),
        render.clouds_animation,
        extent,
        ellipsoid_of,
        name="pose_diffusion",
        duration_ms=frame_duration_ms,
    )
    final_panels = list(frames[-1])
    save_figure(render.settled_figure(final_panels, extent, ellipsoid_of), "pose_diffusion")
    gain_controllers = -mv.scalar(gains[:, None]) * Twist                       # [n_gains] Twist+ <- Twist+
    spreads = position_spread(steady_covariance(gain_controllers, gust_covariance(gusts)))   # [n_gains] Scalar+ <- (Plane+, Plane+)
    save_figure(render.envelope_figure(gains, spreads, tolerance), "pose_diffusion_gain")

    # checks
    # By the final frame the Euler-integrated covariance in each scene converges to its Lyapunov steady state.
    frame = mv.bivector(np.eye(3)).dual()                                       # [3] Line-
    for _, predicted, limit in final_panels:
        residual = (frame[:, None] & (predicted - limit)(frame[None, :])).abs().max()   # [] Scalar+
        scale = (frame[:, None] & limit(frame[None, :])).abs().max()                    # [] Scalar+
        assert bool(residual < 0.05 * scale)


if __name__ == "__main__":
    main()

