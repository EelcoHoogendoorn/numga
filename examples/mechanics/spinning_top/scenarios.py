"""Scenes for the spinning top: a disc, a stem and a tip, spun on flat ground, in four variations of its tip,
its disc, and the friction of the ground."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace

import numpy as np

from examples.mechanics import lie_integrators as lie
from examples.mechanics.spinning_top import core

mv = core.mv
UP = core.mv(core.Direction, np.array([0.0, 0.0, 1.0]))
GRAVITY = 9.81
STATIC_FRICTION = 0.4
# The top's stem and tip: the stem's semi-axes, how far its centre sits above the disc's, its mass, the
# tip's height and its mass; and the disc's semi-axes.
DISC_SEMI = np.array([0.25, 0.25, 0.04])
STEM_SEMI = np.array([0.02, 0.02, 0.15])
STEM_ABOVE_DISC = 0.16
STEM_MASS = 0.05
TIP_HEIGHT = 0.14
TIP_MASS = 0.15
# The tilt it is spun at, and how far the tip starts clear of the ground.
TILT = 0.15
CLEARANCE = 0.002
# The run: its length in seconds, the time step, the steps per frame, and the frames' pixels and duration
# in milliseconds.
RUNTIME = 5.0
STEP = 4e-3
N_STEPS_PER_FRAME = 10
PIXELS = 260
FRAME_DURATION = 33


@dataclass(frozen=True)
class Setup:
    """A top and its ground: the spin it starts with, its tip's radius, its disc's height above the
    tip and mass, the ground's curvature, the dynamic friction, and the air's drag rate."""
    spin_rate: float
    tip_radius: float
    disc_height: float
    disc_mass: float
    ground_curvature: float
    dynamic_friction: float
    drag: float


BASE = Setup(18.0, 0.05, 0.10, 1.0, 0.0, 0.3, 0.13)
VARIATIONS = {
    "base": BASE,
    "sharp_tip": replace(BASE, tip_radius=0.02),
    "heavy_disc": replace(BASE, disc_mass=2.0),
    "slippery": replace(BASE, dynamic_friction=0.1),
}
VIEW, CENTRE, EXTENT = (30, -60), np.array([0.0, 0.0, 0.15]), 0.55


# --- math -----------------------------------------------------------------------------
def top(tip_radius: float, disc_height: float, disc_mass: float):
    """A disc, a stem and a tip, each a solid ellipsoid, in the frame of their centre of mass: the parts,
    the inertia and its inverse, the total mass, and the height of the centre of mass above the tip."""
    centres = np.array([[0, 0, disc_height], [0, 0, disc_height + STEM_ABOVE_DISC], [0, 0, TIP_HEIGHT]])
    semi = np.stack([DISC_SEMI, STEM_SEMI, np.array([tip_radius, tip_radius, TIP_HEIGHT])])
    mass = np.array([disc_mass, STEM_MASS, TIP_MASS])
    com = (centres * mass[:, None]).sum(0) / mass.sum()
    centres = centres - com
    parts = core.ellipsoid(core.mv.point(centres), semi)                       # [parts] Plane <- Point
    points, masses = core.sigma_points(centres, semi, mass)
    inertia = core.inertia(points, masses)                                     # [] Forque <- Rate
    return parts, inertia, inertia.inverse(), mass.sum(), com[2]


def spin(setup: Setup, seconds: float, dt: float, n_steps_per_frame: int) -> Iterator[tuple]:
    """Simulate from a slight tilt; yield the pose, the rate, the parts and the ground every few steps. A frame's steps
    run under `jax.lax.scan`: each is a few hundred small operations on one top, which the compiled loop does at once."""
    import jax
    from numga.backend.jax import JaxContext

    parts, inertia, inverse_inertia, mass, height = top(setup.tip_radius, setup.disc_height, setup.disc_mass)
    ground = core.bowl(setup.ground_curvature)
    origin = core.mv.point(np.zeros(3))
    # `(mv.zw * a).exp()` translates up by `2 * a`, `(mv.zx * a).exp()` turns by `2 * a`: lift the
    # centre of mass so the tip clears, then tilt.
    motor = ((mv.zw * ((height * np.cos(TILT) + CLEARANCE) / 2)).exp() * (mv.zx * (TILT / 2)).exp()).normalized()
    rate = mv.xy * setup.spin_rate

    def forces(motor, rate):
        """The weight, a force line down through the centre of mass, and the air's drag, opposing
        the momentum; in the body frame."""
        weight = motor << ((motor >> origin) & core.mv(core.Direction, np.array([0.0, 0.0, -GRAVITY * mass])))
        return weight - inertia(rate) * setup.drag

    def substep(state: tuple[core.Motor, core.Rate], _) -> tuple[tuple[core.Motor, core.Rate], None]:
        before, rate = state
        # Predict with a free step, then correct against the ground.
        motor, rate = lie.explicit_rk4(before, rate, inertia, inverse_inertia, dt, forces)
        return core.project_contacts(before, motor.normalized(), inverse_inertia, parts, ground,
                                     STATIC_FRICTION, setup.dynamic_friction, dt), None

    # A frame's contact steps compiled into one loop, which carries the motor and the rate as every step returns them:
    # the start, a motor composed of two turns, is retyped to a plain motor.
    frame = jax.jit(lambda state: jax.lax.scan(substep, state, length=n_steps_per_frame)[0])
    state = (motor.retyped(core.Motor).materialize(JaxContext(core.ga)), rate.retyped(core.Rate).materialize(JaxContext(core.ga)))
    for _ in range(int(seconds / dt) // n_steps_per_frame):
        state = frame(state)
        yield state[0].materialize(core.mv.context), state[1].materialize(core.mv.context), parts, ground


def lean(motor: core.Motor) -> core.Scalar:
    """The angle of the top's axis from vertical."""
    return ((motor >> UP).dual() | UP.dual()).abs().arccos()


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.animation import save_animation
    from examples.mechanics.spinning_top import render

    runs = {}
    for name, setup in VARIATIONS.items():
        states = list(spin(setup, RUNTIME, STEP, N_STEPS_PER_FRAME))
        frames = [render.frame(motor, parts, ground, VIEW, CENTRE, EXTENT, PIXELS) for motor, _, parts, ground in states]
        save_animation(frames, f"spinning_top_{name}", FRAME_DURATION)
        runs[name] = states

    # checks
    # Every top stays up over the run, leaning less than it would lying down, and keeps spinning the same
    # way while the air's drag slows it.
    for name, states in runs.items():
        leans = np.array([lean(motor).to_array() for motor, _, _, _ in states])
        spins = np.array([-(mv.xy | rate).to_array() for _, rate, _, _ in states])
        assert leans.max() < np.pi / 4
        assert (spins > 0).all() and spins[-1] < spins[0]


if __name__ == "__main__":
    main()
