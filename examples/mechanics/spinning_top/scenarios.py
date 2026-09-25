"""Scenes for the spinning top: a disc, a stem and a tip, spun on a shallow bowl.

One setup per variation. Builds the top and the ground, runs the mathematics in `core`, and yields
the geometry for `render`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from examples.mechanics import lie_integrators as lie
from examples.mechanics.spinning_top import core

mv = core.mv
UP = core.mv(core.Direction, np.array([0.0, 0.0, 1.0]))
STATIC_FRICTION = 0.4


@dataclass(frozen=True)
class Setup:
    """A top and its ground: the spin it starts with, its tip's radius, its disc's height above the
    tip and mass, the bowl's curvature, the dynamic friction, the air's drag rate, and the
    indentation of the tip that sets the drilling friction."""
    spin_rate: float
    tip_radius: float
    disc_height: float
    disc_mass: float
    bowl_curvature: float
    dynamic_friction: float
    drag: float
    indentation: float


BASE = Setup(18.0, 0.05, 0.10, 1.0, 0.15, 0.3, 0.13, 0.0)
VARIATIONS = {
    "base": BASE,
    "sharp_tip": replace(BASE, tip_radius=0.02),
    "heavy_disc": replace(BASE, disc_mass=2.0),
    "slippery": replace(BASE, dynamic_friction=0.1),
}
VIEW, CENTRE, EXTENT = (30, -60), np.array([0.0, 0.0, 0.15]), 0.55


def top(tip_radius: float, tip_height: float, disc_height: float, disc_mass: float):
    """A disc, a stem and a tip, each a solid ellipsoid, in the frame of their centre of mass."""
    centres = np.array([[0, 0, disc_height], [0, 0, disc_height + 0.16], [0, 0, tip_height]])
    semi = np.array([[0.25, 0.25, 0.04], [0.02, 0.02, 0.15], [tip_radius, tip_radius, tip_height]])
    mass = np.array([disc_mass, 0.05, 0.15])
    com = (centres * mass[:, None]).sum(0) / mass.sum()
    centres = centres - com
    parts = core.ellipsoid(core.point(centres), semi)
    points, masses = core.sigma_points(centres, semi, mass)
    inertia = core.inertia(points, masses)
    return parts, inertia, inertia.inverse(), mass.sum(), com[2]


def spin(setup: Setup, seconds: float, dt: float, every: int):
    """Simulate from a slight tilt; yield the pose, the parts and the ground every few steps."""
    parts, inertia, inertia_inv, mass, height = top(setup.tip_radius, 0.14, setup.disc_height, setup.disc_mass)
    ground = core.bowl(setup.bowl_curvature)
    origin = core.point(np.zeros(3))
    tilt = 0.15
    # `(mv.zw * a).exp()` translates up by `2 * a`, `(mv.zx * a).exp()` turns by `2 * a`: lift the
    # centre of mass so the tip clears, then tilt.
    motor = ((mv.zw * ((height * np.cos(tilt) + 0.002) / 2)).exp() * (mv.zx * (tilt / 2)).exp()).normalized()
    rate = mv.xy * setup.spin_rate

    def forces(motor, rate):
        """The weight, a force line down through the centre of mass, and the air's drag, opposing
        the momentum; in the body frame."""
        weight = motor << ((motor >> origin) & core.mv(core.Direction, np.array([0.0, 0.0, -9.81 * mass])))
        return weight - inertia(rate) * setup.drag

    started = time.time()
    for i in range(int(seconds / dt)):
        before = motor
        # Predict with a free step, then correct against the ground.
        motor, rate = lie.explicit_rk4(motor, rate, inertia, inertia_inv, dt, forces)
        motor, rate = core.project_contacts(before, motor.normalized(), inertia_inv, parts, ground,
                                            STATIC_FRICTION, setup.dynamic_friction, setup.indentation, dt)
        if i % every == 0:
            yield motor, parts, ground
        if i % 1000 == 0:
            # The axis's angle from vertical.
            lean = ((motor >> UP).dual() | UP.dual()).abs().arccos()
            print(f"t={i * dt:5.1f} tilt={np.degrees(lean.to_array()):5.1f} "
                  f"spin={-(mv.xy | rate).to_array():6.1f} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    from examples.animation import save_animation
    from examples.mechanics.spinning_top import render

    for name, setup in VARIATIONS.items():
        frames = [render.frame(motor, parts, ground, VIEW, CENTRE, EXTENT, 260)
                  for motor, parts, ground in spin(setup, 5.0, 1e-3, 40)]
        save_animation(frames, f"spinning_top_{name}", 33)
