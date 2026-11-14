"""A hollow ellipsoid and its rounded stem form one rigid top with two contact surfaces.

An offset, turned cavity rotates the mass distribution relative to the outer shell, so sliding friction at the shell and the tip couples spinning to rocking differently for opposite spins. The same carved quadrics supply both the visible surfaces and the mass samples.
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import partial

import jax
import numpy as np

from numga import stack
from numga.algebras import PGA3D as ga
from numga.backend.jax import JaxContext
from examples.mechanics import lie_integrators as lie
from examples.mechanics.spinning_top import core as contact

jax.config.update("jax_enable_x64", True)
mv = JaxContext(ga, dtype=np.float64).multivector
Point = ga.gatype.antivector()                                               # Antivector-
Plane = ga.gatype.vector()                                                   # Vector+
Rate = ga.gatype.bivector()                                                  # Bivector+
Forque = ga.gatype.antibivector()                                            # Antibivector-
Motor = ga.gatype.rotor()                                                    # Rotor+
Quadric = ga.gatype((Plane, Point))                                          # Plane+ <- Point-
Inertia = ga.gatype((Forque, Rate))                                          # Forque- <- Rate+
origin = mv.w.dual()                                                         # [] Point-
up = mv.z.dual()                                                             # [] Point-
axes = stack((mv.x, mv.y, mv.z))                                             # [3] Plane+
ground = (mv.z * (mv.w & Point) + mv.w * (mv.z & Point)) / 2                 # [] Plane+ <- Point-


def ellipsoid(centre: Point, frame: Plane, semiaxes: np.ndarray) -> Quadric:
    """Planes through `centre` along `frame` (`[3] Plane+`) scaled by `semiaxes` form the ellipsoid's surface quadric (`[] Plane+ <- Point-`)."""
    axial_planes = frame - mv.w * (frame & centre)                           # [3] Plane+
    return (axial_planes * (axial_planes & Point) / semiaxes**2).sum(axis=-1) - mv.w * (mv.w & Point)


@jax.jit
def mass_distribution(parts: Quadric, samples: Point, sample_volume: float) -> tuple[Point, Inertia]:
    """Uniform material inside the carved shell, stem, and tip (`[4] Plane+ <- Point-`) determines the mass point (`[] Point-`) and inertia map (`[] Forque- <- Rate+`)."""
    shell_mask, cavity_mask, stem_mask, tip_mask = (parts[:, None](samples) & samples) <= 0
    material_mask = (shell_mask & ~cavity_mask) | stem_mask | tip_mask       # [...]
    masses = material_mask * sample_volume                                   # [...]
    mass = (samples * masses).sum(axis=-1)                                   # [] Point-
    # Each mass sample joins its velocity under a rate to its position, contributing to the body's momentum.
    inertia = (masses * samples).regressive(samples.commutator(Rate)).sum(axis=-1)  # [] Forque- <- Rate+
    return mass, inertia


def release_states(contacts: Quadric, mass: Point, spins: np.ndarray, lean: float) -> tuple[Motor, Rate]:
    """Tilts the body by `lean` without table penetration and assigns `spins` (`[...]`) about the vertical line through `mass`."""
    tilt = (mv.xz * (lean / 2)).exp()                                        # [] Motor+
    body_up = tilt << up                                                     # [] Point-
    supports = contact.lowest_point(contacts, body_up)                       # [2] Point-
    placed_supports = tilt >> supports                                       # [2] Point-
    height = (mv.z & placed_supports).min(axis=0)                            # [] Scalar+
    lift = (-up * origin.inverse() * height / 2).exp()                       # [] Motor+
    motor = (lift * tilt).broadcast_to(spins.shape)                          # [...] Motor+
    centre = mass.normalized()                                               # [] Point-
    spin_axis = (mv.x - mv.w * (mv.x & centre)) ^ (mv.y - mv.w * (mv.y & centre))  # [] Rate+
    return motor.retyped(Motor), (spin_axis * spins).retyped(Rate)


@partial(jax.jit, static_argnames=("n_steps",))
def frame_step(contacts: Quadric, mass: Point, inertia: Inertia, motor: Motor, rate: Rate,
               gravity: float, static_friction: float, dynamic_friction: float,
               dt: float, n_steps: int) -> tuple[Motor, Rate]:
    """Advances `motor` (`[...] Motor+`) and `rate` (`[...] Rate+`) across `n_steps` free-motion and contact substeps."""
    inverse_inertia = inertia.inverse()                                      # [] Rate+ <- Forque-

    def gravity_forque(placement: Motor, velocity: Rate) -> Forque:
        gravity_force = placement << (-up * gravity)                         # [...] Point-
        return mass & gravity_force

    def substep(step_index: int, state: tuple[Motor, Rate]) -> tuple[Motor, Rate]:
        prior_motor, velocity = state
        predicted_motor, _ = lie.explicit_rk4(prior_motor, velocity, inertia, inverse_inertia, dt, gravity_forque)
        next_motor, next_rate = contact.project_contacts(
            prior_motor, predicted_motor.normalized(), inverse_inertia,
            contacts, ground, static_friction, dynamic_friction, dt,
        )
        return next_motor.retyped(Motor), next_rate.retyped(Rate)

    return jax.lax.fori_loop(0, n_steps, substep, (motor, rate))


def motion(contacts: Quadric, mass: Point, inertia: Inertia, motor: Motor, rate: Rate,
           gravity: float, static_friction: float, dynamic_friction: float,
           dt: float, n_states: int, n_steps_per_frame: int) -> Iterator[tuple[Motor, Rate]]:
    """Yields `n_states` pairs of placement (`[...] Motor+`) and rate (`[...] Rate+`) separated by `n_steps_per_frame` substeps."""
    yield motor, rate
    for _ in range(n_states - 1):
        motor, rate = frame_step(contacts, mass, inertia, motor, rate, gravity,
                                 static_friction, dynamic_friction, dt, n_steps_per_frame)
        yield motor, rate
