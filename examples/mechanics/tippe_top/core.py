"""A hollow ellipsoid and its rounded stem form one rigid top with two contact surfaces.

The cavity turns the mass distribution relative to the outer shell. Sliding friction at the shell or the tip couples spinning to rocking; reversing the spin need not produce the same motion. The same Boolean solid supplies the visible surfaces and the mass samples.
"""
from __future__ import annotations

from collections.abc import Iterator
from functools import partial
from typing import NamedTuple

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
Plane = ga.gatype.vector()                                                  # Vector+
Rate = ga.gatype.bivector()                                                 # Bivector+
Forque = ga.gatype.antibivector()                                           # Antibivector-
Motor = ga.gatype.rotor()                                                   # Rotor+
Scalar = ga.gatype.scalar()                                                 # Scalar+
Quadric = ga.gatype((Plane, Point))
Inertia = ga.gatype((Forque, Rate))
InverseInertia = ga.gatype((Rate, Forque))
origin = mv.w.dual()                                                        # [] Point-
up = mv.z.dual()                                                            # [] Point-
axes = stack((mv.x, mv.y, mv.z))                                           # [axes] Plane+
ground = (mv.z * (mv.w & Point) + mv.w * (mv.z & Point)) / 2                # [] Plane+ <- Point-


class Body(NamedTuple):
    shell: Quadric
    cavity: Quadric
    stem: Quadric
    tip: Quadric
    mass: Point
    inertia: Inertia
    inverse_inertia: InverseInertia

    @property
    def contacts(self) -> Quadric:
        """The two surfaces that meet the table: the shell and the tip."""
        return stack((self.shell, self.tip))


# --- math -----------------------------------------------------------------------------
def ellipsoid(centre: Point, frame: Plane, semiaxes: np.ndarray) -> Quadric:
    """The ellipsoid about a point with its axes square to a frame's planes `[axes]`, as the quadric of its surface."""
    through = frame - mv.w * (frame & centre)                                # [axes] Plane+
    return (through * (through & Point) / semiaxes**2).sum(axis=-1) - mv.w * (mv.w & Point)


@jax.jit
def build(shell: Quadric, cavity: Quadric, stem: Quadric, tip: Quadric,
          samples: Point, sample_volume: float) -> Body:
    """The shell carved by the cavity, joined to the stem and the tip: one solid of uniform density, its pieces
    counted once where they overlap."""
    parts = stack((shell, cavity, stem, tip))                                # [parts] Plane+ <- Point-
    in_shell, in_cavity, in_stem, in_tip = (parts[:, None](samples) & samples) <= 0
    material = (in_shell & ~in_cavity) | in_stem | in_tip                    # [samples]
    masses = material * sample_volume
    mass = (samples * masses).sum(axis=-1)                                  # [] Point-
    # Each mass sample joins its velocity to its position, contributing to the body's momentum.
    inertia = samples.regressive(samples.commutator(Rate) * masses).sum(axis=-1)
    return Body(shell, cavity, stem, tip, mass, inertia, inertia.inverse())


def launch(body: Body, spins: np.ndarray, lean: float) -> tuple[Motor, Rate]:
    """Opposite spins about the mass centre, with the same lean and no initial penetration."""
    tilt = (mv.xz * (lean / 2)).exp()                                        # [] Motor+
    normal = tilt << up                                                      # [] Point-
    supports = contact.lowest_point(body.contacts, normal)                   # [contacts] Point-
    height = (mv.z & (tilt >> supports)).min(axis=0)                         # [] Scalar+
    lift = (-up * origin.inverse() * height / 2).exp()                       # [] Motor+
    motor = (lift * tilt).broadcast_to(spins.shape)
    centre = body.mass.normalized()                                         # [] Point-
    spin_axis = (mv.x - mv.w * (mv.x & centre)) ^ (mv.y - mv.w * (mv.y & centre))
    return motor.retyped(Motor), (spin_axis * spins).cast(Rate).retyped(Rate)


@partial(jax.jit, static_argnames=("n_steps",))
def advance(body: Body, motor: Motor, rate: Rate, gravity: float,
            static: float, dynamic: float, dt: float, n_steps: int) -> tuple[Motor, Rate]:
    """Free rigid motion and XPBD contact corrections advance the top to the next frame."""

    def load(placement: Motor, velocity: Rate) -> Forque:
        return body.mass & (placement << (-up * gravity))

    def substep(index: int, state: tuple[Motor, Rate]) -> tuple[Motor, Rate]:
        before, velocity = state
        predicted, _ = lie.explicit_rk4(before, velocity, body.inertia, body.inverse_inertia, dt, load)
        placement, velocity = contact.project_contacts(before, predicted.normalized(), body.inverse_inertia,
                                                       body.contacts, ground, static, dynamic, dt)
        return placement.retyped(Motor), velocity.retyped(Rate)

    return jax.lax.fori_loop(0, n_steps, substep, (motor, rate))


# --- plumbing -------------------------------------------------------------------------
def evolve(body: Body, motor: Motor, rate: Rate, gravity: float,
           static: float, dynamic: float, dt: float, n_frames: int,
           stride: int) -> Iterator[tuple[Motor, Rate]]:
    """Yield each launch's placement and rate as the experiment advances."""
    yield motor, rate
    for _ in range(n_frames - 1):
        motor, rate = advance(body, motor, rate, gravity, static, dynamic, dt, stride)
        yield motor, rate
