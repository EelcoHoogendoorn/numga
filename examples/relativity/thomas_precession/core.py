"""Thomas precession and the Bargmann–Michel–Telegdi equation in the spacetime algebra.

A spinning relativistic particle carries a unit timelike four-velocity `u`, with `u | u == 1`, and a
unit spacelike four-spin `s` orthogonal to it, `s | u == 0` and `s | s == -1`. In the particle's own
instantaneous rest frame its spin is purely spatial; an observer with four-velocity `TIME = mv.t`
reads out that rest-frame spin by boosting `u` back to `TIME` with the pure boost rotor
`(TIME * u).square_root() >> s`.

When the particle accelerates without any torque in its instantaneous rest frame, each small step
from `u_tail` to `u_head` boosts its frame in the plane `u_head ^ u_tail` by
`(u_head * u_tail).square_root()`, discrete Fermi–Walker transport. Because two boosts in different
directions do not commute, chaining those torque-free boosts around a closed circular orbit at
Lorentz factor `gamma = cosh(rapidity)` returns the four-velocity to where it started but leaves a
spatial turn in the orbital plane by `2 * np.pi * (gamma - 1)`, the hyperbolic area of the cap the
four-velocity sweeps out on the unit hyperboloid: Thomas precession.

In an electromagnetic field bivector `field`, with charge-to-mass ratio one, the particle's
four-acceleration is `acceleration = field | u`. For a gyromagnetic ratio `g`, the single spacetime
bivector

    omega = (g / 2) * field - (g / 2 - 1) * (acceleration ^ u)

governs both the orbit and the spin. Because `u | u == 1` and `acceleration | u == 0`, the wedge
`(acceleration ^ u) | u` equals `acceleration`, so `omega | u == field | u` for every value of `g`.
On any four-spin `s` orthogonal to `u`, `(acceleration ^ u) | s` equals `-(acceleration | s) * u`,
so `omega | Vector` (`Vector <- Vector`) is the Bargmann–Michel–Telegdi rate map. Stepping one
Lorentz rotor `frame` by `(omega * (dt / 2)).exp() * frame` therefore integrates both the worldline
`u = frame >> TIME` and the spin `s = frame >> rest_spin` at once, keeping `u | u == 1`,
`s | s == -1` and `u | s == 0`. For a Dirac particle with `g == 2` the acceleration term drops out
and `omega == field`, so in a magnetic storage ring the spin turns at the cyclotron rate alongside
the velocity; an anomalous moment `g != 2` makes the spin precess relative to the velocity at a rate
proportional to `(g - 2) / 2`.

In the four-tensor index notation of relativistic electrodynamics, the electromagnetic bivector
reads as the antisymmetric field tensor $F^{\\mu\\nu}$ and the Bargmann–Michel–Telegdi equation as
$\\frac{ds^\\mu}{d\\tau} = \\frac{q}{m}\\left(\\frac{g}{2} F^{\\mu}{}_{\\nu} s^\\nu + \\left(\\frac{g}{2} - 1\\right) u^\\mu F_{\\nu\\lambda} u^\\nu s^\\lambda\\right)$.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext, stack
from numga.algebras import STA as ga

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Rotor = ga.gatype.rotor()
RateMap = ga.gatype((Vector, Vector))                                           # Vector <- Vector

TIME = mv.t.normalized()                                                        # [] Vector


# --- math -----------------------------------------------------------------------------
def rest_boost(four_velocity: Vector) -> Rotor:
    """The pure boost rotor carrying `TIME` to `four_velocity`."""
    return (four_velocity * TIME).square_root()                                 # [...] Rotor


def lab_spin(four_velocity: Vector, four_spin: Vector) -> Vector:
    """The particle's spin carried back to the laboratory observer's rest frame by the inverse of
    its pure boost from `TIME`."""
    return (TIME * four_velocity).square_root() >> four_spin                    # [...] Vector


def fermi_walker_step(u_tail: Vector, u_head: Vector) -> Rotor:
    """The pure boost from `u_tail` to `u_head` in their spacetime plane: discrete Fermi–Walker
    transport."""
    return (u_head * u_tail).square_root()                                      # [...] Rotor


def fermi_walker_loop(four_velocities: Vector, start_spin: Vector) -> tuple[Vector, Vector]:
    """Carry `start_spin` along a sequence of four-velocities `[steps + 1, ...] Vector` by discrete
    Fermi–Walker boosts: returns the four-spin and the laboratory rest-frame spin at each step."""
    turns = fermi_walker_step(four_velocities[:-1], four_velocities[1:])        # [steps, ...] Rotor
    spins = [start_spin]
    for turn in turns:
        spins.append(turn >> spins[-1])                                         # [...] Vector
    four_spins = stack(spins)                                                   # [steps + 1, ...] Vector
    return four_spins, lab_spin(four_velocities, four_spins)


def bmt_generator(field: Bivector, four_velocity: Vector, g_factor: np.ndarray) -> Bivector:
    """The spacetime bivector generating both the Lorentz-force acceleration of `four_velocity` and
    the Bargmann–Michel–Telegdi precession of any four-spin orthogonal to it."""
    acceleration = field | four_velocity                                        # [...] Vector
    half_g = g_factor / 2.0
    return half_g * field - (half_g - 1.0) * (acceleration ^ four_velocity)     # [...] Bivector


def bmt_map(field: Bivector, four_velocity: Vector, g_factor: np.ndarray) -> RateMap:
    """The linear rate map `Vector <- Vector` on four-velocity and four-spin."""
    return bmt_generator(field, four_velocity, g_factor) | Vector               # [...] Vector <- Vector


def evolve_bmt(
    field: Bivector,
    start_frame: Rotor,
    rest_spin: Vector,
    g_factor: np.ndarray,
    dt: float,
    steps: int,
) -> tuple[Vector, Vector, Vector]:
    """Integrate the Lorentz frame rotor over `steps` proper-time intervals of length `dt` by the
    midpoint rule: returns the four-velocities, four-spins and laboratory spins `[steps + 1, ...] Vector`."""
    frames = [start_frame]
    for _ in range(steps):
        current = frames[-1]                                                    # [...] Rotor
        omega_start = bmt_generator(field, current >> TIME, g_factor)           # [...] Bivector
        halfway = (omega_start * (dt / 4.0)).exp() * current                    # [...] Rotor
        omega_mid = bmt_generator(field, halfway >> TIME, g_factor)             # [...] Bivector
        frames.append((omega_mid * (dt / 2.0)).exp() * current)                 # [...] Rotor
    history = stack(frames)                                                     # [steps + 1, ...] Rotor
    four_velocities = history >> TIME                                           # [steps + 1, ...] Vector
    four_spins = history >> rest_spin                                           # [steps + 1, ...] Vector
    return four_velocities, four_spins, lab_spin(four_velocities, four_spins)


# --- plumbing -------------------------------------------------------------------------
def circular_orbit_velocities(rapidities: np.ndarray, steps: int) -> tuple[Vector, Vector, Vector]:
    """Spatial unit-circle positions `[steps + 1, 1] Vector`, four-velocities `[steps + 1, orbits] Vector`
    along circular orbits in the xy plane at `rapidities` `[orbits]`, and the initial four-spin along +x."""
    phases = np.linspace(0.0, 2 * np.pi, steps + 1)
    # In STA (+---), mv.x and mv.y square to -1, so `(mv.xy * (phases / 2)).exp()` turns x toward y.
    turn = (mv.xy * (phases[:, None] / 2.0)).exp()                              # [steps + 1, 1] Rotor
    positions = turn >> mv.x                                                    # [steps + 1, 1] Vector
    tangent = turn >> mv.y                                                      # [steps + 1, 1] Vector
    boost = ((tangent ^ TIME) * (rapidities / 2.0)).exp()                       # [steps + 1, orbits] Rotor
    four_velocities = boost >> TIME                                             # [steps + 1, orbits] Vector
    start_spins = (rapidities * 0.0 + 1.0) * mv.x                               # [orbits] Vector
    return positions, four_velocities, start_spins
