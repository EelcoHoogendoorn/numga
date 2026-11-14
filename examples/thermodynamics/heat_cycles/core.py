"""A Carnot engine: an ideal gas taken around a closed cycle between a hot and a cold reservoir converts part of the
heat absorbed from the hot reservoir into mechanical work.

A gas state is modelled as an antivector in the exterior algebra `s0 v0` of entropy and volume, `entropies * entropy +
volumes * volume`, and the gas's internal energy is a scalar function of its state. At each state, temperature and
pressure are extrinsic potentials along `s` and `v`, so the heat absorbed along a state tangent is `(temperature * s) &
tangent` and the work done is `(pressure * v) & tangent`. Their difference equals the differential of the internal
energy, whereas neither one-form alone is the gradient of a state function, so heat and work depend on the path while
their difference depends only on the endpoints. Wedging each potential's state gradient with its direction gives the two
cycle-area two-forms, `temperature_slope ^ s` and `pressure_slope ^ v`. Because both potentials come from one internal
energy, their mixed slopes cancel and the two area forms are identical, so around any closed cycle the area enclosed on
the temperature-entropy chart equals the area enclosed on the pressure-volume chart.

Entropy is measured in units of nR, volume and temperature in reference values, and energy in units of nRT_0; work is
positive when the gas expands.
"""
from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext
from numga.topology import Topology

ga = Algebra("s0v0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                 # Scalar+
State = ga.gatype.antivector()                              # Antivector-
Potential = ga.gatype.vector()                              # Vector+
AreaForm = ga.gatype.bivector()                             # Bivector+

entropy = mv.s.dual()                                       # [] State-
volume = mv.v.dual()                                        # [] State-
unit_patch = entropy & volume                               # [] Antibivector-


# --- math -----------------------------------------------------------------------------
def energy(state: State, capacity: float) -> Scalar:
    """Internal energy of an ideal gas with constant heat capacity at fixed volume."""
    return capacity * (((mv.s & state) - (mv.v & state).log()) / capacity).exp()


def temperature_of(state: State, capacity: float) -> Scalar:
    """Absolute temperature of the ideal gas at each state."""
    return energy(state, capacity) / capacity


def pressure_of(state: State, capacity: float) -> Scalar:
    """Pressure of the ideal gas at each state."""
    return temperature_of(state, capacity) / (mv.v & state)


def equilibrium_states(entropies: Scalar, temperatures: Scalar, capacity: float) -> State:
    """Equilibrium states of the ideal gas at the given entropies and temperatures."""
    return entropy * entropies + volume * (entropies - capacity * temperatures.log()).exp()


def stroke_samples(fractions: Scalar, around: Topology, corner_entropy: Scalar, corner_temperature: Scalar,
                   capacity: float, n_steps: int) -> tuple[State, State]:
    """States and stroke tangents at the given step fractions around the closed loop of corners."""
    V, E = around.n_elements
    by_stroke = around.reading(np.repeat(np.arange(E), n_steps), 1)                       # Scalar+[E * n_steps <- E]
    entropy_starts = by_stroke * (around.incidence[1, 0][0] * corner_entropy)             # Scalar+[E * n_steps]
    entropy_steps = by_stroke * (around.d[0] * corner_entropy)                            # Scalar+[E * n_steps]
    temperature_starts = by_stroke * (around.incidence[1, 0][0] * corner_temperature)     # Scalar+[E * n_steps]
    temperature_steps = by_stroke * (around.d[0] * corner_temperature)                    # Scalar+[E * n_steps]
    entropies = entropy_starts + entropy_steps * fractions                                # Scalar+[E * n_steps]
    temperatures = temperature_starts + temperature_steps * fractions                     # Scalar+[E * n_steps]
    states = equilibrium_states(entropies, temperatures, capacity)                        # State-[E * n_steps]
    # Differentiating V = exp(S - c ln T) along the linear (S, T) stroke gives dV = V * (dS - c dT / T).
    volumes = mv.v & states                                                               # Scalar+[E * n_steps]
    tangents = entropy * entropy_steps + volume * volumes * (entropy_steps - capacity * temperature_steps / temperatures)
    return states, tangents


def exchanges(states: State, tangents: State, capacity: float) -> tuple[Scalar, Scalar]:
    """Heat absorbed and work done along each state tangent."""
    heat_form = temperature_of(states, capacity) * mv.s                                   # [...] Potential+
    work_form = pressure_of(states, capacity) * mv.v                                      # [...] Potential+
    return heat_form & tangents, work_form & tangents


def potential_slopes(state: State, capacity: float) -> tuple[Potential, Potential]:
    """State gradients of temperature and pressure across entropy and volume."""
    temperatures = temperature_of(state, capacity)                                        # [...] Scalar+
    volumes = mv.v & state                                                                # [...] Scalar+
    pressures = temperatures / volumes                                                    # [...] Scalar+
    temperature_slope = (temperatures / capacity) * (mv.s - mv.v / volumes)               # [...] Potential+
    pressure_slope = (pressures / capacity) * mv.s - (pressures * (1 + 1 / capacity) / volumes) * mv.v  # [...] Potential+
    return temperature_slope, pressure_slope


def circulations(state: State, capacity: float) -> tuple[AreaForm, AreaForm]:
    """Heat and work two-forms circulated per oriented patch of state space."""
    temperature_slope, pressure_slope = potential_slopes(state, capacity)                 # [...] Potential+, Potential+
    return temperature_slope ^ mv.s, pressure_slope ^ mv.v
