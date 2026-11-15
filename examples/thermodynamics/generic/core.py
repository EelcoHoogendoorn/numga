"""A mass on a spring warms a thermal reservoir through viscous damping in the GENERIC formulation.

The algebra `q0 p0 s0` gives one degenerate direction each to the position `q`, the momentum `p` and the reservoir
entropy `s`. A state is modelled as an antivector `position * q.dual() + momentum * p.dual() + entropy * s.dual()`,
and a potential is a vector that pairs with changes of state. The energy potential combines the spring restoring force
along `q`, the velocity along `p` and the reservoir temperature along `s`, while the entropic potential is `s`. Two
maps from potentials to rates of state change generate the motion. The skew Poisson map sends the energy potential to
reversible oscillation and maps `s` to zero. The friction mobility is a positive dyad along the step that converts lost
mechanical momentum into reservoir heat, sending `s` to viscous damping while mapping the energy potential to zero.
Together they conserve total energy and produce nonnegative entropy until the mass comes to rest.
"""

from __future__ import annotations

from numga import Algebra, NumpyContext

ga = Algebra("q0p0s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                               # Scalar+
State = ga.gatype.antivector()                            # Antivector-
Potential = ga.gatype.vector()                            # Vector+
Map = ga.gatype((State, Potential))                       # State- <- Potential+

# Canonical Poisson map sending momentum slopes to position rates and position slopes to minus momentum rates.
poisson_map = (
    mv.q.dual() * (Potential & mv.p.dual()) - mv.p.dual() * (Potential & mv.q.dual())
)                                                         # [] State- <- Potential+
# Entropic potential pairing with a change of state to read the change in reservoir entropy.
entropy_potential = mv.s                                  # [] Potential+


# --- math -----------------------------------------------------------------------------
def temperature(state: State, thermal_capacity: float, reference_temperature: float) -> Scalar:
    """Absolute temperature of the thermal reservoir from its entropy and heat capacity."""
    return reference_temperature * ((mv.s & state) / thermal_capacity).exp()


def kinetic_energy(state: State, mass: float) -> Scalar:
    """Kinetic energy of the oscillating mass."""
    return (mv.p & state).squared() / (2 * mass)


def potential_energy(state: State, stiffness: float) -> Scalar:
    """Elastic energy stored in the spring."""
    return stiffness * (mv.q & state).squared() / 2


def internal_energy(state: State, thermal_capacity: float, reference_temperature: float) -> Scalar:
    """Thermal internal energy of the reservoir."""
    return thermal_capacity * temperature(state, thermal_capacity, reference_temperature)


def total_energy(
    state: State, mass: float, stiffness: float, thermal_capacity: float, reference_temperature: float
) -> Scalar:
    """Sum of kinetic, elastic and thermal internal energy."""
    return (
        kinetic_energy(state, mass)
        + potential_energy(state, stiffness)
        + internal_energy(state, thermal_capacity, reference_temperature)
    )


def energy_potential(
    state: State, mass: float, stiffness: float, thermal_capacity: float, reference_temperature: float
) -> Potential:
    """Thermodynamic potential combining spring force along `q`, velocity along `p` and temperature along `s`."""
    return (
        mv.q * (stiffness * (mv.q & state))
        + mv.p * ((mv.p & state) / mass)
        + mv.s * temperature(state, thermal_capacity, reference_temperature)
    )


def friction_mobility(
    state: State, mass: float, friction: float, thermal_capacity: float, reference_temperature: float
) -> Map:
    """Positive dyad mapping potentials along the energy-conserving step from momentum into reservoir entropy."""
    reservoir_temperature = temperature(state, thermal_capacity, reference_temperature)    # [...] Scalar+
    # Losing a unit of momentum at velocity v dissipates heat v into the reservoir, raising its entropy by v / T.
    dissipation_step = -mv.p.dual() + mv.s.dual() * ((mv.p & state) / (mass * reservoir_temperature))   # [...] State-
    return dissipation_step * (Potential & dissipation_step) * (friction * reservoir_temperature)


def rates(
    state: State, mass: float, stiffness: float, friction: float, thermal_capacity: float, reference_temperature: float
) -> tuple[State, State]:
    """Reversible rate from the Poisson map on the energy potential and irreversible rate from friction mobility on `s`."""
    reversible = poisson_map(energy_potential(state, mass, stiffness, thermal_capacity, reference_temperature))
    irreversible = friction_mobility(state, mass, friction, thermal_capacity, reference_temperature)(entropy_potential)
    return reversible, irreversible


def advance(
    state: State,
    mass: float,
    stiffness: float,
    friction: float,
    thermal_capacity: float,
    reference_temperature: float,
    dt: float,
) -> State:
    """State after one fourth-order Runge-Kutta step of the combined reversible and irreversible rates."""

    def rate(current: State) -> State:
        reversible, irreversible = rates(current, mass, stiffness, friction, thermal_capacity, reference_temperature)
        return reversible + irreversible

    k1 = rate(state)
    k2 = rate(state + k1 * (dt / 2))
    k3 = rate(state + k2 * (dt / 2))
    k4 = rate(state + k3 * dt)
    return state + (k1 + 2 * k2 + 2 * k3 + k4) * (dt / 6)
