"""Modal extended position-based dynamics with the couplings between bodies and constraints found by
differentiation.

`core.py` builds by hand how each constraint's gap changes as its two bodies move and deform: each
anchor's commutator with the space of twists, each mode's shape at each anchor, signed for the two sides and
read off the bodies through the incidence of the constraint graph into sparse maps. Here the gaps are written
once, as a function of the coordinates of the step, how far each body moves and how far each mode deforms, and
the couplings are its derivative:
dense field maps from the bodies to the constraints, one for the twists and one for each mode's
amplitudes. The rest of the step is the core's, run on JAX: the same system for all constraints, solved
once per step, here a dense field map over the constraints.
"""

from dataclasses import dataclass, fields, replace
from collections.abc import Iterator

import jax
import numpy as np

from numga.algebras import PGA2D
from numga.backend.jax import JaxContext, derivative
from examples import instantiate
from examples.mechanics import lie_integrators as lie

core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D, JaxContext)
mv = core.mv
Bodies, Constraints, Displacement, Force, Forque, Motor, Scalar, Twist = (
    core.Bodies, core.Constraints, core.Displacement, core.Force, core.Forque, core.Motor, core.Scalar, core.Twist,
)
# A load on a mode is the complement of its amplitude, as a forque is the complement of a twist; a mode
# of unit mass takes it back to an amplitude.
Load = Scalar.dual().output_gatype                                          # Antiscalar-
# A gap is the second anchor's position less the first's.
SIDES = np.array([-1, 1])


@dataclass(frozen=True)
class Coordinates:
    """How far each body moves over the step, and how far each mode deforms."""
    twists: Twist                                                           # [...] Twist+[bodies]
    amplitudes: Scalar                                                      # [..., modes] Scalar+[bodies]


# Bodies are stepped inside jit, and coordinates differentiated, as the pytrees of their fields.
for record in (Bodies, Coordinates):
    jax.tree_util.register_dataclass(record, data_fields=[field.name for field in fields(record)], meta_fields=[])


# --- math -----------------------------------------------------------------------------
def gaps(motor: Motor, constraints: Constraints, coordinates: Coordinates) -> Displacement:
    """Each constraint's gap, the second anchor less the first: each anchor on its body, deformed by the
    body's modes and moved by its motor and its twist."""
    stepped_motor = motor * (coordinates.twists * -0.5).exp()                       # [...] Motor+[bodies]
    local = constraints.anchors + (constraints.modes * (constraints.ends * coordinates.amplitudes[..., None])).sum(axis=-2)   # [..., sides] Point-[constraints]
    anchors = (constraints.ends * stepped_motor[..., None]) >> local                # [..., sides] Point-[constraints]
    return (anchors * SIDES).sum(axis=-1).cast(Displacement)                   # [...] Displacement-[constraints]


def project(bodies: Bodies, constraints: Constraints, previous_amplitudes: Scalar, mode_reactions: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """The bodies displaced to satisfy all constraints, and the modes' reactions."""
    compliance, residual, response = core.modal_terms(bodies, previous_amplitudes, dt)   # [..., modes] Scalar+[bodies], Scalar+[bodies], Scalar+[bodies]

    def gap(coordinates: Coordinates) -> Displacement:
        """The gaps for the bodies as they are, moved and deformed by the coordinates."""
        return gaps(bodies.motor, constraints, coordinates)

    here = Coordinates(twists=bodies.rate * 0.0, amplitudes=bodies.amplitudes)
    # How the gaps change with the coordinates: as the bodies move, and as each mode deforms them.
    change = derivative(gap)(here)
    rigid = change.twists                                                   # [...] (Displacement- <- Twist+)[constraints <- bodies]
    modal = change.amplitudes                                               # [..., modes] (Displacement- <- Scalar+)[constraints <- bodies]
    # The load of each mode's spring, less the reactions it has already received this step, and how far
    # each mode moves under it alone.
    spring_load = -(residual + compliance * mode_reactions)                # [..., modes] Scalar+[bodies]
    unconstrained_step = response * spring_load                            # [..., modes] Scalar+[bodies]
    # How far each mode moves under a unit load from a constraint, its spring holding back the rest.
    mobility = response * compliance * Load.dual()                         # [..., modes] (Scalar+ <- Load-)[bodies]
    # Each constraint's own compliance over the step squared, as a map from force to gap at that constraint.
    constraint_compliance = (constraints.compliance / dt**2 * Force.dual()).on_diagonal()   # (Displacement- <- Force+)[constraints <- constraints]
    # The derivatives are maps from motion to the gaps, displacements, intrinsic; a force at a gap is extrinsic, its
    # work `force & displacement`, and each derivative's adjugate reads that pairing backwards, carrying the forces
    # to the bodies' forques and the modes' loads with no transpose and no metric.
    # The change in the gaps for forces at the constraints: the adjugates carry the forces back to the
    # bodies' forques and the modes' loads, the inverse inertias and mobilities turn those into motion,
    # and the derivatives carry the motion forward to the gaps.
    system = (rigid(bodies.inverse_inertia(rigid.adjugate()))
              + modal(mobility(modal.adjugate())).sum(axis=-1)
              + constraint_compliance)                                      # [...] (Displacement- <- Force+)[constraints <- constraints]
    # The reactions at the constraints that close every gap, given how far the modes move on their own.
    reactions = system.solve(-(gap(here) + modal(unconstrained_step).sum(axis=-1)))   # [...] Force+[constraints]
    # Each body's twist, from the forques the reactions exert on it, and each mode's load.
    displacement = bodies.inverse_inertia(rigid.adjugate()(reactions))     # [...] Twist+[bodies]
    mode_loads = Load.dual()(modal.adjugate()(reactions))                  # [..., modes] Scalar+[bodies]
    reaction_change = response * (spring_load - mode_loads)                # [..., modes] Scalar+[bodies]
    constrained_bodies = replace(
        bodies,
        motor=bodies.motor * (displacement * -0.5).exp(),
        amplitudes=bodies.amplitudes + mode_loads + reaction_change,
    )
    return constrained_bodies, mode_reactions + reaction_change


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Displacement) -> Bodies:
    """The bodies one time step later, as `core.step` steps them."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, the origin of its frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses            # [...] Forque-[bodies]

    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    _, residual, response = core.modal_terms(bodies, previous.amplitudes, dt)
    mode_reactions = -residual * response                                  # [..., modes] Scalar+[bodies]
    bodies, _ = project(replace(bodies, amplitudes=bodies.amplitudes + mode_reactions), constraints, previous.amplitudes, mode_reactions, dt)
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )


def swing(bodies: Bodies, constraints: Constraints, gravity: Displacement, dt: float, substeps: int, frames: int) -> Iterator[Bodies]:
    """The bodies at every frame, under gravity, `substeps` compiled steps per frame."""
    @jax.jit
    def advance(bodies: Bodies) -> Bodies:
        return step(bodies, constraints, dt, gravity)

    for _ in range(frames):
        yield bodies
        for _ in range(substeps):
            bodies = advance(bodies)


def main():
    from examples.util.animation import save_animation
    from . import render, scenarios

    shape = core.girder(scenarios.N_CELLS, scenarios.LENGTH, scenarios.HEIGHT, scenarios.STIFFNESS, scenarios.DENSITY, scenarios.N_MODES)
    bodies, constraints = scenarios.hinged_chain(core, shape, scenarios.DAMPING, scenarios.FLEXIBILITY)
    gravity = mv.direction([0, -scenarios.GRAVITY])                         # [] Displacement-
    points = jax.jit(lambda moment: core.points(moment, shape))
    frames = (points(moment) for moment in swing(bodies, constraints, gravity, scenarios.INTERVAL, scenarios.N_STEPS_PER_FRAME, scenarios.N_FRAMES))
    save_animation(render.swinging_chain(frames, shape.edges, core.points(bodies, shape)), "modal_xpbd_derived_swing", scenarios.DURATION_MS)


if __name__ == "__main__":
    main()
