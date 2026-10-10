"""Modal extended position-based dynamics with the couplings between bodies and constraints found by
differentiation.

`core.py` builds by hand how each constraint's gap changes as its two bodies move and deform: each
anchor's commutator with an open twist, each mode's shape at each anchor, signed for the two sides and
gathered by index into sparse maps. Here the gaps are written once, as a function of how far each body
moves, and of how far each mode deforms; the maps are their derivatives, dense field maps from the
bodies to the constraints. The rest of the step is the core's: the same system for all constraints,
solved once per step, here a dense field map over the constraints.
"""

from dataclasses import fields, replace

import jax
import jax.numpy as jnp
import numpy as np

from numga.backend.jax import JaxContext, derivative
from examples.mechanics import lie_integrators as lie
from examples.mesh import at_sites
from . import core
from .core import Bodies, Constraints, Direction, Force, Forque, Motor, Point, Scalar, Twist

jax.config.update("jax_enable_x64", True)
ctx = JaxContext(core.ga, np.float64)
mv = ctx.multivector
# A load on a mode is the complement of its amplitude, as a forque is the complement of a twist; a mode
# of unit mass takes it back to an amplitude.
Load = core.ga.gatype.pseudoscalar()
# A gap is the second anchor's position less the first's.
SIDES = np.array([-1, 1])
# Bodies are stepped inside jit, as the pytree of their fields.
jax.tree_util.register_dataclass(Bodies, data_fields=[field.name for field in fields(Bodies)], meta_fields=[])


# --- math -----------------------------------------------------------------------------
def anchors(motor: Motor, amplitudes: Scalar, constraints: Constraints) -> Point:
    """Each constraint's two anchors in the world: on each side, its body's anchor deformed by the modes
    and moved by the motor."""
    local = constraints.anchors + (constraints.modes * at_sites(amplitudes, constraints.body_idx)).sum(axis=-2)   # [..., sides] Point[constraints]
    return at_sites(motor, constraints.body_idx) >> local                  # [..., sides] Point[constraints]


def project(bodies: Bodies, constraints: Constraints, previous_amplitudes: Scalar, mode_reactions: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """The bodies displaced to satisfy all constraints, and the modes' reactions."""
    compliance, residual, response = core.modal_terms(bodies, previous_amplitudes, dt)   # [..., modes] Scalar[bodies] each

    def gaps(twists: Twist) -> Direction:
        """Each constraint's gap, with each body moved by a twist."""
        moved = bodies.motor * (twists * -0.5).exp()                        # [...] Motor[bodies]
        return (anchors(moved, bodies.amplitudes, constraints) * SIDES).sum(axis=-1).cast(Direction)   # [...] Direction[constraints]

    def deformations(amplitudes: Scalar) -> Direction:
        """Each constraint's gap from each mode alone."""
        motor = at_sites(bodies.motor, constraints.body_idx)[..., None, :]  # [..., 1, sides] Motor[constraints]
        moved = motor >> (constraints.modes * at_sites(amplitudes, constraints.body_idx))   # [..., modes, sides] Direction[constraints]
        return (moved * SIDES).sum(axis=-1).cast(Direction)                 # [..., modes] Direction[constraints]

    still = bodies.rate * 0.0                                               # [...] Twist[bodies]
    gap = gaps(still)                                                       # [...] Direction[constraints]
    # How the gaps change as the bodies move, and as each mode deforms them:
    rigid = derivative(gaps)(still)                                         # [...] Direction[constraints] <- Twist[bodies]
    modal = derivative(deformations)(bodies.amplitudes)                     # [..., modes] Direction[constraints] <- Scalar[bodies]
    # The load of each mode's spring, less the reactions it has already received this step, and how far
    # each mode moves under it alone.
    spring_load = -(residual + compliance * mode_reactions)                # [..., modes] Scalar[bodies]
    unconstrained_step = response * spring_load                            # [..., modes] Scalar[bodies]
    # How far each mode moves under a unit load from a constraint, its spring holding back the rest.
    mobility = response * compliance * Load.dual()                         # [..., modes] (Scalar <- Load)[bodies]
    # Each constraint's own compliance over the step squared, as a map from force to gap at that constraint.
    constraint_compliance = (constraints.compliance / dt**2 * Force.dual()).on_diagonal()   # Direction[constraints] <- Force[constraints]
    # The change in the gaps for forces at the constraints: the adjugates carry the forces back to the
    # bodies' forques and the modes' loads, the inverse inertias and mobilities turn those into motion,
    # and the derivatives carry the motion forward to the gaps.
    system = (rigid(bodies.inverse_inertia(rigid.adjugate()))
              + modal(mobility(modal.adjugate())).sum(axis=-1)
              + constraint_compliance)                                      # [...] Direction[constraints] <- Force[constraints]
    # The reactions at the constraints that close every gap, given how far the modes move on their own.
    reactions = system.solve(-(gap + modal(unconstrained_step).sum(axis=-1)))   # [...] Force[constraints]
    # Each body's twist, from the forques the reactions exert on it, and each mode's load.
    displacement = bodies.inverse_inertia(rigid.adjugate()(reactions))     # [...] Twist[bodies]
    mode_loads = Load.dual()(modal.adjugate()(reactions))                  # [..., modes] Scalar[bodies]
    reaction_change = response * (spring_load - mode_loads)                # [..., modes] Scalar[bodies]
    moved = replace(
        bodies,
        motor=bodies.motor * (displacement * -0.5).exp(),
        amplitudes=bodies.amplitudes + mode_loads + reaction_change,
    )
    return moved, mode_reactions + reaction_change


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Direction) -> Bodies:
    """One time step, as `core.step`."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, the origin of its frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses            # [...] Forque[bodies]

    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    _, residual, response = core.modal_terms(bodies, previous.amplitudes, dt)
    mode_reactions = -residual * response                                  # [..., modes] Scalar[bodies]
    bodies, _ = project(replace(bodies, amplitudes=bodies.amplitudes + mode_reactions), constraints, previous.amplitudes, mode_reactions, dt)
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )


def swing(bodies: Bodies, constraints: Constraints, gravity: Direction, dt: float, frames: int, substeps: int):
    """The bodies at every frame, under gravity, each frame's substeps one compiled call."""
    @jax.jit
    def advance(bodies: Bodies) -> Bodies:
        return jax.lax.fori_loop(0, substeps, lambda _, bodies: step(bodies, constraints, dt, gravity), bodies)

    for _ in range(frames):
        yield bodies
        bodies = advance(bodies)


# --- plumbing -------------------------------------------------------------------------
def on_jax(bodies: Bodies, constraints: Constraints) -> tuple[Bodies, Constraints]:
    """Bodies and constraints with their extensors on JAX, the rates over every blade of a twist."""
    lift = lambda value: ctx.extensor(value.gatype, jnp.asarray(value.kernel))
    bodies = replace(bodies, rate=bodies.rate.cast(Twist))
    return (
        Bodies(**{field.name: lift(getattr(bodies, field.name)) for field in fields(Bodies)}),
        replace(constraints, anchors=lift(constraints.anchors), modes=lift(constraints.modes), compliance=lift(constraints.compliance)),
    )


def on_numpy(bodies: Bodies) -> Bodies:
    """Bodies with their extensors back on NumPy, for drawing."""
    return Bodies(**{field.name: core.ctx.extensor(getattr(bodies, field.name).gatype, np.asarray(getattr(bodies, field.name).kernel))
                     for field in fields(Bodies)})


def main():
    from examples.animation import save_animation
    from . import render, scenarios

    shape = core.girder(scenarios.CELLS, scenarios.LENGTH, scenarios.HEIGHT, scenarios.STIFFNESS, scenarios.DENSITY, scenarios.MODES)
    bodies, constraints = on_jax(*scenarios.hinged_chain(shape, scenarios.LINKS, scenarios.DAMPING))
    gravity = ctx.extensor(scenarios.GRAVITY.gatype, jnp.asarray(scenarios.GRAVITY.kernel))
    frames = (core.points(on_numpy(moment), shape) for moment in swing(bodies, constraints, gravity, scenarios.INTERVAL, scenarios.FRAMES, scenarios.SUBSTEPS))
    reference = core.points(on_numpy(bodies), shape)
    save_animation(render.swinging_chain(frames, shape.edges, reference), "modal_xpbd_derived_swing", scenarios.DURATION_MS)


if __name__ == "__main__":
    main()
