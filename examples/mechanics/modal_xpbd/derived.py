"""Modal extended position-based dynamics with joint couplings obtained by automatic differentiation
on JAX.

Whereas `core.py` assembles sparse rigid and modal gap maps analytically from anchor commutators and
constraint incidence, this module writes the world-frame constraint gaps once as a function of body
twists and modal amplitudes and differentiates that map with `numga.backend.jax.derivative`. Each
derivative is a dense field map from `n_bodies` to `n_constraints`, whose `.adjugate()` pulls
constraint forces back to body forques and modal loads without a coordinate transpose or twist
metric.
"""

from collections.abc import Iterator
from dataclasses import dataclass, fields, replace

import jax
import numpy as np

from numga import dim
from numga.algebras import PGA2D
from numga.backend.jax import JaxContext, derivative
from examples import instantiate
from examples.mechanics import lie_integrators as lie
from examples.mechanics.modal_xpbd.scenarios import Bodies, Constraints

core = instantiate("examples.mechanics.modal_xpbd.core", PGA2D, JaxContext, units=True)
mv = core.mv
Displacement, Force, Forque, Motor, Scalar, Twist = (
    core.Displacement, core.Force, core.Forque, core.Motor, core.Scalar, core.Twist,
)
# A load on a mode is the dual complement of its scalar amplitude, just as a forque is the dual
# complement of a twist; unit modal mass maps a load back to an amplitude.
Load = Scalar.derive.dual                                                  # Antiscalar-
sides = np.array([-1, 1])                                                  # [2]


@dataclass(frozen=True)
class Coordinates:
    """Incremental body twists and modal amplitudes over one step."""
    twists: Twist                                                          # [...] Bivector+[n_bodies]
    amplitudes: Scalar                                                     # [..., n_modes] Scalar+[n_bodies]


# Register `Bodies` and `Coordinates` as JAX pytrees so `jit` and `derivative` traverse their fields.
for record in (Bodies, Coordinates):
    jax.tree_util.register_dataclass(record, data_fields=[field.name for field in fields(record)], meta_fields=[])


# --- math -----------------------------------------------------------------------------
def constraint_gaps(motor: Motor, constraints: Constraints, coordinates: Coordinates) -> Displacement:
    """Deforming local anchors by `coordinates.amplitudes` and advancing `motor` by
    `coordinates.twists` gives the world-frame gap across every constraint
    (`[...] Antivector-[n_constraints]`)."""
    stepped_motor = motor * (coordinates.twists * -0.5).exp()              # [...] Rotor+[n_bodies]
    local_anchors = constraints.anchors + (constraints.modes * (constraints.ends * coordinates.amplitudes[..., None])).sum(axis=-2)  # [..., 2] Antivector-[n_constraints]
    endpoint_motors = constraints.ends * stepped_motor[..., None]          # [..., 2] Rotor+[n_constraints]
    anchors = endpoint_motors >> local_anchors                             # [..., 2] Antivector-[n_constraints]
    return (anchors * sides).sum(axis=-1).cast(Displacement)               # [...] Antivector-[n_constraints]


def project_constraints(bodies: Bodies, constraints: Constraints, previous_amplitudes: Scalar, mode_reactions: Scalar, dt: float) -> tuple[Bodies, Scalar]:
    """Differentiating `constraint_gaps` supplies the rigid and modal coupling maps whose Schur
    complement closes all constraint gaps in one solve."""
    compliance, residual, response = core.modal_terms(bodies, previous_amplitudes, dt)  # [..., n_modes] Scalar+[n_bodies], Scalar+[n_bodies], Scalar+[n_bodies]

    def gap(coordinates: Coordinates) -> Displacement:
        """World-frame constraint gaps at `coordinates` around the current `bodies.motor`."""
        return constraint_gaps(bodies.motor, constraints, coordinates)

    current = Coordinates(twists=bodies.rate * dt * 0, amplitudes=bodies.amplitudes)
    change = derivative(gap)(current)
    rigid = change.twists                                                  # [...] (Antivector- <- Bivector+)[n_constraints <- n_bodies]
    modal = change.amplitudes                                              # [..., n_modes] (Antivector- <- Scalar+)[n_constraints <- n_bodies]
    spring_load = -(residual + compliance * mode_reactions)                # [..., n_modes] Scalar+[n_bodies]
    unconstrained_step = response * spring_load                            # [..., n_modes] Scalar+[n_bodies]
    mobility = response * compliance * Load.dual()                         # [..., n_modes] (Scalar+ <- Antiscalar-)[n_bodies]
    constraint_compliance = (constraints.compliance / dt**2 * Force.dual()).on_diagonal()  # (Antivector- <- Vector+)[n_constraints <- n_constraints]
    # Each derivative maps motion to intrinsic gap displacements, so its adjugate pulls extrinsic
    # constraint forces back to body forques and modal loads across `force & displacement`.
    system = (rigid(bodies.inverse_inertia(rigid.adjugate()))
              + modal(mobility(modal.adjugate())).sum(axis=-1)
              + constraint_compliance)                                     # [...] (Antivector- <- Vector+)[n_constraints <- n_constraints]
    reactions = system.solve(-(gap(current) + modal(unconstrained_step).sum(axis=-1)))  # [...] Vector+[n_constraints]
    twist_step = bodies.inverse_inertia(rigid.adjugate()(reactions))       # [...] Bivector+[n_bodies]
    mode_loads = Load.dual()(modal.adjugate()(reactions))                  # [..., n_modes] Scalar+[n_bodies]
    reaction_change = response * (spring_load - mode_loads)                # [..., n_modes] Scalar+[n_bodies]
    stepped_motor = bodies.motor * (twist_step * -0.5).exp()               # [...] Rotor+[n_bodies]
    constrained_bodies = replace(
        bodies,
        motor=stepped_motor,
        amplitudes=bodies.amplitudes + mode_loads + reaction_change,
    )
    return constrained_bodies, mode_reactions + reaction_change


def step(bodies: Bodies, constraints: Constraints, dt: float, gravity: Displacement) -> Bodies:
    """Free rigid-body prediction under `gravity` followed by differentiated constraint projection
    advances `bodies` by one step `dt`."""
    previous = bodies

    def weight(motor: Motor, rate: Twist) -> Forque:
        # Gravity acts at each body's centre of mass, placed at the origin of the body frame.
        return (mv.w.dual() & (motor << gravity)) * bodies.masses          # [...] Antibivector-[n_bodies]

    motor, _ = lie.explicit_rk4(bodies.motor, bodies.rate, bodies.inertia, bodies.inverse_inertia, dt, weight)
    bodies = replace(bodies, motor=motor, amplitudes=bodies.amplitudes + bodies.rates * dt)
    _, residual, response = core.modal_terms(bodies, previous.amplitudes, dt)  # [..., n_modes] Scalar+[n_bodies], Scalar+[n_bodies], Scalar+[n_bodies]
    mode_reactions = -residual * response                                  # [..., n_modes] Scalar+[n_bodies]
    bodies, _ = project_constraints(
        replace(bodies, amplitudes=bodies.amplitudes + mode_reactions),
        constraints,
        previous.amplitudes,
        mode_reactions,
        dt,
    )
    return replace(
        bodies,
        rate=(~previous.motor * bodies.motor).log() * (-2 / dt),
        rates=(bodies.amplitudes - previous.amplitudes) / dt,
    )


def swing_states(bodies: Bodies, constraints: Constraints, gravity: Displacement, dt: float, n_substeps: int, n_frames: int) -> Iterator[Bodies]:
    """Yields `n_frames` body states under `gravity`, advancing `n_substeps` compiled steps per
    frame."""
    @jax.jit
    def advance(bodies: Bodies) -> Bodies:
        return step(bodies, constraints, dt, gravity)

    for _ in range(n_frames):
        yield bodies
        for _ in range(n_substeps):
            bodies = advance(bodies)


def main() -> None:
    from examples.util.animation import save_animation
    from . import render, scenarios

    shape = core.girder(
        scenarios.n_bays,
        mv.scalar(scenarios.length, unit=dim.length),
        mv.scalar(scenarios.height, unit=dim.length),
        mv.scalar(scenarios.stiffness, unit=dim.mass * dim.length / dim.time**2),
        mv.scalar(scenarios.density, unit=dim.mass / dim.length**2),
        scenarios.n_modes,
    )
    bodies, constraints = scenarios.hinged_chain(mv, shape, scenarios.damping, scenarios.flexibility)
    gravity = mv.direction([0, -scenarios.gravity_strength], unit=dim.length / dim.time**2)  # [] Antivector-, length/time^2
    dt = mv.scalar(scenarios.dt, unit=dim.time)
    world_points = jax.jit(lambda state: core.world_points(state, shape))
    kept: list = []

    def frames() -> Iterator:
        for state in swing_states(bodies, constraints, gravity, dt, scenarios.n_substeps, scenarios.n_frames):
            kept.append(state)
            yield world_points(state)

    save_animation(render.swinging_chain(shape.edges, core.world_points(bodies, shape), frames()), "modal_xpbd_derived_swing", scenarios.frame_duration_ms)

    # checks
    _, _, _, final_gap = core.coupling(kept[-1], constraints)
    assert (final_gap.dual_inverse().norm() < mv.scalar(1e-4, unit=dim.length)).all()


if __name__ == "__main__":
    main()
