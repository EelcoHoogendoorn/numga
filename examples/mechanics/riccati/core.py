"""Finite-horizon Riccati feedback for planar docking with quadratic twist and forque costs.

Pose error is modelled as a twist with rigid-body motor `(-error / 2).exp()`, while net thruster action is modelled as a
forque. Under strong drag, velocity is proportional to the applied forque, so one step adds the resulting displacement
twist directly to the pose error.

Costs on pose errors and thruster pushes are quadratic forms evaluated as `value(error, error)` and
`effort_cost(push, push)`. Evaluating a quadratic form on linear maps pulls it back without transposes, so
`value(dynamics, actuation)` couples the present error and push through the next step, and minimizing over the push
yields a linear feedback map from twists to forques.
"""

from collections.abc import Iterator

import numpy as np

from numga import NumpyContext, concatenate
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                         # Scalar+
Point = ga.gatype.antivector()                      # Antivector-
Motor = ga.gatype.rotor()                           # Rotor+
Twist = ga.gatype.bivector()                        # Bivector+
Forque = ga.gatype.antibivector()                   # Antibivector-
StateCost = ga.gatype((Scalar, Twist, Twist))       # Scalar+ <- (Twist+, Twist+)
EffortCost = ga.gatype((Scalar, Forque, Forque))    # Scalar+ <- (Forque-, Forque-)
Dynamics = ga.gatype((Twist, Twist))                # Twist+ <- Twist+
Actuation = ga.gatype((Twist, Forque))              # Twist+ <- Forque-
Feedback = ga.gatype((Forque, Twist))               # Forque- <- Twist+


# --- math -----------------------------------------------------------------------------
def tracking_cost(tracking_points: Point, axes: Point, tracking_weights: np.ndarray, dt: float) -> StateCost:
    """Sums weighted squared displacement readouts (`[] Scalar+ <- (Twist+, Twist+)`) over hull sites and axes."""
    tracking_lines = tracking_points[:, None] & axes[None, :]                 # [..., ...] Forque-
    readouts = Twist & tracking_lines                                         # [..., ...] Scalar+ <- Twist+
    return (readouts * readouts * tracking_weights).sum(axis=(0, 1)) * dt


def drag_mobility(centre: Point, axes: Point, drag_weights: np.ndarray) -> Actuation:
    """Inverts the translational and rotational drag dyad sum (`[] Twist+ <- Forque-`) into a planar mobility."""
    drag_lines = concatenate([centre & axes, (axes[0] & axes[1])[None]])      # [3] Forque-
    drag = (drag_lines * (Twist & drag_lines) * drag_weights).sum(axis=0)     # [] Forque- <- Twist+
    return drag.inverse()


def riccati(value: StateCost, dynamics: Dynamics, actuation: Actuation, state_cost: StateCost,
            effort_cost: EffortCost, n_steps: int) -> Iterator[tuple[StateCost, Feedback]]:
    """Yields the remaining cost form and optimal feedback map (`[...]`) across `n_steps` backward steps."""
    for _ in range(n_steps):
        # A push pays effort now and shifts the next error inside the future cost form.
        control_cost = effort_cost + value(actuation, actuation)              # [...] Scalar+ <- (Forque-, Forque-)
        # Solving against the mixed pullback cancels the cost gradient for every error at once.
        feedback = -control_cost.solve(value(dynamics, actuation))            # [...] Forque- <- Twist+
        # Pulling the future cost through the closed-loop step updates the remaining cost form.
        value = state_cost + value(dynamics, dynamics + actuation(feedback))  # [...] Scalar+ <- (Twist+, Twist+)
        yield value, feedback


def rollout(initial_error: Twist, dynamics: Dynamics, actuation: Actuation,
            feedbacks: Feedback) -> Iterator[Twist]:
    """Yields the initial pose error (`[...] Twist+`) and its forward trajectory under `feedbacks` (`[n_steps, ...]`)."""
    error = initial_error                                                     # [...] Twist+
    yield error
    for feedback in feedbacks:
        push = feedback(error)                                                # [...] Forque-
        error = dynamics(error) + actuation(push)                             # [...] Twist+
        yield error
