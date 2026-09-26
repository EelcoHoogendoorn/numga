"""Finite-horizon feedback for a local docking error, with costs as quadratic forms.

A twist describes the remaining displacement and turn, with pose `exp(-error / 2)`. A forque is the
total force and torque from the thrusters. Strong drag makes their relation a mobility rather than an
acceleration: one step adds the mobility of the push to the pose error.

Every cost is a quadratic form, a scalar with two open slots: the cost of an error is
`value(error, error)`. Filling both slots with maps pulls a form back through them, so
`value(dynamics, actuation)` is the future cost seen from the present error and push, with no
transposes. The Riccati recursion pulls the future cost back through one step and minimizes over the
push; its solve returns a feedback map from a twist to a forque.
"""

from collections.abc import Iterator

from numga import NumpyContext
from numga.algebras import PGA2D

ga = PGA2D
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Point = ga.gatype.antivector()
Motor = ga.gatype.rotor()
Twist = ga.gatype.bivector()
Forque = ga.gatype.antibivector()
StateCost = ga.gatype((Scalar, Twist, Twist))       # Scalar <- (Twist, Twist)
EffortCost = ga.gatype((Scalar, Forque, Forque))    # Scalar <- (Forque, Forque)
Dynamics = ga.gatype((Twist, Twist))               # Twist <- Twist
Actuation = ga.gatype((Twist, Forque))              # Twist <- Forque
Feedback = ga.gatype((Forque, Twist))              # Forque <- Twist


# --- math -----------------------------------------------------------------------------
def riccati(value: StateCost, dynamics: Dynamics, actuation: Actuation, state_cost: StateCost,
            effort_cost: EffortCost, steps: int) -> Iterator[tuple[StateCost, Feedback]]:
    """The cost still to pay and the feedback, one more step back from the given cost each time."""
    for _ in range(steps):
        # A push costs effort now and moves the error whose cost is paid next.
        control_cost = effort_cost + value(actuation, actuation)             # [cases] Scalar <- (Forque, Forque)
        # The push that cancels the cost's derivative, for every error at once.
        feedback = -control_cost.solve(value(dynamics, actuation))           # [cases] Forque <- Twist
        # The error's cost now, and its future cost through the step that feedback takes.
        value = state_cost + value(dynamics, dynamics + actuation(feedback))  # [cases] Scalar <- (Twist, Twist)
        yield value, feedback


def rollout(initial: Twist, dynamics: Dynamics, actuation: Actuation,
            feedbacks: Feedback) -> Iterator[Twist]:
    """The initial error and its evolution under the finite-horizon feedback policy."""
    error = initial                                                # [cases] Twist
    yield error
    for feedback in feedbacks:
        push = feedback(error)                                     # [cases] Forque
        error = dynamics(error) + actuation(push)                  # [cases] Twist
        yield error
