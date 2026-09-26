"""Dock with three reversible thrusters, trading speed against effort."""

import numpy as np

from numga import concatenate, stack
from examples.mechanics.riccati import core

mv, Twist, Forque = core.mv, core.Twist, core.Forque

LABELS = ("Lower effort penalty", "Higher effort penalty")
CASES = len(LABELS)
EFFORT_SCALES = np.array([0.05, 5.])                                   # [cases]
THRUSTER_WEIGHTS = np.array([1., 2., 3.])                               # [thrusters]
# Drag along the forward and sideways directions through the centre, and against turning.
DRAG_WEIGHTS = np.array([2., 3., 1.])                                   # [modes]
# Rows: bow and stern; columns: forward and sideways displacement.
TRACKING_WEIGHTS = np.array([[4., 8.], [1., 2.]])                       # [sites, directions]
DT = 0.1
STEPS = 72
# How much stiffer the pose is priced at the deadline than along the way: the vessel must be there.
LANDING = 1e6
FRAME_DURATION = 40
ARROW_SCALE = 0.04

HULL = mv.antivector([[-.45, -.23, 1], [.28, -.23, 1], [.55, 0, 1],
                      [.28, .23, 1], [-.45, .23, 1]])                   # [vertices] Point
CENTRE = mv.antivector([0, 0, 1])                                       # [] Point
AXES = mv.antivector([[1, 0, 0], [0, 1, 0]])                            # [directions] Point at infinity: forward, sideways
MOUNTS = mv.antivector([[-.3, .2, 1], [-.3, -.2, 1], [.35, 0, 1]])      # [thrusters] Point
DIRECTIONS = mv.antivector([[1, 0, 0], [1, 0, 0], [0, 1, 0]])           # [thrusters] Point at infinity
TRACKING_POINTS = mv.antivector([[.45, 0, 1], [-.35, 0, 1]])            # [sites] Point
INITIAL = (mv.xw * .4 + mv.yw * .5 + mv.xy * .35) * np.ones(CASES)      # [cases] Twist


# --- math -----------------------------------------------------------------------------
def docking() -> tuple[core.StateCost, core.Feedback, core.Twist, core.Scalar]:
    """Costs for one through STEPS remaining actions, then the policy, errors, and commands."""
    # Joining a mount to its ideal direction gives its line of force, including the lever arm.
    thrusters = MOUNTS & DIRECTIONS                                    # [thrusters] Forque
    authority = (thrusters * (thrusters & Twist) / THRUSTER_WEIGHTS).sum(axis=0)   # [] Forque <- Twist
    resistance = authority.inverse()                                   # [] Twist <- Forque
    effort_cost = (Forque & resistance(Forque)) * EFFORT_SCALES * DT    # [cases] Scalar <- (Forque, Forque)

    # Each tracking line reads one displacement of a hull point; its square is that displacement's cost.
    tracking_lines = TRACKING_POINTS[:, None] & AXES[None, :]          # [sites, directions] Forque
    readouts = tracking_lines & Twist                                  # [sites, directions] Scalar <- Twist
    state_cost = (readouts * readouts * TRACKING_WEIGHTS).sum(axis=(0, 1)) * DT   # [] Scalar <- (Twist, Twist)

    # Drag resists moving along either axis through the centre, and turning; its inverse turns a push
    # into a velocity twist.
    drag_lines = concatenate([CENTRE & AXES, (AXES[0] & AXES[1])[None]])   # [modes] Forque
    drag = (drag_lines * (drag_lines & Twist) * DRAG_WEIGHTS).sum(axis=0)  # [] Forque <- Twist
    dynamics = mv.rotor() >> Twist                                     # [] Twist <- Twist: without a push, the vessel stays put
    actuation = drag.inverse() * DT                                    # [] Twist <- Forque

    # From the deadline back, where whatever error is left is priced LANDING times as stiffly.
    costs, gains = zip(*core.riccati(state_cost * LANDING * np.ones(CASES), dynamics, actuation, state_cost,
                                     effort_cost, STEPS))
    values = stack(costs)                                              # [steps, cases] Scalar <- (Twist, Twist): horizons 1 through STEPS
    feedbacks = stack(gains)[::-1]                                     # [steps, cases] Forque <- Twist
    errors = stack(tuple(core.rollout(INITIAL, dynamics, actuation, feedbacks)))   # [steps + 1, cases] Twist
    pushes = feedbacks(errors[:-1])                                    # [steps, cases] Forque
    # Allocate each requested push to the thrusters at minimum weighted squared command.
    commands = (thrusters & resistance(pushes)[..., None]) / THRUSTER_WEIGHTS   # [steps, cases, thrusters] Scalar
    return values, feedbacks, errors, commands


# --- plumbing -------------------------------------------------------------------------
if __name__ == "__main__":
    from examples.animation import save_animation, save_figure
    from examples.mechanics.riccati import render

    values, feedbacks, errors, commands = docking()
    poses = (errors * -0.5).exp()                                # [steps + 1, cases] Motor

    save_figure(render.draw_setup(HULL, MOUNTS, DIRECTIONS, TRACKING_POINTS), "riccati_setup")
    save_figure(render.draw_approaches(HULL, poses, LABELS), "riccati_approaches")
    save_figure(render.draw_costs(values, LABELS), "riccati_costs")
    # Include the landed pose with the thrusters off, after the final commanded step.
    commands = concatenate((commands, commands[-1:] * 0))         # [steps + 1, cases, thrusters] Scalar
    save_animation(render.animate(HULL, MOUNTS, DIRECTIONS, poses, commands, LABELS, ARROW_SCALE),
                   "riccati", FRAME_DURATION)
