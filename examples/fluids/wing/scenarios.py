"""Pitching and cylinder-to-wing morphing scenarios for potential flow past a Joukowski wing."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator

import numpy as np

from numga import stack

from examples.fluids.wing import core
from numga.topology import Topology

# Cylinder centre offset behind and above the origin before the map sets wing thickness and camber.
thickness = 0.1
camber = 0.08
chord_scale = 1.0
speed = 1.0
density = 1.0
lowest_attack = np.deg2rad(-4.0)
highest_attack = np.deg2rad(12.0)
morph_attack = np.deg2rad(8.0)
# Sample ring and angle counts for animations and contour checks.
n_rings = 140
n_angles = 360
n_drawn_rings = 35
n_drawn_angles = 90
outer_distance = 6.0
# Frame counts, durations, and palette size for the pitching gallery sweep and morph animations.
n_frames = 120
sweep_duration_ms = 40
n_sweep_colours = 64
n_morph_frames = 90
frame_duration_ms = 40
# Uniform free stream along +x.
stream = speed * core.mv.x.dual()                                            # [] Point-


# --- math -----------------------------------------------------------------------------
def wing_geometry(attack: float) -> tuple[core.Point, core.Point]:
    """Cylinder centre and trailing-edge point for the wing pitched nose up by `attack`."""
    rotor = (core.mv.xy * (attack / 2)).exp()                                # [] Rotor+
    centre = rotor >> (-thickness * core.mv.x + camber * core.mv.y).dual()   # [] Point-
    trailing_edge = chord_scale * (rotor >> core.mv.x.dual())                # [] Point-
    return centre, trailing_edge


def morph_states() -> Iterator[tuple[core.Point, core.Point, core.Bivector, core.Force, core.Point, core.Point]]:
    """Yield flow states as the critical points separate from the origin out to the trailing edge."""
    centre, trailing_edge = wing_geometry(morph_attack)
    swirl = core.kutta_circulation(centre, trailing_edge, stream)            # [] Bivector+
    lift = core.kutta_lift(swirl, stream, density)                           # [] Force+
    positions = core.cylinder_rings(centre, trailing_edge, n_drawn_rings, n_drawn_angles, outer_distance)   # [n_drawn_rings, n_drawn_angles] Point-
    # Outward half-cycle; the saved animation plays forward and back.
    spread = (1 - np.cos(np.linspace(0, np.pi, n_morph_frames // 2 + 1))) / 2
    for fraction in spread:
        critical = fraction * trailing_edge                                  # [] Point-
        critical_images = 2 * stack([critical, -critical])                   # [2] Point-
        points, velocity, _, flux = core.wing_flow(centre, trailing_edge, swirl, stream, positions, critical)
        yield points, velocity, flux, lift, critical_images, centre


def pitch_states() -> Iterator[tuple[core.Point, core.Point, core.Bivector, core.Force]]:
    """Yield flow states over one full period as the wing pitches up and back down."""
    phase = (1 - np.cos(np.linspace(0, 2 * np.pi, n_frames, endpoint=False))) / 2
    for attack in lowest_attack + (highest_attack - lowest_attack) * phase:
        centre, trailing_edge = wing_geometry(attack)
        swirl = core.kutta_circulation(centre, trailing_edge, stream)        # [] Bivector+
        positions = core.cylinder_rings(centre, trailing_edge, n_drawn_rings, n_drawn_angles, outer_distance)   # [n_drawn_rings, n_drawn_angles] Point-
        points, velocity, _, flux = core.wing_flow(centre, trailing_edge, swirl, stream, positions, trailing_edge)
        yield points, velocity, flux, core.kutta_lift(swirl, stream, density)


# --- plumbing -------------------------------------------------------------------------
def save_states(states: Iterable, animate: Callable[..., Iterable[np.ndarray]], *arguments,
                name: str, duration_ms: int, forward_then_back: bool = False) -> list:
    """Save an animation from `states` in a single pass and return the collected states."""
    from examples.util.animation import save_animation

    kept = []

    def passing() -> Iterator:
        for state in states:
            kept.append(state)
            yield state

    save_animation(animate(*arguments, passing()), name, duration_ms, forward_then_back=forward_then_back)
    return kept


def main() -> None:
    from examples.fluids.wing import render

    morphed = save_states(morph_states(), render.animate_marked, speed,
                          name="wing_morph", duration_ms=frame_duration_ms, forward_then_back=True)

    # checks
    # Final morph state is the full Joukowski wing at `morph_attack`.
    wing_points, wing_velocity, wing_flux, lift, critical_images, centre = morphed[-1]
    _, trailing_edge = wing_geometry(morph_attack)
    swirl = core.kutta_circulation(centre, trailing_edge, stream)            # [] Bivector+
    assert ((critical_images[0] - 2 * trailing_edge).norm() < 1e-12).all()
    assert ((wing_flux[0] - wing_flux[0, :1]).norm() < 1e-12).all()
    # Velocity derivative vanishes everywhere past the cylinder, and line integrals around every ring equal circulation.
    positions = core.cylinder_rings(centre, trailing_edge, n_drawn_rings, n_drawn_angles, outer_distance)   # [n_drawn_rings, n_drawn_angles] Point-
    _, gradient, _ = core.cylinder_flow(centre, trailing_edge, swirl, stream, positions)
    assert (core.geometric_derivative(gradient).norm() < 1e-12).all()
    ring = Topology.loops([n_drawn_angles])
    edge_average = ring.incidence[1, 0].mean(axis=0)                         # Scalar+[E <- V]
    loop_points = wing_points.field()                                        # [n_drawn_rings] Point-[V]
    loop_velocities = wing_velocity.field()                                  # [n_drawn_rings] Point-[V]
    edge_steps = ring.d[0] * loop_points                                     # [n_drawn_rings] Point-[E]
    ring_integrals = ((edge_average * loop_velocities) * edge_steps).batch().sum(axis=-1)   # [n_drawn_rings] Even+
    assert ((ring_integrals - ring_integrals[:1]).norm() < 1e-10).all()
    tails, heads = ring.incidence[1, 0][:, None] * loop_points               # [n_drawn_rings] Point-[E], Point-[E]
    ring_areas = (tails ^ heads).batch().sum(axis=-1)                        # [n_drawn_rings] Bivector+
    assert ((ring_integrals * (ring_areas / ring_areas.norm()) - swirl).norm() < 5e-3).all()
    # Integrating Bernoulli surface pressure around the innermost ring recovers Kutta–Joukowski lift with zero drag.
    speed_squared = wing_velocity[0].squared().field()                       # [] Scalar+[V]
    edge_pressures = -0.5 * density * (edge_average * speed_squared)         # [] Scalar+[E]
    winding = (tails[0] & heads[0]).batch().sum(axis=-1)                     # [] Antiscalar-
    pressure_force = (winding / winding.antinorm()) * (edge_pressures * edge_steps[0]).batch().sum(axis=-1)   # [] Force+
    assert ((pressure_force - lift).norm() < 5e-3).all()


def pinned() -> None:
    """Regenerate the gallery pitching-wing animation tracked in `plots/wing.gif`."""
    from examples.util.animation import pin_animation
    from examples.fluids.wing import render

    pin_animation(render.animate(speed, pitch_states()), "wing", sweep_duration_ms, colors=n_sweep_colours)


if __name__ == "__main__":
    main()
