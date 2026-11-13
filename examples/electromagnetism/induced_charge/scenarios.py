"""Oppositely charged conductors relax in a turning applied field."""
from __future__ import annotations

import numpy as np
from numga import stack, where
from examples.electromagnetism.induced_charge import core
from numga.topology import Topology

mv = core.mv
n_across, n_frames = 129, 100
half_width, permittivity, field_strength = 1., 1., .8
conductivities = np.array([3., 3.])
body_charges = np.array([-1., 1.])
n_interface = 6
interface_width = n_interface * half_width / (n_across - 1)
frame_step, n_substeps = .08, 2
turn_start, turn_end, turn_angle = .25, .65, np.pi
ring_centre = core.origin - .40 * mv.x.dual() + .08 * mv.y.dual()
ring_outer, ring_inner = .36, .22
rod_centre = core.origin + .48 * mv.x.dual() - .08 * mv.y.dual()
rod_turn = (mv.xy * (-np.pi / 12)).exp()
rod_lengths = np.array([.16, .48])
frame_duration_ms = 40


def scene(n_across: int, n_frames: int):
    """The isolated conductors have finite conductivity across a resolved material transition."""
    topology = Topology.grid(n_across, n_across)
    stations = np.linspace(-half_width, half_width, n_across)
    vertices = (core.origin + stations[None, :] * mv.x.dual()
                + stations[:, None] * mv.y.dual()).reshape(-1).field()
    ring_radius = (vertices - ring_centre).antinorm()
    ring_distance = where(ring_outer - ring_radius < ring_radius - ring_inner,
                          ring_outer - ring_radius, ring_radius - ring_inner)
    rod_axes = rod_turn >> stack((mv.x, mv.y))
    clearances = rod_lengths - (rod_axes & (vertices - rod_centre)).abs()
    # Rectangular level sets retain the corners throughout the conductivity taper.
    rod_distance = clearances.min(axis=0)
    transition = (stack((ring_distance, rod_distance)) + interface_width) / (2 * interface_width)
    transition = where(transition < 0, 0, where(transition > 1, 0 * transition + 1, transition))
    fractions = transition.squared() * (3 - 2 * transition)

    # This is a readout region: all vertices touched by each body's conducting triangles.
    mean = topology.incidence[2, 0].mean(axis=0)
    active = mv.scalar(((mean * fractions) > 0)[..., None]).field()
    memberships = mv.scalar(((mean.adjugate() * active) > 0)[..., None]).field()
    outer = topology.reading(topology.exterior[0], 0)
    progress = np.linspace(0, 1, n_frames)
    angles = turn_angle * np.clip((progress - turn_start) / (turn_end - turn_start), 0, 1)
    turns = (mv.xy * (-angles / 2)).exp()
    slopes = turns >> (-field_strength * mv.x)
    boundary = outer.adjugate() * (outer * (slopes & vertices))
    return topology, vertices, fractions, memberships, boundary


def main() -> None:
    from examples.electromagnetism.induced_charge import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, fractions, memberships, boundary = scene(n_across, n_frames)
    metal = fractions.sum(axis=0)
    save_figure(render.setup(topology, vertices, metal), "charge_relaxation_setup")
    states = []

    def recorded():
        for state in core.relaxation(topology, vertices, fractions, boundary, permittivity,
                                     conductivities, body_charges, frame_step, n_substeps):
            states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, metal, recorded()), "opposite_charge_potential", frame_duration_ms)
    charges = stack([state[1] for state in states])
    net = (charges[:, None] * memberships).sites.sum()
    negative = (where(charges < 0, -charges, 0)[:, None] * memberships).sites.sum()
    save_figure(render.balance(net, negative, frame_step), "opposite_charge_relaxation_balance")
    save_figure(render.corner_density(topology, vertices, charges[-1], rod_centre, rod_turn,
                                      rod_lengths, interface_width), "rod_corner_charge_density")

    # checks
    assert np.all((net - body_charges).abs() < .01)


if __name__ == "__main__":
    main()
