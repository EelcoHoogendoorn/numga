"""Oppositely charged conductors relax in a turning applied field."""
from __future__ import annotations

from collections.abc import Iterator
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
ring_centre = core.origin - .40 * mv.x.dual() + .08 * mv.y.dual()   # Point-
ring_outer, ring_inner = .36, .22
rod_centre = core.origin + .48 * mv.x.dual() - .08 * mv.y.dual()    # Point-
rod_turn = (mv.xy * (-np.pi / 12)).exp()                            # Rotor+
rod_lengths = np.array([.16, .48])
frame_duration_ms = 40
n_gallery_pixels, n_gallery_colours = 256, 24


def conductor_scene(n_across: int, n_frames: int) -> tuple[Topology, core.Point, core.Scalar, core.Scalar, core.Scalar]:
    """Two isolated conductors with smoothed boundary transitions sit inside an outer boundary whose applied field turns across `n_frames` frames."""
    topology = Topology.grid(n_across, n_across)
    stations = np.linspace(-half_width, half_width, n_across)       # [n_across]
    vertices = (core.origin + stations[None, :] * mv.x.dual()
                + stations[:, None] * mv.y.dual()).reshape(-1).field()  # Point-[V]
    ring_radius = (vertices - ring_centre).antinorm()               # Scalar+[V]
    ring_distance = where(ring_outer - ring_radius < ring_radius - ring_inner,
                          ring_outer - ring_radius, ring_radius - ring_inner)  # Scalar+[V]
    rod_axes = rod_turn >> stack((mv.x, mv.y))                      # [2] Upright+
    clearances = rod_lengths - (rod_axes & (vertices - rod_centre)).abs()  # [2] Scalar+[V]
    # Rectangular level sets retain the corners throughout the conductivity transition.
    rod_distance = clearances.min(axis=0)                           # Scalar+[V]
    transition = (stack((ring_distance, rod_distance)) + interface_width) / (2 * interface_width)  # [2] Scalar+[V]
    transition = where(transition < 0, 0, where(transition > 1, 0 * transition + 1, transition))  # [2] Scalar+[V]
    fractions = transition.squared() * (3 - 2 * transition)         # [2] Scalar+[V]

    # Readout region collecting every vertex touched by each body's conducting triangles.
    corner_mean = topology.incidence[2, 0].mean(axis=0)             # Scalar+[F <- V]
    active_faces = mv.scalar(((corner_mean * fractions) > 0)[..., None]).field()  # [2] Scalar+[F]
    memberships = mv.scalar(((corner_mean.adjugate() * active_faces) > 0)[..., None]).field()  # [2] Scalar+[V]
    outer = topology.reading(topology.exterior[0], 0)               # Scalar+[V_exterior <- V]
    progress = np.linspace(0, 1, n_frames)                          # [n_frames]
    angles = turn_angle * np.clip((progress - turn_start) / (turn_end - turn_start), 0, 1)  # [n_frames]
    turns = (mv.xy * (-angles / 2)).exp()                           # [n_frames] Rotor+
    slopes = turns >> (-field_strength * mv.x)                      # [n_frames] Upright+
    boundary = outer.adjugate() * (outer * (slopes & vertices))     # [n_frames] Scalar+[V]
    return topology, vertices, fractions, memberships, boundary


def main() -> None:
    from examples.electromagnetism.induced_charge import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, fractions, memberships, boundary = conductor_scene(n_across, n_frames)
    metal = fractions.sum(axis=0)                                   # Scalar+[V]
    save_figure(render.setup_figure(topology, vertices, metal), "charge_relaxation_setup")
    kept_states = []

    def streamed_states() -> Iterator[tuple[core.Scalar, core.Scalar, core.Scalar, core.Point]]:
        for state in core.relaxation_states(topology, vertices, fractions, boundary, permittivity,
                                            conductivities, body_charges, frame_step, n_substeps):
            kept_states.append(state)
            yield state

    save_animation(render.relaxation_animation(topology, vertices, metal, streamed_states()),
                   "opposite_charge_potential", frame_duration_ms)
    charges = stack([state[1] for state in kept_states])            # [n_frames] Scalar+[V]
    net_charge = (charges[:, None] * memberships).sites.sum()       # [n_frames, 2] Scalar+
    negative_charge = (where(charges < 0, -charges, 0)[:, None] * memberships).sites.sum()  # [n_frames, 2] Scalar+
    save_figure(render.balance_figure(net_charge, negative_charge, frame_step), "opposite_charge_relaxation_balance")
    final_density = kept_states[-1][2]                              # Scalar+[V]
    save_figure(render.corner_density_figure(vertices, final_density, rod_centre, rod_turn,
                                             rod_lengths, interface_width), "rod_corner_charge_density")

    # checks
    assert ((net_charge - body_charges).abs() < .01).all()
    assert (negative_charge[-1] > negative_charge[0]).all()


def pinned() -> None:
    """Saves the gallery animation of oppositely charged conductors relaxing in a turning field."""
    from examples.electromagnetism.induced_charge import render
    from examples.util.animation import pin_animation

    topology, vertices, fractions, _, boundary = conductor_scene(n_across, n_frames)
    metal = fractions.sum(axis=0)                                   # Scalar+[V]
    states = core.relaxation_states(topology, vertices, fractions, boundary, permittivity,
                                    conductivities, body_charges, frame_step, n_substeps)
    pin_animation(render.relaxation_animation(topology, vertices, metal, states, n_gallery_pixels),
                  "induced_charge", frame_duration_ms, colors=n_gallery_colours)


if __name__ == "__main__":
    main()
