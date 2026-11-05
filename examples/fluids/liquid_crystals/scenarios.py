"""Opposite half-charge nematic defects, relaxing and driven by active stress on a substrate."""

from __future__ import annotations

import numpy as np

from examples.fluids.liquid_crystals import core
from examples.util import shapes
from examples.util.topology import Topology

mv = core.mv
half_width, n_side = 6.0, 49
separation, core_radius = 2.4, 0.45
elastic, bulk = 0.6, 3.0
activities = np.array([0.0, 0.4, -0.4])
time_step, n_substeps, n_frames = 0.015, 2, 91
frame_duration_ms = 40
probe_width, imposed_turn, turn_span, n_turn_frames = 0.8, 1.0, 0.25, 61
rotation_step = time_step / 2
n_relaxation_frames, n_relaxation_substeps = 61, 2
dense_half_width, n_dense_side = 8.0, 65
n_waves, wave_number, softness, seed = 32, 1.8, 0.2, 7
dense_activities = np.array([1.2])
n_dense_substeps = 2


def film(n_side: int, half_width: float) -> tuple[Topology, core.Point]:
    """A square film, triangulated across its width."""
    axis = np.linspace(-half_width, half_width, n_side)
    topology, coordinates = shapes.box([axis, axis])
    return topology, mv.vector(coordinates).dual().field()


def dense() -> None:
    """A crowded film of interacting defects driven by active stress."""
    from examples.fluids.liquid_crystals import render
    from examples.util.animation import save_animation, save_figure
    from numga import stack

    topology, vertices = film(n_dense_side, dense_half_width)
    rng = np.random.default_rng(seed)
    wavevectors = mv.vector(rng.normal(size=(n_waves, 2)) * wave_number)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    bearings = mv.xy * rng.uniform(0, np.pi, n_waves)
    directors = (-bearings / 2).exp() >> mv.x.dual()
    amplitudes = core.alignment(directors, 1 / np.sqrt(n_waves))
    initial = core.disordered_alignment(vertices, wavevectors, phases, amplitudes, softness)
    save_figure(render.draw_dense_setup(topology, vertices, initial), "liquid_crystals_dense_setup")
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    start = initial[None].broadcast_to((len(dense_activities),))
    states = []

    def passing():
        for state in core.evolution(topology, areas, gradients, cells, stiffness, elastic, bulk, dense_activities,
                                    start, time_step, n_frames, n_dense_substeps):
            states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, render.dense_camera, passing()),
                   "liquid_crystals_dense", frame_duration_ms, render.n_colours)
    energies = stack([core.free_energy(stiffness, cells, elastic, bulk, state) for state in states])
    save_figure(render.draw_energy(np.arange(n_frames) * time_step * n_dense_substeps, energies),
                "liquid_crystals_dense_energy")

    # checks
    assert np.all(energies.isfinite())
    np.testing.assert_allclose(states[-1].batch().trace().kernel, 0, atol=1e-8)


def main() -> None:
    from examples.fluids.liquid_crystals import render
    from examples.util.animation import save_animation, save_figure
    from numga import stack

    topology, vertices = film(n_side, half_width)
    centres = mv.vector([[-separation / 2 + 0.03, 0.07], [separation / 2 + 0.03, 0.07]]).dual()
    initial = core.defect_pair(vertices, centres, core_radius, mv.xy * 0)
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    save_figure(render.draw_setup(topology, vertices, initial), "liquid_crystals_setup")
    # A smooth turn near one defect supplies a state with an orientational restoring torque.
    offset = vertices - (centres[0] + mv.y.dual() * probe_width)
    probe = mv.xy * (-offset.scalar_norm_squared() / (2 * probe_width ** 2)).exp()
    imprint = (-probe * imposed_turn / 2).exp()
    strained = imprint >> initial(imprint << core.Point)
    angles = np.linspace(-turn_span, turn_span, n_turn_frames)
    turn_states = []

    def virtual_turns():
        for angle in angles:
            rotor = (-probe * angle / 2).exp()
            state = rotor >> strained(rotor << core.Point)
            turn_states.append(state)
            yield state[None]

    save_animation(render.animate(topology, vertices, render.turn_camera, virtual_turns()),
                   "liquid_crystals_turn", frame_duration_ms, render.n_colours)
    turn_energies = stack([core.free_energy(stiffness, cells, elastic, bulk, state) for state in turn_states])
    molecular = core.molecular_field(stiffness, cells, elastic, bulk, strained)
    slope = (cells * core.turn_slope(strained, molecular)(probe)).sites.sum()
    baseline = core.free_energy(stiffness, cells, elastic, bulk, strained)
    save_figure(render.draw_derivative(angles, turn_energies, baseline, slope), "liquid_crystals_derivative")
    rotation_states = []

    def rotating():
        for state in core.turn_relaxation(topology, stiffness, cells, elastic, bulk, strained[None], rotation_step,
                                          n_relaxation_frames, n_relaxation_substeps):
            rotation_states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, render.turn_camera, rotating()),
                   "liquid_crystals_torque", frame_duration_ms, render.n_colours)
    rotation_energies = stack([core.free_energy(stiffness, cells, elastic, bulk, state) for state in rotation_states])
    save_figure(render.draw_release(np.arange(n_relaxation_frames) * rotation_step * n_relaxation_substeps,
                                   rotation_energies[:, 0]), "liquid_crystals_torque_energy")
    initial = initial[None].broadcast_to((len(activities),))
    states = []

    def passing():
        for state in core.evolution(topology, areas, gradients, cells, stiffness, elastic, bulk, activities,
                                    initial, time_step, n_frames, n_substeps):
            states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, render.defect_camera, passing()),
                   "liquid_crystals_defects", frame_duration_ms, render.n_colours)
    energies = stack([core.free_energy(stiffness, cells, elastic, bulk, state) for state in states])
    save_figure(render.draw_energy(np.arange(n_frames) * time_step * n_substeps, energies), "liquid_crystals_energy")

    # checks
    assert np.all(np.diff(rotation_energies[:, 0].to_array()) <= 0)
    first, last = rotation_states[0].batch(), rotation_states[-1].batch()
    np.testing.assert_allclose(first(first).trace().kernel, last(last).trace().kernel, atol=1e-8)
    assert np.all(np.diff(energies[:, 0].to_array()) <= 0)
    final_winding = core.windings(topology, states[-1][0], mv.x.dual())
    assert np.all(final_winding.norm() < 0.01)


def pinned() -> None:
    """The gallery shows one crowded active film, with its molecular axes and half-charge defects."""
    from examples.fluids.liquid_crystals import render
    from examples.util.animation import pin_animation

    topology, vertices = film(n_dense_side, dense_half_width)
    rng = np.random.default_rng(seed)
    wavevectors = mv.vector(rng.normal(size=(n_waves, 2)) * wave_number)
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    bearings = mv.xy * rng.uniform(0, np.pi, n_waves)
    directors = (-bearings / 2).exp() >> mv.x.dual()
    amplitudes = core.alignment(directors, 1 / np.sqrt(n_waves))
    initial = core.disordered_alignment(vertices, wavevectors, phases, amplitudes, softness)
    areas, gradients, cells = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    states = core.evolution(topology, areas, gradients, cells, stiffness, elastic, bulk, dense_activities,
                            initial[None], time_step, n_frames, n_dense_substeps)
    pin_animation(render.animate(topology, vertices, render.dense_camera, states),
                  "liquid_crystals", frame_duration_ms, render.n_colours)


if __name__ == "__main__":
    main()
    dense()
