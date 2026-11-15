"""Opposite half-charge nematic defects and crowded active films relax and move on a frictional substrate."""

from __future__ import annotations

import numpy as np

from examples.fluids.liquid_crystals import core
from examples.util import shapes
from numga.topology import Topology

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
    """Regular grid triangulation builds a square film of `n_side` points per side (`Point-[V]`) across
    `[-half_width, half_width]`."""
    axis = np.linspace(-half_width, half_width, n_side)
    topology, coordinates = shapes.box([axis, axis])
    return topology, mv.vector(coordinates).dual().field()


def dense() -> None:
    """Active stress drives a disordered film of interacting half-charge defects, saving the setup
    figure, animations, and free-energy trace."""
    from examples.fluids.liquid_crystals import render
    from examples.util.animation import save_animation, save_figure
    from numga import stack

    topology, vertices = film(n_dense_side, dense_half_width)
    rng = np.random.default_rng(seed)
    wavevectors = mv.vector(rng.normal(size=(n_waves, 2)) * wave_number)    # [n_waves] Plane+
    phases = rng.uniform(0, 2 * np.pi, n_waves)
    bearings = mv.xy * rng.uniform(0, np.pi, n_waves)                       # [n_waves] Turn+
    directors = (-bearings / 2).exp() >> mv.x.dual()                        # [n_waves] Point-
    amplitudes = core.alignment(directors, 1 / np.sqrt(n_waves))            # [n_waves] Point- <- Point-
    initial = core.disordered_alignment(vertices, wavevectors, phases, amplitudes, softness)
    defects = core.defects(topology, vertices, initial, mv.x.dual())
    save_figure(render.draw_dense_setup(topology, vertices, initial, defects), "liquid_crystals_dense_setup")
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    initial_films = initial[None].broadcast_to((len(dense_activities),))
    states = []

    def state_stream():
        for state in core.evolution(topology, areas, gradients, dual_areas, stiffness, elastic, bulk,
                                    dense_activities, initial_films, time_step, n_frames, n_dense_substeps):
            states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, render.dense_camera,
                                  core.with_defects(topology, vertices, state_stream(), mv.x.dual())),
                   "liquid_crystals_dense", frame_duration_ms, render.n_colours)
    save_animation(render.animate(topology, vertices, render.dense_camera,
                                  core.with_defects(topology, vertices, states, mv.x.dual()), marker_radius=0),
                   "liquid_crystals_dense_texture", frame_duration_ms, render.n_colours)
    energies = stack([core.free_energy(stiffness, dual_areas, elastic, bulk, state) for state in states])
    save_figure(render.draw_energy(np.arange(n_frames) * time_step * n_dense_substeps, energies),
                "liquid_crystals_dense_energy")

    # checks
    assert energies.isfinite().all()
    assert (states[-1].batch().trace().abs() < 1e-6).all()


def main() -> None:
    """Defect-pair virtual turns, orientational relaxation, and active-stress runs write the example's
    animations and charts."""
    from examples.fluids.liquid_crystals import render
    from examples.util.animation import save_animation, save_figure
    from numga import stack

    topology, vertices = film(n_side, half_width)
    centres = mv.vector([[-separation / 2 + 0.03, 0.07], [separation / 2 + 0.03, 0.07]]).dual()  # [2] Point-
    initial = core.defect_pair(vertices, centres, core_radius, mv.xy * 0)
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    defects = core.defects(topology, vertices, initial, mv.x.dual())
    save_figure(render.draw_setup(topology, vertices, initial, defects), "liquid_crystals_setup")
    # A smooth turn near one defect supplies a state with an orientational restoring torque.
    offset = vertices - (centres[0] + mv.y.dual() * probe_width)
    probe = mv.xy * (-offset.scalar_norm_squared() / (2 * probe_width ** 2)).exp()
    imprint = (-probe * imposed_turn / 2).exp()
    strained_order = imprint >> initial(imprint << core.Point)
    angles = np.linspace(-turn_span, turn_span, n_turn_frames)
    turn_states = []

    def virtual_turns():
        for angle in angles:
            rotor = (-probe * angle / 2).exp()
            state = rotor >> strained_order(rotor << core.Point)
            turn_states.append(state)
            yield state[None]

    save_animation(render.animate(topology, vertices, render.turn_camera,
                                  core.with_defects(topology, vertices, virtual_turns(), mv.x.dual())),
                   "liquid_crystals_turn", frame_duration_ms, render.n_colours)
    turn_energies = stack([core.free_energy(stiffness, dual_areas, elastic, bulk, state) for state in turn_states])
    molecular = core.molecular_field(stiffness, dual_areas, elastic, bulk, strained_order)
    couple = core.work.solve(-core.turn_slope(strained_order, molecular))
    slope = -(dual_areas * (probe & couple)).sites.sum()
    baseline = core.free_energy(stiffness, dual_areas, elastic, bulk, strained_order)
    save_figure(render.draw_derivative(angles, turn_energies, baseline, slope), "liquid_crystals_derivative")
    rotation_states = []

    def relaxation_stream():
        for state in core.turn_relaxation(topology, stiffness, dual_areas, elastic, bulk, strained_order[None],
                                          rotation_step, n_relaxation_frames, n_relaxation_substeps):
            rotation_states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, render.turn_camera,
                                  core.with_defects(topology, vertices, relaxation_stream(), mv.x.dual())),
                   "liquid_crystals_torque", frame_duration_ms, render.n_colours)
    rotation_energies = stack([core.free_energy(stiffness, dual_areas, elastic, bulk, state)
                               for state in rotation_states])
    save_figure(render.draw_release(np.arange(n_relaxation_frames) * rotation_step * n_relaxation_substeps,
                                    rotation_energies[:, 0]), "liquid_crystals_torque_energy")
    initial_films = initial[None].broadcast_to((len(activities),))
    states = []

    def evolution_stream():
        for state in core.evolution(topology, areas, gradients, dual_areas, stiffness, elastic, bulk, activities,
                                    initial_films, time_step, n_frames, n_substeps):
            states.append(state)
            yield state

    save_animation(render.animate(topology, vertices, render.defect_camera,
                                  core.with_defects(topology, vertices, evolution_stream(), mv.x.dual())),
                   "liquid_crystals_defects", frame_duration_ms, render.n_colours)
    energies = stack([core.free_energy(stiffness, dual_areas, elastic, bulk, state) for state in states])
    save_figure(render.draw_energy(np.arange(n_frames) * time_step * n_substeps, energies), "liquid_crystals_energy")

    # checks
    assert (rotation_energies[1:, 0] <= rotation_energies[:-1, 0]).all()
    first_order, last_order = rotation_states[0].batch(), rotation_states[-1].batch()
    assert ((first_order(first_order).trace() - last_order(last_order).trace()).abs() < 1e-6).all()
    assert (energies[1:, 0] <= energies[:-1, 0]).all()
    final_winding = core.windings(topology, states[-1][0], mv.x.dual())
    assert (final_winding.norm() < 0.01).all()


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
    areas, gradients, dual_areas = core.triangles(topology, vertices)
    stiffness = core.elasticity(topology, areas, gradients)
    states = core.evolution(topology, areas, gradients, dual_areas, stiffness, elastic, bulk, dense_activities,
                            initial[None], time_step, n_frames, n_dense_substeps)
    pin_animation(render.animate(topology, vertices, render.dense_camera,
                                 core.with_defects(topology, vertices, states, mv.x.dual())),
                  "liquid_crystals", frame_duration_ms, render.n_colours)


if __name__ == "__main__":
    main()
    dense()
