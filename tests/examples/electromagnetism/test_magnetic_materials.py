"""A minimal coil-pulse step exercises magnetic storage, axial 4-currents and rendering."""
import matplotlib.pyplot as plt
import numpy as np
from numga import stack
from examples.electromagnetism.magnetic_materials import core, render, scenarios


def test_material_field_and_pulse_step():
    """One coarse pulse step preserves zero net axial current and non-negative Joule heating."""
    n_across, n_up, n_frames = 9, 7, 2
    topology, vertices, fractions, permeability, conductivity, remanence, coils, _ = scenarios.material_scene(
        n_across, n_up, n_frames,
    )
    areas = core.triangle_areas(topology, vertices)               # Scalar+[F]
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas  # Scalar+[V]
    gradients = core.simplex_gradients(topology, vertices)        # [3] Upright+[F]
    stiffness = core.magnetic_stiffness(topology, areas, gradients, permeability)  # Scalar+[V <- V]
    source = core.remanent_source(topology, areas, gradients, remanence)  # Scalar+[V]
    extension = topology.reading(topology.interior[0], 0).adjugate()  # Scalar+[V <- V_interior]
    coil_load = dual_areas * (coils / (dual_areas * coils).sites.sum() * scenarios.coil_signs).sum(axis=0)  # Scalar+[V]
    drive = np.array([0.0, scenarios.peak_coil_current])

    states = []

    def recorded_states():
        for state in core.evolution(stiffness, source, dual_areas, conductivity, extension, coil_load, drive, scenarios.frame_step):
            states.append(state)
            yield state

    frames = list(render.field_animation(topology, vertices, fractions, coils, recorded_states()))
    potentials = stack([potential for potential, _ in states])    # [n_frames] Scalar+[V]
    currents = stack([current for _, current in states])          # [n_frames] Current-[V]
    heating = core.joule_heating(potentials, dual_areas, conductivity, scenarios.frame_step)  # [n_frames - 1] Scalar+
    relaxation_plot = render.relaxation_figure(heating, scenarios.frame_step, scenarios.pulse_fall)
    setup_plot = render.setup_figure(topology, vertices, fractions, coils)
    plt.close(relaxation_plot)
    plt.close(setup_plot)

    corners = topology.incidence[2, 0]                            # [3] Scalar+[F <- V]
    reluctance = core.magnetic_response / (corners.mean(axis=0) * permeability)  # (Excitation- <- Field+)[F]
    final_field = ((gradients ^ core.mv.z) * (corners * potentials[-1])).sum(axis=0)  # Field+[F]

    assert len(frames) == n_frames
    assert potentials.isfinite().all()
    assert ((final_field & reluctance(final_field)) >= 0).all()
    assert source.sites.sum().abs() < 1e-8
    assert ((dual_areas * (core.mv.z & currents)).sites.sum().abs() < 1e-8).all()
    assert (heating >= 0).all()
