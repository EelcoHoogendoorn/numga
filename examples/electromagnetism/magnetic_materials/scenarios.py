"""A coil pulse disturbs a permanent magnet's field across soft iron and copper."""
from __future__ import annotations

import numpy as np
from numga import stack, where
from examples.util import shapes
from examples.electromagnetism.magnetic_materials import core

n_across, n_up, n_frames = 97, 73, 100
frame_duration_ms = 40
half_width, half_height = 1.65, 1.25
frame_step = 0.06
interface_width = 0.035
magnet_centre = core.origin - 0.8 * core.mv.x.dual()          # Point-
magnet_half_lengths = np.array([0.15, 0.36])
iron_outer_centre = core.origin + 0.1 * core.mv.x.dual()      # Point-
iron_inner_centre = core.origin - 0.1 * core.mv.x.dual()      # Point-
iron_outer_lengths = np.array([1.0, 0.85])
iron_inner_lengths = np.array([1.0, 0.62])
copper_centre = core.origin + 0.15 * core.mv.x.dual()         # Point-
copper_outer_radius, copper_inner_radius = 0.32, 0.20
relative_permeability, conductance, remanent_strength = 35.0, 90.0, 2.0
coil_centres = core.origin + core.mv.x.dual() * np.array([0.76, 1.24])  # [2] Point-
coil_radius, peak_coil_current = 0.09, -0.55
coil_signs = np.array([1, -1])
pulse_start, pulse_rise, pulse_hold, pulse_fall = 0.5, 0.8, 2.0, 3.4


def material_scene(n_across: int, n_up: int, n_frames: int):
    """Smooth transitions place the permanent magnet, iron core, copper ring and coil pair on one triangulated rectangle."""
    topology, coordinates = shapes.box([
        np.linspace(-half_width, half_width, n_across),
        np.linspace(-half_height, half_height, n_up),
    ])
    vertices = (core.mv(core.Upright, coordinates) + core.mv.w).dual().field()  # Point-[V]
    axes = stack((core.mv.x, core.mv.y))                          # [2] Upright+
    centres = stack((magnet_centre, iron_outer_centre, iron_inner_centre))  # [3] Point-
    half_lengths = np.stack((magnet_half_lengths, iron_outer_lengths, iron_inner_lengths))
    clearances = half_lengths - (axes & (vertices - centres[:, None])).abs()  # [3, 2] Scalar+[V]
    box_clearance = where(clearances[:, 0] < clearances[:, 1], clearances[:, 0], clearances[:, 1])  # [3] Scalar+[V]
    box_transition = (box_clearance + interface_width) / (2 * interface_width)  # [3] Scalar+[V]
    box_transition = where(box_transition < 0, 0, where(box_transition > 1, 0 * box_transition + 1, box_transition))  # [3] Scalar+[V]
    boxes = box_transition.squared() * (3 - 2 * box_transition)   # [3] Scalar+[V]
    magnet, outer_iron, inner_iron = boxes                        # Scalar+[V], Scalar+[V], Scalar+[V]
    iron = outer_iron * (1 - inner_iron)                          # Scalar+[V]
    radius = (vertices - copper_centre).antinorm()                # Scalar+[V]
    ring_clearance = where(
        copper_outer_radius - radius < radius - copper_inner_radius,
        copper_outer_radius - radius,
        radius - copper_inner_radius,
    )                                                             # Scalar+[V]
    ring_transition = (ring_clearance + interface_width) / (2 * interface_width)  # Scalar+[V]
    ring_transition = where(ring_transition < 0, 0, where(ring_transition > 1, 0 * ring_transition + 1, ring_transition))  # Scalar+[V]
    copper = ring_transition.squared() * (3 - 2 * ring_transition)  # Scalar+[V]
    fractions = stack((magnet, iron, copper))                     # [3] Scalar+[V]
    permeability = 1 + (relative_permeability - 1) * iron         # Scalar+[V]
    conductivity = conductance * copper                           # Scalar+[V]
    remanence = remanent_strength * magnet * core.mv.zx.dual()    # Excitation-[V]
    distances = (vertices - coil_centres).antinorm()              # [2] Scalar+[V]
    coils = (-(distances / coil_radius).squared()).exp()          # [2] Scalar+[V]
    times = np.arange(n_frames) * frame_step
    rise = np.clip((times - pulse_start) / (pulse_rise - pulse_start), 0, 1)
    fall = np.clip((times - pulse_hold) / (pulse_fall - pulse_hold), 0, 1)
    pulse = rise ** 2 * (3 - 2 * rise) - fall ** 2 * (3 - 2 * fall)
    drive = peak_coil_current * pulse
    return topology, vertices, fractions, permeability, conductivity, remanence, coils, drive


def main() -> None:
    """Stream the coil-pulse field evolution to disk and save the Joule-heating relaxation curve."""
    from examples.electromagnetism.magnetic_materials import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, fractions, permeability, conductivity, remanence, coils, drive = material_scene(
        n_across, n_up, n_frames,
    )
    areas = core.triangle_areas(topology, vertices)               # Scalar+[F]
    dual_areas = topology.incidence[2, 0].mean(axis=0).adjugate() * areas  # Scalar+[V]
    gradients = core.simplex_gradients(topology, vertices)        # [3] Upright+[F]
    stiffness = core.magnetic_stiffness(topology, areas, gradients, permeability)  # Scalar+[V <- V]
    source = core.remanent_source(topology, areas, gradients, remanence)  # Scalar+[V]
    extension = topology.reading(topology.interior[0], 0).adjugate()  # Scalar+[V <- V_interior]
    coil_load = dual_areas * (coils / (dual_areas * coils).sites.sum() * coil_signs).sum(axis=0)  # Scalar+[V]

    states = []

    def recorded_states():
        for state in core.evolution(stiffness, source, dual_areas, conductivity, extension, coil_load, drive, frame_step):
            states.append(state)
            yield state

    movie = save_animation(
        render.field_animation(topology, vertices, fractions, coils, recorded_states()),
        "magnetic_materials", frame_duration_ms, colors=128,
    )
    potentials = stack([potential for potential, _ in states])    # [n_frames] Scalar+[V]
    currents = stack([current for _, current in states])          # [n_frames] Current-[V]
    heating = core.joule_heating(potentials, dual_areas, conductivity, frame_step)  # [n_frames - 1] Scalar+
    save_figure(render.relaxation_figure(heating, frame_step, pulse_fall), "magnetic_materials_relaxation")
    print(movie, flush=True)

    # checks
    assert potentials.isfinite().all()
    assert (heating >= 0).all()
    assert ((dual_areas * (core.mv.z & currents)).sites.sum().abs() < 1e-8).all()


if __name__ == "__main__":
    main()
