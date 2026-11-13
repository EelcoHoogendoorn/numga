"""One magnetic-material step exercises the common solve and its drawing paths."""
import numpy as np
from numga import stack
from examples.electromagnetism.magnetic_materials import core, scenarios, render
from examples.util.animation import save_animation


def test_material_field_and_pulse_step():
    topology, vertices, fractions, permeability, conductivity, remanence, coils, _ = scenarios.scene(9, 7, 2)
    areas, cells, gradients, response, stiffness, source, mass, extension = core.medium(
        topology, vertices, permeability, conductivity, remanence)
    coil_shape = coils / (cells * coils).sites.sum()
    coil = cells * (coil_shape * scenarios.coil_signs).sum(axis=0)
    drive = np.array([0., scenarios.coil_current])
    states = list(core.evolution(stiffness, source, mass, extension, coil, drive, scenarios.frame_step))
    corners = topology.incidence[2, 0]
    hat_fields = gradients ^ core.mv.z
    pole_readings = gradients | core.mv.xyz
    permanent = corners.mean(axis=0) * remanence
    frames = []
    for previous, potential, applied in zip([states[0], *states[:-1]], states, drive):
        rate = (potential - previous) / scenarios.frame_step
        field = (hat_fields * (corners * potential)).sum(axis=0)
        magnetization = permanent + core.magnetic_response(field) - response(field)
        pole_charge = (corners.adjugate() * (areas * (pole_readings & magnetization))).sum(axis=0)
        pole_density = pole_charge / cells
        mean_rate = (mass * rate).sites.sum() / mass.sites.sum()
        current = conductivity * core.electric_response(-(rate - mean_rate) * core.mv.zt)
        frames.append((potential, pole_density, current, applied))
    save_animation(render.animate(topology, vertices, fractions, coils, scenarios.coil_current, frames),
                   "magnetic_materials_test", 40)
    render.relaxation(stack(states), mass, scenarios.frame_step, scenarios.pulse_fall)
    render.setup(topology, vertices, fractions, coils, scenarios.coil_current)
    assert np.all(potential.isfinite())
    assert np.all((field & response(field)) >= 0)
    assert np.all((cells * (core.mv.zt & current)).sites.sum().abs() < .01)
    assert np.all(pole_charge.sites.sum().abs() < .01)
