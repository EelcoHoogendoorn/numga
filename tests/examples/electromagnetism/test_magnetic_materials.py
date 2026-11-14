"""One magnetic-material step exercises the common solve and its drawing paths."""
import numpy as np
from numga import stack
from examples.electromagnetism.magnetic_materials import core, scenarios, render


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
    bound = core.sweep * (source / cells)
    prescribed = core.sweep * (coil / cells)
    frames = []
    for previous, potential, applied in zip([states[0], *states[:-1]], states, drive):
        rate = (potential - previous) / scenarios.frame_step
        field = (hat_fields * (corners * potential)).sum(axis=0)
        mean_rate = (mass * rate).sites.sum() / mass.sites.sum()
        current = conductivity * core.electric_response(-(rate - mean_rate) * core.mv.zt)
        frames.append((potential, bound + applied * prescribed + current))
    list(render.animate(topology, vertices, fractions, coils, frames))
    render.relaxation(stack(states), mass, scenarios.frame_step, scenarios.pulse_fall)
    render.setup(topology, vertices, fractions, coils)
    assert np.all(potential.isfinite())
    assert np.all((field & response(field)) >= 0)
    assert np.all((cells * (core.mv.zt & current)).sites.sum().abs() < .01)
    assert np.all(source.sites.sum().abs() < .01)
