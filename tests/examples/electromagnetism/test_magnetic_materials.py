"""One magnetic-material step exercises the common solve and its drawing paths."""
import numpy as np
from numga import stack
from examples.electromagnetism.magnetic_materials import core, scenarios, render


def test_material_field_and_pulse_step():
    topology, vertices, fractions, permeability, conductivity, remanence, coils, _ = scenarios.scene(9, 7, 2)
    areas, cells, hats, response, stiffness, source, mass, extension = core.medium(
        topology, vertices, permeability, conductivity, remanence)
    coil_shape = coils / (cells * coils).sites.sum()
    coil = cells * (coil_shape * scenarios.coil_signs).sum(axis=0)
    drive = np.array([0., scenarios.coil_current])
    states = list(core.evolution(stiffness, source, mass, extension, coil, drive, scenarios.frame_step))
    potential = states[-1]
    rate = (potential - states[0]) / scenarios.frame_step
    field = (hats * (topology.incidence[2, 0] * potential)).sum(axis=0)
    induced = core.magnetic_response(field) - response(field)
    mean_rate = (mass * rate).sites.sum() / mass.sites.sum()
    current = conductivity * core.electric_response(-(rate - mean_rate) * core.mv.zt)
    next(render.animate(topology, vertices, fractions, coils, scenarios.coil_current,
                        [(potential, induced, current, drive[-1])]))
    render.relaxation(stack(states), mass, scenarios.frame_step, scenarios.pulse_fall)
    render.setup(topology, vertices, fractions, coils, scenarios.coil_current)
    assert np.all(potential.isfinite())
    assert np.all((field & response(field)) >= 0)
    assert np.all((cells * (core.mv.zt & current)).sites.sum().abs() < .01)
