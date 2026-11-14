"""A coil pulse disturbs a permanent magnet's field in soft iron and copper."""
from __future__ import annotations

import numpy as np
from numga import stack, where
from examples.util import shapes
from examples.electromagnetism.magnetic_materials import core

n_across, n_up, n_frames = 97, 73, 100
half_width, half_height = 1.65, 1.25
frame_step = .06
interface_width = .035
magnet_centre = core.origin - .8 * core.mv.x.dual()
magnet_half_lengths = np.array([.15, .36])
iron_outer_centre = core.origin + .1 * core.mv.x.dual()
iron_inner_centre = core.origin - .1 * core.mv.x.dual()
iron_outer_lengths = np.array([1., .85])
iron_inner_lengths = np.array([1., .62])
copper_centre = core.origin + .15 * core.mv.x.dual()
copper_outer_radius, copper_inner_radius = .32, .20
relative_permeability, conductance, remanent_strength = 35., 90., 2.
coil_centres = core.origin + core.mv.x.dual() * np.array([.76, 1.24])
coil_radius, coil_current = .09, -.55
coil_signs = np.array([1, -1])
pulse_start, pulse_rise, pulse_hold, pulse_fall = .5, .8, 2., 3.4


def scene(across: int, up: int, frames: int):
    """Place all material properties and the coil's two cross-sections on one mesh."""
    topology, coordinates = shapes.box([np.linspace(-half_width, half_width, across),
                                       np.linspace(-half_height, half_height, up)])
    spatial = core.ga.gatype.from_blades("x y")
    vertices = core.origin + core.mv(spatial, coordinates).dual().field()
    axes = stack((core.mv.x, core.mv.y))
    centres = stack((magnet_centre, iron_outer_centre, iron_inner_centre))
    half_lengths = np.stack((magnet_half_lengths, iron_outer_lengths, iron_inner_lengths))
    clearances = half_lengths - (axes & (vertices - centres[:, None])).abs()
    clearance = where(clearances[:, 0] < clearances[:, 1], clearances[:, 0], clearances[:, 1])
    transition = (clearance + interface_width) / (2 * interface_width)
    transition = where(transition < 0, 0, where(transition > 1, 0 * transition + 1, transition))
    boxes = transition.squared() * (3 - 2 * transition)
    magnet, outer, inner = boxes
    iron = outer * (1 - inner)
    radius = (vertices - copper_centre).antinorm()
    distance = where(copper_outer_radius - radius < radius - copper_inner_radius,
                     copper_outer_radius - radius, radius - copper_inner_radius)
    ring_transition = (distance + interface_width) / (2 * interface_width)
    ring_transition = where(ring_transition < 0, 0, where(ring_transition > 1, 0 * ring_transition + 1, ring_transition))
    copper = ring_transition.squared() * (3 - 2 * ring_transition)
    fractions = stack((magnet, iron, copper))
    permeability = 1 + (relative_permeability - 1) * iron
    conductivity = conductance * copper
    remanence = remanent_strength * magnet * core.mv.zx.dual()
    distances = (vertices - coil_centres).antinorm()
    coil_shape = (-(distances / coil_radius).squared()).exp()
    times = np.arange(frames) * frame_step
    rise = np.clip((times - pulse_start) / (pulse_rise - pulse_start), 0, 1)
    fall = np.clip((times - pulse_hold) / (pulse_fall - pulse_hold), 0, 1)
    pulse = rise ** 2 * (3 - 2 * rise) - fall ** 2 * (3 - 2 * fall)
    return topology, vertices, fractions, permeability, conductivity, remanence, coil_shape, coil_current * pulse


def run(across: int, up: int, frames: int, name: str):
    from examples.electromagnetism.magnetic_materials import render
    from examples.util.animation import save_animation, save_figure
    topology, vertices, fractions, permeability, conductivity, remanence, coils, drive = scene(across, up, frames)
    areas, cells, gradients, response, stiffness, source, mass, extension = core.medium(
        topology, vertices, permeability, conductivity, remanence)
    coil_shape = coils / (cells * coils).sites.sum()
    coil = cells * (coil_shape * coil_signs).sum(axis=0)
    # The remanence's source in the solve is its bound current along z through each vertex's cell.
    bound = core.sweep * (source / cells)
    prescribed = core.sweep * (coil / cells)
    states = list(core.evolution(stiffness, source, mass, extension, coil, drive, frame_step))
    def observed():
        for previous, potential, applied in zip([states[0], *states[:-1]], states, drive):
            rate = (potential - previous) / frame_step
            mean_rate = (mass * rate).sites.sum() / mass.sites.sum()
            electric = -(rate - mean_rate) * core.mv.zt
            current = conductivity * core.electric_response(electric)
            yield potential, bound + applied * prescribed + current
    movie = save_animation(render.animate(topology, vertices, fractions, coils, observed()), name, 40, colors=128)
    history = stack(states)
    save_figure(render.relaxation(history, mass, frame_step, pulse_fall), name + "_relaxation")
    print(movie, flush=True)
    # checks
    assert np.all(history.isfinite())
    return movie


if __name__ == "__main__":
    run(n_across, n_up, n_frames, "magnetic_materials")
