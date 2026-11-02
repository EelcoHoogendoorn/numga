"""Channels whose width and wall charge vary along them, on a strip of triangles, and the best efficiency of slits as
generators over their width and wall charge.

The ions are those of potassium chloride. The channels are those of Ghosal's lubrication theory, Journal of Fluid
Mechanics 459, 103 (2002), here solved whole with the double layer resolved; the generators follow van der Heyden,
Bonthuis, Stein, Meyer and Dekker, Nano Letters 6, 2232 (2006).
"""
from __future__ import annotations

import numpy as np

from numga import stack
from examples.fluids.electrokinetics import core

mv = core.mv
# Potassium chloride, its bulk concentrations giving a Debye length of one, `(valences ** 2 * bulk).sum() == 1`, and the
# ions' mobilities through water, in units of the permittivity times a thermal volt over the viscosity; the viscosity.
valences = np.array([1, -1])
bulk = np.array([0.5, 0.5])
mobilities = np.array([4.27, 4.43])
viscosity = 1.0
# The vertices across a strip and the damped Newton steps that settle its double layer.
n_across, n_newton_steps = 41, 20
# The channels: their columns along one wavelength of the walls' waves; the mean half width in Debye lengths, its swing
# and the swing of the wall charge per area, whose mean is nothing; and the phases of the charge's wave behind the
# width's, one per channel.
n_stations, wavelength = 64, 16.0
mean_half_width, half_width_swing, charge_swing = 2.0, 0.6, 2.0
phases = np.array([0.0, np.pi / 2, np.pi])
# The generators: their half widths and wall charges, each a stretch of three columns as long as it is wide.
half_widths = np.geomspace(0.02, 10.0, 28)
surface_charges = -np.geomspace(0.3, 1000.0, 32)
n_generator_columns = 3

# A dye released as a band across a straight channel of uniform wall charge, many Debye lengths wide, so that its double
# layers are thin beside it: the channel's length, its half width and its charge per area, and its vertices along and
# across, four to a Debye length across; the band's middle along it and its width; the dye's mobility, the same for
# either charge, and its diffusivity, that of a large molecule; the runs' drives, by their place among the experiments,
# and the dyes' charges; the time step, the frames after the first and the steps between frames, and the frames'
# duration in milliseconds.
straight_length, straight_half_width, straight_charge = 120.0, 15.0, -2.0
n_straight_stations, n_straight_across = 192, 121
band_middle, band_width = 20.0, 4.0
dye_mobility, dye_diffusivity = 0.4, 0.05
run_drives, dye_charges = np.array([0, 1, 1, 1]), np.array([0, 0, 1, -1])
dye_dt, n_dye_frames, dye_stride = 0.2, 100, 3
frame_duration_ms = 40

pump, battery = mv.xv, mv.xt                                                                          # [] Drive+, Drive+
experiments = stack((pump, battery))                                                                  # [experiments] Drive+


# --- plumbing -------------------------------------------------------------------------
def generators() -> tuple[core.Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The generators' strips `[widths, charges]`: their topology, their vertices' places, their edges' reach past
    them, their lower and upper walls, the charge their walls put on their vertices, and their lengths."""
    widths = np.multiply.outer(np.multiply.outer(half_widths, np.ones(len(surface_charges))),
                               np.ones(n_generator_columns))                                          # [widths, charges, along]
    lengths = 2 * widths[..., 0]
    charges = np.multiply.outer(surface_charges, np.ones(n_generator_columns))                       # [charges, along]
    return (*core.strip(n_generator_columns, n_across, lengths, widths),
            core.wall_charge(n_across, lengths, widths, charges), lengths)


def channels() -> tuple[core.Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The channels' strip, one for all three: its topology, its vertices' places, its edges' reach past them, its
    lower and upper walls, the charge their walls put on its vertices `[channels, vertices]`, and the charge per area
    along their walls `[channels, stations]`."""
    turns = 2 * np.pi * np.arange(n_stations) / n_stations
    widths = mean_half_width + half_width_swing * np.sin(turns)
    wall_charges = charge_swing * np.sin(turns + phases[:, None])                                     # [channels, stations]
    return (*core.strip(n_stations, n_across, np.array(wavelength), widths),
            core.wall_charge(n_across, np.array(wavelength), widths, wall_charges), wall_charges)


def operators(topology: core.Topology, edges: core.Point):
    """A strip's triangles, their areas' sizes, sides and hat gradients; the field's outflow from each vertex's cell
    under each vertex's potential, and the circulation around each vertex's cell of each vertex's stream function; and
    its cells' areas."""
    double_areas, sides, gradients = core.triangles(topology, edges)
    areas = double_areas.norm() / 2                                                                   # [...] Scalar+[faces]
    # A potential rising by one at a corner drives the field down its hat's gradient, through time, and the vacuum
    # answers each drive with its excitation, its push on a unit charge swept through time, the permittivity one: each
    # hat's drive reading every other's excitation over the triangle is the field's outflow.
    pushes = -(gradients ^ mv.t)                                                                      # [..., corners] Drive+[faces]
    excitations = pushes.commutator(core.time).regressive(core.time)                                  # [..., corners] Flux-[faces]
    outflow = core.assembled(topology, (pushes[..., :, None] & excitations[..., None, :]) * areas[..., None, None])
    # A stream function rising by one at a corner moves the liquid along the facing side; each corner's velocity, read
    # as the plane of its circulation and paired with every other's over the triangle, is its circulation around the
    # other's cell.
    velocities = sides / (2 * areas[..., None])                                                       # [..., corners] Point-[faces]
    circulation = core.assembled(topology, (core.circulating(velocities[..., :, None]) & velocities[..., None, :])
                                 * areas[..., None, None])
    cells = ~topology.incidence[2, 0].sum(axis=0) * (areas / 3)                                       # [...] Scalar+[vertices]
    return areas, sides, gradients, outflow, circulation, cells


# --- math -----------------------------------------------------------------------------
def generator_efficiency() -> core.Scalar:
    """Each slit's best efficiency as a generator `[widths, charges] Scalar+`, at the load that gives the most."""
    topology, coordinates, shifts, lower, upper, charge, lengths = generators()
    vertices, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, outflow, circulation, cells = operators(topology, edges)
    charge = mv.scalar(charge[..., None]).field()
    potentials = core.double_layer(outflow, cells, charge, valences, bulk, n_newton_steps)
    _, _, carried, *_ = core.transport(topology, areas, sides, gradients, circulation, cells, potentials, valences, bulk,
                                       viscosity, mobilities, lengths, lower, upper, experiments[:, None, None])
    readings = experiments[:, None, None, None] & carried[None]                                      # [drives, experiments, widths, charges] Scalar+
    coupling = readings[0, 1].squared() / (readings[0, 0] * readings[1, 1])                          # [widths, charges] Scalar+
    # The square root of what is left, as the exponential of half its logarithm.
    root = ((1 - coupling).log() / 2).exp()                                                          # [widths, charges] Scalar+
    return coupling / (2 - coupling + 2 * root)


def straight():
    """The straight channel's strip: its topology, its vertices' places, its edges' reach past them, its lower and upper
    walls, the charge its walls put on its vertices `[vertices]`, and the charge per area along its walls `[stations]`."""
    widths = np.full(n_straight_stations, straight_half_width)
    wall_charges = np.full(n_straight_stations, straight_charge)
    return (*core.strip(n_straight_stations, n_straight_across, np.array(straight_length), widths),
            core.wall_charge(n_straight_across, np.array(straight_length), widths, wall_charges), wall_charges)


def dyed():
    """The straight channel's strip, its vertices and wall charges; the drives `[experiments] Drive+`, each scaled to
    move the liquid at a unit of speed on average, and the liquid's velocity under each `[experiments] Point-[faces]`;
    and the dye at each frame `[frames, runs] Scalar+[vertices]`: a band of it carried by the pressure drop's flow, by
    the field's electro-osmotic flow, and, charged either way, by that flow while it drifts under the field."""
    topology, coordinates, shifts, lower, upper, charge, wall_charges = straight()
    vertices, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, outflow, circulation, cells = operators(topology, edges)
    potentials = core.double_layer(outflow, cells, mv.scalar(charge[:, None]).field(), valences, bulk,
                                   n_newton_steps)
    _, _, carried, velocities, field, _ = core.transport(topology, areas, sides, gradients, circulation, cells,
                                                         potentials, valences, bulk, viscosity, mobilities,
                                                         np.array(straight_length), lower, upper, experiments)
    # The volume each experiment carries along per length, read by the pressure's drive, over the channel's width, is
    # the liquid's mean speed; the flow is linear in its drive, so each is scaled to a unit of it.
    speeds = (pump & carried) / (2 * straight_half_width)                     # [experiments] Scalar+
    band = np.exp(-((coordinates[:, 0] - band_middle) / band_width) ** 2 / 2)
    dye = np.ones(len(run_drives)) * mv.scalar(band[:, None]).field()        # [runs] Scalar+[vertices]
    velocities, field = velocities / speeds, field / speeds                  # [experiments] Point-[faces], Drive+[faces]
    return topology, vertices, wall_charges, experiments / speeds, velocities, core.dyeing(
        topology, areas, gradients, velocities[run_drives], field[run_drives], dye_charges, dye_mobility,
        dye_diffusivity, dye, dye_dt, n_dye_frames, dye_stride)


def channel_flows():
    """The channels' strip and walls, their liquid's charge density `[channels] Scalar+[vertices]`, its velocity at
    each vertex `[channels] Point-[vertices]`, the heat made `[channels] Scalar+[vertices]`, what each carries
    `[channels] Flux-`, and the first channel's double layer's own pushes on the liquid, its osmotic pressure's and its
    field's, `Point-[vertices]` each."""
    topology, coordinates, shifts, lower, upper, charge, wall_charges = channels()
    vertices, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, outflow, circulation, cells = operators(topology, edges)
    charge = mv.scalar(charge[..., None]).field()
    potentials = core.double_layer(outflow, cells, charge, valences, bulk, n_newton_steps)
    density = (valences * bulk * (-valences * potentials[..., None]).exp()).sum(axis=-1)            # [channels] Scalar+[vertices]
    drives = battery * np.ones(len(phases))                                                          # [channels] Drive+
    stream, _, carried, velocities, field, conductivity = core.transport(
        topology, areas, sides, gradients, circulation, cells, potentials, valences, bulk, viscosity, mobilities,
        np.array(wavelength), lower, upper, drives)
    viscous, joule = core.entropy_production(topology, areas, circulation, cells, viscosity, stream, field, conductivity)
    # Each vertex's velocity, its triangles' weighed by their areas.
    speeds = (~topology.incidence[2, 0].sum(axis=0) * (velocities * (areas / 3))) / cells             # [channels] Point-[vertices]
    osmotic, electric = core.equilibrium_pushes(topology, areas, gradients, cells, potentials[0], valences, bulk)
    return topology, vertices, wall_charges, density, speeds, viscous + joule, carried, osmotic, electric


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.electrokinetics import render

    topology, vertices, wall_charges, density, speeds, heat, carried, osmotic, electric = channel_flows()
    save_figure(render.draw_mesh(topology, vertices, n_across, wavelength, wall_charges[0]), "electrokinetics_channel")
    save_figure(render.draw_charge(topology, vertices, n_across, wavelength, density, wall_charges),
                "electrokinetics_charge")
    save_figure(render.draw_balance(topology, vertices, n_across, wavelength, osmotic, electric, wall_charges[0]),
                "electrokinetics_balance")
    save_figure(render.draw_flow(topology, vertices, n_across, wavelength, speeds, wall_charges),
                "electrokinetics_flow")
    save_figure(render.draw_heat(topology, vertices, n_across, wavelength, heat, wall_charges), "electrokinetics_heat")
    save_figure(render.draw_efficiency(half_widths, surface_charges, generator_efficiency()), "electrokinetics_efficiency")
    topology, vertices, wall_charges, _, velocities, frames = dyed()
    save_figure(render.draw_profiles(topology, vertices, n_straight_across, velocities), "electrophoresis_profiles")
    save_animation(render.animate_dyes(topology, vertices, n_straight_across, straight_length, wall_charges, frames),
                   "electrophoresis", frame_duration_ms)
