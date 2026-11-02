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

# A dye released as a band across a channel of uniform wall charge, many Debye lengths wide, so that its double layers
# are thin beside it, and narrowing to a neck midway along it: the channel's length, its mean half width, the swing of
# its half width and its charge per area, and its vertices along and
# across, four to a Debye length across; the band's middle along it and its width; the dye's mobility, the same for
# either charge, and its diffusivity, that of a large molecule; the runs' drives, by their place among the experiments,
# and the dyes' charges; the time step, the frames after the first and the steps between frames, and the frames'
# duration in milliseconds.
capillary_length, capillary_half_width, neck_swing, capillary_charge = 60.0, 10.0, 3.0, -2.0
n_capillary_stations, n_capillary_across = 192, 121
band_middle, band_width = 6.0, 0.8
dye_mobility, dye_diffusivity = 0.4, 0.02
run_drives, dye_charges = np.array([0, 1, 1, 1]), np.array([0, 0, 1, -1])
dye_dt, n_dye_frames, dye_stride = 0.1, 100, 3
frame_duration_ms = 40

# Dye stripes pumped through the channels by the battery, which is scaled so that the first channel pumps at a unit of
# speed: their spacing along the channels and how sharp they are, the dye's diffusivity, and the time step, the frames
# after the first and the steps between frames, half a wavelength's travel at that speed.
stripe_spacing, stripe_sharpness = 8.0, 8
pumped_diffusivity = 0.005
pumped_dt, n_pumped_frames, pumped_stride = 0.02, 100, 4

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
    # gives each drive its excitation, its push on a unit charge swept through time, the permittivity one: each
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


def capillary(swing: float):
    """The capillary's strip, its half width swinging by `swing` about its mean, widest at the ends of its period and
    narrowest midway along it: its topology, its vertices' places, its edges' reach past them, its lower and upper
    walls, the charge its walls put on its vertices `[vertices]`, and the charge per area along its walls `[stations]`."""
    # Widest at the ends of the period, narrowest midway along it.
    turns = 2 * np.pi * np.arange(n_capillary_stations) / n_capillary_stations
    widths = capillary_half_width + swing * np.cos(turns)
    wall_charges = np.full(n_capillary_stations, capillary_charge)
    return (*core.strip(n_capillary_stations, n_capillary_across, np.array(capillary_length), widths),
            core.wall_charge(n_capillary_across, np.array(capillary_length), widths, wall_charges), wall_charges)


def dyed(swing: float):
    """The capillary's strip, its half width swinging by `swing`, its vertices and wall charges; the liquid's charge density `Scalar+[vertices]`, its double
    layers; the drives `[experiments] Drive+`, each scaled to move the liquid at a unit of speed on average, and the
    liquid's velocity under each, at each vertex `[experiments] Point-[vertices]`; the potential each sets up
    `[experiments] Scalar+[vertices]` and how far it falls over the channel's length `[experiments] Scalar+`; the
    liquid's velocity in each triangle `[experiments] Point-[faces]`; and the dye at each frame `[frames, runs] Scalar+[vertices]`: a band of it carried by the pressure drop's flow, by
    the field's electro-osmotic flow, and, charged either way, by that flow while it drifts under the field."""
    topology, coordinates, shifts, lower, upper, charge, wall_charges = capillary(swing)
    vertices, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, outflow, circulation, cells = operators(topology, edges)
    potentials = core.double_layer(outflow, cells, mv.scalar(charge[:, None]).field(), valences, bulk,
                                   n_newton_steps)
    _, added, carried, velocities, field, _ = core.transport(topology, areas, sides, gradients, circulation, cells,
                                                         potentials, valences, bulk, viscosity, mobilities,
                                                         np.array(capillary_length), lower, upper, experiments)
    # The volume each experiment carries along per length, read by the pressure's drive, over the channel's width, is
    # the liquid's mean speed; the flow is linear in its drive, so each is scaled to a unit of it.
    speeds = (pump & carried) / (2 * capillary_half_width)                     # [experiments] Scalar+
    band = np.exp(-((coordinates[:, 0] - band_middle) / band_width) ** 2 / 2)
    dye = np.ones(len(run_drives)) * mv.scalar(band[:, None]).field()        # [runs] Scalar+[vertices]
    velocities, field = velocities / speeds, field / speeds                  # [experiments] Point-[faces], Drive+[faces]
    density = (valences * bulk * (-valences * potentials[..., None]).exp()).sum(axis=-1)   # Scalar+[vertices]
    # Each vertex's velocity, its triangles' weighed by their areas.
    flows = (~topology.incidence[2, 0].sum(axis=0) * (velocities * (areas / 3))) / cells   # [experiments] Point-[vertices]
    # The potential each drive sets up, the applied one falling steadily along the channel and the one the liquid adds,
    # and how far it falls over the channel's length.
    along = mv.x & experiments.commutator(core.time)                         # [experiments] Scalar+
    drive_potentials = (added - along * (mv.x & vertices)) / speeds         # [experiments] Scalar+[vertices]
    falls = along * capillary_length / speeds                                # [experiments] Scalar+
    return topology, vertices, wall_charges, density, flows, drive_potentials, falls, experiments / speeds, velocities, core.dyeing(
        topology, areas, gradients, velocities[run_drives], field[run_drives], dye_charges, dye_mobility,
        dye_diffusivity, dye, dye_dt, n_dye_frames, dye_stride)


def channel_flows():
    """The channels' strip and walls, their liquid's charge density `[channels] Scalar+[vertices]`, its velocity at
    each vertex `[channels] Point-[vertices]`, the potential the battery sets up `[channels] Scalar+[vertices]` and how
    far it falls over a wavelength `[channels] Scalar+`, the heat made `[channels] Scalar+[vertices]`, what each carries
    `[channels] Flux-`, the first channel's double layer's own pushes on the liquid, its osmotic pressure's and its
    field's, `Point-[vertices]` each, and the stripes of dye the battery pumps through the channels at each frame
    `[frames, channels] Scalar+[vertices]`."""
    topology, coordinates, shifts, lower, upper, charge, wall_charges = channels()
    vertices, edges = core.placed(topology, coordinates, shifts)
    areas, sides, gradients, outflow, circulation, cells = operators(topology, edges)
    charge = mv.scalar(charge[..., None]).field()
    potentials = core.double_layer(outflow, cells, charge, valences, bulk, n_newton_steps)
    density = (valences * bulk * (-valences * potentials[..., None]).exp()).sum(axis=-1)            # [channels] Scalar+[vertices]
    drives = battery * np.ones(len(phases))                                                          # [channels] Drive+
    stream, added, carried, velocities, field, conductivity = core.transport(
        topology, areas, sides, gradients, circulation, cells, potentials, valences, bulk, viscosity, mobilities,
        np.array(wavelength), lower, upper, drives)
    viscous, joule = core.entropy_production(topology, areas, circulation, cells, viscosity, stream, field, conductivity)
    # Each vertex's velocity, its triangles' weighed by their areas.
    speeds = (~topology.incidence[2, 0].sum(axis=0) * (velocities * (areas / 3))) / cells             # [channels] Point-[vertices]
    osmotic, electric = core.equilibrium_pushes(topology, areas, gradients, cells, potentials[0], valences, bulk)
    # The potential the battery sets up, its steady fall along the channel and what the liquid adds, and its fall over
    # a wavelength.
    along = mv.x & battery.commutator(core.time)                                                    # [] Scalar+
    drive_potentials = added - along * (mv.x & vertices)                                            # [channels] Scalar+[vertices]
    falls = along * wavelength * np.ones(len(phases))                                               # [channels] Scalar+
    # Neutral dye in stripes across the channels, carried by the liquid at the battery's pace, scaled so that the first
    # channel's mean speed, its volume per length read by the pump over its mean width, is one.
    first_speed = (pump & carried[0]) / (2 * mean_half_width)                                       # [] Scalar+
    stripes = ((1 + np.cos(2 * np.pi * coordinates[:, 0] / stripe_spacing)) / 2) ** stripe_sharpness
    dye = np.ones(len(phases)) * mv.scalar(stripes[:, None]).field()                                # [channels] Scalar+[vertices]
    pumped = core.dyeing(topology, areas, gradients, velocities / first_speed, field / first_speed, np.zeros(len(phases)),
                         dye_mobility, pumped_diffusivity, dye, pumped_dt, n_pumped_frames, pumped_stride)
    return (topology, vertices, wall_charges, density, speeds, drive_potentials, falls, viscous + joule, carried, osmotic,
            electric, pumped)


if __name__ == "__main__":
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.electrokinetics import render

    (topology, vertices, wall_charges, density, speeds, drive_potentials, falls, heat, carried, osmotic, electric,
     pumped) = channel_flows()
    save_figure(render.draw_channels(topology, vertices, n_across, wavelength, wall_charges), "electrokinetics_channels")
    save_figure(render.draw_flow(topology, vertices, n_across, wavelength, density, speeds, drive_potentials, falls,
                                 wall_charges),
                "electrokinetics_flow")
    save_figure(render.draw_heat(topology, vertices, n_across, wavelength, heat, drive_potentials, falls, wall_charges),
                "electrokinetics_heat")
    save_animation(render.animate_pumping(topology, vertices, n_across, wavelength, density, speeds, drive_potentials,
                                          falls, wall_charges, pumped), "electrokinetics_pumping", frame_duration_ms)
    save_figure(render.draw_efficiency(half_widths, surface_charges, generator_efficiency()), "electrokinetics_efficiency")
    topology, vertices, wall_charges, density, flows, drive_potentials, falls, _, velocities, frames = dyed(neck_swing)
    save_figure(render.draw_profiles(topology, vertices, n_capillary_across, velocities), "electrophoresis_profiles")
    save_animation(render.animate_dyes(topology, vertices, n_capillary_across, capillary_length, wall_charges, density,
                                       flows, drive_potentials, falls, frames),
                   "electrophoresis", frame_duration_ms)
