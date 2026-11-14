"""Scenario parameters and figure drivers cover wavy charged channels, straight-slit electrokinetic generators, and necked-capillary electrophoresis.

Electrolyte constants model aqueous potassium chloride at unit bulk Debye length. Wavy channels with phase-shifted wall charges extend the lubrication problem of Ghosal, Journal of Fluid Mechanics 459, 103 (2002), to full two-dimensional channels with resolved double layers, while straight-slit power conversion follows van der Heyden, Bonthuis, Stein, Meyer and Dekker, Nano Letters 6, 2232 (2006).
"""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.topology import Topology
from examples.fluids.electrokinetics import core

mv = core.mv
# Potassium and chloride valences, bulk concentrations for unit Debye length, and dimensionless aqueous mobilities.
valences = np.array([1, -1])
bulk = np.array([0.5, 0.5])
mobilities = np.array([4.27, 4.43])
viscosity = 1.0
n_across, n_newton_steps = 41, 20

# Three wavy channels sharing one periodic geometry with surface-charge waves shifted in phase behind the width wave.
n_stations, wavelength = 64, 16.0
mean_half_width, half_width_swing, charge_swing = 2.0, 0.6, 2.0
phases = np.array([0.0, np.pi / 2, np.pi])
n_channels = len(phases)

# Straight-slit generator parameter grid over half widths and negative surface-charge densities.
n_widths, n_charges = 28, 32
half_widths = np.geomspace(0.02, 10.0, n_widths)
surface_charges = -np.geomspace(0.3, 1000.0, n_charges)
n_generator_columns = 3

# Wide necked capillary and four dye-band runs comparing pressure-driven and electro-osmotic transport.
capillary_length, capillary_half_width, neck_swing, capillary_charge = 60.0, 10.0, 3.0, -2.0
n_capillary_stations, n_capillary_across = 192, 121
band_middle, band_width = 6.0, 0.8
dye_mobility, dye_diffusivity = 0.4, 0.02
run_drives, dye_charges = np.array([0, 1, 1, 1]), np.array([0, 0, 1, -1])
n_runs = len(run_drives)
dye_dt, n_dye_frames, dye_stride = 0.1, 100, 3
frame_duration_ms = 40

# Neutral dye stripes advected through the three wavy channels under battery pumping.
stripe_spacing, stripe_sharpness = 8.0, 8
pumped_diffusivity = 0.005
pumped_dt, n_pumped_frames, pumped_stride = 0.02, 100, 4

pump, battery = mv.xv, mv.xt                                                  # [] Drive+, Drive+
experiments = stack((pump, battery))                                          # [2] Drive+


# --- plumbing -------------------------------------------------------------------------
def generators() -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Logarithmic sweeps of `half_widths` and `surface_charges` define `[n_widths, n_charges]` short periodic slit strips, vertex wall charges, and slit lengths."""
    widths = np.multiply.outer(
        np.multiply.outer(half_widths, np.ones(n_charges)), np.ones(n_generator_columns)
    )                                                                         # [n_widths, n_charges, n_generator_columns]
    lengths = 2 * widths[..., 0]                                              # [n_widths, n_charges]
    charges = np.multiply.outer(surface_charges, np.ones(n_generator_columns))  # [n_charges, n_generator_columns]
    return (
        *core.strip(n_generator_columns, n_across, lengths, widths),
        core.wall_charge(n_across, lengths, widths, charges),
        lengths,
    )


def channels(
    n_stations: int,
    n_across: int,
) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sinusoidal width modulation over `n_stations` stations and `n_channels` phase-shifted charge waves define the shared periodic channel strip and vertex wall charges."""
    angles = 2 * np.pi * np.arange(n_stations) / n_stations
    widths = mean_half_width + half_width_swing * np.sin(angles)
    wall_charges = charge_swing * np.sin(angles + phases[:, None])            # [n_channels, n_stations]
    return (
        *core.strip(n_stations, n_across, np.array(wavelength), widths),
        core.wall_charge(n_across, np.array(wavelength), widths, wall_charges),
        wall_charges,
    )


def capillary(
    swing: float,
) -> tuple[Topology, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cosine half-width modulation of amplitude `swing` forms a uniformly charged capillary widest at the ends and narrowest at its midpoint neck."""
    angles = 2 * np.pi * np.arange(n_capillary_stations) / n_capillary_stations
    widths = capillary_half_width + swing * np.cos(angles)
    wall_charges = np.full(n_capillary_stations, capillary_charge)
    return (
        *core.strip(n_capillary_stations, n_capillary_across, np.array(capillary_length), widths),
        core.wall_charge(n_capillary_across, np.array(capillary_length), widths, wall_charges),
        wall_charges,
    )


# --- math -----------------------------------------------------------------------------
def generator_efficiency() -> core.Scalar:
    """Two-drive Onsager responses across the `[n_widths, n_charges]` slit grid determine each slit's maximum power-conversion efficiency (`[n_widths, n_charges] Scalar+`)."""
    topology, coordinates, edge_shifts, lower_wall, upper_wall, charges, lengths = generators()
    vertices, edges = core.strip_placement(topology, coordinates, edge_shifts)
    areas, sides, gradients, outflow, circulation, dual_areas = core.channel_operators(topology, edges)
    wall_charges = mv.scalar(charges[..., None]).field()                      # [n_widths, n_charges] Scalar+[V]
    potentials = core.double_layer(outflow, dual_areas, wall_charges, valences, bulk, n_newton_steps)
    _, _, channel_fluxes, *_ = core.transport(
        topology, areas, sides, gradients, circulation, dual_areas, potentials, valences, bulk,
        viscosity, mobilities, lengths, lower_wall, upper_wall, experiments[:, None, None],
    )
    readings = experiments[:, None, None, None] & channel_fluxes[None]        # [2, 2, n_widths, n_charges] Scalar+
    coupling = readings[0, 1].squared() / (readings[0, 0] * readings[1, 1])   # [n_widths, n_charges] Scalar+
    return coupling / (1 + (1 - coupling).square_root()).squared()


def capillary_transport(swing: float):
    """Pressure-driven and electro-osmotic solves in a capillary of neck amplitude `swing` normalize both drives to unit mean speed and stream `n_runs` dye-band evolutions."""
    topology, coordinates, edge_shifts, lower_wall, upper_wall, charges, wall_charges = capillary(swing)
    vertices, edges = core.strip_placement(topology, coordinates, edge_shifts)
    areas, sides, gradients, outflow, circulation, dual_areas = core.channel_operators(topology, edges)
    potentials = core.double_layer(
        outflow, dual_areas, mv.scalar(charges[:, None]).field(), valences, bulk, n_newton_steps)
    _, induced_potentials, channel_fluxes, velocities, field, _ = core.transport(
        topology, areas, sides, gradients, circulation, dual_areas, potentials, valences, bulk,
        viscosity, mobilities, np.array(capillary_length), lower_wall, upper_wall, experiments,
    )
    mean_speeds = (pump & channel_fluxes) / (2 * capillary_half_width)        # [2] Scalar+
    band = np.exp(-((coordinates[:, 0] - band_middle) / band_width) ** 2 / 2)
    dye = np.ones(n_runs) * mv.scalar(band[:, None]).field()                  # [n_runs] Scalar+[V]
    velocities, field = velocities / mean_speeds, field / mean_speeds         # [2] Point-[F], Drive+[F]
    density = (valences * bulk * (-valences * potentials[..., None]).exp()).sum(axis=-1)  # Scalar+[V]
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    flows = (corners.sum(axis=0).adjugate() * (velocities * (areas / 3))) / dual_areas    # [2] Point-[V]
    axial_fields = mv.x & experiments.commutator(core.time)                   # [2] Scalar+
    drive_potentials = (induced_potentials - axial_fields * (mv.x & vertices)) / mean_speeds  # [2] Scalar+[V]
    falls = axial_fields * capillary_length / mean_speeds                     # [2] Scalar+
    frames = core.dye_transport(
        topology, areas, gradients, velocities[run_drives], field[run_drives], dye_charges,
        dye_mobility, dye_diffusivity, dye, dye_dt, n_dye_frames, dye_stride,
    )
    return (
        topology, vertices, dual_areas, wall_charges, density, flows,
        drive_potentials, falls, experiments / mean_speeds, velocities, dye, frames,
    )


def channel_flows(n_stations: int, n_across: int):
    """Battery-driven electro-osmosis across the `n_channels` wavy strips determines double-layer charge densities, vertex speeds, potentials, entropy production, fluxes, equilibrium forces, and advected dye stripes."""
    topology, coordinates, edge_shifts, lower_wall, upper_wall, charges, wall_charges = channels(n_stations, n_across)
    vertices, edges = core.strip_placement(topology, coordinates, edge_shifts)
    areas, sides, gradients, outflow, circulation, dual_areas = core.channel_operators(topology, edges)
    wall_charge_field = mv.scalar(charges[..., None]).field()                 # [n_channels] Scalar+[V]
    potentials = core.double_layer(outflow, dual_areas, wall_charge_field, valences, bulk, n_newton_steps)
    density = (valences * bulk * (-valences * potentials[..., None]).exp()).sum(axis=-1)  # [n_channels] Scalar+[V]
    drives = battery * np.ones(n_channels)                                    # [n_channels] Drive+
    stream, induced_potentials, channel_fluxes, velocities, field, conductivity = core.transport(
        topology, areas, sides, gradients, circulation, dual_areas, potentials, valences, bulk,
        viscosity, mobilities, np.array(wavelength), lower_wall, upper_wall, drives,
    )
    viscous_heat, joule_heat = core.entropy_production(
        topology, areas, circulation, dual_areas, viscosity, stream, field, conductivity)
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    speeds = (corners.sum(axis=0).adjugate() * (velocities * (areas / 3))) / dual_areas   # [n_channels] Point-[V]
    osmotic_force, electric_force = core.equilibrium_forces(
        topology, areas, gradients, dual_areas, potentials[0], valences, bulk)
    axial_field = mv.x & battery.commutator(core.time)                        # [] Scalar+
    drive_potentials = induced_potentials - axial_field * (mv.x & vertices)   # [n_channels] Scalar+[V]
    falls = axial_field * wavelength * np.ones(n_channels)                    # [n_channels] Scalar+
    first_speed = (pump & channel_fluxes[0]) / (2 * mean_half_width)          # [] Scalar+
    stripes = ((1 + np.cos(2 * np.pi * coordinates[:, 0] / stripe_spacing)) / 2) ** stripe_sharpness
    dye = np.ones(n_channels) * mv.scalar(stripes[:, None]).field()           # [n_channels] Scalar+[V]
    stripe_frames = core.dye_transport(
        topology, areas, gradients, velocities / first_speed, field / first_speed,
        np.zeros(n_channels), dye_mobility, pumped_diffusivity, dye, pumped_dt, n_pumped_frames, pumped_stride,
    )
    return (
        topology, vertices, dual_areas, wall_charges, density, speeds, drive_potentials,
        falls, viscous_heat + joule_heat, channel_fluxes, osmotic_force, electric_force, stripe_frames,
    )


def main() -> None:
    """Renders all wavy-channel, slit-generator, and capillary-electrophoresis figures and animations and checks physical balances."""
    from examples.util.animation import save_animation, save_figure
    from examples.fluids.electrokinetics import render

    (
        channel_topology, channel_vertices, channel_dual_areas, wall_charges, density, speeds,
        drive_potentials, falls, heat, channel_fluxes, osmotic_force, electric_force, stripe_frames,
    ) = channel_flows(n_stations, n_across)
    save_figure(
        render.channels_figure(channel_topology, channel_vertices, n_across, wavelength, wall_charges),
        "electrokinetics_channels",
    )
    save_figure(
        render.flow_figure(
            channel_topology, channel_vertices, n_across, wavelength,
            density, speeds, drive_potentials, falls, wall_charges,
        ),
        "electrokinetics_flow",
    )
    save_figure(
        render.heat_figure(
            channel_topology, channel_vertices, n_across, wavelength,
            heat, drive_potentials, falls, wall_charges,
        ),
        "electrokinetics_heat",
    )
    save_animation(
        render.pumping_animation(
            channel_topology, channel_vertices, n_across, wavelength,
            density, speeds, drive_potentials, falls, wall_charges, stripe_frames,
        ),
        "electrokinetics_pumping",
        frame_duration_ms,
    )
    efficiencies = generator_efficiency()
    save_figure(render.efficiency_figure(half_widths, surface_charges, efficiencies), "electrokinetics_efficiency")

    (
        capillary_topology, capillary_vertices, capillary_dual_areas, capillary_walls, capillary_density,
        capillary_flows, capillary_potentials, capillary_falls, _, capillary_velocities, initial_dye, dye_frames,
    ) = capillary_transport(neck_swing)
    save_figure(
        render.profiles_figure(capillary_topology, capillary_vertices, n_capillary_across, capillary_velocities),
        "electrophoresis_profiles",
    )
    final_dye = initial_dye

    def tracked_frames():
        nonlocal final_dye
        for state in dye_frames:
            final_dye = state
            yield state

    save_animation(
        render.dyes_animation(
            capillary_topology, capillary_vertices, n_capillary_across, capillary_length,
            capillary_walls, capillary_density, capillary_flows, capillary_potentials, capillary_falls, tracked_frames(),
        ),
        "electrophoresis",
        frame_duration_ms,
    )

    # checks
    assert (mv.y & (osmotic_force + electric_force)).abs().max() < 0.1 * (mv.y & osmotic_force).abs().max()
    pumped_rates = pump & channel_fluxes
    assert pumped_rates[0] > 0 and pumped_rates[2] < 0 and pumped_rates[1].abs() < 0.05 * pumped_rates[0]
    heat_power = (heat * channel_dual_areas).sites.sum()
    electric_power = (battery & channel_fluxes) * wavelength
    assert ((heat_power - electric_power).abs() < 1e-10 * electric_power).all()
    assert (efficiencies.max() - 0.064).abs() < 3e-3
    initial_totals = (capillary_dual_areas * initial_dye).sites.sum()
    assert (((capillary_dual_areas * final_dye).sites.sum() - initial_totals).abs() < 1e-10 * initial_totals).all()


if __name__ == "__main__":
    main()
