"""Coverage tests exercise electrokinetic transport and rendering on a minimal periodic channel strip."""

from __future__ import annotations

import matplotlib.pyplot as plt

from examples.fluids.electrokinetics import core, render, scenarios


def test_minimal_channel_flows_and_rendering():
    """Coupled double-layer, electro-osmotic flow, heat dissipation, and frame rendering succeed on a four-station strip."""
    n_stations, n_across = 4, 3
    (
        topology, vertices, dual_areas, wall_charges, density, speeds, drive_potentials,
        falls, heat, channel_fluxes, osmotic_force, electric_force, stripe_frames,
    ) = scenarios.channel_flows(n_stations, n_across)
    heat_power = (heat * dual_areas).sites.sum()
    electric_power = (scenarios.battery & channel_fluxes) * scenarios.wavelength
    assert ((heat_power - electric_power).abs() < 1e-10 * electric_power).all()
    assert (((core.mv.w & vertices.batch()) - 1).abs() < 1e-12).all()
    figure = render.flow_figure(
        topology, vertices, n_across, scenarios.wavelength,
        density, speeds, drive_potentials, falls, wall_charges,
    )
    plt.close(figure)
    frame = next(render.pumping_animation(
        topology, vertices, n_across, scenarios.wavelength,
        density, speeds, drive_potentials, falls, wall_charges, stripe_frames,
    ))
    assert frame.ndim == 3 and frame.shape[-1] == 3
