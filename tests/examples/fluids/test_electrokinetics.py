"""Electrokinetic transport on a mesh of triangles: the channels' flows on the smallest strip, and one frame of their
pumping drawn."""

from __future__ import annotations

from examples.fluids.electrokinetics import render, scenarios


def test_the_channels_flow_and_one_frame_draws():
    n_stations, n_across = 4, 3
    (topology, vertices, wall_charges, density, speeds, drive_potentials, falls, heat, carried, osmotic, electric,
     pumped) = scenarios.channel_flows(n_stations, n_across)
    next(render.animate_pumping(topology, vertices, n_across, scenarios.wavelength, density, speeds, drive_potentials,
                                falls, wall_charges, pumped))
