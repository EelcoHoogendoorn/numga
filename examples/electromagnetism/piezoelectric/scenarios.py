"""A PZT bimorph driven through its electrodes and read as a bending sensor."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import stack
from examples.electromagnetism.piezoelectric import core

length_unit = 1e-3
stress_unit = 1e10
permittivity_unit = 1e-8
field_unit = np.sqrt(stress_unit / permittivity_unit)
voltage_unit = length_unit * field_unit
force_unit = stress_unit * length_unit**2
charge_unit = force_unit * length_unit / voltage_unit

length, thickness, width = 30, 1, 5
n_length, n_thickness = 180, 12
applied_voltage, applied_force = 100 / voltage_unit, 0.06 / force_unit
n_frames, duration_ms = 90, 40
displacement_scale, strain_scale = 100, 2e-5
# Gallery picture: this makes plots/piezoelectric.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
# The gallery's actuator: the colours of its palette.
gallery_colours = 64


def pzt5a(directions: core.Plane) -> tuple[core.Elastic, core.Polarization, core.Permittivity]:
    """PZT-5A under plane stress, with the second material plane across its poling direction.

    https://dael.euracoustics.org/bin/EAA/aaua_dl?document_id=52105
    These reduced stress-charge constants follow from the compliance data in Table A3,
    with zero stress through the width and permittivity measured at fixed in-plane strain.
    """
    # Entries follow the two material directions: transverse and parallel to poling.
    stretch_stiffness = np.array([73.38258, 64.01459]) * 1e9 / stress_unit
    cross_stiffness = 28.18203 * 1e9 / stress_unit
    shear_stiffness = 21.05263 * 1e9 / stress_unit
    stretch_polarization = np.array([-2.008341, 19.12233]) * field_unit / stress_unit
    shear_polarization = 12.29474 * field_unit / stress_unit
    permittivities = np.array([8.137619, 7.556942]) * 1e-9 / permittivity_unit
    return core.constitutive(directions, stretch_stiffness, cross_stiffness, shear_stiffness,
                            stretch_polarization, shear_polarization, permittivities)


def device():
    mesh, vertices, poling = core.strip(length, thickness, n_length, n_thickness)
    areas, gradients = core.triangles(mesh, vertices)
    material = pzt5a(stack((core.mv.x, core.mv.y)))
    blocks = core.assemble(mesh, areas, gradients, *material, poling, width)
    return mesh, vertices, areas, gradients, poling, material, blocks


def actuation(mesh: core.Strip, gradients: core.Plane, blocks):
    movement, voltage = core.boundary_maps(mesh, np.array([], dtype=int))
    displacement, potential = core.equilibrium(*blocks, movement, voltage,
                                             core.electrode_potential(mesh, applied_voltage), core.tip_load(mesh, 0))
    axial_strain = core.axis & core.strain(mesh, displacement, gradients)(core.axis)
    return displacement, potential, axial_strain


def sensing(mesh: core.Strip, gradients: core.Plane, blocks):
    movement, voltage = core.boundary_maps(mesh, mesh.top)
    displacement, potential = core.equilibrium(*blocks, movement, voltage,
                                             core.electrode_potential(mesh, 0), core.tip_load(mesh, applied_force))
    axial_strain = core.axis & core.strain(mesh, displacement, gradients)(core.axis)
    return displacement, potential, axial_strain


def cycle(displacement: core.Point, potential: core.Scalar, axial_strain: core.Scalar,
          n_frames: int) -> Iterator[tuple[core.Point, core.Scalar, core.Scalar]]:
    """A slow sinusoidal load sweep through equilibria of this linear material."""
    for phase in np.linspace(0, 2 * np.pi, n_frames, endpoint=False):
        amplitude = np.sin(phase)
        yield displacement * amplitude, potential * amplitude, axial_strain * amplitude


if __name__ == "__main__":
    from examples.electromagnetism.piezoelectric import render
    from examples.util.animation import save_animation, save_figure

    mesh, vertices, areas, gradients, poling, material, blocks = device()
    actuator = actuation(mesh, gradients, blocks)
    sensor = sensing(mesh, gradients, blocks)
    save_figure(render.draw_device(mesh, vertices, areas, *actuator, displacement_scale, applied_voltage, strain_scale),
                "piezoelectric_actuator")
    save_figure(render.draw_device(mesh, vertices, areas, *sensor, displacement_scale, 10 / voltage_unit, strain_scale),
                "piezoelectric_sensor")
    save_animation(render.animate(mesh, vertices, areas, cycle(*actuator, n_frames), displacement_scale, applied_voltage,
                                  strain_scale), "piezoelectric_actuator", duration_ms)
    save_animation(render.animate(mesh, vertices, areas, cycle(*sensor, n_frames), displacement_scale, 10 / voltage_unit,
                                  strain_scale), "piezoelectric_sensor", duration_ms)
    # Gallery picture: this makes plots/piezoelectric.gif for docs/extensors.md, tuned as a visual for the gallery. Don't change it; to experiment, copy it.
    save_animation(render.animate(mesh, vertices, areas, cycle(*actuator, n_frames), displacement_scale, applied_voltage,
                                  strain_scale), "piezoelectric", duration_ms, colors=gallery_colours)
