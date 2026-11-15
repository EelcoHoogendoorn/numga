"""Scene parameters, laminated strip topology, and PZT-5A actuator and sensor experiments."""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.sparse import Sparse, SparseExtensor
from numga.topology import Topology
from examples.electromagnetism.piezoelectric import core

length_unit = 1e-3
stress_unit = 1e10
permittivity_unit = 1e-8
field_unit = float(np.sqrt(stress_unit / permittivity_unit))
voltage_unit = length_unit * field_unit
force_unit = stress_unit * length_unit**2
charge_unit = force_unit * length_unit / voltage_unit

length, thickness, width = 30, 1, 5
n_length, n_thickness = 180, 12
applied_voltage, applied_force = 100 / voltage_unit, 0.06 / force_unit
sensor_voltage_scale = 10 / voltage_unit
n_frames, frame_duration_ms = 90, 40
displacement_scale, strain_scale = 100, 2e-5
gallery_colours = 64

# Reduced plane-stress constants for PZT-5A (Martins and Cordioli, Table A3),
# ordered transverse then parallel to the poling direction at fixed in-plane strain.
stretch_stiffnesses = np.array([73.38258, 64.01459]) * 1e9 / stress_unit
cross_stiffness = 28.18203 * 1e9 / stress_unit
shear_stiffness = 21.05263 * 1e9 / stress_unit
stretch_polarizations = np.array([-2.008341, 19.12233]) * field_unit / stress_unit
shear_polarization = 12.29474 * field_unit / stress_unit
permittivities = np.array([8.137619, 7.556942]) * 1e-9 / permittivity_unit


class Strip(Topology):
    """Triangulated bimorph strip topology carrying clamp, tip, electrode and bond vertex index sets."""

    def __init__(self, n_vertices: int, triangles_table: np.ndarray, clamp_nodes: np.ndarray,
                 tip_nodes: np.ndarray, bottom_nodes: np.ndarray, top_nodes: np.ndarray,
                 interface_nodes: np.ndarray) -> None:
        super().__init__(n_vertices, triangles_table)
        self.clamp_nodes, self.tip_nodes = clamp_nodes, tip_nodes
        self.bottom_nodes, self.top_nodes, self.interface_nodes = bottom_nodes, top_nodes, interface_nodes


def bimorph(strip_length: float, strip_thickness: float, n_along: int, n_across: int
            ) -> tuple[Strip, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Alternating grid diagonals split the strip into triangles with vertex coordinates (`[V, 2]`), layer poling signs (`[F, 1]`),
    top-electrode weights (`[V, 1]`) and trapezoidal tip-load weights (`[V, 1]`)."""
    horizontal = np.linspace(0, strip_length, n_along + 1)
    vertical = np.linspace(-strip_thickness / 2, strip_thickness / 2, n_across + 1)
    grid_x, grid_y = np.meshgrid(horizontal, vertical, indexing="ij")
    coordinates = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)     # [V, 2]

    nodes = np.arange((n_along + 1) * (n_across + 1)).reshape(n_along + 1, n_across + 1)
    lower_left, lower_right = nodes[:-1, :-1], nodes[1:, :-1]
    upper_left, upper_right = nodes[:-1, 1:], nodes[1:, 1:]
    parity = np.indices(lower_left.shape).sum(axis=0) % 2
    first = np.stack((lower_left, lower_right, np.where(parity, upper_left, upper_right)), axis=-1)
    second = np.stack((np.where(parity, lower_right, lower_left), upper_right, upper_left), axis=-1)
    triangles_table = np.stack((first, second), axis=-2).reshape(-1, 3)           # [F, 3]

    layer = np.broadcast_to(np.where(np.arange(n_across) < n_across // 2, -1, 1), lower_left.shape)
    layer_poling = np.repeat(layer.reshape(-1), 2)[:, None]                       # [F, 1]
    topology = Strip(
        nodes.size, triangles_table, nodes[0], nodes[-1], nodes[:, 0], nodes[:, -1], nodes[:, n_across // 2],
    )
    V, E, F = topology.n_elements
    top_weights = np.zeros((V, 1))                                                # [V, 1]
    top_weights[topology.top_nodes, 0] = 1
    tip_weights = np.zeros((V, 1))                                                # [V, 1]
    tip_weights[topology.tip_nodes, 0] = 1
    tip_weights[topology.tip_nodes[[0, -1]], 0] = 0.5
    tip_weights /= tip_weights.sum()
    return topology, coordinates, layer_poling, top_weights, tip_weights


def boundary_maps(topology: Strip, floating_nodes: np.ndarray) -> tuple[Sparse, Sparse]:
    """Unconstrained vertices and shared electrode potentials define prolongation maps (`Scalar+[V <- V_free], Scalar+[V <- V_potential]`)."""
    V, E, F = topology.n_elements
    free_nodes = np.setdiff1d(np.arange(V), topology.clamp_nodes)
    free_motion = topology.reading(free_nodes, 0).adjugate()                      # Scalar+[V <- V_free]
    interior_nodes = np.setdiff1d(np.arange(V), np.concatenate((topology.top_nodes, topology.bottom_nodes)))
    potential_nodes = np.concatenate((interior_nodes, floating_nodes))
    groups = np.concatenate((np.arange(len(interior_nodes)), np.full(len(floating_nodes), len(interior_nodes))))
    n_potentials = len(interior_nodes) + int(len(floating_nodes) > 0)
    free_voltage = topology.reading(potential_nodes, 0).adjugate() * Sparse.selection(groups, n_potentials)  # Scalar+[V <- V_potential]
    return free_motion, free_voltage


def pzt5a(directions: core.Plane) -> tuple[core.Elastic, core.Polarization, core.Permittivity]:
    """Plane-stress PZT-5A constants instantiate the constitutive maps for two orthogonal material planes (`[2] Plane+`)."""
    return core.constitutive(
        directions, stretch_stiffnesses, cross_stiffness, shear_stiffness,
        stretch_polarizations, shear_polarization, permittivities,
    )


def device() -> tuple[Strip, core.Point, core.Scalar, core.Plane, core.Scalar, core.Scalar, core.Scalar,
                      tuple[core.Elastic, core.Polarization, core.Permittivity],
                      tuple[SparseExtensor, SparseExtensor, SparseExtensor]]:
    """Standard bimorph geometry and PZT-5A constants produce the strip fields and assembled enthalpy blocks."""
    topology, coordinates, layer_poling, top_weights, tip_weights = bimorph(length, thickness, n_length, n_thickness)
    vertices = core.mv(core.Plane, coordinates).dual().field()                    # Point-[V]
    poling = core.mv.scalar(layer_poling).field()                                 # Scalar+[F]
    top_electrode = core.mv.scalar(top_weights).field()                           # Scalar+[V]
    tip_distribution = core.mv.scalar(tip_weights).field()                        # Scalar+[V]
    areas, gradients = core.triangles(topology, vertices)
    material = pzt5a(stack((core.mv.x, core.mv.y)))
    blocks = core.assemble(topology, areas, gradients, *material, poling, width)
    return topology, vertices, areas, gradients, poling, top_electrode, tip_distribution, material, blocks


def actuation(topology: Strip, gradients: core.Plane, top_electrode: core.Scalar, tip_distribution: core.Scalar,
              blocks: tuple[SparseExtensor, SparseExtensor, SparseExtensor]
              ) -> tuple[core.Point, core.Scalar, core.Scalar]:
    """Voltage across the electrodes with zero tip load bends the strip into its actuator equilibrium."""
    free_motion, driven_voltage = boundary_maps(topology, np.array([], dtype=int))
    displacement, potential = core.equilibrium(
        *blocks, free_motion, driven_voltage, applied_voltage * top_electrode, 0 * core.mv.y * tip_distribution,
    )
    axial_strain = core.strain(topology, displacement, gradients)(core.axis) & core.axis  # [F] Scalar+
    return displacement, potential, axial_strain


def sensing(topology: Strip, gradients: core.Plane, top_electrode: core.Scalar, tip_distribution: core.Scalar,
            blocks: tuple[SparseExtensor, SparseExtensor, SparseExtensor]
            ) -> tuple[core.Point, core.Scalar, core.Scalar]:
    """Transverse tip force with a floating zero-charge top electrode bends the strip into its open-circuit sensor equilibrium."""
    free_motion, floating_voltage = boundary_maps(topology, topology.top_nodes)
    displacement, potential = core.equilibrium(
        *blocks, free_motion, floating_voltage, 0 * top_electrode, (core.mv.y * applied_force) * tip_distribution,
    )
    axial_strain = core.strain(topology, displacement, gradients)(core.axis) & core.axis  # [F] Scalar+
    return displacement, potential, axial_strain


def main() -> None:
    """Renders the actuator and sensor stills and drive-cycle animations and verifies reciprocity and charge balance."""
    from examples.electromagnetism.piezoelectric import render
    from examples.util.animation import save_animation, save_figure

    topology, vertices, areas, gradients, poling, top_electrode, tip_distribution, material, blocks = device()
    actuator = actuation(topology, gradients, top_electrode, tip_distribution, blocks)
    sensor = sensing(topology, gradients, top_electrode, tip_distribution, blocks)
    save_figure(
        render.draw_device(topology, vertices, areas, *actuator, displacement_scale, applied_voltage, strain_scale),
        "piezoelectric_actuator",
    )
    save_figure(
        render.draw_device(topology, vertices, areas, *sensor, displacement_scale, sensor_voltage_scale, strain_scale),
        "piezoelectric_sensor",
    )
    save_animation(
        render.animate(
            topology, vertices, areas, displacement_scale, applied_voltage, strain_scale,
            core.load_cycle(*actuator, n_frames),
        ),
        "piezoelectric_actuator",
        frame_duration_ms,
    )
    save_animation(
        render.animate(
            topology, vertices, areas, displacement_scale, sensor_voltage_scale, strain_scale,
            core.load_cycle(*sensor, n_frames),
        ),
        "piezoelectric_sensor",
        frame_duration_ms,
    )

    # checks
    mechanical, coupling, dielectric = blocks
    actuator_motion, actuator_potential, _ = actuator
    sensor_motion, sensor_potential, _ = sensor
    free_motion, driven_voltage = boundary_maps(topology, np.array([], dtype=int))
    short_motion, short_potential = core.equilibrium(
        *blocks, free_motion, driven_voltage, 0 * top_electrode, (core.mv.y * applied_force) * tip_distribution,
    )
    actuator_tip = ((core.mv.y & actuator_motion) * tip_distribution).sites.sum()
    short_charges = -(coupling.adjugate()(short_motion) + dielectric(short_potential))
    short_charge = (top_electrode & short_charges).sites.sum()
    sensor_charges = -(coupling.adjugate()(sensor_motion) + dielectric(sensor_potential))
    assert (actuator_tip / applied_voltage - short_charge / applied_force).abs() < 1e-9 * (actuator_tip / applied_voltage).abs()
    assert (top_electrode & sensor_charges).sites.sum().abs() < 1e-10 * short_charge.abs()
    assert ((topology.reading(topology.clamp_nodes, 0) * actuator_motion).antinorm() < 1e-12).all()


def pinned() -> None:
    """The gallery animation of the open-circuit piezoelectric bimorph bending sensor."""
    from examples.electromagnetism.piezoelectric import render
    from examples.util.animation import pin_animation

    topology, vertices, areas, gradients, poling, top_electrode, tip_distribution, material, blocks = device()
    sensor = sensing(topology, gradients, top_electrode, tip_distribution, blocks)
    pin_animation(
        render.animate(
            topology, vertices, areas, displacement_scale, sensor_voltage_scale, strain_scale,
            core.load_cycle(*sensor, n_frames),
        ),
        "piezoelectric",
        frame_duration_ms,
        colors=gallery_colours,
    )


if __name__ == "__main__":
    main()
