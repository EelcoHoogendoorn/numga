"""Piezoelectric continuum elasticity: a laminated strip as actuator and sensor.

Displacements are intrinsic directions; electric fields are planes through time. A material's
elastic, piezoelectric and dielectric maps give one electric enthalpy. Pulling their work back
through the finite-element readings gives equilibrium and charge balance. Metal electrodes
have a common potential; a floating electrode fixes total charge while its voltage responds.
"""

from __future__ import annotations

import numpy as np

from numga import NumpyContext
from numga.algebras import STA2D as ga
from numga.sparse import SparseExtensor

from examples.electromagnetism.piezoelectric.numerical import solve_coupled

# The spacetime of the strip's plane, x+y+t-: two positive spatial directions, one negative time direction.
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()
# Spatial planes read positions, displacements and polarization, their intrinsic complements.
Plane = ga.gatype.from_blades("x y")
Point = Plane.dual().output_gatype
# Electric fields are extrinsic planes through time; charge is an intrinsic amount.
Field = ga.gatype.from_blades("xt yt")
Amount = ga.gatype.antiscalar()
Strain = ga.gatype((Plane, Point))
Elastic = ga.gatype((Scalar, Point, Plane, Point, Plane))
Polarization = ga.gatype((Point, Point, Plane))
Permittivity = ga.gatype((Point, Field))
time = -mv.t.dual()
axis = mv.x.dual()
pole = mv.y.dual()


# --- math -----------------------------------------------------------------------------
def triangles(vertices: Point, faces: np.ndarray) -> tuple[Scalar, Plane]:
    """Each triangle's area, and each of its corners' shape-function gradients `[faces, corners]`."""
    corners = vertices.batch()[faces]
    # Each corner's opposite edge, from the next corner to the one after it.
    edges = corners[:, [1, 2, 0]] - corners[:, [2, 0, 1]]
    # Twice a triangle's signed area: its two edges joined, read against time, intrinsic as a winding is.
    double_area = (edges[:, 0] & edges[:, 1]) & time
    # A corner's shape function rises from zero along the opposite edge to one at the corner: its gradient is the
    # plane through that edge and time, over twice the signed area.
    gradients = (edges & time) / double_area[:, None]
    # The integrals weight by the area's size, an ordinary number.
    return double_area.norm() / 2, gradients


def strain(displacement: Point, gradients: Plane, faces: np.ndarray) -> Strain:
    """The symmetric displacement gradient on each triangle, as a map from directions to stretch readings."""
    movement = displacement.batch()[..., faces]
    gradient = (movement.dual_inverse() * (gradients & Point)).sum(axis=-1)
    return (gradient + gradient.adjugate()) / 2


def constitutive(directions: Plane, stretch_stiffness: np.ndarray, cross_stiffness: float,
                 shear_stiffness: float, stretch_polarization: np.ndarray,
                 shear_polarization: float, permittivities: np.ndarray
                 ) -> tuple[Elastic, Polarization, Permittivity]:
    """Constitutive extensors from orthonormal material planes, transverse then parallel to poling.

    A displacement and its gradient direction describe a change of shape. The elastic form
    pairs two such changes; the piezoelectric map sends one to a polarization direction.
    """
    transverse, poled = directions[..., 0], directions[..., 1]
    # Each material plane reads extension along its own direction, the shape gradient read by the plane's direction;
    # the mixed reading is shear.
    stretches = (directions & Point) * (Plane & directions.dual())
    shear = (transverse & Point) * (Plane & poled.dual()) + (poled & Point) * (Plane & transverse.dual())
    elastic = ((stretches * stretches * stretch_stiffness).sum(axis=-1)
               + cross_stiffness * (stretches[..., 0] * stretches[..., 1]
                                    + stretches[..., 1] * stretches[..., 0])
               + shear_stiffness * shear * shear)
    # Extension polarizes along the poling direction; shear polarizes across it.
    polarization = (poled.dual() * (stretches * stretch_polarization).sum(axis=-1)
                    + transverse.dual() * shear_polarization * shear)
    # The electric field is read on each material direction swept through time.
    permittivity = (directions.dual() * (Field & (directions.dual() & time)) * permittivities).sum(axis=-1)
    return elastic, polarization, permittivity


def enthalpy(strain: Strain, electric: Field, elastic: Elastic, polarization: Polarization,
             permittivity: Permittivity, poling: Scalar) -> Scalar:
    """Electric enthalpy density, contracting the constitutive maps with the local strain."""
    deformation = strain.dual()
    field = electric
    # Each trace supplies a whole deformation map to its displacement–gradient pair.
    stored = elastic(deformation, Plane, deformation, Plane).trace(1, 2).trace(1, 2) / 2
    polarized = polarization(deformation, Plane).trace(1, 2) * poling
    induced = permittivity(field)
    return stored - (field & (polarized & time)) - (field & (induced & time)) / 2


def assemble(mesh: Mesh, areas: Scalar, gradients: Plane, elastic: Elastic, polarization: Polarization,
             permittivity: Permittivity, poling: Scalar, width: float
             ) -> tuple[SparseExtensor, SparseExtensor, SparseExtensor]:
    """The three blocks of electric enthalpy, as maps to the work-duals of the nodal fields."""
    # Binding each shape gradient turns the continuum laws into nodal work and polarization maps.
    work = elastic(Point, gradients[:, None, :], Point, gradients[:, :, None])
    polarized = polarization(Point, gradients) * poling[:, None]
    electric = -(gradients ^ mv.t) * Scalar

    # Read the work form as the force paired with a test displacement.
    elastic_cells = (Plane & Point).solve(work)
    # Polarization sweeps a plane through time, the electric field's work-dual.
    # Reading this map backwards supplies the converse effect from the same coupling.
    mixed_cells = -(polarized[:, :, None] & time).adjugate()(electric[:, None, :])
    electric_cells = -electric[:, :, None].adjugate()(permittivity(electric[:, None, :]) & time)

    # Local test/trial pairs become couplings between global nodes at assembly.
    weights = width * areas[..., None, None]
    return tuple(SparseExtensor.from_indices(cells * weights, mesh.faces[:, :, None], mesh.faces[:, None, :],
                                             (mesh.n_nodes, mesh.n_nodes))
                 for cells in (elastic_cells, mixed_cells, electric_cells))


def equilibrium(mechanical: SparseExtensor, coupling: SparseExtensor, dielectric: SparseExtensor,
                movement: SparseExtensor, voltage: SparseExtensor,
                prescribed: Scalar, force: Plane) -> tuple[Point, Scalar]:
    """Equilibrium with grounded or prescribed electrodes and zero free charge on floating ones."""
    # The same mixed map supplies forces from voltage and, read backwards, charge from motion.
    elastic = movement.adjugate()(mechanical(movement))
    mixed = movement.adjugate()(coupling(voltage))
    capacitance = voltage.adjugate()(dielectric(voltage))
    load = movement.adjugate()(force - coupling(prescribed))
    charge = -voltage.adjugate()(dielectric(prescribed))
    displacement, potential = solve_coupled(elastic, mixed, capacitance, load, charge)
    return movement(displacement), prescribed + voltage(potential)


# --- plumbing -------------------------------------------------------------------------
class Mesh:
    """A triangulated strip's topology: its faces, by corner nodes, and the nodes of its clamped edge, its tip, its
    bottom and top electrodes and its bonded interface."""

    def __init__(self, n_nodes: int, faces: np.ndarray, clamped: np.ndarray, tip: np.ndarray,
                 bottom: np.ndarray, top: np.ndarray, interface: np.ndarray) -> None:
        self.n_nodes, self.faces = n_nodes, faces
        self.clamped, self.tip = clamped, tip
        self.bottom, self.top, self.interface = bottom, top, interface


def strip(length: float, thickness: float, n_length: int, n_thickness: int) -> tuple[Mesh, Point, Scalar]:
    """A rectangular grid split into alternating triangles, with a material interface halfway up: its topology, its
    vertices, and each triangle's poling sign."""
    horizontal = np.linspace(0, length, n_length + 1)
    vertical = np.linspace(-thickness / 2, thickness / 2, n_thickness + 1)
    vertices = (axis * horizontal[:, None] + pole * vertical[None, :]).reshape(-1).field()
    nodes = np.arange((n_length + 1) * (n_thickness + 1)).reshape(n_length + 1, n_thickness + 1)
    lower_left, lower_right = nodes[:-1, :-1], nodes[1:, :-1]
    upper_left, upper_right = nodes[:-1, 1:], nodes[1:, 1:]
    parity = np.indices(lower_left.shape).sum(axis=0) % 2
    first = np.stack((lower_left, lower_right, np.where(parity, upper_left, upper_right)), axis=-1)
    second = np.stack((np.where(parity, lower_right, lower_left), upper_right, upper_left), axis=-1)
    faces = np.stack((first, second), axis=-2).reshape(-1, 3)
    layer = np.broadcast_to(np.where(np.arange(n_thickness) < n_thickness // 2, -1, 1), lower_left.shape)
    poling = mv.scalar(np.repeat(layer.reshape(-1), 2)[:, None])
    mesh = Mesh(nodes.size, faces, nodes[0], nodes[-1], nodes[:, 0], nodes[:, -1], nodes[:, n_thickness // 2])
    return mesh, vertices, poling


def boundary_maps(mesh: Mesh, floating: np.ndarray) -> tuple[SparseExtensor, SparseExtensor]:
    """Free displacement nodes and potential groups; the named electrode shares one free voltage."""
    n_nodes = mesh.n_nodes
    free_movement = np.setdiff1d(np.arange(n_nodes), mesh.clamped)
    movement = SparseExtensor.from_indices(
        mv.scalar(np.ones((len(free_movement), 1))) * Point,
        free_movement, np.arange(len(free_movement)), (n_nodes, len(free_movement)))
    interior = np.setdiff1d(np.arange(n_nodes), np.concatenate((mesh.top, mesh.bottom)))
    rows = np.concatenate((interior, floating))
    columns = np.concatenate((np.arange(len(interior)), np.full(len(floating), len(interior))))
    n_voltages = len(interior) + int(len(floating) > 0)
    voltage = SparseExtensor.from_indices(mv.scalar(np.ones((len(rows), 1))) * Scalar,
                                        rows, columns, (n_nodes, n_voltages))
    return movement, voltage


def electrode_potential(mesh: Mesh, voltage: float) -> Scalar:
    values = np.zeros(mesh.n_nodes)
    values[mesh.top] = voltage
    return mv.scalar(values[:, None]).field()


def tip_load(mesh: Mesh, force: float) -> Plane:
    weights = np.zeros(mesh.n_nodes)
    weights[mesh.tip] = 1
    weights[mesh.tip[[0, -1]]] = 0.5
    return (mv.y * (force * weights / weights.sum())).field()
