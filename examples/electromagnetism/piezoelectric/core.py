"""Piezoelectric continuum elasticity: a laminated strip as actuator and sensor.

Displacements are intrinsic directions; electric fields are planes through time. A material's
elastic, piezoelectric and dielectric maps give one electric enthalpy. Pulling their work back
through the finite-element readings gives equilibrium and charge balance. Metal electrodes
have a common potential; a floating electrode fixes total charge while its voltage responds.
"""

from __future__ import annotations

import numpy as np

from numga.algebras import STA2D as ga
from numga.sparse import SparseExtensor, spdiag

from examples import instantiate
from examples.electromagnetism.piezoelectric.numerical import solve_coupled

# The spacetime of the strip's plane, x+y+t-: two positive spatial directions, one negative time direction.
mesh_module = instantiate("examples.util.mesh", ga)
Mesh, ctx, mv = mesh_module.Mesh, mesh_module.ctx, mesh_module.mv
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
def triangles(mesh: Mesh, vertices: Point) -> tuple[Scalar, Plane]:
    """Each triangle's area, `Scalar+[faces]`, and each of its corners' shape-function gradients,
    `[corners] Plane+[faces]`."""
    # The side facing each corner, signed as the triangle's winding runs along it.
    sides = mesh.boundary[2, 1] * (mesh.d[0] * vertices)                         # [corners] Point-[faces]
    # Twice a triangle's signed area: its two sides joined, read against time, intrinsic as a winding is.
    double_area = (sides[0] & sides[1]) & time                                    # Scalar-[faces]
    # A corner's shape function rises from zero along the facing side to one at the corner: its gradient is the
    # plane through that side and time, over twice the signed area.
    gradients = (time & sides) / double_area                                      # [corners] Plane+[faces]
    # The integrals weight by the area's size, an ordinary number.
    return double_area.norm() / 2, gradients


def strain(mesh: Mesh, displacement: Point, gradients: Plane) -> Strain:
    """The symmetric displacement gradient on each triangle, `[faces]`, as a map from directions to stretch readings:
    a map per triangle, so read on the batch of triangles, where the adjugate acts on each map."""
    movement = (mesh.incidence[2, 0] * displacement).batch()                      # [..., corners, faces] Point-
    gradient = (movement.dual_inverse() * (gradients.batch() & Point)).sum(axis=-2)
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
    # Binding each shape gradient turns the continuum laws into nodal work and polarization maps, test corners
    # first and trial corners second.
    gradients = gradients.batch()                                                 # [corners, faces] Plane+
    work = elastic(Point, gradients[None], Point, gradients[:, None])
    polarized = polarization(Point, gradients) * poling.batch()
    electric = -(gradients ^ mv.t) * Scalar

    # Read the work form as the force paired with a test displacement.
    elastic_cells = (Plane & Point).solve(work)
    # Polarization sweeps a plane through time, the electric field's work-dual.
    # Reading this map backwards supplies the converse effect from the same coupling.
    mixed_cells = -(polarized[:, None] & time).adjugate()(electric[None])
    electric_cells = -electric[:, None].adjugate()(permittivity(electric[None]) & time)

    weights = width * areas.batch()
    corners = mesh.incidence[2, 0]                                                # [corners] Scalar+[faces <- vertices]
    # The pullback onto the nodes: each test corner's incidence run back, the triangles' blocks, each trial corner's
    # incidence, summed over both corners of the pair.
    return tuple((~corners[:, None] * spdiag((cells * weights).field()) * corners[None, :]).sum(axis=0).sum(axis=0)
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
class Strip(Mesh):
    """A triangulated strip's mesh, with the vertices of its clamped edge, its tip, its bottom and top electrodes and
    its bonded interface."""

    def __init__(self, n_vertices: int, faces: np.ndarray, clamped: np.ndarray, tip: np.ndarray,
                 bottom: np.ndarray, top: np.ndarray, interface: np.ndarray) -> None:
        super().__init__(n_vertices, faces)
        self.clamped, self.tip = clamped, tip
        self.bottom, self.top, self.interface = bottom, top, interface


def strip(length: float, thickness: float, n_length: int, n_thickness: int) -> tuple[Strip, Point, Scalar]:
    """A rectangular grid split into alternating triangles, with a material interface halfway up: its mesh, its
    vertices, and each triangle's poling sign, `Scalar+[faces]`."""
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
    poling = mv.scalar(np.repeat(layer.reshape(-1), 2)[:, None]).field()
    mesh = Strip(nodes.size, faces, nodes[0], nodes[-1], nodes[:, 0], nodes[:, -1], nodes[:, n_thickness // 2])
    return mesh, vertices, poling


def boundary_maps(mesh: Strip, floating: np.ndarray) -> tuple[SparseExtensor, SparseExtensor]:
    """Free displacement nodes and potential groups; the named electrode shares one free voltage."""
    free_movement = np.setdiff1d(np.arange(mesh.n_vertices), mesh.clamped)
    # Each free displacement is put back at its vertex.
    movement = ~mesh.reading(free_movement) * Point
    interior = np.setdiff1d(np.arange(mesh.n_vertices), np.concatenate((mesh.top, mesh.bottom)))
    potential_nodes = np.concatenate((interior, floating))
    # Each interior node reads a voltage of its own, and the floating electrode's nodes all read the one after them.
    groups = np.concatenate((np.arange(len(interior)), np.full(len(floating), len(interior))))
    n_voltages = len(interior) + int(len(floating) > 0)
    voltage = ~mesh.reading(potential_nodes) * SparseExtensor.selection(ctx, groups, n_voltages) * Scalar
    return movement, voltage


def electrode_potential(mesh: Strip, voltage: float) -> Scalar:
    values = np.zeros(mesh.n_vertices)
    values[mesh.top] = voltage
    return mv.scalar(values[:, None]).field()


def tip_load(mesh: Strip, force: float) -> Plane:
    weights = np.zeros(mesh.n_vertices)
    weights[mesh.tip] = 1
    weights[mesh.tip[[0, -1]]] = 0.5
    return (mv.y * (force * weights / weights.sum())).field()
