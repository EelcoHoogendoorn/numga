"""Below the lambda point, liquid helium flows through a triangulated planar cavity as two interpenetrating components. An inviscid superfluid component carries mass with zero entropy, while a normal fluid component carries the remaining mass and all of the entropy. Local heating drives normal fluid outward and draws superfluid inward as an undamped temperature wave called second sound. Clamping the normal fluid inside a porous wall leaves only the superfluid mobile, converting thermal driving into a coupled pressure-temperature wave called fourth sound that travels roughly ten times faster. Viscosity, thermal conduction, and mutual friction are omitted so the coupled dynamics remain conservative.

The algebra joins the projective plane with two degenerate material directions for mass and entropy. Cavity vertices sit at unit projective weight, while each component's parcel combines its mass and entropy per unit volume along the two material directions. Thermodynamic potentials combine chemical potential per unit mass and temperature in the conjugate material plane, and minus their spatial gradient wedged with the potentials forms a bivector drive whose commutator accelerates each parcel according to the mass and entropy it holds. Joining each component's velocity with its parcel gives the material flux, and wedging each corner's simplex gradient with that flux integrates the mass and entropy leaving each vertex's dual cell. The thermodynamic stiffness maps vertex states to potentials as a sum of compression and heating dyads that satisfies Maxwell symmetry automatically.

Lengths are measured in millimetres, times in milliseconds, speeds in metres per second, and extensive state perturbations per unit volume.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext
from numga.extensor import Extensor
from numga.topology import Topology

# The projective plane for spatial placement, joined with degenerate directions for mass and entropy.
space = Algebra("x+y+w0")
quantities = Algebra("m0s0")
ga = space * quantities
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# Spatial coordinate planes, extensive mass-entropy states, and their conjugate thermodynamic potentials.
Upright = Plane.derive.within("x y")                                          # Vector+
State = Plane.derive.within(quantities).derive.dual                           # Antivector-
Potential = Plane.derive.within(quantities)                                   # Vector+
# Reference origin at unit projective weight, unit material quantities, and their joint complement for spatial areas.
origin = mv.w.dual()                                                          # [] Point-
mass = mv.m.dual()                                                            # [] State-
entropy = mv.s.dual()                                                         # [] State-
unread = mass & entropy                                                       # [] Antibivector-


# --- plumbing -------------------------------------------------------------------------
def cavity(n_columns: int, n_rows: int, width: float, height: float) -> tuple[Topology, np.ndarray]:
    """The cavity triangulation spans a `width` by `height` rectangle on an `n_rows` by `n_columns` grid of vertex coordinates (`[V, 2]`)."""
    grid_axes = np.meshgrid(np.linspace(0, width, n_columns), np.linspace(0, height, n_rows))
    return Topology.grid(n_rows, n_columns), np.stack([grid_axis.ravel() for grid_axis in grid_axes], axis=-1)


# --- math -----------------------------------------------------------------------------
def thermodynamic_stiffness(
    density: np.ndarray,
    first_sound: np.ndarray,
    temperature: np.ndarray,
    entropy_per_mass: np.ndarray,
    heat_capacity: np.ndarray,
) -> Extensor:
    """Compression at fixed specific entropy and heating at fixed density form the symmetric thermodynamic stiffness (`[...] Potential+ <- State-`) mapping mass and entropy per unit volume to chemical potential and temperature."""
    # Unit rise in entropy per unit mass measures entropy added beyond what incoming mass carries at fixed specific entropy.
    heating = mv.s - entropy_per_mass * mv.m                                  # [...] Potential+
    return (first_sound ** 2 / density) * (mv.m * (mv.m & State)) + (
        temperature / (heat_capacity * density)
    ) * (heating * (heating & State))


def sound_speeds(stiffness: Extensor, parcels: State, masses: np.ndarray) -> Scalar:
    """Generalized eigenvalues of the parcel mobility form against the thermodynamic compliance give the squared sound speeds (`[..., 2] Scalar+`) in ascending order."""
    mobility_form = (parcels * (Potential & parcels) / masses).sum(axis=-1)   # [...] State- <- Potential+
    compliance = stiffness.inverse()                                          # [...] State- <- Potential+
    speeds_squared, modes = (Potential & mobility_form(Potential)).eigh(Potential & compliance(Potential))
    return speeds_squared


def triangle_geometry(topology: Topology, vertices: Point) -> tuple[Scalar, Plane, Scalar]:
    """Projective vertex joins determine each triangle's area (`Scalar+[F]`), pure spatial simplex gradients (`[3] Plane+[F]`), and lumped vertex dual areas (`Scalar+[V]`)."""
    corners = topology.incidence[2, 0] * vertices                             # [3] Point-[F]
    facing = topology.off[2, 0] * vertices                                    # [2, 3] Point-[F]
    sides = facing[0] & facing[1]                                             # [3] Antibivector-[F]
    # Joining all three corners with the material complement gives twice each triangle's signed spatial area.
    double_areas = corners[0] & corners[1] & corners[2] & unread              # Antiscalar-[F]
    areas = double_areas.norm() / 2                                           # Scalar+[F]
    gradients = ((sides & unread) / (sides & unread & corners)).cast(Upright)  # [3] Plane+[F]
    dual_areas = (topology.incidence[2, 0].adjugate() * (areas / 3)).sum(axis=0)  # Scalar+[V]
    return areas, gradients, dual_areas


def cavity_states(
    topology: Topology,
    vertices: Point,
    stiffness: Extensor,
    parcels: State,
    masses: np.ndarray,
    normal_clamp: Scalar,
    state: State,
    supply: State,
    dt: float,
    n_steps: int,
    stride: int,
) -> Iterator[State]:
    """Yields the mass and entropy perturbations per unit volume (`State-[V]`) every `stride` steps of `dt` as potential drives accelerate the two fluid components and their flux redistributes state across dual cells."""
    incidence = topology.incidence[2, 0]                                      # [3] Scalar+[F <- V]
    areas, gradients, dual_areas = triangle_geometry(topology, vertices)
    # Superfluid moves freely everywhere while the normal fluid is immobilized where normal_clamp equals one.
    mobility = (1 - normal_clamp * np.array([0, 1])) / masses                 # [2] Scalar+[F]
    velocities = 0 * ((mv.x + mv.y).dual() * mobility)                        # [2] Point-[F]
    for step in range(n_steps + 1):
        if step % stride == 0:
            yield state
        potentials = stiffness(state)                                         # Potential+[V]
        drive = -(gradients ^ (incidence * potentials)).sum(axis=0)           # Bivector+[F]
        velocities = velocities + dt * mobility * drive.commutator(parcels)   # [2] Point-[F]
        flux = (velocities & parcels).sum(axis=0)                             # Antibivector-[F]
        # Dual-cell outflow through each corner's simplex gradient balances external entropy supply.
        outflow = (incidence.adjugate() * (areas * (gradients ^ flux))).sum(axis=0) / dual_areas  # State-[V]
        state = state + dt * (supply - outflow)
