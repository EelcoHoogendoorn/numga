"""Active bonds that push sideways under extension give a planar lattice an odd elastic response.

Displacements are modelled as intrinsic antivectors and forces as extrinsic vectors paired with them
by mechanical work `force & displacement`. Each stretched bond reads its longitudinal extension with
the unit force along the bond and answers with both a restoring force and a transverse push turned in
the actuation plane, so the linear map `Force <- Displacement` delivers net work around closed
deformation loops. Pulling forces back through `motion.adjugate()` preserves work from single bonds
to macroscopic shear strains and to overdamped lattice waves.
"""

from __future__ import annotations

from collections.abc import Iterator

from numga import NumpyContext, stack
from numga.algebras import VGA2D
from numga.sparse import Sparse, SparseExtensor

ga = VGA2D
ctx = NumpyContext(ga)
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                  # Scalar+
# Displacements and shear amplitudes are intrinsic antivectors; forces are extrinsic vectors paired by work.
Displacement = ga.gatype.antivector()                                        # Antivector-
Force = ga.gatype.vector()                                                   # Vector+
Bivector = ga.gatype.bivector()                                              # Bivector+
Response = ga.gatype((Force, Displacement))                                  # Force+ <- Displacement-
Shear = ga.gatype.antivector()                                               # Antivector-

# Unit displacements dual to the coordinate force planes.
x, y = stack((mv.x, mv.y)).dual()                                            # [] Displacement-, Displacement-

# Pure shear along the axes and at forty-five degrees, combined into a bilinear strain map.
axial_shear = x * (mv.x & Displacement) - y * (mv.y & Displacement)          # Displacement- <- Displacement-
diagonal_shear = x * (mv.y & Displacement) + y * (mv.x & Displacement)       # Displacement- <- Displacement-
shear_strain = (mv.x & Shear) * axial_shear + (mv.y & Shear) * diagonal_shear  # Displacement- <- (Shear-, Displacement-)


def bond_response(direction: Displacement, actuation_plane: Bivector, stiffness: float, activity: float) -> Response:
    """Each stretched bond answers with a longitudinal restoring force and a transverse active push (`[...] Force+ <- Displacement-`)."""
    # Unit force along the bond whose pairing reads longitudinal extension.
    unit_force = direction.dual_inverse()                                    # [...] Force+
    push = -stiffness * unit_force + activity * (unit_force | actuation_plane)  # [...] Force+
    return push * (unit_force & Displacement)                                # [...] Force+ <- Displacement-


def cycle_states(response: Response, cycle: Displacement) -> Iterator[tuple[Displacement, Force, Scalar]]:
    """Yields `(displacement, force, work)` along consecutive straight segments of `cycle` (`[...] Displacement-`)."""
    work = mv.scalar([0])                                                    # Scalar+
    yield cycle[0], response(cycle[0]), work
    for start, end in zip(cycle[:-1], cycle[1:]):
        # Midpoint evaluation integrates a linear force response without quadrature error on each segment.
        work = work + (response((start + end) / 2) & (end - start))          # Scalar+
        yield end, response(end), work


def lattice_response(incidence: Sparse, response: Response) -> SparseExtensor:
    """Adjugate pullback across `incidence` (`Scalar+[E <- V]`) maps bond responses (`(Force+ <- Displacement-)[E]`) to nodal forces (`(Force+ <- Displacement-)[V <- V]`)."""
    bond_displacement = incidence * Displacement                             # (Displacement- <- Displacement-)[E <- V]
    return bond_displacement.adjugate()(response(bond_displacement))         # (Force+ <- Displacement-)[V <- V]


def velocity_map(incidence: Sparse, response: Response, drags: Scalar) -> SparseExtensor:
    """Metric mobility against `drags` maps nodal bond forces to overdamped velocities (`(Displacement- <- Displacement-)[V <- V]`)."""
    # Viscous drag maps each nodal force to a velocity along its metric dual direction.
    mobility = Force.dual() / drags                                          # (Displacement- <- Force+)[V]
    return mobility(lattice_response(incidence, response))                   # (Displacement- <- Displacement-)[V <- V]


def rk4_step(velocities: SparseExtensor, displacement: Displacement, dt: float) -> Displacement:
    """Fourth-order Runge-Kutta integration advances `displacement` (`Displacement-[V]`) by one step of duration `dt`."""
    first_rate = velocities(displacement)                                    # Displacement-[V]
    second_rate = velocities(displacement + first_rate * (dt / 2))           # Displacement-[V]
    third_rate = velocities(displacement + second_rate * (dt / 2))          # Displacement-[V]
    fourth_rate = velocities(displacement + third_rate * dt)                 # Displacement-[V]
    return displacement + (first_rate + 2 * second_rate + 2 * third_rate + fourth_rate) * (dt / 6)


def wave_states(
    reference: Displacement, incidence: Sparse, bond_directions: Displacement, velocities: SparseExtensor,
    initial: Displacement, n_frames: int, n_steps_per_frame: int, dt: float,
) -> Iterator[tuple[Displacement, Scalar]]:
    """Yields nodal positions (`Displacement-[V]`) and signed bond extensions (`Scalar+[E]`) over `n_frames` frames of `n_steps_per_frame` steps."""
    displacement = initial                                                   # Displacement-[V]
    for _ in range(n_frames):
        extensions = bond_directions.dual_inverse() & (incidence * displacement)  # Scalar+[E]
        yield reference + displacement, extensions
        for _ in range(n_steps_per_frame):
            displacement = rk4_step(velocities, displacement, dt)            # Displacement-[V]
