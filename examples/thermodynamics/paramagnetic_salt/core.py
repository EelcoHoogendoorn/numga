"""Adiabatic demagnetization cools a triangulated paramagnetic crystal around a magnetic refrigeration cycle between a
warm bath at one end and a cold load at the other.

A paramagnetic salt is modelled as non-interacting two-level spins in a crystal lattice with a small heat capacity.
Applying a magnetic field aligns the spins and lowers their mixing entropy; in an isolated crystal that entropy flows
into the lattice and warms it, whereas removing the field returns entropy to the spins and cools the lattice through the
magnetocaloric effect. Because an ideal paramagnet's internal energy resides solely in its lattice heat, the equilibrium
magnetic field arises entropically from the spin-entropy slope multiplied by temperature, just as an ideal rubber band's
tension arises from chain entropy. Across a finite crystal, lattice conduction takes time to carry heat, so rapid field
sweeps warm or cool the interior ahead of the linked reservoir and send temperature fronts inward from the contact end.

Planar projective spacetime weights spatial positions with the null plane `w` and joins with a metric-free direction `s`
for entropy. Each vertex's dual cell carries one combined state (`State-`) summing its magnetization patch
`moment * share * magnet_plane.dual()` in the turning plane and its entropy quantity `entropies * s.dual()` per spin.
Its conjugate potential (`Potential+`) sums the magnetic field plane and the temperature along `s`, so the first law
pairs potential and state increment as `potentials & change` without cross-talk between magnetic work `field & spin` and
heat. Differentiating the lattice energy over the state yields `temperature * (s - spin_slope)`, giving the spin-half
hyperbolic-tangent equation of state and Curie's linear law at weak fields. Differentiating a second time yields the
thermodynamic stiffness (`Potential+ <- State-`) as a symmetric sum of two dyads whose cross term
`field / heat_capacity` expresses the magnetocaloric Maxwell relation; its per-vertex inverse gives the compliance
mapping field and temperature increments to state increments.

Joining each triangle corner's opposite side with time and entropy and normalizing at the corner defines an affine plane
equal to one at the corner and zero on the opposite side, whose spatial projection gives the corner's simplex gradient
(`[3] Plane+[F]`). Isotropic lattice conduction pulls the triangle conductivity dyad back through the simplex-gradient
adjugate as `gradient.adjugate()(medium(gradient))`, forming a symmetric map `(Antiscalar- <- Scalar+)[V <- V]` on
vertex temperatures. Each implicit time step solves a single linear heat balance coupling dual-cell heat capacities,
lattice conduction, reservoir switch conductances, and magnetocaloric heating from the applied field step.

Temperatures are measured in kelvin, magnetic fields in tesla, entropy in units of Boltzmann's constant per spin
referenced to zero lattice entropy at `reference`, energy in kelvin per spin, and magnetization in kelvin per tesla per
spin with saturation `moment`. Spatial lengths are measured in slab widths at unit areal spin density, and times in
thermal diffusion units across one slab width at unit heat capacity."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, dim, stack, where
from numga.sparse import SparseExtensor, spdiag
from numga.topology import Topology

# Planar projective spacetime with null weight w joined with a metric-free direction s for entropy.
space = Algebra("x+y+")
ga = Algebra("x+y+t-w0") * Algebra("s0")
mv = NumpyContext(ga, units=True).multivector

Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
# Spatial planes along and across the crystal slab.
Plane = ga.gatype.vector().derive.within(space)                               # Vector+
# Magnetic field plane and intrinsic spin-turning patch.
Magnetic = ga.gatype.bivector().derive.within(space)                          # Bivector+
Spin = Magnetic.derive.dual                                                   # Antibivector-
# Thermal plane along s for temperature and its dual entropy quantity.
Thermal = ga.gatype.vector().derive.within(Algebra("s0"))                     # Vector+
Quantity = Thermal.derive.dual                                                # Antivector-
# Combined thermodynamic state, conjugate potential, and energy Hessian stiffness.
State = ga.gatype(Spin + Quantity)                                            # Antibivector- + Antivector-
Potential = ga.gatype(Magnetic + Thermal)                                     # Bivector+ + Vector+
Stiffness = ga.gatype((Potential, State))                                     # Potential+ <- State-

boltzmann = mv.scalar(1.0, unit=dim.boltzmann)                                # [] Scalar+, boltzmann
one_spin = mv.scalar(1.0, unit=dim.extent)                                    # [] Scalar+, extent
# Unit turning plane, intrinsic spin patch, entropy quantity, time direction, and unit heat antiscalar.
magnet_plane = mv.xy                                                          # [] Magnetic+
spin = magnet_plane.dual()                                                    # [] Spin-
entropy = mv.s.dual()                                                         # [] Quantity-
time = mv.t.dual()                                                            # [] Point-
heat = mv.scalar([1]).dual()                                                  # [] Antiscalar-
# Orthogonal crystal axes along which the lattice conducts heat isotropically.
crystal_axes = stack((mv.x.dual(), mv.y.dual()))                              # [2] Point-


# --- math -----------------------------------------------------------------------------
def spin_entropy(shares: Scalar) -> Scalar:
    """Two-level mixing entropy in nats (`[...] Scalar+`) falls from `ln 2` at zero magnetization to zero as `shares`
    (`[...] Scalar+`) approaches saturation."""
    aligned_share, opposed_share = (1 + shares) / 2, (1 - shares) / 2         # [...] Scalar+, Scalar+
    return -(aligned_share * aligned_share.log() + opposed_share * opposed_share.log())


def potentials(state: State, spins: Scalar, moment: Scalar, heat_capacity: Scalar,
               reference: Scalar) -> Potential:
    """Differentiating the lattice internal energy at extensive `state` (`[...] State-`) for `spins` (`[...] Scalar+, extent`)
    yields the intensive equilibrium magnetic field plane and temperature (`[...] Potential+, field | temperature`) as
    `temperature * (s - spin_slope)`."""
    shares = (magnet_plane & state) / (spins * moment)                        # [...] Scalar+
    # Constant lattice heat capacity makes temperature grow exponentially with the non-spin entropy.
    lattice_entropy = (mv.s & state) - spins * boltzmann * spin_entropy(shares)   # [...] Scalar+, boltzmann*extent
    temperature = reference * (lattice_entropy / (spins * heat_capacity)).exp()   # [...] Scalar+, temperature
    # Spin entropy decreases with alignment, giving an entropic restoring field proportional to temperature.
    spin_slope = -(boltzmann * shares.arctanh() / moment) * magnet_plane      # [...] Magnetic+, field/temperature
    return temperature * (mv.s - spin_slope)


def stiffness(state: State, spins: Scalar, moment: Scalar, heat_capacity: Scalar,
              reference: Scalar) -> Stiffness:
    """The second derivative of internal energy at extensive `state` (`[...] State-`) sums the thermal and spin-ordering
    dyads into a symmetric stiffness map (`[...] Potential+ <- State-`)."""
    shares = (magnet_plane & state) / (spins * moment)                        # [...] Scalar+
    salt_potentials = potentials(state, spins, moment, heat_capacity, reference)   # [...] Potential+, field | temperature
    temperature = salt_potentials & entropy                                   # [...] Scalar+, temperature
    thermal_stiffness = salt_potentials * (salt_potentials & State) / ((spins * heat_capacity) * temperature)   # [...] Stiffness
    spin_stiffness = boltzmann * temperature / (spins * moment ** 2 * (1 - shares ** 2)) * magnet_plane * (magnet_plane & State)   # [...] Stiffness
    return thermal_stiffness + spin_stiffness


def equilibrium_state(field: Magnetic, temperature: Scalar, spins: Scalar, moment: Scalar,
                      heat_capacity: Scalar, reference: Scalar) -> State:
    """Spin-half magnetization saturation and logarithmic lattice entropy determine the extensive equilibrium state
    (`[...] State-`) for `spins` (`[...] Scalar+, extent`) at `field` (`[...] Magnetic+, field`) and `temperature`
    (`[...] Scalar+, temperature`)."""
    shares = (moment * (field & spin) / (boltzmann * temperature)).tanh()     # [...] Scalar+
    entropies = spins * (boltzmann * spin_entropy(shares) + heat_capacity * (temperature / reference).log())   # [...] Scalar+, boltzmann*extent
    return (spins * moment) * shares * spin + entropies * entropy


def drawn_spins(states: State, spins: Scalar, draws: Scalar, moment: Scalar) -> Spin:
    """Thresholding fixed uniform `draws` (`Scalar+[V]`) against the aligned population fraction of extensive `states`
    (`State-[V]`) assigns one two-level spin (`Spin-[V]`) per vertex that flips smoothly as magnetization changes."""
    shares = (magnet_plane & states) / (spins * moment)                       # Scalar+[V]
    is_aligned = draws < (1 + shares) / 2
    return moment * where(is_aligned, spin, -spin)


def corner_fields(bath_temperature: Scalar, load_temperature: Scalar, highest_field: Scalar,
                  lowest_field: Scalar, moment: Scalar, heat_capacity: Scalar,
                  reference: Scalar, n_halvings: int) -> Magnetic:
    """Adiabatic entropy conservation between `bath_temperature` and `load_temperature` determines the four corner
    magnetic fields (`[4] Magnetic+, field`) of the ideal rectangular refrigeration cycle."""
    held_fields = stack((highest_field * magnet_plane, lowest_field * magnet_plane))   # [2] Magnetic+, field
    held_entropies = mv.s & equilibrium_state(held_fields, stack((bath_temperature, load_temperature)), one_spin,
                                              moment, heat_capacity, reference)        # [2] Scalar+, boltzmann*extent
    # Adiabatic legs preserve total entropy, so the spin share absorbs the lattice entropy shift to the other bath.
    target_temperatures = stack((load_temperature, bath_temperature))                  # [2] Scalar+, temperature
    corner_shares = shares_at_entropy(
        (held_entropies - one_spin * heat_capacity * (target_temperatures / reference).log()) / (one_spin * boltzmann),
        n_halvings)                                                                    # [2] Scalar+
    corner_potentials = potentials((one_spin * moment) * corner_shares * spin + held_entropies * entropy, one_spin,
                                   moment, heat_capacity, reference)                   # [2] Potential+, field | temperature
    cooled_field, warmed_field = (corner_potentials & spin) * magnet_plane             # [] Magnetic+, field
    return stack((warmed_field, held_fields[0], cooled_field, held_fields[1]))


def magnet_fields(corner_fields: Magnetic, from_corners: np.ndarray, to_corners: np.ndarray,
                  phase_progress: np.ndarray) -> Magnetic:
    """Logarithmic interpolation across `phase_progress` between `from_corners` and `to_corners` of `corner_fields`
    (`[4] Magnetic+, field`) gives the applied magnetic field (`[...] Magnetic+, field`) at each step."""
    start, end = corner_fields[from_corners], corner_fields[to_corners]       # [n_steps] Magnetic+, field
    return start * ((end / start).log() * phase_progress).exp()


def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane]:
    """Joining opposite triangle corners across `vertices` (`Point-[V], 1 | length`) yields triangle areas (`Scalar+[F], length^2`) and spatial
    simplex gradients (`[3] Plane+[F], 1/length`) on `topology`."""
    corners = topology.incidence[2, 0] * vertices                             # [3] Point-[F], 1 | length
    opposite_vertices = topology.off[2, 0] * vertices                         # [2, 3] Point-[F], 1 | length
    # Joining the opposite side with time and entropy and normalizing at the corner yields the affine plane whose
    # spatial projection is the corner's simplex gradient.
    sides = opposite_vertices[0] & opposite_vertices[1]                       # [3] Antibivector-[F]
    gradients = ((sides & time & entropy) / (sides & time & entropy & corners)).cast(Plane)   # [3] Plane+[F], 1/length
    double_areas = corners[0] & corners[1] & corners[2] & time & entropy      # Antiscalar-[F], length^2
    return double_areas.norm() / 2, gradients


def conduction(topology: Topology, areas: Scalar, gradients: Plane, conductivity: Scalar) -> SparseExtensor:
    """Pulling isotropic `conductivity` on triangle `areas` (`Scalar+[F], length^2`) back through the adjugate of `gradients`
    (`[3] Plane+[F], 1/length`) assembles the vertex heat-conduction map (`(Antiscalar- <- Scalar+)[V <- V]`)."""
    gradient = (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar   # (Plane+ <- Scalar+)[F <- V], 1/length
    medium = areas * conductivity * (crystal_axes * (Plane & crystal_axes)).sum(axis=0)   # (Point- <- Plane+)[F]
    return gradient.adjugate()(medium(gradient))


def step(cell_spins: Scalar, lattice_conduction: SparseExtensor, states: State, field: Magnetic,
         links: Scalar, reservoirs: Scalar, moment: Scalar, heat_capacity: Scalar,
         reference: Scalar, dt: Scalar) -> State:
    """An implicit heat-balance solve over `dt` advances extensive dual-cell `states` (`State-[V]`) for
    `cell_spins` (`Scalar+[V], extent`) under applied `field` (`[] Magnetic+, field`), `lattice_conduction`, and reservoir
    `links` (`Scalar+[V], boltzmann*extent/time`)."""
    salt_potentials = potentials(states, cell_spins, moment, heat_capacity, reference)   # Potential+[V], field | temperature
    temperatures = salt_potentials & entropy                                  # Scalar+[V], temperature
    # Per-vertex compliance maps field and temperature increments to extensive state increments at held temperature or field.
    compliance = stiffness(states, cell_spins, moment, heat_capacity, reference).batch().inverse().field()   # (State- <- Potential+)[V]
    field_response = compliance(magnet_plane)                                 # State-[V]
    thermal_response = compliance(mv.s)                                       # State-[V]
    # Magnetizing at fixed temperature expels spin entropy and releases magnetocaloric heat into the dual cells.
    field_shortfall = (field - salt_potentials) & spin                        # Scalar+[V], field
    cell_capacities = temperatures * (mv.s & thermal_response)                # Scalar+[V], boltzmann*extent
    magnetocaloric_heat = temperatures * (mv.s & field_response) * field_shortfall   # Scalar+[V], boltzmann*extent*temperature
    # Implicit heat balance equates dual-cell warming to reservoir inflow and conduction minus isothermal heat intake.
    reservoir_coupling = spdiag(links * heat * Scalar)                        # (Antiscalar- <- Scalar+)[V <- V]
    heat_balance = spdiag(cell_capacities * heat * Scalar) + dt * (lattice_conduction + reservoir_coupling)   # (Antiscalar- <- Scalar+)[V <- V]
    net_heat = dt * (reservoir_coupling(reservoirs - temperatures) - lattice_conduction(temperatures)) - magnetocaloric_heat * heat   # Antiscalar-[V]
    temperature_step = heat_balance.solve(net_heat)                           # Scalar+[V], temperature
    return states + field_shortfall * field_response + temperature_step * thermal_response


def run(cell_spins: Scalar, lattice_conduction: SparseExtensor, states: State, fields: Magnetic,
        links: Scalar, reservoirs: Scalar, durations: Scalar, steps_per_frame: np.ndarray,
        moment: Scalar, heat_capacity: Scalar, reference: Scalar
        ) -> Iterator[tuple[State, Potential, Magnetic, Scalar]]:
    """Yields extensive vertex states (`State-[V]`), intensive potentials (`Potential+[V], field | temperature`), applied
    field (`[] Magnetic+, field`), and switch conductances (`Scalar+[V], boltzmann*extent/time`) initially and after each frame's
    `steps_per_frame` implicit steps."""
    yield states, potentials(states, cell_spins, moment, heat_capacity, reference), fields[0], links[0]
    frame_ends = np.cumsum(steps_per_frame)
    for start, end in zip(frame_ends - steps_per_frame, frame_ends):
        for index in range(start, end):
            states = step(cell_spins, lattice_conduction, states, fields[index], links[index], reservoirs, moment,
                          heat_capacity, reference, durations[index])
        yield states, potentials(states, cell_spins, moment, heat_capacity, reference), fields[end - 1], links[end - 1]


# --- plumbing -------------------------------------------------------------------------
def shares_at_entropy(spin_entropies: Scalar, n_halvings: int) -> Scalar:
    """Bisection over `n_halvings` intervals inverts the two-level mixing entropy to find the positive magnetization
    shares (`[...] Scalar+`) matching `spin_entropies` (`[...] Scalar+`)."""
    lower_share, upper_share = 0 * spin_entropies, 0 * spin_entropies + 1     # [...] Scalar+, Scalar+
    for _ in range(n_halvings):
        mid_share = (lower_share + upper_share) / 2                           # [...] Scalar+
        # Larger mixing entropy than the target indicates too little spin alignment.
        above_target = spin_entropy(mid_share) > spin_entropies
        lower_share = where(above_target, mid_share, lower_share)
        upper_share = where(above_target, upper_share, mid_share)
    return (lower_share + upper_share) / 2
