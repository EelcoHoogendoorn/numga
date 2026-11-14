"""Scene parameters and armchair honeycomb flake geometry for the Kane–Mele topological insulator:
bulk gap closure across staggered mass, eigenstates on either side of the transition, and helical
edge transport around the flake.

Energies are measured in units of the nearest-neighbour hop, and lengths in lattice constants.
"""

from __future__ import annotations

import numpy as np

from numga import stack
from numga.sparse import Sparse
from numga.topology import Topology
from examples.quantum.kane_mele import core

mv = core.mv
# Half-width of the armchair hexagon in half lattice constants, and the outer fraction counted as the rim.
flake_size = 12
rim_fraction = 0.3
# Spin-orbit coupling strength, its bulk-gap threshold 3 * sqrt(3) * spin_orbit, and the swept masses.
spin_orbit = 0.1
gap_threshold = 3 * 3**0.5 * spin_orbit
n_masses = 41
masses = np.linspace(0.0, 0.8, n_masses)
# Spin-up levels nearest zero tracked across the mass sweep (each energy is doubly degenerate in Up).
n_levels = 32
# Representative masses below and above the topological transition.
n_cases = 2
sample_masses = np.array([0.0, 0.8])
# In-gap edge levels at zero mass (fourfold per energy across both spins) and initial +x spinor.
n_in_gap = 32
initial_spinor = (1 - mv.zx) * 0.5**0.5                                        # [] Even+
# Duration of one full circuit around the flake, frame count, and GIF frame interval.
lap_time = 50.0
n_frames = 150
times = np.linspace(0.0, lap_time, n_frames + 1)
frame_duration_ms = 40
# Pixel width of the pinned gallery animation.
gallery_pixels = 400


# --- plumbing -------------------------------------------------------------------------
class Flake:
    """Armchair hexagonal graphene flake with vertex positions (`Point-[V]`), bond and
    second-neighbour topologies (`E`, `E_next`), intermediate turn sites (`Point-[E_next]`),
    sublattice signs (`Scalar+[V]`), rim indicator (`Scalar+[V]`), and initial edge site (`Scalar+[V]`)."""

    def __init__(self, bonds: Topology, next_bonds: Topology, positions: core.Point, between: core.Point,
                 sublattice: core.Scalar, rim: core.Scalar, start_site: core.Scalar) -> None:
        self.bonds = bonds
        self.next_bonds = next_bonds
        self.positions = positions
        self.between = between
        self.sublattice = sublattice
        self.rim = rim
        self.start_site = start_site


def honeycomb_flake(size: int, rim_width: float) -> Flake:
    """Honeycomb atoms within `size` half-lattice constants along the three armchair axes, trimmed
    to coordination at least two, with outer fraction `rim_width` marked as the rim."""
    # Unit cell (i, j) along two axes 60 degrees apart, with the second sublattice offset by (1/3, 1/3).
    span = np.arange(-size, size + 1)
    grid_i, grid_j, sub = (index.ravel() for index in np.meshgrid(span, span, [0, 1], indexing="ij"))
    coord_u, coord_v = grid_i + sub / 3, grid_j + sub / 3
    distance = np.maximum.reduce([np.abs(2 * coord_u + coord_v), np.abs(coord_u + 2 * coord_v), np.abs(coord_v - coord_u)])
    inside = distance <= size + 1e-9
    # Trim singly-bonded corner stubs so every boundary atom belongs to the armchair rim.
    inside[inside] = _coordination(grid_i[inside], grid_j[inside], sub[inside], size) >= 2
    grid_i, grid_j, sub = grid_i[inside], grid_j[inside], sub[inside]
    coord_u, coord_v, distance = coord_u[inside], coord_v[inside], distance[inside]
    V = len(grid_i)

    lookup = _site_lookup(grid_i, grid_j, sub, size)
    sign = 1 - 2 * sub
    bond_offsets = np.array([[0, 0], [-1, 0], [0, -1]])
    neighbours = lookup(grid_i[:, None] + sign[:, None] * bond_offsets[:, 0],
                        grid_j[:, None] + sign[:, None] * bond_offsets[:, 1], 1 - sub[:, None])
    first_pairs = _directed_pairs(neighbours)
    bonds = Topology(V, first_pairs[first_pairs[:, 0] < first_pairs[:, 1]])

    # Second neighbours sit one lattice vector away on the same sublattice across a shared opposite-sublattice site.
    next_offsets = np.array([[1, 0], [-1, 1], [0, -1], [-1, 0], [1, -1], [0, 1]])
    seconds = lookup(grid_i[:, None] + next_offsets[:, 0], grid_j[:, None] + next_offsets[:, 1], sub[:, None])
    second_pairs = _directed_pairs(seconds)
    lattice_indices = np.stack([grid_i, grid_j], axis=-1)                      # [V, 2]
    far_indices = lattice_indices[:, None, :] + next_offsets                   # [V, 6, 2]
    own_bridges = lattice_indices[:, None, :] + sign[:, None, None] * bond_offsets   # [V, 3, 2]
    far_bridges = far_indices[:, :, None, :] + sign[:, None, None, None] * bond_offsets   # [V, 6, 3, 2]
    shared = (own_bridges[:, None, :, None, :] == far_bridges[:, :, None, :, :]).all(axis=-1).any(axis=-1)   # [V, 6, 3]
    bridge_indices = own_bridges[np.arange(V)[:, None], shared.argmax(axis=-1)]   # [V, 6, 2]

    # Lattice basis vectors 60 degrees apart in the xy plane.
    basis_u = mv.x.dual()                                                      # [] Point-
    turn_sixty = (mv.xy * (-np.pi / 6)).exp()                                  # [] Rotor+
    basis_v = turn_sixty >> basis_u                                            # [] Point-
    positions = (basis_u * coord_u + basis_v * coord_v).field()                # Point-[V]

    forward = second_pairs[:, 0] < second_pairs[:, 1]
    next_bonds = Topology(V, second_pairs[forward])
    bridge_coords = (bridge_indices + (1 - sub[:, None, None]) / 3)[seconds >= 0][forward]
    between = (basis_u * bridge_coords[:, 0] + basis_v * bridge_coords[:, 1]).field()   # Point-[E_next]

    facing = np.flatnonzero(2 * coord_u + coord_v == (2 * coord_u + coord_v).max())
    start_index = int(facing[np.argmin(np.abs(coord_v[facing]))])
    sublattice = mv.scalar(sign.astype(float)[:, None]).field()                # Scalar+[V]
    rim = mv.scalar((distance > (1 - rim_width) * size)[:, None]).field()      # Scalar+[V]
    start_site = Sparse.selection(np.array([start_index]), V).adjugate() * mv.scalar([[1]]).field()   # Scalar+[V]
    return Flake(bonds, next_bonds, positions, between, sublattice, rim, start_site)


def _site_lookup(grid_i: np.ndarray, grid_j: np.ndarray, sub: np.ndarray, size: int):
    """Maps lattice coordinates `(i, j, sub)` to their atom index in `[0, V)`, or `-1` outside the flake."""
    table = -np.ones((2 * size + 3, 2 * size + 3, 2), dtype=int)
    table[grid_i + size + 1, grid_j + size + 1, sub] = np.arange(len(grid_i))
    return lambda ii, jj, ss: table[np.clip(ii + size + 1, 0, 2 * size + 2),
                                    np.clip(jj + size + 1, 0, 2 * size + 2), ss]


def _coordination(grid_i: np.ndarray, grid_j: np.ndarray, sub: np.ndarray, size: int) -> np.ndarray:
    """Nearest-neighbour count of each candidate atom in `[V]`."""
    sign = 1 - 2 * sub
    bond_offsets = np.array([[0, 0], [-1, 0], [0, -1]])
    lookup = _site_lookup(grid_i, grid_j, sub, size)
    neighbours = lookup(grid_i[:, None] + sign[:, None] * bond_offsets[:, 0],
                        grid_j[:, None] + sign[:, None] * bond_offsets[:, 1], 1 - sub[:, None])
    return (neighbours >= 0).sum(axis=1)


def _directed_pairs(partners: np.ndarray) -> np.ndarray:
    """Directed `(atom, partner)` index pairs where `partners >= 0`."""
    atoms, slots = np.nonzero(partners >= 0)
    return np.stack([atoms, partners[atoms, slots]], axis=-1)


# --- math -----------------------------------------------------------------------------
def mass_sweep(flake: Flake, sweep_masses: np.ndarray, n_modes: int) -> tuple[core.Scalar, core.Scalar]:
    """Spin-up energies nearest zero and their rim fractions across `sweep_masses`."""
    spectra = [
        core.sector_levels(
            core.hamiltonian(flake.bonds, flake.next_bonds, flake.positions, flake.between,
                             flake.sublattice, spin_orbit, float(mass)),
            core.Up, n_modes,
        )
        for mass in sweep_masses
    ]
    energies = stack([energy for energy, _ in spectra])                        # [..., n_modes] Scalar+
    shares = stack([core.rim_share(flake.rim, states) for _, states in spectra])   # [..., n_modes] Scalar+
    return energies, shares


def gap_edge_densities(flake: Flake, case_masses: np.ndarray) -> core.Scalar:
    """Atomic probability density of the lowest-magnitude spin-up eigenstate at each mass in `case_masses`."""
    states = [
        core.sector_levels(
            core.hamiltonian(flake.bonds, flake.next_bonds, flake.positions, flake.between,
                             flake.sublattice, spin_orbit, float(mass)),
            core.Up, 2,
        )[1][0]
        for mass in case_masses
    ]
    return stack(states).scalar_norm_squared()                                 # [...] Scalar+[V]


def main() -> None:
    from examples.util.animation import save_animation, save_figure
    from examples.quantum.kane_mele import render

    flake = honeycomb_flake(flake_size, rim_fraction)
    save_figure(render.flake_figure(flake), "kane_mele_flake")

    energies, shares = mass_sweep(flake, masses, n_levels)
    save_figure(render.levels_figure(masses, energies, shares, gap_threshold), "kane_mele_levels")

    densities = gap_edge_densities(flake, sample_masses)
    save_figure(render.states_figure(flake, densities), "kane_mele_states")

    zero_mass_energy = core.hamiltonian(
        flake.bonds, flake.next_bonds, flake.positions, flake.between, flake.sublattice, spin_orbit, 0.0,
    )
    in_gap, edge_states = core.sector_levels(zero_mass_energy, core.Even, n_in_gap)   # [n_in_gap] Scalar+, [n_in_gap] Even+[V]
    recorded_spins: list[core.Spin] = []

    def streamed_spins():
        for spin in core.spin_evolution(edge_states, in_gap, flake.start_site, initial_spinor, times):
            recorded_spins.append(spin)
            yield spin

    save_animation(
        render.spin_animation(flake, render.flake_pixels, streamed_spins()),
        "kane_mele_helical", frame_duration_ms,
    )
    spins = stack(recorded_spins)                                              # [n_frames + 1] Spin-[V]

    # checks
    # The Hamiltonian equals its own reverse because every coupling reverses into the opposite direction's coupling.
    assert (((~zero_mass_energy - zero_mass_energy) * edge_states).scalar_norm_squared().sites.sum() < 1e-20).all()
    # Every spin-up level is doubly degenerate (psi and psi * xy share the same energy).
    sweep_order = energies.argsort(axis=-1)
    ordered_energies = stack([energies[row][sweep_order[row]] for row in range(n_masses)])   # [n_masses, n_levels] Scalar+
    assert ((ordered_energies[:, ::2] - ordered_energies[:, 1::2]).abs() < 1e-8).all()
    # At zero mass the levels inside the bulk gap sit on the rim; well past the transition the gap is empty.
    half_gaps = np.abs(gap_threshold - masses)
    inside_gap = energies[0].abs() < 0.9 * half_gaps[0]
    assert inside_gap.sum() >= 8 and (shares[0][inside_gap] > 0.9).all()
    assert bool(energies[-1].abs().min() > 0.5 * half_gaps[-1])
    # The state nearest zero sits on the rim below the transition and spreads into the interior above it.
    case_shares = (densities * flake.rim).sites.sum() / densities.sites.sum()   # [n_cases] Scalar+
    assert bool(case_shares[0] > 0.9) and bool(case_shares[-1] < 0.6)
    # In the full Even space at zero mass, every in-gap energy is fourfold degenerate and localized on the rim.
    gap_order = in_gap.argsort()
    ordered_gap = in_gap[gap_order]                                            # [n_in_gap] Scalar+
    assert bool(ordered_gap.abs().max() < gap_threshold)
    assert ((ordered_gap[::4] - ordered_gap[3::4]).abs() < 1e-8).all()
    assert bool(core.rim_share(flake.rim, edge_states).min() > 0.9)
    # Total edge-projected weight is conserved and stays localized on the rim throughout the circuit.
    density = spins.norm()                                                     # [n_frames + 1] Scalar+[V]
    totals = density.sites.sum()                                               # [n_frames + 1] Scalar+
    assert ((totals - totals[0]).abs() < 1e-9 * totals[0]).all()
    assert ((density * flake.rim).sites.sum() > 0.9 * totals).all()
    # A quarter lap in, the spin-up packet has moved counterclockwise and the spin-down packet clockwise.
    quarter = n_frames // 4
    spin_z = mv.xy & spins[quarter]                                            # Scalar+[V]
    up_half = (density[quarter] + spin_z) * 0.5                                # Scalar+[V]
    down_half = (density[quarter] - spin_z) * 0.5                              # Scalar+[V]
    start_point = (flake.positions * flake.start_site).sites.sum().normalized()   # [] Point-
    up_centre = ((up_half * flake.positions).sites.sum() / up_half.sites.sum()).normalized()   # [] Point-
    down_centre = ((down_half * flake.positions).sites.sum() / down_half.sites.sum()).normalized()   # [] Point-
    assert bool(mv.xy & (start_point & up_centre) > 0.5)
    assert bool(mv.xy & (start_point & down_centre) < -0.5)


def pinned() -> None:
    """Regenerates the gallery animation of counter-propagating spin-polarized edge packets."""
    from examples.util.animation import pin_animation
    from examples.quantum.kane_mele import render

    flake = honeycomb_flake(flake_size, rim_fraction)
    zero_mass_energy = core.hamiltonian(
        flake.bonds, flake.next_bonds, flake.positions, flake.between, flake.sublattice, spin_orbit, 0.0,
    )
    in_gap, edge_states = core.sector_levels(zero_mass_energy, core.Even, n_in_gap)   # [n_in_gap] Scalar+, [n_in_gap] Even+[V]
    spins = core.spin_evolution(edge_states, in_gap, flake.start_site, initial_spinor, times)
    pin_animation(render.spin_animation(flake, gallery_pixels, spins), "kane_mele", frame_duration_ms)


if __name__ == "__main__":
    main()
