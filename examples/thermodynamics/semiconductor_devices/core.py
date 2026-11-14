"""Drift, diffusion, and recombination govern carrier transport across semiconductor devices on a simplicial complex of
any spatial dimension. Spacetime weighed by a null point weight joins with null quantity directions for electrons,
holes, heat, and light to model a p-n junction along a bar, a planar MOSFET across an insulated gate, and a repeating
solar-cell period between front-contact fingers.

Acceptors capture crystal electrons and leave mobile holes behind, whereas donors supply mobile electrons. Where p-type
and n-type regions meet, electrons diffuse into the p side and holes diffuse into the n side before recombining and
handing the band-gap energy to the lattice as heat. The uncovered dopant ions establish an electric field across the
carrier-depleted junction that balances diffusion once each carrier's electrochemical potential, or Fermi level, is
uniform and the bands bend by the built-in voltage. Forward bias lowers that barrier so current grows exponentially with
voltage, reverse bias raises it so current saturates at thermal generation, and absorbed light generates electron-hole
pairs that the junction field sweeps apart as a photocurrent opposing forward injection. Across an insulating gate
oxide that holds no mobile carriers, a gate voltage repels majority holes and gathers an inversion channel of electrons
between source and drain wells.

Each instantiation binds projective spacetime with one or two spatial axes, time, and a null point weight joined with
four null quantity directions for electrons, holes, heat, and light. The term ***parcel*** here means an extensive
state combining carrier counts, charge along `t`, heat, or photon energy: an electron combines a unit of electron count
with negative charge, a hole combines a unit of hole count with positive charge, and ionized dopants carry fixed charge
along time alone. A single potential plane at each vertex holds the electric potential along `t` and the carrier
chemical potentials along `n` and `p`, so pairing `potentials & electron` reads the electron quasi-Fermi level
directly. The term ***simplex gradient*** here means the spatial covector at each corner of a simplex whose pairing with a spatial
displacement gives the linear change of that corner's interpolation weight across the simplex. Wedging the assembled
simplex gradient operator with the potential plane forms one combined drift-diffusion drive `-(gradient ^ potentials)`.
A carrier drifts along `drive.commutator(carrier)` and sweeps out the isotropic flux
`drive.commutator(carrier).regressive(carrier)` scaled by its mobility and density, while a unit charge swept the same
way and scaled by permittivity gives the electric displacement. Pulling fluxes back through the adjugate of
`-(gradient ^ Potentials)` preserves pairing with the potential plane,
`potentials & drive_map.adjugate()(flux) == drive_map(potentials) & flux`, and sums simplex fluxes into dual-cell
outflows. Gauss's law supplies the time component of the dual-cell balance where electron and hole conservation repeat
each other, and each implicit time or steady-state step solves the coupled balance by damped Newton iteration.

Recombination acts along `gap * heat - (electron + hole)` at the mass-action rate
`rate_constant * (electrons * holes - intrinsic ** 2)`, and optical absorption acts along `(electron + hole) - gap * light`.
Both transitions pair to zero with the energy plane. Pairing the thermodynamic entropy slope with recombination and
solar absorption, and pairing the transport drive with the carrier flux it drives, yields non-negative entropy
production everywhere. Energies and potentials are scaled by the thermal energy, carrier densities by the reference
doping density, lengths by the corresponding Debye length, and times by the Debye electron diffusion time.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from functools import reduce
from operator import and_

import numpy as np

from numga import Algebra, stack
from numga.backend.context import Context
from numga.gatype import GAType
from numga.sparse import Sparse, SparseExtensor, spdiag
from numga.topology import Topology

# Supplied by examples.instantiate: projective spacetime joined with null quantity directions for electrons, holes,
# heat, and light, together with the compute context.
ga: Algebra
ctx: Context
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# Electric potential along t and carrier chemical potentials along n and p.
Potentials = Plane.derive.within(Algebra("t-n0p0"))                           # Vector+
# Unit charge along time, charged carrier parcels, and neutral heat and photon energy parcels.
time = -mv.t.dual()                                                           # [] Point-
electron = mv.n.dual() - time                                                 # [] Point-
hole = mv.p.dual() + time                                                     # [] Point-
heat = mv.q.dual()                                                            # [] Point-
light = mv.l.dual()                                                           # [] Point-
pairs = electron + hole                                                       # [] Point-
carriers = stack((electron, hole))                                            # [2] Point-
counts = stack((mv.n.dual(), mv.p.dual()))                                    # [2] Point-
charges = carriers - counts                                                   # [2] Point-
count_planes = stack((mv.n, mv.p))                                            # [2] Plane+
# Span of time and every material quantity along which purely spatial fields stay constant.
unread = time & counts[0] & counts[1] & heat & light                          # [] Span-


# --- math -----------------------------------------------------------------------------
def simplex_maps(topology: Topology, vertices: Point, Space: GAType) -> tuple[Scalar, SparseExtensor]:
    """Corner joins on `topology` determine the simplex measures (`Scalar+[N[-1]]`) and the simplex gradient operator
    (`Plane+[N[-1] <- N[0]]`) over unit-weight `vertices` (`Point-[N[0]]`) projected onto `Space`."""
    n_dim = topology.n_dim
    N = topology.n_elements
    corners = topology.incidence[-1, 0] * vertices                            # [n_dim + 1] Point-[N[-1]]
    # Joining the opposite corners with time and all quantities gives a hyperplane vanishing at those corners;
    # normalizing by its pairing with the corner itself and projecting onto Space extracts the simplex gradient.
    complements = reduce(and_, topology.off[-1, 0] * vertices, unread)        # [n_dim + 1] Plane-[N[-1]]
    gradients = (complements / (complements & corners)).cast(Space)           # [n_dim + 1] Plane+[N[-1]]
    volumes = reduce(and_, corners, unread).norm() / math.factorial(n_dim)    # Scalar+[N[-1]]
    gradient = (spdiag(gradients) * topology.incidence[-1, 0]).sum(axis=0)    # Plane+[N[-1] <- N[0]]
    return volumes, gradient


def carrier_densities(potentials: Potentials, intrinsic: float) -> Scalar:
    """Boltzmann's law relates the electron and hole densities (`[2] Scalar+[N[0]]`) to the chemical potentials."""
    return intrinsic * (potentials & counts).exp()


def energy_plane(potentials: Potentials, gap: float) -> Plane:
    """Electrostatic potential and half-gap offsets combine with thermal and photon directions into the energy covector
    field (`Plane+[N[0]]`)."""
    return (potentials & time) * mv.t + gap / 2 * (mv.n + mv.p) + mv.q + mv.l


def band_energies(potentials: Potentials, gap: float) -> Scalar:
    """Pairing the energy and potential planes with an electron and a hole gives the conduction band edge, valence band
    edge, and electron and hole quasi-Fermi levels (`[4] Scalar+[N[0]]`)."""
    N = potentials.gatype.site_shape
    energies = energy_plane(potentials, gap)                                  # Plane+[N[0]]
    return stack((energies & electron, -(energies & hole), potentials & electron, -(potentials & hole)))


def conductances(topology: Topology, gradient: SparseExtensor, semiconductor: Scalar, potentials: Potentials,
                 intrinsic: float, mobilities: np.ndarray) -> SparseExtensor:
    """Corner-mean carrier densities and `mobilities` (`[2]`) weight the isotropic drift-diffusion conductance map
    (`(Flux- <- Drive+)[N[-1]]`) on each simplex where `semiconductor` (`Scalar+[N[-1]]`) is nonzero."""
    N = topology.n_elements
    Drive = ga.gatype(gradient.gatype ^ Potentials)
    corner_mean = topology.incidence[-1, 0].mean(axis=0)                      # Scalar+[N[-1] <- N[0]]
    densities = corner_mean * carrier_densities(potentials, intrinsic)        # [2] Scalar+[N[-1]]
    return (semiconductor * densities * mobilities * Drive.commutator(carriers).regressive(carriers)).sum(axis=0)


def carrier_flows(topology: Topology, gradient: SparseExtensor, semiconductor: Scalar, potentials: Potentials,
                  intrinsic: float, mobilities: np.ndarray) -> Point:
    """The combined drive `-(gradient ^ potentials)` sweeps electrons and holes along each semiconductor simplex into
    their flow vectors (`[2] Point-[N[-1]]`)."""
    N = topology.n_elements
    corner_mean = topology.incidence[-1, 0].mean(axis=0)                      # Scalar+[N[-1] <- N[0]]
    densities = corner_mean * carrier_densities(potentials, intrinsic)        # [2] Scalar+[N[-1]]
    return semiconductor * densities * mobilities * (-(gradient ^ potentials)).commutator(carriers)


def potential_frames(topology: Topology, volumes: Scalar, gradient: SparseExtensor, permittivities: Scalar,
                     semiconductor: Scalar, dopants: Point, potentials: Potentials, held_reading: Sparse,
                     free_reading: Sparse, contacts: Potentials, generations: Scalar, intrinsic: float,
                     mobilities: np.ndarray, rate_constant: float, dt: float, limit: float,
                     tolerance: float) -> Iterator[Potentials]:
    """Yields the initial vertex potentials (`Potentials+[N[0]]`) and the implicit Newton solution after each step `dt`
    under held contact potentials (`[n_frames] Potentials+[V_held]`) and optical pair generation
    (`[n_frames] Scalar+[N[0]]`), solving directly for steady states when `dt` is infinite."""
    N = topology.n_elements
    corner_mean = topology.incidence[-1, 0].mean(axis=0)                      # Scalar+[N[-1] <- N[0]]
    # Each vertex's dual cell holds an equal corner share of the surrounding semiconductor simplices.
    dual_volumes = corner_mean.adjugate() * (volumes * semiconductor)         # Scalar+[N[0]]
    Drive = ga.gatype(gradient.gatype ^ Potentials)
    drive_map = -(gradient ^ Potentials)                                      # (Drive+ <- Potentials+)[N[-1] <- N[0]]
    # Pulling back across the regressive product satisfies `potentials & outflow(flux) == drive_map(potentials) & flux`
    # and gathers simplex fluxes into dual-cell outflows.
    outflow = drive_map.adjugate()                                            # (Point- <- Flux-)[N[0] <- N[-1]]
    drift_sweeps = Drive.commutator(carriers).regressive(carriers)            # [2] Flux- <- Drive+
    unit_conductances = (semiconductor * mobilities) * drift_sweeps           # [2] (Flux- <- Drive+)[N[-1]]
    displacement = permittivities * Drive.commutator(time).regressive(time)   # (Flux- <- Drive+)[N[-1]]
    yield potentials
    for contact, generation in zip(contacts, generations):
        previous_parcels = (carrier_densities(potentials, intrinsic) * carriers).sum(axis=0)   # Point-[N[0]]
        potentials = free_reading.adjugate() * (free_reading * potentials) + held_reading.adjugate() * contact
        while True:
            densities = carrier_densities(potentials, intrinsic)              # [2] Scalar+[N[0]]
            parcels = (densities * carriers).sum(axis=0)                      # Point-[N[0]]
            drive = drive_map(potentials)                                     # Drive+[N[-1]]
            conductance = ((corner_mean * densities) * unit_conductances).sum(axis=0)   # (Flux- <- Drive+)[N[-1]]
            rates = rate_constant * (densities[0] * densities[1] - intrinsic ** 2) - generation   # Scalar+[N[0]]
            # Along time, electron and hole conservation repeat each other, and Gauss's law supplies the independent
            # charge balance between electric displacement outflow and net dual-cell charge.
            charge = (densities * charges).sum(axis=0) + dopants              # Point-[N[0]]
            gauss = outflow(volumes * displacement(drive)) - dual_volumes * charge   # Point-[N[0]]
            residual = (dual_volumes * (parcels - previous_parcels) / dt + outflow(volumes * conductance(drive))
                        + dual_volumes * rates * pairs + gauss)               # Point-[N[0]]
            # Linearizing storage, charge, simplex fluxes (both through the drive and through corner densities), and
            # mass-action recombination gives the exact tangent map on the potential plane.
            capacitance = dual_volumes * (densities * (carriers / dt - charges) * (Potentials & counts)).sum(axis=0)
            drive_response = (volumes * (conductance + displacement))(drive_map)
            density_response = (spdiag(volumes * unit_conductances(drive)) * corner_mean
                                * spdiag(densities * (Potentials & counts))).sum(axis=0)
            pair_response = dual_volumes * rate_constant * densities[0] * densities[1] * (pairs * (Potentials & pairs))
            jacobian = capacitance + pair_response + outflow(drive_response + density_response)
            # Dividing each carrier balance by its local density keeps scarce minority carriers well conditioned beside
            # majority carriers many orders of magnitude denser.
            balance_weights = time * (mv.t & Point) + (counts * (count_planes & Point) / densities).sum(axis=0)
            free_jacobian = free_reading * balance_weights(jacobian) * free_reading.adjugate()
            step_change = -(free_reading.adjugate() * free_jacobian.solve(free_reading * balance_weights(residual)))
            # Because carrier densities depend exponentially on chemical potentials, damping large chemical steps
            # prevents overshoot while leaving purely electrostatic updates undamped.
            chemical_shift = (counts & step_change).abs().sum(axis=0)         # Scalar+[N[0]]
            potentials = potentials + step_change / (1 + chemical_shift / limit)
            if chemical_shift.max() < tolerance:
                break
        yield potentials


def terminal_readings(topology: Topology, volumes: Scalar, gradient: SparseExtensor, semiconductor: Scalar,
                      high_contact: np.ndarray, low_contact: np.ndarray, potentials: Potentials, intrinsic: float,
                      mobilities: np.ndarray) -> tuple[Scalar, Scalar]:
    """Quasi-Fermi levels and dual-cell outflows determine the voltage drop (`[] Scalar+`) from `high_contact` to
    `low_contact` and the terminal current (`[] Scalar+`) entering `low_contact`."""
    N = topology.n_elements
    drive_map = -(gradient ^ Potentials)                                      # (Drive+ <- Potentials+)[N[-1] <- N[0]]
    conductance = conductances(topology, gradient, semiconductor, potentials, intrinsic, mobilities)
    net_outflow = drive_map.adjugate()(volumes * conductance(drive_map(potentials)))   # Point-[N[0]]
    levels = potentials & electron                                            # Scalar+[N[0]]
    low_reading = topology.reading(low_contact, 0)                            # Scalar+[V_low <- N[0]]
    high_reading = topology.reading(high_contact, 0)                          # Scalar+[V_high <- N[0]]
    voltage = (low_reading * levels).sites.mean() - (high_reading * levels).sites.mean()
    return voltage, -(low_reading * (mv.t & net_outflow)).sites.sum()


def entropy_production(topology: Topology, volumes: Scalar, gradient: SparseExtensor, semiconductor: Scalar,
                       potentials: Potentials, generation: Scalar, intrinsic: float, mobilities: np.ndarray,
                       rate_constant: float, gap: float, sun: float) -> tuple[Scalar, Scalar, Scalar]:
    """Pairing transport drives with fluxes and pairing the thermodynamic entropy slope with reaction transitions gives
    the entropy production rates (`Scalar+[N[-1]]`, `Scalar+[N[0]]`, `Scalar+[N[0]]`) from transport, recombination,
    and solar absorption at relative solar temperature `sun`."""
    N = topology.n_elements
    dual_volumes = topology.incidence[-1, 0].mean(axis=0).adjugate() * (volumes * semiconductor)   # Scalar+[N[0]]
    densities = carrier_densities(potentials, intrinsic)                      # [2] Scalar+[N[0]]
    conductance = conductances(topology, gradient, semiconductor, potentials, intrinsic, mobilities)
    drive = -(gradient ^ potentials)                                          # Drive+[N[-1]]
    # Subtracting the potential plane and the solar Carnot loss `(1 - 1 / sun)` from the energy plane forms the
    # thermodynamic entropy slope over parcel quantities in units of Boltzmann's constant.
    entropy_slope = energy_plane(potentials, gap) - potentials - (1 - 1 / sun) * mv.l   # Plane+[N[0]]
    recombination = gap * heat - pairs                                        # [] Point-
    absorption = pairs - gap * light                                          # [] Point-
    rates = rate_constant * (densities[0] * densities[1] - intrinsic ** 2)    # Scalar+[N[0]]
    return (volumes * (drive & conductance(drive)), dual_volumes * rates * (entropy_slope & recombination),
            dual_volumes * generation * (entropy_slope & absorption))
