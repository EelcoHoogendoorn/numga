"""Semiconductor devices by drift and diffusion, on a mesh of simplices of any dimension, in space and time joined with
directions for the electrons, the holes, the heat and the light: a p-n junction along a bar, its junction forming as
electrons and holes spill across and recombine, its current under a bias following Shockley's diode law, and under
light a solar cell; a MOSFET across a cross-section, its gate gathering electrons into a channel between two wells;
and a solar cell across a cross-section between the fingers of its front contact, the light making pairs that fade
with depth and none in the fingers' shadow.

Acceptors take electrons from the crystal and leave mobile holes behind; donors give it electrons. Where the two meet,
electrons spill from the n side into the p side and holes the other way, and recombine there, each pair giving its
energy, the band gap, to the lattice as heat. The dopants they leave behind are charged, and their field, across a
depletion region empty of carriers, holds the rest back: in equilibrium each carrier's electrochemical potential, the
Fermi level, is level throughout, and the bands bend by the built-in voltage. A forward bias lowers that barrier and
the current grows exponentially with it; a reverse bias raises it, and the current saturates at what thermal
generation supplies. Light makes pairs, which the field sweeps across the junction: a photocurrent against the diode's
own, the solar cell. A gate, a conductor held at its voltage across an oxide that holds no carriers, draws electrons to
the surface beneath it: past a threshold they form a channel joining two electron-rich wells, which a drain voltage
drives a current along.

The algebra is space and time, with a degenerate w that weighs a place, x+t-w0 along a bar or x+y+t-w0 across a
cross-section, joined with four degenerate directions, for electrons, holes, heat and light, n0p0q0l0; the module is
instantiated per algebra. Charge rides time: a unit of charge is `-mv.t.dual()`, an electron is a unit along n with a
negative charge, a hole a unit along p with a positive one, and the dopants' fixed charge lies along time alone. The
potentials at each vertex are one plane, the electric potential along t and the carriers' chemical potentials along n
and p, and it reads each carrier's electrochemical potential, its quasi-Fermi level, `potentials & electron`. Each
vertex is a point of weight one, and a field over space varies across each simplex by its corners' ramps: a corner's
ramp is the join of the corners off it with time and every amount, over that join closed by the corner itself, and its
gradient is what it reads across space. The potentials' drive, `-(gradients ^ potentials)`, pushes each carrier by what
it holds, drift and diffusion in one; a carrier drifts along the force its drive puts on it,
`drive.commutator(carrier)`, and its flux, the carrier swept along that drift, `drive.commutator(carrier).regressive(
carrier)`, times its mobility and density, is a material map the same in every direction. The same map on a unit of
charge, times the permittivity, makes the drive the field's displacement. The adjugate of the map that makes the drive
reads a flux back across the regressive product, the potentials first, as a reading does, `potentials &
driving.adjugate()(flux) == driving(potentials) & flux`, so that the potentials read what it gathers as they read every
amount; it gathers the fluxes into what flows out of each cell. Gauss's law, the field's outflow from each cell against
the charge it holds, joins the carriers' balance along time, where that balance only repeats itself. Each step is
implicit, solved by Newton's method until its step is small, whose map is the carriers' capacitance and the response of
their fluxes and of the field gathered the same way. A contact holds the potentials at its vertices: an ohmic contact
all three, at the neutral semiconductor's chemical potentials, and a gate all three too, the electric potential at the
gate's voltage and the carriers' idle, its vertices on the oxide alone, where no carriers are; the oxide carries the
field's displacement and holds no carriers, its simplices of no semiconductor.

Recombination is a direction among the amounts, `gap * heat - (electron + hole)`, at the rate mass action gives,
`rate_constant * (electrons * holes - intrinsic ** 2)`; absorption is its reverse with its energy drawn from the light,
`electron + hole - gap * light`. The energy plane pairs with both to nothing. The entropy's slope over the amounts is
the energy plane less the potentials, over the temperature, the light's free energy being its energy less the part the
sun's temperature carries away; paired with each process it gives the entropy the process produces, and the drive
paired with the flux it drives gives the entropy transport produces, none of them negative.

Energies and potentials are in units of the thermal energy, densities in the doping's, lengths in the Debye length of
that doping, and times in the time an electron takes to diffuse across a Debye length, which is also the dielectric
relaxation time. The lattice holds the device at its temperature, taking the heat away.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from functools import reduce
from operator import and_

import numpy as np

from numga import Algebra, stack
from numga.gatype import GAType
from numga.backend.context import Context
from numga.sparse import SparseExtensor, spdiag

from numga.topology import Topology

# Supplied by examples.instantiate: space and time, with a degenerate w that weighs a place, joined with the electrons,
# the holes, the heat and the light a cell holds or trades, each a direction of its own with no metric; and the context
# to compute in. Space is the example's to place its vertices in, points of weight one, and the planes that read a place
# there its to name; the core reads space through the vertices and those planes.
ga: Algebra
ctx: Context
mv = ctx.multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# The potentials, the electric potential along t and the carriers' chemical potentials along n and p.
Potentials = Plane.derive.within(Algebra("t-n0p0"))                           # Vector+
# A unit of charge, along time; an electron, a unit of the electrons' count with a negative charge, and a hole, a unit
# of the holes' count with a positive one; a unit of heat, and of the light's energy.
time = -mv.t.dual()                                                           # [] Point-
electron = mv.n.dual() - time                                                 # [] Point-
hole = mv.p.dual() + time                                                     # [] Point-
heat = mv.q.dual()                                                            # [] Point-
light = mv.l.dual()                                                           # [] Point-
# An electron and a hole, what recombination takes and absorption makes.
pairs = electron + hole                                                       # [] Point-
# The carriers; their counts, which their chemical potentials read, and their charges, along time.
carriers = stack((electron, hole))                                            # [carriers] Point-
counts = stack((mv.n.dual(), mv.p.dual()))                                    # [carriers] Point-
charges = carriers - counts                                                   # [carriers] Point-
# The planes that read each carrier's count off an amount.
counting = stack((mv.n, mv.p))                                                # [carriers] Plane+
# Time and every amount, joined: what a field over space does not vary along.
unread = time & counts[0] & counts[1] & heat & light                          # [] Span-


# --- math -----------------------------------------------------------------------------
def simplices(topology: Topology, vertices: Point, Space: GAType) -> tuple[Scalar, SparseExtensor, SparseExtensor]:
    """Each simplex's volume `Scalar+[simplices]`; the drive across each simplex of the potentials on its corners
    `(Drive+ <- Potentials+)[simplices <- vertices]`, minus their slope, their corners' values on their ramps'
    gradients, wedged with what each acts on; and what flows out of each vertex's cell of what flows in the simplices
    about it `(Point- <- Flux-)[vertices <- simplices]`, the drive's map read backwards: at these vertices, points of
    weight one, the gradients read across space by the planes `Space`."""
    corners = topology.incidence[-1, 0] * vertices                            # [corners] Point-[simplices]
    # Each corner's ramp, which reads one at the corner and nothing at the others: the join of the corners off it, with
    # time and every amount, over that join closed by the corner; the join's order cancels between the two.
    complements = reduce(and_, topology.off[-1, 0] * vertices, unread)        # [corners] Plane-[simplices]
    ramps = complements / (complements & corners)                             # [corners] Plane+[simplices]
    # A ramp's gradient, what it reads across space, its value at the origin, read along w, dropped.
    gradients = ramps.cast(Space)                                             # [corners] Plane+[simplices]
    # A simplex's volume: the join of its corners, with time and every amount, over the factorial of its dimension, an
    # ordinary number for the integrals to weigh by.
    joins = reduce(and_, corners, unread)                                     # Scalar-[simplices]
    volumes = joins.norm() / math.factorial(topology.n_dim)                   # Scalar+[simplices]
    slopes = (spdiag(gradients) * topology.incidence[-1, 0]).sum(axis=0)      # Plane+[simplices <- vertices]
    driving = -(slopes ^ Potentials)                                          # (Drive+ <- Potentials+)[simplices <- vertices]
    # The drive's adjugate reads a flux back across the regressive product, the potentials first, as a reading does,
    # `potentials & driving.adjugate()(flux) == driving(potentials) & flux`, so that the potentials read what it gathers
    # as they read every amount, `potentials & electron`: it gathers each flux into what flows out of the cells of its
    # simplex's corners.
    return volumes, driving, driving.adjugate()


def carrier_densities(potentials: Potentials, intrinsic: float) -> Scalar:
    """The electrons and the holes per volume at these potentials `[carriers] Scalar+[vertices]`: the intrinsic
    density times the exponential of each carrier's chemical potential, Boltzmann's distribution."""
    return intrinsic * (potentials & counts).exp()


def energy(potentials: Potentials, gap: float) -> Plane:
    """The plane reading a parcel's energy at these potentials `Plane+[vertices]`: its charge at the electric
    potential, each electron at the conduction band's edge and each hole at the valence band's, a gap apart with the
    middle of the gap at nothing, and heat and light at their own energy."""
    return (potentials & time) * mv.t + gap / 2 * (mv.n + mv.p) + mv.q + mv.l


def bands(potentials: Potentials, gap: float) -> Scalar:
    """The conduction band's edge, the valence band's, and the electrons' and the holes' quasi-Fermi levels at these
    potentials `[4] Scalar+[vertices]`: the energy plane and the potentials read on an electron and on a hole, the
    holes' read as the electrons missing."""
    energies = energy(potentials, gap)                                        # Plane+[vertices]
    return stack((energies & electron, -(energies & hole), potentials & electron, -(potentials & hole)))


def conductances(topology: Topology, driving: SparseExtensor, semiconductor: Scalar, potentials: Potentials,
                 intrinsic: float, mobilities: np.ndarray) -> SparseExtensor:
    """The carriers' conductance in each simplex at these potentials `(Flux- <- Drive+)[simplices]`, for the drives this
    map makes `(Drive+ <- Potentials+)[simplices <- vertices]`: each carrier swept along the drift a drive puts it to,
    times its mobility `[carriers]` and its density, the mean of the simplex's corners', where the simplex is
    semiconductor `Scalar+[simplices]`, one, and none where it holds no carriers."""
    Drive = ga.gatype(driving.cells)
    corner_mean = topology.incidence[-1, 0].mean(axis=0)                      # Scalar+[simplices <- vertices]
    densities = corner_mean * carrier_densities(potentials, intrinsic)        # [carriers] Scalar+[simplices]
    return (semiconductor * densities * mobilities * Drive.commutator(carriers).regressive(carriers)).sum(axis=0)


def flows(topology: Topology, driving: SparseExtensor, semiconductor: Scalar, potentials: Potentials, intrinsic: float,
          mobilities: np.ndarray) -> Point:
    """Each carrier's flow in each simplex at these potentials `[carriers] Point-[simplices]`, for the drives this map
    makes `(Drive+ <- Potentials+)[simplices <- vertices]`: the force its drive puts on it, the direction it drifts
    along, times its mobility `[carriers]` and its density, the mean of the simplex's corners', where the simplex is
    semiconductor `Scalar+[simplices]`."""
    corner_mean = topology.incidence[-1, 0].mean(axis=0)                      # Scalar+[simplices <- vertices]
    densities = corner_mean * carrier_densities(potentials, intrinsic)        # [carriers] Scalar+[simplices]
    return semiconductor * densities * mobilities * driving(potentials).commutator(carriers)


def run(topology: Topology, volumes: Scalar, driving: SparseExtensor, outflow: SparseExtensor, permittivities: Scalar,
        semiconductor: Scalar, dopants: Point, potentials: Potentials, holding: SparseExtensor, free: SparseExtensor,
        contacts: Potentials, generations: Scalar, intrinsic: float, mobilities: np.ndarray, rate_constant: float,
        dt: float, limit: float, tolerance: float) -> Iterator[Potentials]:
    """The potentials `Potentials+[vertices]` from these, and after each frame's step of `dt`, in a device of these
    dopants `Point-[vertices]`, its simplices of these permittivities `Scalar+[simplices]`, semiconductor where
    `semiconductor` `Scalar+[simplices]` is one and holding no carriers where it is nothing: the vertices `holding`
    reads `Scalar+[held <- vertices]` at that frame's potentials of the contacts `[frames] Potentials+[held]`, and the
    rest, which `free` reads `Scalar+[free <- vertices]`, free; the light making that frame's generation of pairs per
    volume and time at each vertex `[frames] Scalar+[vertices]`; the carriers of these mobilities `[carriers]` at this
    intrinsic density, recombining at this rate constant. Each step is implicit, solved by Newton's steps, each
    changing the chemical potentials at a vertex by not much more than `limit`, until one changes none by more than
    `tolerance`; at an infinite step, the steady state."""
    corner_mean = topology.incidence[-1, 0].mean(axis=0)                      # Scalar+[simplices <- vertices]
    # Each vertex's cell of the semiconductor, which holds the carriers: a share of each simplex around it, one over the
    # simplex's count of corners.
    cells = corner_mean.adjugate() * (volumes * semiconductor)                # Scalar+[vertices]
    # The drives the map makes, a slope in space wedged with what each potential acts on. A carrier drifts along the
    # force a drive puts on it, and its flux is the carrier swept along that drift, the same in every direction: each
    # carrier's conductance per unit of its density, where a simplex is semiconductor. A unit of charge swept so is the
    # field's displacement.
    Drive = ga.gatype(driving.cells)
    drifting = Drive.commutator(carriers).regressive(carriers)                # [carriers] Flux- <- Drive+
    conducting = (semiconductor * mobilities) * drifting                      # [carriers] (Flux- <- Drive+)[simplices]
    displacement = permittivities * Drive.commutator(time).regressive(time)   # (Flux- <- Drive+)[simplices]
    yield potentials
    for contact, generation in zip(contacts, generations):
        previous_parcels = (carrier_densities(potentials, intrinsic) * carriers).sum(axis=0)   # Point-[vertices]
        # The contacts take their potentials; the rest start from where they were.
        potentials = free.adjugate() * (free * potentials) + holding.adjugate() * contact  # Potentials+[vertices]
        while True:
            densities = carrier_densities(potentials, intrinsic)              # [carriers] Scalar+[vertices]
            parcels = (densities * carriers).sum(axis=0)                      # Point-[vertices]
            drive = driving(potentials)                                       # Drive+[simplices]
            conductance = ((corner_mean * densities) * conducting).sum(axis=0)   # (Flux- <- Drive+)[simplices]
            # Recombination by mass action, less the light's generation, pairs per volume and time.
            rates = rate_constant * (densities[0] * densities[1] - intrinsic ** 2) - generation   # Scalar+[vertices]
            # Gauss's law, along time: the field's outflow from each cell against the charge it holds.
            charge = (densities * charges).sum(axis=0) + dopants              # Point-[vertices]
            gauss = outflow(volumes * displacement(drive)) - cells * charge  # Point-[vertices]
            # What each cell holds of the carriers, against what it held, what their fluxes take out and what
            # recombines; along time this only repeats the electrons' and the holes' balance, which Gauss's law
            # completes there.
            residual = (cells * (parcels - previous_parcels) / dt + outflow(volumes * conductance(drive))
                        + cells * rates * pairs + gauss)                      # Point-[vertices]
            # Newton's map: the carriers' capacitance over the step and their charge's in Gauss's law; the response
            # of the carriers' fluxes and of the field through the drive, and of the fluxes through the densities
            # on each simplex, by each carrier's flux per unit of its density, gathered into each cell's outflow; and
            # recombination's, by the pairs' dyad.
            capacitance = cells * (densities * (carriers / dt - charges) * (Potentials & counts)).sum(axis=0)
            through_drive = (volumes * (conductance + displacement))(driving)
            through_densities = (spdiag(volumes * conducting(drive)) * corner_mean
                                 * spdiag(densities * (Potentials & counts))).sum(axis=0)
            recombining = cells * rate_constant * densities[0] * densities[1] * (pairs * (Potentials & pairs))
            newton = capacitance + recombining + outflow(through_drive + through_densities)
            # Each carrier's balance taken per carrier, over its density, the charge's as it is: a carrier far scarcer
            # than the other, by many powers of e, still weighs alike in the solve.
            per_carrier = time * (mv.t & Point) + (counts * (counting & Point) / densities).sum(axis=0)
            solving = free * per_carrier(newton) * free.adjugate()            # (Point- <- Potentials+)[free <- free]
            change = -(free.adjugate() * solving.solve(free * per_carrier(residual)))  # Potentials+[vertices]
            # Newton's step scaled down at each vertex where it would change the chemical potentials there by much
            # more than the limit: the densities follow them exponentially, and a full step from far off overshoots by
            # powers of e. The residual is linear in the electric potential, whose change needs no holding back.
            changes = (counts & change).abs().sum(axis=0)                     # Scalar+[vertices]
            potentials = potentials + change / (1 + changes / limit)
            if changes.max() < tolerance:
                break
        yield potentials


def terminal(topology: Topology, volumes: Scalar, driving: SparseExtensor, outflow: SparseExtensor,
             semiconductor: Scalar, raised: np.ndarray, grounded: np.ndarray, potentials: Potentials, intrinsic: float,
             mobilities: np.ndarray) -> tuple[Scalar, Scalar]:
    """The voltage between two contacts at these potentials `[] Scalar+`, the rise of the electrons' quasi-Fermi level
    from the contact at the vertices `raised` to the one at the vertices `grounded`, each at one level, and the current
    through the device from the first to the second `[] Scalar+`, the charge the carriers' fluxes bring into the
    second's cells, against what flows out of them."""
    conductance = conductances(topology, driving, semiconductor, potentials, intrinsic, mobilities)
    outflowing = outflow(volumes * conductance(driving(potentials)))          # Point-[vertices]
    levels = potentials & electron                                            # Scalar+[vertices]
    rise = (topology.reading(grounded, 0) * levels).sites.mean() - (topology.reading(raised, 0) * levels).sites.mean()
    return rise, -(topology.reading(grounded, 0) * (mv.t & outflowing)).sites.sum()


def entropy_production(topology: Topology, volumes: Scalar, driving: SparseExtensor, semiconductor: Scalar,
                       potentials: Potentials, generation: float, intrinsic: float, mobilities: np.ndarray,
                       rate_constant: float, gap: float, sun: float) -> tuple[Scalar, Scalar, Scalar]:
    """The entropy produced per time at these potentials, in units of Boltzmann's constant, by the carriers' transport
    across each simplex `Scalar+[simplices]`, the drive read on the flux it drives, by recombination in each cell
    `Scalar+[vertices]` and by the light's absorption in each cell `Scalar+[vertices]`, each process's rate times the
    entropy's slope read on it; the light making this generation of pairs per volume and time, its energy arriving at
    the sun's temperature `sun`, in the lattice's."""
    cells = topology.incidence[-1, 0].mean(axis=0).adjugate() * (volumes * semiconductor)  # Scalar+[vertices]
    densities = carrier_densities(potentials, intrinsic)                      # [carriers] Scalar+[vertices]
    conductance = conductances(topology, driving, semiconductor, potentials, intrinsic, mobilities)
    drive = driving(potentials)                                               # Drive+[simplices]
    # The entropy's slope over a parcel's amounts, at the lattice's temperature: the energy plane less the
    # potentials, the light's free energy being its energy less the share the sun's temperature takes along.
    entropy_slope = energy(potentials, gap) - potentials - (1 - 1 / sun) * mv.l   # Plane+[vertices]
    recombination = gap * heat - pairs                                        # [] Point-
    absorption = pairs - gap * light                                          # [] Point-
    rates = rate_constant * (densities[0] * densities[1] - intrinsic ** 2)    # Scalar+[vertices]
    return (volumes * (drive & conductance(drive)), cells * rates * (entropy_slope & recombination),
            cells * generation * (entropy_slope & absorption))
