"""Adiabatic demagnetization: a crystal of a paramagnetic salt, a slab in the plane, its spins along the axis out of it,
taken around the magnetic refrigerator's cycle between a bath at one end and a load at the other; on a mesh of
triangles, in planar projective spacetime joined with a direction for the entropy the salt holds.

A paramagnetic salt holds spins that turn freely, each a two-level moment, and a crystal lattice that holds a little
heat. A field lines the spins up and so takes their entropy away; at a fixed entropy that entropy goes to the lattice,
which warms, and removing the field at a fixed entropy gives it back to the spins, so that the lattice cools: the
magnetocaloric effect. An ideal paramagnet's internal energy does not depend on its magnetization, as an ideal rubber's
does not depend on its stretch: the field it holds its magnetization against comes from its entropy, the temperature
times the entropy the spins would regain, as a rubber band's tension does. In a crystal heat takes time to cross the
lattice, so a field that changes faster warms or cools the crystal before its ends' reservoirs can take or give the
heat, and fronts of temperature run in from the end in contact.

The algebra is planar projective spacetime, x+y+t-w0, its degenerate w weighing a place, joined with one degenerate
direction for entropy: x+y+t-w0s0. A cell's state is one element: its magnetization, the patch of the plane its spins
turn in, `moment * share * mv.xy.dual()`, and the entropy it holds, `entropies * mv.s.dual()`, both per spin. Its
potentials are one element too, the field the magnet holds it at, a plane of space, and its temperature, along s; the
first law is their pairing with a change of state, `potentials & change`, the field's part `field & spin` the magnet's
work, and neither part reads the other's. The spins' entropy is that of two-level moments, a function of the
magnetization's share of saturation alone, and the lattice holds the rest at a constant heat capacity, so that the
salt's energy is the lattice's heat alone; its derivative over the state, the potentials, is the temperature times the
plane that reads the lattice's entropy, `temperature * (mv.s - spin_slope)`: the field is minus the temperature times
the spins' entropy's slope, Curie's law at small fields and the tanh of spin one half beyond. The energy's second
derivative, the stiffness, a map from a change of state to the change of the potentials, is two dyads: the potentials
with themselves over the heat capacity times the temperature, and the magnet's plane with itself over the spins'
susceptibility. Its symmetry is the Maxwell relation of the magnetocaloric effect, and its cross term, the temperature's
rise per magnetization at fixed entropy, is the field over the heat capacity. Inverted at each vertex it is the
compliance, the change of state a change of the potentials asks for, which gives the heat capacity at a held field and
the heat a change of field releases at a held temperature.

Each vertex is a point of weight one, and a corner's ramp is the facing side joined with time and entropy, along which
the temperature does not vary off the mesh, over that join closed by the corner; its gradient is what it reads across
space. The lattice conducts heat down the temperature's gradient, by dyads over the crystal's two axes, and the
gradient's adjugate gathers what it carries into what each cell loses: `gradient.adjugate()(medium(gradient))`, a form
on the vertices' temperatures. Each step is implicit in the temperatures: every cell's change of state is the compliance
on the field's shortfall from the magnet's and on the change of its temperature, and the cells' heat balance, conduction
and the links to the bath and the load included, is one solve.

The magnet follows the ideal cycle, a rectangle on the temperature-entropy chart between a bath and a load: magnetized
to its highest field at the bath's temperature, the salt gives the bath heat; isolated and demagnetized, it cools to the
load's temperature; demagnetized further there, to its lowest field, it takes heat from the load; isolated again and
magnetized, it warms back to the bath's temperature. The fields at the corners are the ideal salt's, and between them
the magnet's field runs evenly in its logarithm, so that at a fixed entropy the salt cools evenly in the logarithm of
its temperature.

Temperatures are in kelvin and fields in tesla; entropy is in Boltzmann's constant per spin, measured so that the
lattice holds none at the reference temperature, and energy in kelvin per spin; a magnetization is in kelvin per tesla
per spin, its saturation the spin's moment. Lengths are in the crystal's width, which holds a unit of spins on each unit
of its area, and times in the time a unit of heat capacity per spin takes to spread heat across a unit of length."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from numga import Algebra, NumpyContext, stack, where
from numga.sparse import SparseExtensor, spdiag

from numga.topology import Topology

# Planar projective spacetime, its degenerate w weighing a place, and the entropy the salt holds, a direction of its
# own with no metric.
space = Algebra("x+y+")
ga = Algebra("x+y+t-w0") * Algebra("s0")
mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
# The planes reading a place's run along the crystal and across it.
Plane = ga.gatype.vector().derive.within(space)                               # Vector+
# The magnetic field, a plane of space, and a magnetization, the patch of the plane its spins turn in.
Magnetic = ga.gatype.bivector().derive.within(space)                          # Bivector+
Spin = Magnetic.derive.dual                                                   # Antibivector-
# The plane that reads entropy, the temperature's, and an amount of entropy.
Thermal = ga.gatype.vector().derive.within(Algebra("s0"))                     # Vector+
Amount = Thermal.derive.dual                                                  # Antivector-
# The salt's state, its magnetization and its entropy; its potentials, the field and the temperature; and its
# stiffness, the map from a change of state to the change of its potentials.
State = ga.gatype(Spin + Amount)                                              # Antibivector- + Antivector-
Potential = ga.gatype(Magnetic + Thermal)                                     # Bivector+ + Vector+
Stiffness = ga.gatype((Potential, State))                                     # Potential+ <- State-
# The plane the magnet's field turns the spins in, a unit of magnetization along it, and a unit of entropy; a direction
# along time, and a unit of heat.
magnet_plane = mv.xy                                                          # [] Magnetic+
spin = magnet_plane.dual()                                                    # [] Spin-
entropy = mv.s.dual()                                                         # [] Amount-
time = mv.t.dual()                                                            # [] Point-
heat = mv.scalar([1.0]).dual()                                                # [] Antiscalar-
# The crystal's two axes, along it and across it, along which its lattice conducts alike.
axes = stack((mv.x.dual(), mv.y.dual()))                                      # [axes] Point-


# --- math -----------------------------------------------------------------------------
def spin_entropy(shares: Scalar) -> Scalar:
    """The entropy of two-level spins `[...] Scalar+` at these shares of their saturated magnetization `[...] Scalar+`:
    the mixing of their two populations, `(1 + shares) / 2` of them along the field and `(1 - shares) / 2` against it,
    the logarithm of two with no magnetization and nothing at saturation."""
    along, against = (1 + shares) / 2, (1 - shares) / 2                       # [...] Scalar+, Scalar+
    return -(along * along.log() + against * against.log())


def potentials(state: State, moment: float, heat_capacity: float, reference: float) -> Potential:
    """The salt's potentials `[...] Potential+` in this state `[...] State-`, its spins of this moment and its lattice
    of this heat capacity, holding no entropy at the reference temperature: the field the magnet holds it at, along the
    magnet's plane, and its temperature, along s. The salt's energy is its lattice's heat alone, so the potentials, the
    energy's derivative over the state, are the temperature times the plane that reads the lattice's entropy."""
    shares = (magnet_plane & state) / moment                                  # [...] Scalar+
    # The lattice holds the entropy the spins do not, and its temperature grows exponentially with it.
    lattice_entropy = (mv.s & state) - spin_entropy(shares)                   # [...] Scalar+
    temperature = reference * (lattice_entropy / heat_capacity).exp()         # [...] Scalar+
    # The spins' entropy's slope over the state, a reading of its changes: the spins lose entropy as they line up.
    spin_slope = -(shares.arctanh() / moment) * magnet_plane                  # [...] Magnetic+
    return temperature * (mv.s - spin_slope)


def stiffness(state: State, moment: float, heat_capacity: float, reference: float) -> Stiffness:
    """The salt's stiffness `[...] Potential+ <- State-` in this state `[...] State-`, the energy's second derivative, a
    map from a change of state to the change of the potentials: a change of state warms the lattice by the energy it
    adds over the heat capacity, which scales the potentials, and turns the spins' entropy's slope, against which the
    spins' susceptibility at fixed temperature holds them."""
    shares = (magnet_plane & state) / moment                                  # [...] Scalar+
    salt_potentials = potentials(state, moment, heat_capacity, reference)     # [...] Potential+
    temperature = salt_potentials & entropy                                   # [...] Scalar+
    warming = salt_potentials * (salt_potentials & State) / (heat_capacity * temperature)   # [...] Potential+ <- State-
    ordering = temperature / (moment ** 2 * (1 - shares ** 2)) * magnet_plane * (magnet_plane & State)   # [...] Potential+ <- State-
    return warming + ordering


def state_at(field: Magnetic, temperature: Scalar, moment: float, heat_capacity: float, reference: float) -> State:
    """The state `[...] State-` of the salt held at this field `[...] Magnetic+` and temperature `[...] Scalar+`: its
    spins' share of saturation the tanh of the moment times the field over the temperature, and its entropy the spins'
    and the lattice's."""
    shares = (moment * (field & spin) / temperature).tanh()                   # [...] Scalar+
    entropies = spin_entropy(shares) + heat_capacity * (temperature / reference).log()   # [...] Scalar+
    return moment * shares * spin + entropies * entropy


def drawn_spins(states: State, draws: Scalar, moment: float) -> Spin:
    """One spin of each cell `Spin-[vertices]`, drawn from the two populations of these states `State-[vertices]`: a
    two-level spin lies along the field or against it, a patch of the magnet's plane turning one way or the other, and
    of a cell's spins `(1 + shares) / 2` lie along it. Each spin keeps its draw `Scalar+[vertices]`, a number between
    none and one, and lies along the field while its draw is under that population: a spin of a low draw nearly always,
    one of a high draw only near saturation, so that the spins turn over one at a time as the states change."""
    shares = (magnet_plane & states) / moment                                 # Scalar+[vertices]
    along = draws < (1 + shares) / 2
    return moment * where(along, spin, -spin)


def corner_fields(bath_temperature: float, load_temperature: float, highest_field: float, lowest_field: float,
                  moment: float, heat_capacity: float, reference: float, n_halvings: int) -> Magnetic:
    """The fields at the ideal cycle's corners `[corners] Magnetic+`, of a salt in perfect contact with its reservoirs:
    where it reaches the bath's temperature, magnetized isolated from the load's corner; the highest field, at the
    bath's temperature; where it reaches the load's temperature, demagnetized isolated from the bath's corner; and the
    lowest field, at the load's temperature."""
    held_fields = stack((highest_field * magnet_plane, lowest_field * magnet_plane))   # [2] Magnetic+
    held_entropies = mv.s & state_at(held_fields, mv.scalar([[bath_temperature], [load_temperature]]), moment,
                                     heat_capacity, reference)                # [2] Scalar+
    # Isolated, the salt keeps its entropy, and at the other reservoir's temperature its spins hold what the lattice
    # does not: the spins' share there, and the field that holds it.
    others = mv.scalar([[load_temperature], [bath_temperature]])              # [2] Scalar+
    shares = shares_holding(held_entropies - heat_capacity * (others / reference).log(), n_halvings)   # [2] Scalar+
    reached = potentials(moment * shares * spin + held_entropies * entropy, moment, heat_capacity, reference)   # [2] Potential+
    cooled, warmed = (reached & spin) * magnet_plane                          # [] Magnetic+, Magnetic+
    return stack((warmed, held_fields[0], cooled, held_fields[1]))


def magnet(corners: Magnetic, starts: np.ndarray, ends: np.ndarray, fractions: np.ndarray) -> Magnetic:
    """The magnet's field at each step `[steps] Magnetic+`, these fractions `[steps]` of the way from the corner each
    step's phase starts at to the one it ends at `[steps]`, of these corners' fields `[corners] Magnetic+`: evenly in
    the logarithm of its strength, so that a salt at a fixed entropy cools evenly in the logarithm of its temperature."""
    strengths = (corners & spin).log()                                       # [corners] Scalar+
    return (strengths[starts] * (1 - fractions) + strengths[ends] * fractions).exp() * magnet_plane


def triangles(topology: Topology, vertices: Point) -> tuple[Scalar, Plane]:
    """Each triangle's area `Scalar+[faces]`, and each corner's hat gradient `[corners] Plane+[faces]`, at these
    vertices, points of weight one: what the corner's ramp reads across space, the ramp reading one at the corner and
    nothing along the facing side."""
    corners = topology.incidence[2, 0] * vertices                             # [corners] Point-[faces]
    facing = topology.off[2, 0] * vertices                                    # [2, corners] Point-[faces]
    # Each corner's ramp: the facing side, joined with time and entropy, along which the temperature does not vary off
    # the mesh, over that join closed by the corner.
    sides = facing[0] & facing[1]                                             # [corners] Antibivector-[faces]
    ramps = (sides & time & entropy) / (sides & time & entropy & corners)     # [corners] Plane+[faces]
    # Twice a triangle's area: its corners joined, along time and entropy.
    double_areas = corners[0] & corners[1] & corners[2] & time & entropy      # Antiscalar-[faces]
    # A ramp's gradient, what it reads across space, its value at the origin, read along w, dropped.
    return double_areas.norm() / 2, ramps.cast(Plane)


def conduction(topology: Topology, areas: Scalar, gradients: Plane, conductivity: float) -> SparseExtensor:
    """The heat each vertex's cell loses as the temperature rises at each vertex `(Antiscalar- <- Scalar+)[vertices <-
    vertices]`, through a lattice of this conductivity on the triangles of these areas `Scalar+[faces]` and hat
    gradients `[corners] Plane+[faces]`: the temperature's gradient drives a flux down it, the same along both of the
    crystal's axes, which the gradient's adjugate gathers into what each cell gives up."""
    gradient = (spdiag(gradients) * topology.incidence[2, 0]).sum(axis=0) * Scalar   # (Plane+ <- Scalar+)[faces <- vertices]
    medium = areas * conductivity * (axes * (Plane & axes)).sum(axis=0)       # (Point- <- Plane+)[faces]
    return gradient.adjugate()(medium(gradient))


def step(cells: Scalar, conducting: SparseExtensor, states: State, field: Magnetic, links: Scalar, reservoirs: Scalar,
         moment: float, heat_capacity: float, reference: float, dt: float) -> State:
    """The crystal's states `State-[vertices]` a step of `dt` on from these, the magnet's field moved to this one `[]
    Magnetic+` and each vertex's cell `Scalar+[vertices]` tied by these links' conductances `Scalar+[vertices]` to its
    reservoir's temperature `Scalar+[vertices]`, the cells conducting heat to each other as `conducting` has it
    `(Antiscalar- <- Scalar+)[vertices <- vertices]`: implicit in the temperatures' change."""
    salt_potentials = potentials(states, moment, heat_capacity, reference)    # Potential+[vertices]
    temperatures = salt_potentials & entropy                                  # Scalar+[vertices]
    # The compliance, the stiffness inverted vertex by vertex: the change of state per unit of field at a held
    # temperature, and per unit of temperature at a held field.
    compliance = stiffness(states, moment, heat_capacity, reference).batch().inverse().field()   # (State- <- Potential+)[vertices]
    along_field = compliance(magnet_plane)                                    # State-[vertices]
    along_temperature = compliance(mv.s)                                      # State-[vertices]
    # The salt's field's shortfall from the magnet's; the heat capacity at a held field, and the heat a cell takes in
    # to keep its temperature as the field changes, negative as it is magnetized.
    shortfall = (field - salt_potentials) & spin                              # Scalar+[vertices]
    capacities = cells * temperatures * (mv.s & along_temperature)            # Scalar+[vertices]
    taken = cells * temperatures * (mv.s & along_field) * shortfall           # Scalar+[vertices]
    # The heat balance over the step, implicit in the temperatures' change: what warms the cells is what the links and
    # the conduction bring in, less what the change of field takes.
    linking = spdiag(links * heat * Scalar)                                   # (Antiscalar- <- Scalar+)[vertices <- vertices]
    balance = spdiag(capacities * heat * Scalar) + dt * (conducting + linking)   # (Antiscalar- <- Scalar+)[vertices <- vertices]
    brought = dt * (linking(reservoirs - temperatures) - conducting(temperatures)) - taken * heat   # Antiscalar-[vertices]
    warming = balance.solve(brought)                                          # Scalar+[vertices]
    return states + shortfall * along_field + warming * along_temperature


def run(cells: Scalar, conducting: SparseExtensor, states: State, fields: Magnetic, links: Scalar,
        reservoirs: Scalar, durations: np.ndarray, n_steps: np.ndarray, moment: float, heat_capacity: float,
        reference: float) -> Iterator[tuple[State, Potential, Magnetic, Scalar]]:
    """The crystal's states `State-[vertices]` and potentials `Potential+[vertices]`, the magnet's field `[] Magnetic+`
    and the links' conductances `Scalar+[vertices]`, at the start and after each frame's steps, these many to each frame
    `[frames]`: one step for each of the magnet's fields `[steps] Magnetic+`, links `[steps] Scalar+[vertices]` and
    durations `[steps]`, from these states; the cells, the conduction and the reservoirs as `step` takes them."""
    yield states, potentials(states, moment, heat_capacity, reference), fields[0], links[0]
    ends = np.cumsum(n_steps)
    for start, end in zip(ends - n_steps, ends):
        for index in range(start, end):
            states = step(cells, conducting, states, fields[index], links[index], reservoirs, moment, heat_capacity,
                          reference, durations[index])
        yield states, potentials(states, moment, heat_capacity, reference), fields[end - 1], links[end - 1]


# --- plumbing -------------------------------------------------------------------------
def shares_holding(spin_entropies: Scalar, n_halvings: int) -> Scalar:
    """The shares of saturation `[...] Scalar+` at which two-level spins hold these entropies `[...] Scalar+`, found by
    this many halvings of the shares from none to saturation: the spins' entropy falls from the logarithm of two with no
    magnetization to nothing at saturation."""
    lower, upper = 0 * spin_entropies, 0 * spin_entropies + 1                 # [...] Scalar+, Scalar+
    for _ in range(n_halvings):
        middle = (lower + upper) / 2                                          # [...] Scalar+
        # Spins at the middle share holding more entropy than these are short of their share.
        short = spin_entropy(middle) > spin_entropies
        lower, upper = where(short, middle, lower), where(short, upper, middle)
    return (lower + upper) / 2
