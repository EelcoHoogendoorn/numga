"""A snowflake growing: a seed of ice in undercooled water growing into a six-armed dendrite, by a phase field on an
unstructured mesh of triangles, in the plane joined with three directions, for the volume, the ice and the heat a
parcel of water holds.

Water cooled below its melting point freezes onto any ice it touches, and freezing releases latent heat, which warms
the water at the ice's edge and slows the freezing there. A bump on the edge reaches into colder water and sheds its
heat to more sides than the flat edge around it, so it grows faster: the tips outrun the sides, and each tip sprouts
branches of its own. The ice's surface energy depends on how its edge lies against the crystal's axes, six-fold for
ice, and steers the fastest tips along six directions, the six arms of a snowflake. The phase is 0 in water and 1 in
ice, with a smooth edge between, far wider than real ice's so that the mesh resolves it; the temperature is measured
from the cold water's, in units of its undercooling, so that the ice melts at one, and absolute zero lies at
`absolute_zero` on that scale.

The algebra is the plane, x+y+, joined with three degenerate directions, for volume, ice and heat: x+y+v0i0h0. A parcel
of water is a point weighed by its volume, offset along the ice's direction by the ice it holds and along the heat's by
its heat, which `mv.i &` and `mv.h &` read off it. Freezing is one fixed direction in that space of amounts,
`freezing = ice + latent_heat * heat`: ice gained, and its latent heat released. Ice is taken at water's density, so
that freezing leaves the volume alone, `mv.v & freezing == 0`, and the volume is a parcel's weight; ice and water share
one heat capacity and one conductivity. Two planes read what freezing does. The plane that reads the energy a parcel
holds, its heat less the latent heat its ice has given up, `mv.h - latent_heat * mv.i`, pairs with freezing to nothing:
freezing only moves energy from the water's latent store into its heat. The entropy's slope over a parcel's amounts,
`mv.h / absolute_temperatures - latent_heat / absolute_melting * mv.i`, heat raising the entropy by one over the
absolute temperature and ice holding the latent heat over the melting temperature less of it than water, pairs with
freezing to the entropy that freezing a unit of ice produces: positive in undercooled water, nothing at the melting
temperature. Times the melting temperature, that pairing is the push to freeze, so the push comes from the same latent
heat that warms the water.

The phase is an order parameter, not an amount, and carries the edge's free energy; the ice a parcel holds is a smooth
step in it, `phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)`, flat at water and at ice. That step is a device, not a
fact about ice: with its slope vanishing at water, the push to freeze vanishes there too, and the water stays a local
minimum of the free energy at any undercooling, metastable until the ice's edge reaches it. The phase relaxes up the
total entropy at fixed energy, the parcels' entropy less the edge's free energy over the melting temperature, and each
parcel moves along freezing by the ice its phase gained, so that the energy summed over the cells holds and the total
entropy only rises.

A drive is minus the gradient of a field wedged with the amount it acts on, and a material map, written as dyads over
the plane's directions, takes a drive to the flux it drives. The heat's drive, `-(gradients ^ mv.h)`, is conducted out
of each cell. The phase's drive, `-(gradients ^ mv.i)`, paired with the flux the same map makes of it, is the squared
steepness of the ice's edge, and half of that, times the edge's width squared, is its gradient energy; the width
depends on the edge's normal through the turn from the crystal's axis to the steepness, `steepness * crystal_axis`, an
even element whose sixth power has the scalar part `|steepness| ** 6 * cos(6 angle)`: the six-fold anisotropy, with no
angle taken. With a double well, that makes the edge's free energy, whose derivative at each vertex holds the phase
back. The derivative of an anisotropic gradient energy is notoriously fiddly, its cross terms coming from the normal's
dependence on the gradient; here it is written out: the six-fold term's slope with respect to the steepness is a
rotor's fifth power, `6 * crystal_axis * turn ** 5`, the Cahn-Hoffman vector of materials science, and each vertex's
share is its hat's drive paired with the energy's slope, a flux. The turn is the one place the plane's own metric
enters, as an angle needs it; its slope comes out as a plane, which the material map makes a flux, right here since
the isotropic map is that metric itself. A triangle's area stays intrinsic, like the join that measures it, and so
does what a cell holds; dividing by the cell gives back a rate. The stepping is jitted and scanned by JAX.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax

from numga import Algebra, stack
from numga.backend.jax import JaxContext
from numga.extensor import Extensor
from numga.sparse import spdiag

from examples.util.topology import Topology

jax.config.update("jax_enable_x64", True)

# The plane, and the volume, the ice and the heat a parcel holds, each a direction of its own with no metric.
ga = Algebra("x+y+") * Algebra("v0i0h0")
mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
# An amount a cell holds, intrinsic like the join that measures its area: the area, or the energy in it.
Amount = ga.gatype.antipseudoscalar()                                         # Antipseudoscalar-
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Drive = ga.gatype.from_blades("xi yi xh yh")                                  # Bivector+
across = mv.x.dual()                                                          # [] Point-
up = mv.y.dual()                                                              # [] Point-
# The plane's two directions, across and up.
directions = stack((across, up))                                              # [2] Point-
# A unit of volume, the weight of a place, a unit of ice and a unit of heat.
volume = mv.v.dual()                                                          # [] Point-
ice = mv.i.dual()                                                             # [] Point-
heat = mv.h.dual()                                                            # [] Point-


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Amount, Plane, Amount]:
    """Each triangle's area `Scalar-[faces]`, each corner's hat gradient `[corners] Plane+[faces]`, the plane through
    the facing side, along volume, ice and heat, over twice the area, so that it reads one from the side to the corner,
    and each vertex's share of the area around it `Scalar-[vertices]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, at unit volume, ice and heat.
    double_areas = (((sides[0] & sides[1]) & volume) & ice) & heat            # Scalar-[faces]
    areas = double_areas / 2                                                  # Scalar-[faces]
    gradients = (volume & ice & heat & sides) / double_areas                  # [corners] Plane+[faces]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=0)


def held_ice(phase: Scalar) -> Scalar:
    """The ice a parcel holds at this phase `Scalar+[vertices]`: none in water and all in ice, a smooth step between
    whose slope, `30 * phase ** 2 * (1 - phase) ** 2`, vanishes at both."""
    return phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)


def edge_free_energy(areas: Amount, cells: Amount, gradient: Extensor, spreading: Extensor, crystal_axis: Plane,
                     mean_width: float, anisotropy: float, regularization: float, well_height: float,
                     phase: Scalar) -> Amount:
    """The free energy of the ice's edge `[] Scalar-` at this phase `Scalar+[vertices]`, with this field's gradient
    `(Plane+ <- Scalar+)[faces <- vertices]` and the material map `Flux- <- Drive+`: the gradient energy, half the
    edge's width squared times its squared steepness, over the triangles, and the double well over the cells."""
    steepness = gradient(phase)                                               # Plane+[faces]
    drive = -(steepness ^ mv.i)                                               # Drive+[faces]
    squared_steepness = drive & spreading(drive)                              # Scalar+[faces]
    third_power = (steepness * crystal_axis) ** 3                             # Even+[faces]
    six_fold = third_power.scalar_product(third_power)                        # Scalar+[faces]
    reach = squared_steepness + regularization ** 2                           # Scalar+[faces]
    widths = mean_width * (1 + anisotropy * six_fold / reach ** 3)            # Scalar+[faces]
    well = well_height * phase ** 2 * (1 - phase) ** 2                        # Scalar+[vertices]
    return (areas * widths ** 2 * squared_steepness / 2).sites.sum() + (cells * well).sites.sum()


def edge_slope(topology: Topology, areas: Amount, gradients: Plane, cells: Amount, gradient: Extensor,
               spreading: Extensor, crystal_axis: Plane, mean_width: float, anisotropy: float, regularization: float,
               well_height: float, phase: Scalar) -> Amount:
    """The derivative of the edge's free energy with respect to the phase at each vertex `Scalar-[vertices]`, at this
    phase `Scalar+[vertices]`, with this field's gradient `(Plane+ <- Scalar+)[faces <- vertices]` and the material map
    `Flux- <- Drive+`: the gradient energy's share, each hat's drive paired with the energy's slope, a flux, and the
    double well's slope."""
    steepness = gradient(phase)                                               # Plane+[faces]
    # The phase's drive and the flux it drives; paired, the edge's squared steepness, whose slope is twice the flux.
    drive = -(steepness ^ mv.i)                                               # Drive+[faces]
    flux = spreading(drive)                                                   # Flux-[faces]
    squared_steepness = drive & flux                                          # Scalar+[faces]
    # The turn from the crystal's axis to the edge's normal, scaled by the steepness: the scalar part of its sixth power,
    # its third power against itself, is the steepness to the sixth times the six-fold cos(6 angle).
    turn = steepness * crystal_axis                                           # Even+[faces]
    third_power = turn ** 3                                                   # Even+[faces]
    six_fold = third_power.scalar_product(third_power)                        # Scalar+[faces]
    reach = squared_steepness + regularization ** 2                           # Scalar+[faces]
    widths = mean_width * (1 + anisotropy * six_fold / reach ** 3)            # Scalar+[faces]
    # The six-fold term's slope with respect to the steepness, a rotor's fifth power: a plane, measured in the plane's
    # metric, and the flux the material map, that same metric here, makes of it.
    slope = 6 * crystal_axis * turn ** 5                                      # Plane+[faces]
    slope_flux = spreading(-(slope ^ mv.i))                                   # Flux-[faces]
    # The widths' change as the edge turns; the gradient energy's slope, the widths squared times the flux and that
    # change times the steepness squared.
    turning = mean_width * anisotropy * (slope_flux / reach ** 3 - 6 * six_fold * flux / reach ** 4)   # Flux-[faces]
    pull = widths ** 2 * flux + squared_steepness * widths * turning          # Flux-[faces]
    # Each vertex's share of the gradient energy's change: its hat's drive paired with the energy's slope.
    hat_drives = -(gradients ^ mv.i)                                          # [corners] Drive+[faces]
    shares = (~topology.incidence[2, 0] * (areas * (hat_drives & pull))).sum(axis=0)   # Scalar-[vertices]
    well_slope = 2 * well_height * phase * (1 - phase) * (1 - 2 * phase)      # Scalar+[vertices]
    return shares + cells * well_slope


def total_entropy(areas: Amount, cells: Amount, gradient: Extensor, spreading: Extensor, crystal_axis: Plane,
                  mean_width: float, anisotropy: float, regularization: float, well_height: float, latent_heat: float,
                  melting_temperature: float, absolute_zero: float, parcels: Point, phase: Scalar) -> Amount:
    """The total entropy `[] Scalar-` of these parcels `Point-[vertices]` at this phase `Scalar+[vertices]`: each
    parcel's, the logarithm of its absolute temperature less the latent heat over the absolute melting temperature
    for each unit of its ice, summed over the cells, less the edge's free energy over the absolute melting
    temperature."""
    absolute_melting = melting_temperature - absolute_zero
    parcel_entropies = ((mv.h & parcels) - absolute_zero).log() - latent_heat / absolute_melting * (mv.i & parcels)
    edge = edge_free_energy(areas, cells, gradient, spreading, crystal_axis, mean_width, anisotropy, regularization,
                            well_height, phase)                               # [] Scalar-
    return (cells * parcel_entropies).sites.sum() - edge / absolute_melting


def operators(topology: Topology, areas: Amount, gradients: Plane) -> tuple[Extensor, Extensor, Extensor]:
    """A field's gradient across each triangle `(Plane+ <- Scalar+)[faces <- vertices]`, its corners' values on their
    hats' gradients; the material map taking a drive to the flux it drives `Flux- <- Drive+`, the same for heat and ice,
    each of the plane's directions swept through the amount, read by the drive and driving along it; and the heat each
    vertex's cell loses to each vertex's warmth `Scalar-[vertices <- vertices]`, its hat's drive conducted, read for its
    heat."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    gradient = (spdiag(gradients) * corners).sum(axis=0) * Scalar             # (Plane+ <- Scalar+)[faces <- vertices]
    sweeps = stack((directions & heat, directions & ice))                    # [amounts, 2] Flux-
    spreading = (sweeps * (sweeps & Drive)).sum(axis=-1).sum(axis=0)          # [] Flux- <- Drive+
    heat_drives = -(gradients ^ mv.h)                                         # [corners] Drive+[faces]
    local = areas * (mv.h & (gradients[:, None] ^ spreading(heat_drives[None, :])))   # [corners, corners] Scalar-[faces]
    conduction = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)
    return gradient, spreading, conduction


def phase_drive(topology: Topology, areas: Amount, gradients: Plane, cells: Amount, gradient: Extensor,
                spreading: Extensor, crystal_axis: Plane, mean_width: float, anisotropy: float, regularization: float,
                well_height: float, latent_heat: float, melting_temperature: float, absolute_zero: float,
                freezing: Point, parcels: Point, phase: Scalar) -> Amount:
    """The total entropy's slope with respect to the phase at each vertex at fixed energy, times the absolute melting
    temperature `Scalar-[vertices]`, for these parcels `Point-[vertices]` at this phase `Scalar+[vertices]`: the push
    to freeze, where the ice held follows the phase, less the edge's free energy's slope."""
    absolute_melting = melting_temperature - absolute_zero
    # The entropy's slope over a parcel's amounts: heat raises it by one over the absolute temperature, and ice holds
    # the latent heat over the absolute melting temperature less of it than water.
    absolute_temperatures = (mv.h & parcels) - absolute_zero                 # Scalar+[vertices]
    entropy_slope = mv.h / absolute_temperatures - latent_heat / absolute_melting * mv.i   # Plane+[vertices]
    # Paired with freezing, the entropy freezing a unit of ice produces; times the melting temperature, a free energy,
    # and times the slope of the ice held, the push to freeze.
    held_slope = 30 * phase ** 2 * (1 - phase) ** 2                           # Scalar+[vertices]
    push = absolute_melting * cells * (entropy_slope & freezing) * held_slope   # Scalar-[vertices]
    return push - edge_slope(topology, areas, gradients, cells, gradient, spreading, crystal_axis, mean_width,
                             anisotropy, regularization, well_height, phase)


def growth(topology: Topology, areas: Amount, gradients: Plane, cells: Amount, parcels: Point, phase: Scalar,
           crystal_axis: Plane, relaxation_time: float, mean_width: float, anisotropy: float, regularization: float,
           well_height: float, latent_heat: float, melting_temperature: float, absolute_zero: float, dt: float,
           n_frames: int, stride: int) -> Iterator[tuple[Point, Scalar]]:
    """The water's parcels `Point-[vertices]` and the phase `Scalar+[vertices]` at each of `n_frames` frames, `stride`
    steps of `dt` apart, from these: the phase relaxing up the total entropy at fixed energy in `relaxation_time`, its
    edge's width `mean_width` varied six-fold by `anisotropy` about the crystal's axis, a plane whose normal one arm
    grows along, the steepness regularized by `regularization` where it vanishes, its double well `well_height` high;
    each parcel moving along freezing, which releases `latent_heat` per unit of ice, by the ice its phase gained; ice
    melting at `melting_temperature` on a scale whose absolute zero is `absolute_zero`; and the heat conducted at unit
    conductivity."""
    gradient, spreading, conduction = operators(topology, areas, gradients)
    # Freezing: ice gained, and its latent heat released.
    freezing = ice + latent_heat * heat                                       # [] Point-

    def step(state: tuple[Point, Scalar], _) -> tuple[tuple[Point, Scalar], None]:
        """One step: the phase relaxes up the total entropy, each parcel takes up the ice its phase holds, its latent
        heat with it, and the heat is conducted."""
        parcels, phase = state
        drive = phase_drive(topology, areas, gradients, cells, gradient, spreading, crystal_axis, mean_width,
                            anisotropy, regularization, well_height, latent_heat, melting_temperature, absolute_zero,
                            freezing, parcels, phase)                         # Scalar-[vertices]
        next_phase = phase + dt * drive / (relaxation_time * cells)           # Scalar+[vertices]
        ice_gained = held_ice(next_phase) - (mv.i & parcels)                  # Scalar+[vertices]
        conducted = -(conduction * (mv.h & parcels)) / cells                  # Scalar+[vertices]
        return (parcels + ice_gained * freezing + dt * conducted * heat, next_phase), None

    @jax.jit
    def strided(state: tuple[Point, Scalar]) -> tuple[Point, Scalar]:
        return jax.lax.scan(step, state, None, length=stride)[0]

    state = (parcels, phase)
    for _ in range(n_frames):
        yield state
        state = strided(state)
