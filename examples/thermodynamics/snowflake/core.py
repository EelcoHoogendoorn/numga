"""A snowflake growing: a seed of ice in undercooled water growing into a six-armed dendrite, by a phase field on an
unstructured mesh of triangles, in the plane joined with three directions, for the volume, the ice and the heat a
parcel of water holds.

Water cooled below its melting point freezes onto any ice it touches, and freezing releases latent heat, which warms
the water at the ice's edge and slows the freezing there. A bump on the edge reaches into colder water and sheds its
heat to more sides than the flat edge around it, so it grows faster: the tips outrun the sides, and each tip sprouts
branches of its own. The ice's surface energy depends on how its edge lies against the crystal's axes, six-fold for
ice, and steers the fastest tips along six directions, the six arms of a snowflake. The ice fraction is 0 in water
and 1 in ice, with a thin, smooth edge between; the temperature is measured from the cold water's, in units of its
undercooling, so that the ice melts at one.

The algebra is the plane, x+y+, joined with three degenerate directions, for volume, ice and heat: x+y+v0i0h0. A
parcel of water is a point weighed by its volume, offset along the ice's direction by the ice it holds and along the
heat's by its heat, which `mv.i &` and `mv.h &` read off it. Freezing is one fixed direction in that space of amounts,
`freezing = ice + latent_heat * heat`: ice gained, and its latent heat released. The plane that reads the energy a
parcel holds, its heat less the latent heat its ice has given up, `mv.h - latent_heat * mv.i`, pairs with freezing to
nothing: freezing only moves energy from the water's latent store into its heat.

A drive is minus the gradient of a field wedged with the amount it acts on, and a material map, written as dyads over
the plane's directions, takes a drive to the flux it drives. The heat's drive, `-(gradients ^ mv.h)`, is conducted out
of each cell. The ice's drive, `-(gradients ^ mv.i)`, paired with the flux the same map makes of it, is the squared
steepness of the ice's edge, and half of that, times the edge's width squared, is its gradient energy; the width
depends on the edge's normal through the turn from the crystal's axis to the steepness, `steepness * crystal_axis`, an
even element whose sixth power has the scalar part `|steepness| ** 6 * cos(6 angle)`: the six-fold anisotropy, with no
angle taken. With a double well tilted towards ice by the undercooling, that makes the ice's free energy, whose
derivative at each vertex drives the freezing. The derivative of an anisotropic gradient energy is notoriously fiddly,
its cross terms coming from the normal's dependence on the gradient; here it is written out: the six-fold term's slope
with respect to the steepness is a rotor's fifth power, `6 * crystal_axis * turn ** 5`, the Cahn-Hoffman vector of
materials science, and each vertex's share is its hat's drive paired with the energy's slope, a flux. The turn is the
one place the plane's own metric enters, as an angle needs it; its slope comes out as a plane, which the material map
makes a flux, right here since the isotropic map is that metric itself. A triangle's area stays intrinsic, like the
join that measures it, and so does what a cell holds; dividing by the cell gives back a rate. The stepping is jitted
and scanned by JAX.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax
import numpy as np

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


def free_energy_slope(topology: Topology, areas: Amount, gradients: Plane, cells: Amount, gradient: Extensor,
                      spreading: Extensor, crystal_axis: Plane, mean_width: float, anisotropy: float,
                      regularization: float, tilt_strength: float, tilt_steepness: float, melting_temperature: float,
                      fraction: Scalar, temperatures: Scalar) -> Amount:
    """The derivative of the ice's free energy with respect to its fraction at each vertex `Scalar-[vertices]`, at this
    ice fraction and these temperatures `Scalar+[vertices]`, with this field's gradient `(Plane+ <- Scalar+)[faces <-
    vertices]` and the material map `Flux- <- Drive+`: the gradient energy's share, each hat's drive paired with the
    energy's slope, a flux, and the double well's slope."""
    steepness = gradient(fraction)                                            # Plane+[faces]
    # The ice's drive and the flux it drives; paired, the edge's squared steepness, whose slope is twice the flux.
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
    # The double well's slope, tilted toward ice by the undercooling.
    tilt = tilt_strength / np.pi * (tilt_steepness * (melting_temperature - temperatures)).arctan()   # Scalar+[vertices]
    well_slope = fraction * (1 - fraction) * (1 - 2 * fraction) / 2 - tilt * fraction * (1 - fraction)
    return shares + cells * well_slope


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


def growth(topology: Topology, areas: Amount, gradients: Plane, cells: Amount, parcels: Point, crystal_axis: Plane,
           relaxation_time: float, mean_width: float, anisotropy: float, regularization: float,
           tilt_strength: float, tilt_steepness: float, melting_temperature: float, latent_heat: float,
           dt: float, n_frames: int, stride: int) -> Iterator[Point]:
    """The water's parcels `Point-[vertices]` at each of `n_frames` frames, `stride` steps of `dt` apart, from these
    parcels: the ice relaxing down its free energy in `relaxation_time`, its edge's width `mean_width` varied six-fold
    by `anisotropy` about the crystal's axis, a plane whose normal one arm grows along, the steepness regularized by
    `regularization` where it vanishes; the double well tilted toward ice by `tilt_strength / pi` times the arctangent
    of `tilt_steepness` times the undercooling below `melting_temperature`; each parcel moving along freezing, which
    releases `latent_heat` per unit of ice; and the heat conducted at unit conductivity."""
    gradient, spreading, conduction = operators(topology, areas, gradients)
    # Freezing: ice gained, and its latent heat released.
    freezing = ice + latent_heat * heat                                       # [] Point-

    def step(parcels: Point, _) -> tuple[Point, None]:
        """One step: the ice relaxes down its free energy, moving each parcel along freezing, and the heat is conducted."""
        fraction, temperatures = mv.i & parcels, mv.h & parcels               # Scalar+[vertices], Scalar+[vertices]
        slope = free_energy_slope(topology, areas, gradients, cells, gradient, spreading, crystal_axis, mean_width,
                                  anisotropy, regularization, tilt_strength, tilt_steepness, melting_temperature,
                                  fraction, temperatures)                     # Scalar-[vertices]
        frozen = -slope / (relaxation_time * cells)                           # Scalar+[vertices]
        conducted = -(conduction * temperatures) / cells                      # Scalar+[vertices]
        return parcels + dt * (frozen * freezing + conducted * heat), None

    @jax.jit
    def strided(parcels: Point) -> Point:
        return jax.lax.scan(step, parcels, None, length=stride)[0]

    for _ in range(n_frames):
        yield parcels
        parcels = strided(parcels)
