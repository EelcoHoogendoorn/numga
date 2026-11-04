"""A cellular flame: a flame spreading through a lean mixture of fuel and air from where it was lit, breaking into cells
as it grows, on a mesh of triangles, in the plane joined with three directions, for the volume, the fuel and the heat
a parcel of the mixture holds.

A premixed flame is a thin front where fuel burns, moving into the unburnt mixture as the heat it conducts ahead ignites
the fuel there. Burning grows steeply with temperature, as Arrhenius's law has it, so the front's speed hangs on how hot
it burns. A part of the front bulging into the mixture sheds its heat into more of the mixture around it and draws its
fuel from more of it; where fuel spreads faster than heat, as hydrogen does through lean air, the fuel wins, and the
bulge burns hotter and bulges further, while the creases between bulges lose more heat than the fuel they gain makes
up and burn cooler, until the flame there all but goes out. The front breaks into cells. The Lewis number, heat's
diffusivity over fuel's, measures this; below one a smooth front is unstable.

The algebra is the plane, x+y+, joined with three degenerate directions, for volume, fuel and heat: x+y+v0f0h0. A parcel
of the mixture is a point weighed by its volume, `mv.v.dual()`, offset along the fuel's direction by its fuel and along
the heat's by its heat, which `mv.f &` and `mv.h &` read off it. Burning is one fixed direction in that space of
amounts, `reaction = heat - fuel`: what burns moves a parcel along it, from fuel towards heat, and leaves its volume
alone. The plane `mv.h + mv.f` reads the energy a parcel holds, its heat and the heat its fuel would release, and pairs
with the reaction to nothing, `(mv.h + mv.f) & reaction == 0`: burning only moves energy from the fuel into the heat. A
drive is minus the gradient of a potential wedged with the amount it pushes: the temperature's, wedged with heat, and
the fuel's own amount's, wedged with fuel, each mapped to its amount's flux at its own rate. The energy a parcel holds
changes only where the two spread unlike, and that is what gathers energy into the bulges and drains it from the
creases. A triangle's area stays intrinsic, like the join that measures it, and so does what a cell holds; dividing by
the cell gives back a rate.

Lengths are in units of the flame's thickness, heat's diffusivity over the speed of a flat flame, times in units of the
time such a flame takes to cross its thickness, and temperatures in units of the rise burning a parcel's fuel brings, so
that burning a unit of fuel releases a unit of heat; the heat a unit volume holds per degree is one.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax

from numga import Algebra, stack
from numga.backend.jax import JaxContext
from numga.sparse import spdiag

from examples.util.topology import Topology

jax.config.update("jax_enable_x64", True)

# The plane, and the volume, the fuel and the heat a parcel holds, each a direction of its own with no metric.
space = Algebra("x+y+")
amounts = Algebra("v0f0h0")
ga = space * amounts
mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
Point = ga.gatype.antivector()                                                # Antivector-
# An amount a cell holds, intrinsic like the join that measures its area: the area, or the heat or fuel in it.
Amount = ga.gatype.antipseudoscalar()                                         # Antipseudoscalar-
Plane = ga.gatype.vector()                                                    # Vector+
# The planes reading a place's run across the plane and up it.
Upright = Plane.derive.within(space)                                          # Vector+
Drive = ga.gatype.bivector().derive.touching(space, Algebra("f0h0"))          # Bivector+
Flux = ga.gatype.antibivector()                                               # Antibivector-
# The plane's two directions, across and up.
directions = stack((mv.x.dual(), mv.y.dual()))                                # [2] Point-
# A unit of volume, the weight of a place, a unit of fuel and a unit of heat.
volume = mv.v.dual()                                                          # [] Point-
fuel = mv.f.dual()                                                            # [] Point-
heat = mv.h.dual()                                                            # [] Point-
# Burning a unit of fuel releases a unit of heat: the reaction's one direction in the space of amounts.
reaction = heat - fuel                                                        # [] Point-
# The energy a parcel holds: its heat, and the heat its fuel would release.
energy = mv.h + mv.f                                                          # [] Plane+


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Amount, Plane, Amount]:
    """Each triangle's area `Scalar-[faces]`, each corner's hat gradient `[corners] Plane+[faces]`, the plane through
    the facing side, along volume, fuel and heat, over twice the area, so that it reads one from the side to the corner,
    and each vertex's share of the area around it `Scalar-[vertices]`."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [corners] Point-[faces]
    # Twice a triangle's area: two of its sides joined, at unit volume, fuel and heat.
    double_areas = (((sides[0] & sides[1]) & volume) & fuel) & heat           # Scalar-[faces]
    areas = double_areas / 2                                                  # Scalar-[faces]
    gradients = (volume & fuel & heat & sides) / double_areas                 # [corners] Plane+[faces]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=0)


def burning(topology: Topology, areas: Amount, gradients: Plane, cells: Amount, fuel_diffusivity: float,
            prefactor: float, activation: float, unburnt_temperature: float, parcels: Point, dt: float, n_frames: int,
            n_stride: int) -> Iterator[tuple[Point, Scalar]]:
    """The mixture's parcels `Point-[vertices]`, and the rate their fuel burns at per volume `Scalar+[vertices]`, at
    each of `n_frames` frames, `n_stride` steps of `dt` apart, from these parcels: each step every parcel burns along
    the reaction at the rate its temperature sets, while the temperature's drive spreads its heat at unit diffusivity
    and the fuel's its fuel at `fuel_diffusivity`, with no flux through the rim."""
    corners = topology.incidence[2, 0]                                        # [corners] Scalar+[faces <- vertices]
    # The temperature's drive at each corner's hat, minus its gradient wedged with the heat it pushes, and the fuel's,
    # wedged with fuel; heat's conductivity and fuel's diffusivity, each drive read across the plane's two directions
    # driving its amount along them.
    heat_drives, fuel_drives = -(gradients ^ mv.h), -(gradients ^ mv.f)       # [corners] Drive+[faces], [corners] Drive+[faces]
    heat_sweeps = directions & heat                                           # [2] Flux-
    fuel_sweeps = directions & fuel                                           # [2] Flux-
    conductivity = (heat_sweeps * (heat_sweeps & Drive)).sum(axis=0)          # [] Flux- <- Drive+
    diffusivity = fuel_diffusivity * (fuel_sweeps * (fuel_sweeps & Drive)).sum(axis=0)   # [] Flux- <- Drive+
    # What a flux carries out of each corner's cell: its hat gradient wedged with the flux, read for its heat or its
    # fuel, gathered onto the vertices.
    local = areas * (mv.h & (gradients[:, None] ^ conductivity(heat_drives[None, :])))   # [corners, corners] Scalar-[faces]
    conduction = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar-[vertices <- vertices]
    local = areas * (mv.f & (gradients[:, None] ^ diffusivity(fuel_drives[None, :])))   # [corners, corners] Scalar-[faces]
    diffusion = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Scalar-[vertices <- vertices]

    def burnt(parcels: Point) -> Scalar:
        """The fuel each parcel burns over a step `Scalar+[vertices]`: its temperature, the unburnt mixture's raised by
        its heat, sets the rate each unit of its fuel burns at, growing steeply with it, and its fuel decays at that
        rate."""
        temperature = unburnt_temperature + (mv.h & parcels)                  # Scalar+[vertices]
        rates = prefactor * (-activation / temperature).exp()                 # Scalar+[vertices]
        return (mv.f & parcels) * (1 - (-dt * rates).exp())

    def step(parcels: Point, _) -> tuple[Point, None]:
        """One step: what burns moves each parcel along the reaction, and what each amount's drive carries out of a
        cell leaves it."""
        conducted = -(conduction * (mv.h & parcels)) / cells                  # Scalar+[vertices]
        diffused = -(diffusion * (mv.f & parcels)) / cells                    # Scalar+[vertices]
        return parcels + burnt(parcels) * reaction + dt * (conducted * heat + diffused * fuel), None

    @jax.jit
    def strided(parcels: Point) -> Point:
        return jax.lax.scan(step, parcels, None, length=n_stride)[0]

    for _ in range(n_frames):
        yield parcels, burnt(parcels) / dt
        parcels = strided(parcels)
