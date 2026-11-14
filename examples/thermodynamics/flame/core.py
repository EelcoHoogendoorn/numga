"""A premixed flame spreads from sparks through a lean mixture of fuel and air and breaks into cells on a triangulated
disc, in the plane joined with three degenerate directions for volume, fuel and heat.

A premixed flame is a thin front where fuel burns, advancing into the unburnt mixture as the heat conducted ahead
ignites the fuel there. Because the reaction rate grows steeply with temperature under Arrhenius's law, the front's
speed depends strongly on its temperature. A bulge in the front sheds heat into a wider sector of unburnt gas and draws
fuel from a wider sector as well. Where fuel diffuses faster than heat, as hydrogen does in lean air, fuel transport
prevails, so the bulge burns hotter and advances faster while the creases between bulges lose more heat than incoming
fuel replaces, cool down and nearly extinguish. The Lewis number, thermal diffusivity divided by fuel diffusivity,
governs this balance, and below one a smooth front is unstable. Constant density leaves out the gas's thermal expansion
flow across the front.

The algebra joins the plane `x+ y+` with three degenerate directions `v0 f0 h0` for the quantities volume, fuel and
heat. A parcel of the mixture is modelled as a sum of those units, one unit of volume `v.dual()` plus its fuel and its
heat, read off by `f & parcel` and `h & parcel`. Burning is one fixed direction among the quantities,
`reaction = heat - fuel`, moving each parcel from fuel towards heat while leaving its volume unchanged. The plane
`energy = h + f` reads a parcel's enthalpy, the sum of its sensible heat and the chemical heat stored in its fuel, and
pairs with the reaction to zero, `energy & reaction == 0`, so burning converts chemical energy into sensible heat
without creating energy. A drive `-gradient ^ potentials` wedges minus a simplex gradient with a quantity's direction,
giving `-gradients ^ h` for heat and `-gradients ^ f` for fuel, and separate isotropic material maps take each drive to
its flux. Diagonal drives leave out Soret and Dufour cross-diffusion between temperature and concentration gradients.
Because reaction leaves `energy & parcels` unchanged, enthalpy varies only because fuel and heat diffuse at different
rates, concentrating behind the bulges and depleting along the creases. Triangle areas and cell integrals carry
intrinsic orientation, and dividing a cell outflow by the cell's area gives a local rate.

Lengths are measured in flame thicknesses (thermal diffusivity divided by the flat-flame speed), times in flame transit
times across one thickness, and temperatures in units of the adiabatic temperature rise, so burning one unit of fuel
releases one unit of heat at unit volumetric heat capacity.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax

from numga import Algebra, stack
from numga.backend.jax import JaxContext
from numga.sparse import spdiag

from numga.topology import Topology

jax.config.update("jax_enable_x64", True)

# The plane `x+ y+` joined with three degenerate directions `v0 f0 h0` for the quantities volume, fuel and heat.
space = Algebra("x+y+")
quantities = Algebra("v0f0h0")
ga = space * quantities
mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
# Cell-integrated areas and quantities carry the intrinsic orientation of the five-way join.
Antiscalar = ga.gatype.antipseudoscalar()                                     # Antipseudoscalar-
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
# Spatial planes reading horizontal and vertical offsets.
Upright = Plane.derive.within(space)                                          # Vector+
Drive = ga.gatype.bivector().derive.touching(space, Algebra("f0h0"))          # Bivector+
# Horizontal and vertical directions in the plane.
directions = stack((mv.x.dual(), mv.y.dual()))                                # [2] Antivector-
# Units of the three quantities: volume, fuel and heat.
volume = mv.v.dual()                                                          # [] Antivector-
fuel = mv.f.dual()                                                            # [] Antivector-
heat = mv.h.dual()                                                            # [] Antivector-
# Reaction direction among the quantities, converting one unit of fuel into one unit of heat.
reaction = heat - fuel                                                        # [] Antivector-
# Enthalpy plane reading the sum of sensible heat and chemical fuel energy.
energy = mv.h + mv.f                                                          # [] Vector+


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Antiscalar, Plane, Antiscalar]:
    """Triangle areas, corner simplex gradients, and dual vertex cell areas across the triangulated disc."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Antivector-[F]
    # Joining two sides with unit volume, fuel and heat gives twice the oriented triangle area.
    double_areas = (((sides[0] & sides[1]) & volume) & fuel) & heat           # Antipseudoscalar-[F]
    areas = double_areas / 2                                                  # Antipseudoscalar-[F]
    # Joining the facing side with volume, fuel and heat over twice the area gives each corner's simplex gradient.
    gradients = (volume & fuel & heat & sides) / double_areas                 # [3] Vector+[F]
    return areas, gradients, (topology.incidence[2, 0].adjugate() * (areas / 3)).sum(axis=0)


def burning(topology: Topology, areas: Antiscalar, gradients: Plane, cells: Antiscalar, fuel_diffusivity: float,
            prefactor: float, activation: float, unburnt_temperature: float, parcels: Point, dt: float, n_frames: int,
            n_stride: int) -> Iterator[tuple[Point, Scalar]]:
    """Yield the mixture's parcel field and volumetric reaction rate across frames as fuel burns along the reaction
    direction and heat and fuel diffuse across the insulated disc."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    # Corner thermal and fuel drives, minus each simplex gradient wedged with its quantity, and isotropic transport maps.
    heat_drives, fuel_drives = -gradients ^ mv.h, -gradients ^ mv.f           # [3] Bivector+[F], [3] Bivector+[F]
    heat_sweeps = directions & heat                                           # [2] Antibivector-
    fuel_sweeps = directions & fuel                                           # [2] Antibivector-
    conductivity = (heat_sweeps * (heat_sweeps & Drive)).sum(axis=0)          # [] Antibivector- <- Bivector+
    diffusivity = fuel_diffusivity * (fuel_sweeps * (fuel_sweeps & Drive)).sum(axis=0)   # [] Antibivector- <- Bivector+
    # Cell heat and fuel outflow operators assembled by wedging the corner simplex gradients with the diffusive fluxes.
    local = areas * (mv.h & (gradients[:, None] ^ conductivity(heat_drives[None, :])))   # [3, 3] Antipseudoscalar-[F]
    conduction = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Antipseudoscalar-[V <- V]
    local = areas * (mv.f & (gradients[:, None] ^ diffusivity(fuel_drives[None, :])))   # [3, 3] Antipseudoscalar-[F]
    diffusion = (corners[:, None].adjugate() * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Antipseudoscalar-[V <- V]

    def fuel_loss(parcels: Point) -> Scalar:
        """Fuel consumed in each parcel over one time step under Arrhenius decay at the parcel's temperature."""
        temperature = unburnt_temperature + (mv.h & parcels)                  # Scalar+[V]
        rates = prefactor * (-activation / temperature).exp()                 # Scalar+[V]
        return (mv.f & parcels) * (1 - (-dt * rates).exp())

    def step(parcels: Point, _) -> tuple[Point, None]:
        """Parcel field after one step of chemical reaction and explicit heat and fuel diffusion."""
        heat_rate = -(conduction * (mv.h & parcels)) / cells                  # Scalar+[V]
        fuel_rate = -(diffusion * (mv.f & parcels)) / cells                   # Scalar+[V]
        return parcels + fuel_loss(parcels) * reaction + dt * (heat_rate * heat + fuel_rate * fuel), None

    @jax.jit
    def advance(parcels: Point) -> Point:
        return jax.lax.scan(step, parcels, None, length=n_stride)[0]

    for _ in range(n_frames):
        yield parcels, fuel_loss(parcels) / dt
        parcels = advance(parcels)
