"""A seed of ice in undercooled water grows into a six-armed dendrite by a phase field on a triangulated disc, in the
plane joined with three degenerate directions for volume, ice and heat.

Water cooled below its melting point freezes onto any ice it touches, and freezing releases latent heat, which warms
the water at the ice's edge and slows the freezing there. A bump on the edge reaches into colder water and sheds its
heat to more sides than the flat edge around it, so it grows faster: the tips outrun the sides, and each tip sprouts
branches of its own. The ice's surface energy depends on how its edge lies against the crystal's axes, six-fold for
ice, and steers the fastest tips along six directions, the six arms of a snowflake. The phase is 0 in water and 1 in
ice, with a smooth edge between, far wider than real ice's so that the triangles resolve it; temperatures are measured
from the cold water's in units of its undercooling, so ice melts at one.

The algebra joins the plane `x+ y+` with three degenerate directions `v0 i0 h0` for the quantities volume, ice and
heat. A parcel of water is modelled as a sum of those units, one unit of volume plus its ice and its heat, read off by
`i & parcel` and `h & parcel`. Freezing is one fixed direction among the quantities, `freezing = ice + latent_heat * heat`:
ice gained, and its latent heat released. Ice is taken at water's density, so freezing leaves the volume alone,
`v & freezing == 0`, and ice and water share one heat capacity and one conductivity. Two planes read the effect of
freezing. The energy plane `h - latent_heat * i` reads a parcel's energy, its heat less the latent heat given up by its
ice, and pairs with freezing to zero: freezing only moves energy from the water's latent store into its heat. The
entropy's slope across the quantities pairs with freezing to the entropy produced when a unit of ice freezes, positive
in undercooled water and zero at the melting temperature, because heat raises entropy inversely with absolute
temperature while ice carries less entropy than liquid water by the latent heat over the melting temperature.
Multiplied by the melting temperature, that pairing pushes the phase to freeze, so the push comes from the same latent
heat that warms the water.

The phase is an order parameter, not a quantity, and carries the edge's free energy. A parcel's ice follows the phase
as a smooth quintic step that is flat in bulk water and bulk ice. Because its slope vanishes in liquid water, the push
to freeze vanishes there too, and the cold water stays a local minimum of the free energy at any undercooling,
metastable until the ice's edge reaches it. Relaxing the phase up the total entropy at fixed energy while stepping each
parcel along the freezing direction conserves total energy across the cells and raises total entropy.

A drive is minus a field's simplex gradient wedged with its quantity direction, and an isotropic material map written
as dyads over the plane's directions maps drives to fluxes. Heat conduction carries the thermal drive `-gradients ^ h`
out of each cell. Pairing the phase's drive `-gradients ^ i` with its flux under the same map gives the squared
steepness of the ice's edge, and half of that times the squared edge width is its gradient energy. Multiplying the
steepness by the crystal's axis forms an even element whose sixth power has scalar part proportional to `cos(6 angle)`,
giving six-fold anisotropy without taking an angle. Combined with a double well, that forms the edge's free energy,
whose derivative at each vertex holds the phase back. Differentiating the six-fold term directly gives a fifth power of
that turn, the Cahn-Hoffman vector of materials science, and each vertex's share pairs its corner drive with the
resulting flux. That turn is the one place where the plane's Euclidean metric enters, because crystal angles need one;
its plane-valued derivative is wedged with `i` and converted into a flux by the isotropic material map. Triangle areas
and cell integrals have intrinsic orientation, and dividing by a vertex's cell area gives a rate.
"""

from __future__ import annotations

from collections.abc import Iterator

import jax

from numga import Algebra, stack
from numga.backend.jax import JaxContext
from numga.sparse import SparseExtensor, spdiag

from examples.util.topology import Topology

jax.config.update("jax_enable_x64", True)

# The plane `x+ y+` and three degenerate directions `v0 i0 h0` for the quantities volume, ice and heat.
space = Algebra("x+y+")
quantities = Algebra("v0i0h0")
ga = space * quantities
mv = JaxContext(ga).multivector
Scalar = ga.gatype.scalar()                                                   # Scalar+
# A cell integral, intrinsic like the join that measures a triangle's area.
Antiscalar = ga.gatype.antipseudoscalar()                                     # Antipseudoscalar-
Point = ga.gatype.antivector()                                                # Antivector-
Plane = ga.gatype.vector()                                                    # Vector+
Drive = ga.gatype.bivector().derive.touching(space, Algebra("i0h0"))          # Bivector+
Flux = ga.gatype.antibivector().derive.touching(space, Algebra("i0h0"))       # Antibivector-
Spreading = ga.gatype((Flux, Drive))
across = mv.x.dual()                                                          # [] Point-
up = mv.y.dual()                                                              # [] Point-
directions = stack((across, up))                                              # [2] Point-
volume = mv.v.dual()                                                          # [] Point-
ice = mv.i.dual()                                                             # [] Point-
heat = mv.h.dual()                                                            # [] Point-


# --- math -----------------------------------------------------------------------------
def triangles(topology: Topology, vertices: Point) -> tuple[Antiscalar, Plane, Antiscalar]:
    """Triangle areas, corner simplex gradients, and vertex cell areas across the triangulated dish."""
    sides = topology.boundary[2, 1] * (topology.d[0] * vertices)              # [3] Point-[F]
    # Joining two sides with unit volume, ice and heat gives twice the oriented triangle area.
    double_areas = (((sides[0] & sides[1]) & volume) & ice) & heat            # Antiscalar-[F]
    areas = double_areas / 2                                                  # Antiscalar-[F]
    # Joining the facing side with volume, ice and heat over twice the area gives each corner's simplex gradient.
    gradients = (volume & ice & heat & sides) / double_areas                  # [3] Plane+[F]
    return areas, gradients, (~topology.incidence[2, 0] * (areas / 3)).sum(axis=0)


def ice_content(phase: Scalar) -> Scalar:
    """A parcel's ice fraction as a smooth quintic step between zero in water and one in ice, with vanishing slope at both."""
    return phase ** 3 * (10 - 15 * phase + 6 * phase ** 2)


def edge_free_energy(areas: Antiscalar, cells: Antiscalar, gradient: SparseExtensor, spreading: Spreading,
                     crystal_axis: Plane, mean_width: float, anisotropy: float, regularization: float,
                     well_height: float, phase: Scalar) -> Antiscalar:
    """The free energy of the ice's edge, summing the anisotropic gradient energy over the triangles and the double
    well over the vertex cells."""
    steepness = gradient(phase)                                               # Plane+[F]
    drive = -steepness ^ mv.i                                                 # Drive+[F]
    squared_steepness = drive & spreading(drive)                              # Scalar+[F]
    third_power = (steepness * crystal_axis) ** 3                             # Even+[F]
    six_fold = third_power.scalar_product(third_power)                        # Scalar+[F]
    reach = squared_steepness + regularization ** 2                           # Scalar+[F]
    widths = mean_width * (1 + anisotropy * six_fold / reach ** 3)            # Scalar+[F]
    well = well_height * phase ** 2 * (1 - phase) ** 2                        # Scalar+[V]
    return (areas * widths ** 2 * squared_steepness / 2).sites.sum() + (cells * well).sites.sum()


def edge_slope(topology: Topology, areas: Antiscalar, gradients: Plane, cells: Antiscalar, gradient: SparseExtensor,
               spreading: Spreading, crystal_axis: Plane, mean_width: float, anisotropy: float, regularization: float,
               well_height: float, phase: Scalar) -> Antiscalar:
    """The derivative of the edge's free energy with respect to the vertex phases, combining the anisotropic gradient
    energy's pull across the triangles with the double well's slope on the cells."""
    steepness = gradient(phase)                                               # Plane+[F]
    # Pairing the phase's drive with its flux gives the squared steepness, whose derivative is twice that flux.
    drive = -steepness ^ mv.i                                                 # Drive+[F]
    flux = spreading(drive)                                                   # Flux-[F]
    squared_steepness = drive & flux                                          # Scalar+[F]
    # The sixth power of `steepness * crystal_axis` has scalar part `|steepness| ** 6 * cos(6 angle)`.
    turn = steepness * crystal_axis                                           # Even+[F]
    third_power = turn ** 3                                                   # Even+[F]
    six_fold = third_power.scalar_product(third_power)                        # Scalar+[F]
    reach = squared_steepness + regularization ** 2                           # Scalar+[F]
    widths = mean_width * (1 + anisotropy * six_fold / reach ** 3)            # Scalar+[F]
    # Differentiating the six-fold term gives `6 * crystal_axis * turn ** 5`, mapped to a flux by `spreading`.
    slope = 6 * crystal_axis * turn ** 5                                      # Plane+[F]
    slope_flux = spreading(-slope ^ mv.i)                                     # Flux-[F]
    # Combine the width's angular variation with the isotropic flux to form the total flux pull on each triangle.
    turning = mean_width * anisotropy * (slope_flux / reach ** 3 - 6 * six_fold * flux / reach ** 4)   # Flux-[F]
    pull = widths ** 2 * flux + squared_steepness * widths * turning          # Flux-[F]
    # Pair each corner's simplex drive with the triangle's flux pull to gather the vertex derivatives.
    corner_drives = -gradients ^ mv.i                                         # [3] Drive+[F]
    shares = (~topology.incidence[2, 0] * (areas * (corner_drives & pull))).sum(axis=0)   # Antiscalar-[V]
    well_slope = 2 * well_height * phase * (1 - phase) * (1 - 2 * phase)      # Scalar+[V]
    return shares + cells * well_slope


def total_entropy(areas: Antiscalar, cells: Antiscalar, gradient: SparseExtensor, spreading: Spreading,
                  crystal_axis: Plane, mean_width: float, anisotropy: float, regularization: float, well_height: float,
                  latent_heat: float, melting_temperature: float, absolute_zero: float, parcels: Point,
                  phase: Scalar) -> Antiscalar:
    """The total entropy of the dish, summing the parcels' thermal and latent entropy over the cells and subtracting
    the edge's free energy over the absolute melting temperature."""
    absolute_melting = melting_temperature - absolute_zero
    parcel_entropies = ((mv.h & parcels) - absolute_zero).log() - latent_heat / absolute_melting * (mv.i & parcels)   # Scalar+[V]
    edge = edge_free_energy(areas, cells, gradient, spreading, crystal_axis, mean_width, anisotropy, regularization,
                            well_height, phase)                               # [] Antiscalar-
    return (cells * parcel_entropies).sites.sum() - edge / absolute_melting


def operators(topology: Topology, areas: Antiscalar,
              gradients: Plane) -> tuple[SparseExtensor, Spreading, SparseExtensor]:
    """The simplex gradient operator from vertex scalars to triangle planes, the isotropic material map for heat and
    ice, and the assembled heat conduction operator across the vertex cells."""
    corners = topology.incidence[2, 0]                                        # [3] Scalar+[F <- V]
    gradient = (spdiag(gradients) * corners).sum(axis=0) * Scalar             # (Plane+ <- Scalar+)[F <- V]
    sweeps = stack((directions & heat, directions & ice))                     # [2, 2] Flux-
    spreading = (sweeps * (sweeps & Drive)).sum(axis=-1).sum(axis=0)          # [] Flux- <- Drive+
    heat_drives = -gradients ^ mv.h                                           # [3] Drive+[F]
    local = areas * (mv.h & (gradients[:, None] ^ spreading(heat_drives[None, :])))   # [3, 3] Antiscalar-[F]
    conduction = (~corners[:, None] * spdiag(local) * corners[None, :]).sum(axis=0).sum(axis=0)   # Antiscalar-[V <- V]
    return gradient, spreading, conduction


def phase_drive(topology: Topology, areas: Antiscalar, gradients: Plane, cells: Antiscalar, gradient: SparseExtensor,
                spreading: Spreading, crystal_axis: Plane, mean_width: float, anisotropy: float, regularization: float,
                well_height: float, latent_heat: float, melting_temperature: float, absolute_zero: float,
                freezing: Point, parcels: Point, phase: Scalar) -> Antiscalar:
    """The entropy derivative with respect to the vertex phases at fixed energy, scaled by the absolute melting
    temperature: the thermodynamic push to freeze minus the edge's free-energy derivative."""
    absolute_melting = melting_temperature - absolute_zero
    # Heat raises entropy by `1 / T`, while ice carries `latent_heat / T_m` less entropy than liquid water.
    absolute_temperatures = (mv.h & parcels) - absolute_zero                  # Scalar+[V]
    entropy_slope = mv.h / absolute_temperatures - latent_heat / absolute_melting * mv.i   # Plane+[V]
    # Pairing `entropy_slope` with `freezing` gives the entropy produced per unit of ice; `held_slope` vanishes in bulk phases.
    held_slope = 30 * phase ** 2 * (1 - phase) ** 2                           # Scalar+[V]
    push = absolute_melting * cells * (entropy_slope & freezing) * held_slope   # Antiscalar-[V]
    return push - edge_slope(topology, areas, gradients, cells, gradient, spreading, crystal_axis, mean_width,
                             anisotropy, regularization, well_height, phase)


def growth(topology: Topology, areas: Antiscalar, gradients: Plane, cells: Antiscalar, parcels: Point, phase: Scalar,
           crystal_axis: Plane, relaxation_time: float, mean_width: float, anisotropy: float, regularization: float,
           well_height: float, latent_heat: float, melting_temperature: float, absolute_zero: float, dt: float,
           n_frames: int, stride: int) -> Iterator[tuple[Point, Scalar]]:
    """Yield parcel and phase fields across frames separated by fixed-energy phase relaxation, latent heat release
    along the freezing direction, and heat conduction across the cells."""
    gradient, spreading, conduction = operators(topology, areas, gradients)
    freezing = ice + latent_heat * heat                                       # [] Point-

    def step(state: tuple[Point, Scalar], _) -> tuple[tuple[Point, Scalar], None]:
        """One time step of phase relaxation, latent heat release, and heat conduction."""
        parcels, phase = state
        drive = phase_drive(topology, areas, gradients, cells, gradient, spreading, crystal_axis, mean_width,
                            anisotropy, regularization, well_height, latent_heat, melting_temperature, absolute_zero,
                            freezing, parcels, phase)                         # Antiscalar-[V]
        next_phase = phase + dt * drive / (relaxation_time * cells)           # Scalar+[V]
        ice_gained = ice_content(next_phase) - (mv.i & parcels)                  # Scalar+[V]
        conducted = -(conduction * (mv.h & parcels)) / cells                  # Scalar+[V]
        return (parcels + ice_gained * freezing + dt * conducted * heat, next_phase), None

    @jax.jit
    def advance(state: tuple[Point, Scalar]) -> tuple[Point, Scalar]:
        return jax.lax.scan(step, state, None, length=stride)[0]

    state = (parcels, phase)
    for _ in range(n_frames):
        yield state
        state = advance(state)
