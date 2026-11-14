r"""Hydrostatic stability of floating bodies, from small tilts to capsizing, in the projective geometric
algebra of three-dimensional space.

A floating body is modelled as a closed triangulated surface, fixed in its own frame. Still water pushes on
every wet triangle, square to it and in proportion to its depth, as a line of force through the surface.
Summing those lines over the wet surface gives the buoyancy as a single line of force upward through the
centre of the displaced water, holding the displaced weight with no volume integral. At equilibrium draft the
buoyancy balances the weight, while at a tilt the two parallel lines separate into a righting couple.

The water surface is modelled as a plane, centres of gravity as points, forces as forques and turns as twists.
The pressure push on a wet triangle is the interior product `samples | planes`, summed into the buoyancy
forque and added directly to the weight line `weight * (centre & (turns << -up))`. Differentiating that sum
over `Tilt` and pairing back with `Tilt` forms the tilt stiffness `Tilt & (pushing + weighing)`, whose
eigenvalues divided by the weight are the roll and pitch metacentric heights.

In the Newton-Euler notation of naval architecture and hydrostatics, vertical equilibrium integrates pressure
into the heave force $\rho g V_{\mathrm{disp}}(z_0, \phi, \theta) = W$, while rotational stability tracks the
shift of the centre of buoyancy $\mathbf r_B(\phi) = V_{\mathrm{disp}}^{-1} \int_{V_{\mathrm{disp}}} \mathbf r \, \mathrm dV$
relative to the centre of gravity $\mathbf r_G$ to form the righting arm $GZ(\phi) = (y_B(\phi) - y_G)\cos\phi + (z_B(\phi) - z_G)\sin\phi$
and the upright metacentric height $GM = I_w / V_{\mathrm{disp}} - BG$ from the waterplane second moment $I_w$
and vertical separation $BG = z_G - z_B$.
"""

from __future__ import annotations

import numpy as np

from numga import Algebra, NumpyContext, stack, where
from numga.algebras import PGA3D as ga
from examples.mechanics.simplex import simplex_inertia_weights

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()                                             # Scalar+
Plane = ga.gatype.vector()                                              # Vector+
Point = ga.gatype.antivector()                                          # Antivector-
Twist = ga.gatype.bivector()                                            # Bivector+
Forque = ga.gatype.antibivector()                                       # Antibivector-
Motor = ga.gatype.rotor()                                               # Rotor+
# Turns about horizontal lines through the origin: heel and trim.
Tilt = Twist.derive.nondegenerate.derive.touching(Algebra("z+"))        # Bivector+
# The energy of a small tilt.
Stiffness = ga.gatype((Scalar, Tilt, Tilt))                             # Scalar+ <- (Tilt+, Tilt+)

# Unit scalar, still water surface, upward direction, vertical shift, heel axis, and origin.
unit = mv.z | mv.z                                                      # [] Scalar+
water = mv.z                                                            # [] Plane+
up = water.dual()                                                       # [] Point-
heave = mv.w ^ water                                                    # [] Twist+
heel_axis = mv.yz                                                       # [] Twist+
origin = mv.point([0, 0, 0])                                            # [] Point-
# Per triangle, the points halfway between each corner and the centroid, each standing for a third of its area.
quadrature_rule = 3 * simplex_inertia_weights(3)


# --- math -----------------------------------------------------------------------------
def wet_triangles(triangles: Point, surface: Plane) -> Point:
    """The submerged part of each surface triangle below each water plane, split into two oriented triangles
    by fanning the deepest corner to the waterline crossings."""
    depths = -(surface[..., None] & triangles)                          # [..., 3] Scalar+[F]
    first_depth, second_depth, third_depth = depths[..., 0], depths[..., 1], depths[..., 2]
    deep, middle, high = triangles[0], triangles[1], triangles[2]

    def ordered_pair(swap: np.ndarray, left, right):
        return where(swap, right, left), where(swap, left, right)

    # Sort the three corners from deepest to highest and record whether the swaps flip the triangle winding.
    first_swap = first_depth < second_depth
    first_depth, second_depth = ordered_pair(first_swap, first_depth, second_depth)
    deep, middle = ordered_pair(first_swap, deep, middle)
    second_swap = second_depth < third_depth
    second_depth, third_depth = ordered_pair(second_swap, second_depth, third_depth)
    middle, high = ordered_pair(second_swap, middle, high)
    third_swap = first_depth < second_depth
    first_depth, second_depth = ordered_pair(third_swap, first_depth, second_depth)
    deep, middle = ordered_pair(third_swap, deep, middle)
    winding_flip = first_swap ^ second_swap ^ third_swap

    def water_crossing(start: Point, end: Point, start_depth: Scalar, end_depth: Scalar) -> Point:
        """The point along an edge where depth passes through zero, clamped to the segment."""
        drop = start_depth - end_depth
        share = (start_depth / where(drop.abs() > 0, drop, mv.w & start)).clip(0, 1)
        return start + share * (end - start)

    first_cut = where(second_depth < 0, water_crossing(deep, middle, first_depth, second_depth), middle)
    second_cut = where(
        second_depth < 0,
        first_cut,
        where(third_depth < 0, water_crossing(middle, high, second_depth, third_depth), high),
    )
    third_cut = where(third_depth < 0, water_crossing(deep, high, first_depth, third_depth), high)
    first_part = stack((deep, where(winding_flip, second_cut, first_cut), where(winding_flip, first_cut, second_cut)), axis=-1)
    second_part = stack((deep, where(winding_flip, third_cut, second_cut), where(winding_flip, second_cut, third_cut)), axis=-1)
    return stack((first_part, second_part), axis=-2)                    # [..., 2, 3] Point-[F]


def pressure_lines(triangles: Point, surface: Plane) -> tuple[Point, Forque]:
    """Quadrature sample points on the wet parts of each triangle and their inward normal lines per unit
    pressure, each weighted by one third of its wet triangle's area."""
    parts = wet_triangles(triangles, surface)                           # [..., 2, 3] Point-[F]
    samples = (parts[..., None, :] * quadrature_rule).sum(axis=-1)      # [..., 2, 3] Point-[F]
    # A part's corner join has weight equal to twice its area; multiplying by the sign of the enclosed volume
    # makes every plane face outward regardless of winding.
    volume = (origin & triangles[0] & triangles[1] & triangles[2]).sites.sum()  # [] Scalar-
    planes = (parts[..., 0] & parts[..., 1] & parts[..., 2]) * (volume / volume.norm())  # [..., 2] Plane+[F]
    return samples, -(samples | planes[..., None]) / 6                  # [..., 2, 3] Point-[F], Forque-[F]


def buoyancy(triangles: Point, surface: Plane, fluid_weight: float) -> tuple[Forque, Forque]:
    """The net hydrostatic push on the wet hull below each water surface, and its rate of growth with draft."""
    samples, lines = pressure_lines(triangles, surface)                 # [..., 2, 3] Point-[F], Forque-[F]
    depths = -(surface[..., None, None] & samples)                      # [..., 2, 3] Scalar+[F]
    push = (fluid_weight * depths * lines).sites.sum().sum(axis=(-2, -1))    # [...] Forque-
    growth = (fluid_weight * lines).sites.sum().sum(axis=(-2, -1))           # [...] Forque-
    return push, growth


def equilibrium_heights(triangles: Point, turns: Motor, weight: Scalar, heights: Scalar, fluid_weight: float,
                        n_steps: int) -> Scalar:
    """The water-surface height over the body's origin at which buoyancy balances weight under each turn,
    refined from initial heights by Newton steps."""
    for _ in range(n_steps):
        push, growth = buoyancy(triangles, turns << (water - heights * mv.w), fluid_weight)
        upward_shift = turns << heave                                   # [...] Twist+
        heights = heights + (weight - (push & upward_shift)) / (growth & upward_shift)
    return heights                                                      # [...] Scalar+


def world_placement(turns: Motor, heights: Scalar) -> Motor:
    """The motor placing the body in the world frame with the water surface at `z = 0`."""
    return (heave * (heights / 2)).exp() * turns                        # [...] Motor+


def body_forces(triangles: Point, turns: Motor, heights: Scalar, weight: Scalar, centre: Point,
                fluid_weight: float) -> tuple[Forque, Forque]:
    """The buoyancy and gravity lines acting on the floating body in its own frame, whose sum is the righting
    couple."""
    push, _ = buoyancy(triangles, turns << (water - heights * mv.w), fluid_weight)
    return push, weight * (centre & (turns << -up))                     # [...] Forque-, Forque-


def tilt_stiffness(triangles: Point, heights: Scalar, weight: Scalar, centre: Point, fluid_weight: float) -> Stiffness:
    """The quadratic tilt stiffness of the upright body floating at each height, pairing small tilts with the
    induced change in buoyancy and weight lines."""
    surface = water - heights * mv.w                                    # [...] Plane+
    samples, lines = pressure_lines(triangles, surface)                 # [..., 2, 3] Point-[F], Forque-[F]
    # In the body's frame a small tilt turns the water plane against the hull, changing each sample's depth,
    # and turns the downward weight line with it.
    surface_rate = surface[..., None, None].commutator(Tilt)            # [..., 1, 1] Plane+ <- Tilt+
    buoyancy_change = (fluid_weight * -(surface_rate & samples) * lines).sites.sum().sum(axis=(-2, -1))  # [...] Forque- <- Tilt+
    weight_change = weight * centre.regressive((-up).commutator(Tilt))  # [...] Forque- <- Tilt+
    return Tilt & (buoyancy_change + weight_change)                     # [...] Scalar+ <- (Tilt+, Tilt+)
