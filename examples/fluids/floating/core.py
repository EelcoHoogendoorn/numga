"""This example computes the hydrostatic stability of floating bodies, from small tilts to capsizing, in the
projective geometric algebra of three-dimensional space.

A floating body is a closed surface of triangles, fixed in its own frame. Still water pushes on each wet
patch of the surface, square to it and in proportion to its depth: a force along a line through the patch.
The buoyancy is the sum of those lines over the wet surface, a single line of force, upward and through the
centre of the displaced water; for a closed surface it holds the displaced weight, Archimedes' principle,
with no volume integral. The body floats at the height where the buoyancy balances its weight. Tilted, the
buoyancy line and the weight line no longer coincide, and their sum is a couple: the righting couple,
which turns the body back upright, or further over.

The water surface is a plane, the centres are points, forces are forques and turns are twists. Over `Tilt`,
the space of all small tilts, the same sum is a map from tilts to the change of the net force, and paired
with the tilt it is the body's stiffness against tilting: its eigenvalues over the weight are the
metacentric heights.

In the notation of naval architecture the righting arm is $GZ(\\phi)$, the horizontal distance between the
centre of gravity and the line of buoyancy, its slope at upright the metacentric height
$GM = KB + I_{\\mathrm{wp}}/V - KG$, and the stiffness against a small heel $\\Delta \\cdot GM$.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from numga.algebras import PGA3D as ga
from examples import instantiate
from examples.mechanics.simplex import simplex_inertia_weights

mesh_module = instantiate("examples.util.mesh", ga)
Mesh, mv = mesh_module.Mesh, mesh_module.mv
Scalar = ga.gatype.scalar()                                             # Scalar+
Plane = ga.gatype.vector()                                              # Vector+
Point = ga.gatype.antivector()                                          # Antivector-
Twist = ga.gatype.bivector()                                            # Bivector+
Forque = ga.gatype.antibivector()                                       # Antibivector-
Motor = ga.gatype.rotor()                                               # Rotor+
# Turns about horizontal lines through the origin: heel and trim.
Tilt = Twist.derive.nondegenerate.derive.containing("z")                # Bivector+
# The energy of a small tilt.
Stiffness = ga.gatype((Scalar, Tilt, Tilt))                             # Scalar+ <- (Tilt+, Tilt+)

# The still water surface, the point at infinity upward, the vertical shift, and the axis of heel through
# the origin, along the body: the meet of the planes y = 0 and z = 0.
water = mv.z                                                            # [] Plane+
up = water.dual()                                                       # [] Point-
heave = mv.w ^ water                                                    # [] Twist+
heel_axis = mv.yz                                                       # [] Twist+
# The origin, from which the hull's enclosed volume is summed.
origin = mv.point([0.0, 0.0, 0.0])                                      # [] Point-
# Per triangle, the points halfway between each corner and the centroid: they integrate anything quadratic
# over the triangle exactly, each standing for a third of its area.
RULE = 3 * simplex_inertia_weights(3)


# --- math -----------------------------------------------------------------------------
def pressing(triangles: Point, surface: Plane) -> tuple[Point, Forque]:
    """Sample points of the wet part of each triangle, its corners `[3] Point-[triangles]`, under each water
    surface, and the water's push at each per unit of pressure `[..., triangles, parts, samples]`: a line through
    the point, square to the triangle, against its outward normal, with a third of the part's area."""
    parts = wet(triangles, surface)                                     # [..., triangles, parts, 3] Point-
    samples = (parts[..., None, :] * RULE).sum(axis=-1)                 # [..., triangles, parts, samples] Point-
    # The plane through a part's corners, wound as its triangle, has twice the part's area as its weight.
    # The hull's enclosed volume, signed by the same winding, is positive exactly when those planes face
    # outward: times its sign, every plane faces outward, whichever way the hull is wound.
    first, second, third = triangles                    # Point-[triangles], Point-[triangles], Point-[triangles]
    volume = (origin & first & second & third).sites.sum()              # [] Scalar+
    planes = (parts[..., 0] & parts[..., 1] & parts[..., 2]) * (volume / volume.norm())   # [..., triangles, parts] Plane+
    return samples, -(samples | planes[..., None]) / 6


def buoyancy(triangles: Point, surface: Plane, fluid_weight: float) -> tuple[Forque, Forque]:
    """The water's push on the body below each water surface, the sum of the pressure lines over its wet
    surface, each pressure the fluid's weight times the depth; and how the push grows as the surface rises."""
    samples, lines = pressing(triangles, surface)                       # [..., triangles, parts, samples]
    depths = -(surface[..., None, None, None] & samples)                # [..., triangles, parts, samples] Scalar+
    push = (fluid_weight * depths * lines).sum(axis=(-3, -2, -1))       # [...] Forque-
    growth = (fluid_weight * lines).sum(axis=(-3, -2, -1))              # [...] Forque-
    return push, growth


def afloat(triangles: Point, turns: Motor, weight: Scalar, heights: Scalar, fluid_weight: float,
           steps: int) -> Scalar:
    """The height of the water surface over the body's origin at which the body, turned by each motor,
    floats: its buoyancy balancing its weight, by Newton's method from the given heights."""
    for _ in range(steps):
        push, growth = buoyancy(triangles, turns << (water - heights * mv.w), fluid_weight)
        lifting = turns << heave                                        # [...] Twist+
        heights = heights + (weight - (push & lifting)) / (growth & lifting)
    return heights                                                      # [...] Scalar+


def placement(turns: Motor, heights: Scalar) -> Motor:
    """The motion taking the body's frame to the world's, the water surface at the world's `water`: the
    turn, then down by the height of the water surface over the body's origin."""
    return (heave * (heights / 2)).exp() * turns                        # [...] Motor+


def forces(triangles: Point, turns: Motor, heights: Scalar, weight: Scalar, centre: Point,
           fluid_weight: float) -> tuple[Forque, Forque]:
    """The forces on the body, turned by each motor and floating at each height, in the body's frame: its
    buoyancy, and its weight, a line straight down through its centre of gravity. Their sum is the righting
    couple."""
    push, _ = buoyancy(triangles, turns << (water - heights * mv.w), fluid_weight)
    return push, weight * (centre & (turns << -up))                     # [...] Forque-


def stiffness(triangles: Point, heights: Scalar, weight: Scalar, centre: Point, fluid_weight: float) -> Stiffness:
    """The energy of a small tilt of the upright body floating at each height: the change of the net force
    under the tilt, the water surface turning against the body and the weight turning with it, paired with
    the tilt."""
    surface = water - heights * mv.w                                    # [...] Plane+
    samples, lines = pressing(triangles, surface)                       # [..., triangles, parts, samples]
    # In the body's frame a small tilt turns the water surface, deepening each wet point by its height above
    # the change of the surface, and turns the direction of the weight with it.
    tilted = surface[..., None, None, None].commutator(Tilt)            # [..., 1, 1, 1] Plane+ <- Tilt+
    pushing = (fluid_weight * -(tilted & samples) * lines).sum(axis=(-3, -2, -1))   # [...] Forque- <- Tilt+
    weighing = weight * centre.regressive((-up).commutator(Tilt))       # [...] Forque- <- Tilt+
    return Tilt & (pushing + weighing)                                  # [...] Scalar+ <- (Tilt+, Tilt+)


# --- the step outside the algebra ---------------------------------------------------------
def wet(triangles: Point, surface: Plane) -> Point:
    """The wet part of each triangle, its corners `[3] Point-[triangles]`, below each water surface, as two
    triangles wound as the triangle, either or both of no area `[..., triangles, parts, 3]`: with the corners
    ordered from deepest to highest, the deepest corner fanned to the points where the edges leave the water."""
    depths = -(surface[..., None] & triangles)                          # [..., 3] Scalar+[triangles]
    # Each triangle's corners last.
    head = np.moveaxis(depths.cast(ga.subspace.scalar()).kernel[..., 0], -2, -1)   # [..., triangles, 3]
    corners = np.broadcast_to(np.moveaxis(triangles.cast(ga.subspace.antivector()).kernel, 0, -2), head.shape + (4,))
    order = np.argsort(-head, axis=-1)
    head = np.take_along_axis(head, order, axis=-1)
    deep, middle, high = np.moveaxis(np.take_along_axis(corners, order[..., None], axis=-2), -2, 0)
    first, second, third = np.moveaxis(head, -1, 0)

    def leaving(start: np.ndarray, end: np.ndarray, start_head: np.ndarray, end_head: np.ndarray) -> np.ndarray:
        """The point of an edge where it leaves the water, kept within the edge."""
        drop = start_head - end_head
        share = np.clip(start_head / np.where(drop == 0, 1, drop), 0, 1)[..., None]
        return start + share * (end - start)

    turning = np.where((second < 0)[..., None], leaving(deep, middle, first, second), middle)
    rising = np.where((second < 0)[..., None], turning,
                      np.where((third < 0)[..., None], leaving(middle, high, second, third), high))
    closing = np.where((third < 0)[..., None], leaving(deep, high, first, third), high)
    parts = np.stack([np.stack([deep, turning, rising], axis=-2), np.stack([deep, rising, closing], axis=-2)], axis=-3)
    # An ordering that is not a rotation of the corners reverses the winding; reversing the parts restores it.
    reversed_order = (order[..., 1] != (order[..., 0] + 1) % 3)[..., None, None, None]
    return mv.antivector(np.where(reversed_order, parts[..., ::-1, :], parts))


# --- plumbing -------------------------------------------------------------------------
def patch(grid: np.ndarray) -> np.ndarray:
    """The triangles of a grid of vertices `[u, v]`, two per cell, wound from u towards v
    `[2 * (u - 1) * (v - 1), 3]`."""
    corner, along_u, across, along_v = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
    return np.concatenate([np.stack([corner, along_u, across], axis=-1).reshape(-1, 3),
                           np.stack([corner, across, along_v], axis=-1).reshape(-1, 3)])


def box(sizes: np.ndarray) -> tuple[Mesh, Point]:
    """A box of the given sizes along x, y and z, centred on the origin: its eight corners `Point-[vertices]`, and
    two triangles on each face, wound outward."""
    # Each corner is a choice of side along x, y and z, its index the choices read as a binary number.
    sides = np.array(list(product([0, 1], repeat=3)))                  # [vertices, 3]
    u, v = np.meshgrid([0, 1], [0, 1], indexing="ij")
    faces = []
    for axis in range(3):
        for side in (0, 1):
            grid = np.zeros(u.shape + (3,), dtype=int)
            # The face's u and v run along the next two axes, swapped on the negative side, so that every
            # face winds outward.
            first, second = (axis + 1) % 3, (axis + 2) % 3
            grid[..., axis] = side
            grid[..., first], grid[..., second] = (u, v) if side else (v, u)
            faces.append(patch(grid @ np.array([4, 2, 1])))
    return Mesh(len(sides), np.concatenate(faces)), mv.point((sides - 0.5) * sizes).field()


def wigley(length: float, beam: float, draft: float, freeboard: float, cells: int) -> tuple[Mesh, Point]:
    """A Wigley hull: below the waterline its half breadth is `beam / 2 * (1 - (2 x / length)**2) *
    (1 - (z / draft)**2)`, above it the sides rise straight to a flat deck. Its vertices `Point-[vertices]` lie on
    grids along the hull and up each side, and along the hull and across the deck, and its triangles close them."""
    along = np.linspace(-length / 2, length / 2, 2 * cells + 1)
    height = np.concatenate([np.linspace(-draft, 0, cells + 1), np.linspace(0, freeboard, cells // 2 + 1)[1:]])
    x, z = np.meshgrid(along, height, indexing="ij")                    # [along, height]
    half = beam / 2 * (1 - (2 * x / length) ** 2) * (1 - (np.minimum(z, 0) / draft) ** 2)
    across = np.linspace(-1, 1, cells + 1)
    deck_x, share = np.meshgrid(along, across, indexing="ij")           # [along, across]
    deck_half = share * beam / 2 * (1 - (2 * deck_x / length) ** 2)
    # The port side shares the keel and the stems with the starboard side, where the half breadth vanishes; the deck
    # shares its edges with the sides' tops, and closes to the top of each stem. The corner of the keel and the
    # stern's stem is no vertex: the grid's triangles there form a fin of no thickness, one on each side, which the
    # closed hull leaves out.
    own_starboard = np.ones(x.shape, dtype=bool)
    own_starboard[-1, 0] = False
    own_port = np.zeros(x.shape, dtype=bool)
    own_port[1:-1, 1:] = True
    own_deck = np.zeros(deck_x.shape, dtype=bool)
    own_deck[1:-1, 1:-1] = True
    places = np.concatenate([np.stack([x, half, z], axis=-1)[own_starboard], np.stack([x, -half, z], axis=-1)[own_port],
                             np.stack([deck_x, deck_half, np.full_like(deck_x, freeboard)], axis=-1)[own_deck]])
    counts = np.cumsum([0, own_starboard.sum(), own_port.sum()])
    starboard = np.full(x.shape, -1)
    starboard[own_starboard] = counts[0] + np.arange(own_starboard.sum())
    port = starboard.copy()
    port[own_port] = counts[1] + np.arange(own_port.sum())
    deck = np.empty(deck_x.shape, dtype=int)
    deck[:, 0], deck[:, -1] = port[:, -1], starboard[:, -1]
    deck[[0, -1]] = starboard[[0, -1], -1:]
    deck[own_deck] = counts[2] + np.arange(own_deck.sum())
    faces = np.concatenate([patch(starboard.T), patch(port), patch(deck)])
    # The triangles of the closed hull: those not on the fin, and not collapsed into the top of a stem, where two of
    # their corners meet.
    closing = (faces >= 0).all(axis=-1) & (faces != np.roll(faces, 1, axis=-1)).all(axis=-1)
    return Mesh(len(places), faces[closing]), mv.point(places).field()
