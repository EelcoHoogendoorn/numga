"""Compiled ray-stepping and disk-crossing integrators for Kerr light-ray paths and camera images."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from numga import Algebra, NumpyContext, concatenate, where
from numga.algebras import STA as ga
from numga.extensor import Extensor

mv = NumpyContext(ga).multivector
Scalar = ga.gatype.scalar()
Vector = ga.gatype.vector()
Bivector = ga.gatype.bivector()
Antivector = ga.gatype.antivector()
Spin = Bivector.derive.within(Algebra("x+y+z+")).derive.dual
# The crossings of the disk a camera ray keeps: its direct, secondary and tertiary images, and one more.
n_max_crossings = 4


@dataclass(frozen=True)
class Image:
    """Stores disk crossings by pixel (`disk_pixels`, `disk`, `disk_radii`, `disk_weights`) and escaped sky rays (`sky_pixels`, `sky_directions`, `sky_weights`)."""
    disk_pixels: np.ndarray
    disk: tuple[Extensor, ...]
    disk_radii: Scalar
    disk_weights: Scalar
    sky_pixels: np.ndarray
    sky_directions: Antivector
    sky_weights: Scalar


def oblate_radius(position: Antivector, spin: Spin) -> Scalar:
    """The Kerr–Schild oblate radius (`[...] Scalar+`) follows from the spatial anticommutator with the turning plane."""
    time = -mv.t.dual()                                                                     # [] Antivector-
    spatial = position - time * (mv.t & position)                                           # [...] Antivector-
    turning = spin.dual_inverse()                                                           # [...] Bivector+
    volume = spatial.anticommutator(turning)                                                # [...] Vector+
    offset = spatial.squared() - spin.squared()                                             # [...] Scalar+
    discriminant = (offset.squared() - 4 * volume.squared()).square_root()                  # [...] Scalar+
    return ((offset + discriminant) / 2).square_root()                                      # [...] Scalar+


def crossing_fraction(start: Antivector, end: Antivector, normal: Vector) -> Scalar:
    """Finds the linear segment fraction (`[...] Scalar+`) at which the chord from `start` to `end` crosses `normal`."""
    return -(normal & start) / (normal & (end - start))                                     # [...] Scalar+


def disk_crossing(start: tuple[Extensor, ...], end: tuple[Extensor, ...], normal: Vector,
                  spin: Spin) -> tuple[tuple[Extensor, ...], Scalar]:
    """Interpolates a ray's state tuple across `normal` and evaluates its oblate radius (`[...] Scalar+`) at the crossing."""
    fraction = crossing_fraction(start[0], end[0], normal)                                  # [...] Scalar+
    state = tuple(value + fraction * (stop - value) for value, stop in zip(start, end))     # [...] state
    return state, oblate_radius(state[0], spin)                                             # [...] state, [...] Scalar+


def runge_kutta(rates: Callable[..., tuple[Extensor, ...]], state: tuple[Extensor, ...],
                step: Scalar) -> tuple[Extensor, ...]:
    """Advances `state` through one fourth-order Runge–Kutta step of size `step`."""
    first = rates(*state)
    second = rates(*(value + rate * step / 2 for value, rate in zip(state, first)))
    third = rates(*(value + rate * step / 2 for value, rate in zip(state, second)))
    fourth = rates(*(value + rate * step for value, rate in zip(state, third)))
    return tuple(value + (a + 2 * b + 2 * c + d) * step / 6
                 for value, a, b, c, d in zip(state, first, second, third, fourth))


def trace(position: Antivector, momentum: Vector, spin: Spin, mass: float,
          step_size: float, n_steps: int, stop_fraction: float,
          dynamics: Callable[[Antivector, Vector, Spin, float], tuple[Antivector, Vector]]) -> tuple[Antivector, Vector]:
    """Integrates `n_steps` batched fourth-order steps (`[n_steps + 1, ...] Antivector-`, `[n_steps + 1, ...] Vector+`) while slowing captured rays inside the horizon."""
    import jax
    from numga.backend.jax import JaxContext

    horizon = mass + (mass**2 - spin.squared()).square_root()                               # [...] Scalar+

    def step(state: tuple[Antivector, Vector], _) -> tuple[tuple[Antivector, Vector], tuple[Antivector, Vector]]:
        radius = oblate_radius(state[0], spin)                                              # [...] Scalar+
        clock = ((radius / horizon - stop_fraction) / (1 - stop_fraction)).clip(0, 1)       # [...] Scalar+
        state = runge_kutta(lambda *values: dynamics(*values, spin, mass), state, step_size * clock)
        return state, state

    # Sparse execution spells each product as its nonzero terms, which XLA fuses across the wide batch of rays.
    context = JaxContext(ga, execution="sparse")
    start = (position.retyped(Antivector).materialize(context), momentum.retyped(Vector).materialize(context))
    _, later = jax.jit(lambda start: jax.lax.scan(step, start, length=n_steps))(start)      # [n_steps, ...] state
    return tuple(concatenate([first[None], after]).materialize(mv.context) for first, after in zip(start, later))


def resolve(state: tuple[Extensor, ...], spin: Spin, mass: float, step_fraction: float, n_steps: int,
            dynamics: Callable[..., tuple[Extensor, ...]], normal: Vector, inner_radius: Scalar, outer_radius: float,
            opacity: Callable[[Antivector, Vector, Scalar], Scalar], escape_radius: float,
            capture_radius: float) -> tuple[Image, tuple[Extensor, ...]]:
    """Traces camera rays by fourth-order radius-scaled steps in one compiled loop, recording up to `n_max_crossings` disk crossings per ray and escaped sky tangents."""
    import jax
    from numga.backend.jax import JaxContext

    # Sparse execution spells each product as its nonzero terms, which XLA fuses across the wide batch of rays.
    context = JaxContext(ga, execution="sparse")
    n_rays = state[0].shape[0]
    # The loop carries the event in the full antivector layout, whatever layout the rays start in.
    event, *rest = state
    initial_state = tuple(value.materialize(context) for value in (event.retyped(Antivector), *rest))
    remaining = mv.scalar(np.ones((n_rays, 1))).materialize(context)                        # [n_rays] Scalar+
    # Each ray's crossings, empty while their weight is zero, and its escape, empty likewise.
    crossed_states = tuple(value[:, None].broadcast_to((n_rays, n_max_crossings)) for value in initial_state)  # [n_rays, 4] state
    crossed_radii = remaining[:, None] * np.zeros(n_max_crossings)                          # [n_rays, 4] Scalar+
    crossed_weights = remaining[:, None] * np.zeros(n_max_crossings)                        # [n_rays, 4] Scalar+
    crossing_counts = np.zeros(n_rays, dtype=int)                                           # [n_rays]
    sky, sky_weights = initial_state[0], remaining * 0                                      # [n_rays] Antivector-, [n_rays] Scalar+

    def active_mask(current_state: tuple[Extensor, ...], remaining_light: Scalar) -> np.ndarray:
        """Identifies rays (`[n_rays]`) that still carry light between `capture_radius` and `escape_radius`."""
        radius = oblate_radius(current_state[0], spin)                                      # [n_rays] Scalar+
        return (remaining_light > 0) & (radius <= escape_radius) & (radius >= capture_radius)  # [n_rays]

    def step(carry):
        index, current_state, remaining_light, crossed_states, crossed_radii, crossed_weights, crossing_counts, sky, sky_weights = carry
        active = active_mask(current_state, remaining_light)                                # [n_rays]
        # A ray that has ended takes no step.
        size = where(active, step_fraction * oblate_radius(current_state[0], spin), remaining_light * 0)  # [n_rays] Scalar+
        end = runge_kutta(lambda *values: dynamics(*values, spin, mass), current_state, size)
        # A crossing of the disk between its radii, where the segment changes side of its plane.
        at_plane, radii = disk_crossing(current_state, end, normal, spin)                   # [n_rays] state, [n_rays] Scalar+
        crossing = active & ((normal & current_state[0]) * (normal & end[0]) <= 0) & (radii >= inner_radius) & (radii <= outer_radius)
        weights = where(crossing, remaining_light * opacity(at_plane[0], at_plane[1], radii), remaining_light * 0)
        remaining_light = remaining_light - weights
        # The crossing is stored as its ray's next one.
        into = crossing[:, None] & (np.arange(n_max_crossings) == crossing_counts[:, None])  # [n_rays, 4]
        crossed_states = tuple(where(into, value[:, None], crossed_value) for value, crossed_value in zip(at_plane, crossed_states))
        crossed_radii = where(into, radii[:, None], crossed_radii)
        crossed_weights = where(into, weights[:, None], crossed_weights)
        crossing_counts = crossing_counts + crossing
        # A ray leaving the escape radius with light left gives the sky its future-directed tangent and that light.
        escaping = active & (remaining_light > 0) & (oblate_radius(end[0], spin) > escape_radius)
        past_tangent = dynamics(*end, spin, mass)[0]                                        # [n_rays] Antivector-
        sky = where(escaping, -past_tangent, sky)
        sky_weights = where(escaping, remaining_light, sky_weights)
        return index + 1, end, remaining_light, crossed_states, crossed_radii, crossed_weights, crossing_counts, sky, sky_weights

    def unfinished(carry) -> bool:
        """Checks whether step budget remains and at least one ray is still active."""
        index, current_state, remaining_light = carry[:3]
        return (index < n_steps) & active_mask(current_state, remaining_light).any()

    carry = (0, initial_state, remaining, crossed_states, crossed_radii, crossed_weights, crossing_counts, sky, sky_weights)
    _, *carry = jax.jit(lambda carry: jax.lax.while_loop(unfinished, step, carry))(carry)
    final_state, _, crossed_states, crossed_radii, crossed_weights, _, sky, sky_weights = jax.tree_util.tree_map(
        lambda value: value.materialize(mv.context) if isinstance(value, Extensor) else np.asarray(value), carry,
        is_leaf=lambda value: isinstance(value, Extensor))
    ray_indices, crossing_idx = np.nonzero(crossed_weights > 0)
    escaped_indices = np.flatnonzero(sky_weights > 0)
    image = Image(ray_indices, tuple(value[ray_indices, crossing_idx] for value in crossed_states),
                  crossed_radii[ray_indices, crossing_idx], crossed_weights[ray_indices, crossing_idx],
                  escaped_indices, sky[escaped_indices], sky_weights[escaped_indices])
    return image, final_state
