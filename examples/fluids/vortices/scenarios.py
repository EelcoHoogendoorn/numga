"""Planar vortex scenes: leapfrogging pairs, a rolling shear layer, a vortex gas, a walkabout lap,
and worldline braids."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from numga import stack
from examples.fluids.vortices import core

# Two coaxial counter-rotating pairs travelling along x; with the inner span exceeding 0.38 of the
# outer span, the two pairs take turns passing through each other stably.
leapfrog_spans = np.array([1.0, -1.0, 0.55, -0.55])
leapfrog_strengths = np.array([1.0, -1.0, 1.0, -1.0])
leapfrog_core = 0.15

# Periodic shear layer: a row of equal vortices repeating along x with a two-roll perturbation and a
# weaker single-roll perturbation that seeds vortex pairing.
layer_period = 1.0
n_layer_vortices = 64
n_layer_copies = 4
layer_jump = 1.0
layer_core = 0.05
layer_amplitude = 0.01
layer_seed = 0.5

# Vortex gas scattered over a disc with net circulation of one sign keeping the cloud together while
# minority counter-rotating vortices pair with neighbours and shoot across.
n_gas_turning = 32
n_gas_counter = 8
gas_radius = 1.5
gas_core = 0.08
gas_seed = 3

# Walkabout: inner span at 0.35 of the outer span (below the Behring-Goodman stability threshold)
# with one inner vortex nudged along x to break mirror symmetry.
walkabout_spans = np.array([1.0, -1.0, 0.35, -0.35])
walkabout_nudges = np.array([0.0, 0.0, 0.015, 0.0])
walkabout_dt = 1 / 6
n_walkabout_frames, n_walkabout_substeps = 203, 4
walkabout_x, walkabout_y = 10.0, 1.4
walkabout_half_width, walkabout_half_height = 10.6, 2.9
walkabout_pixels_per_unit = 20
walkabout_vorticity_scale = 14.0

# Worldlines: five equal vortices on a regular ring and the leapfrogging pairs traced over time.
n_ring = 5
ring_strengths = np.ones(n_ring)
ring_radius = 1.0
ring_core = 0.05
worldline_dt = 0.2
n_ring_steps, n_leapfrog_steps = 150, 75
braid_height = 2.5

frame_duration_ms = 40
n_colours = 64


@dataclass(frozen=True)
class Scene:
    name: str
    centres: core.Point
    circulations: core.Circulation
    core_radius: float
    copies: core.Point
    half_width: float
    half_height: float
    n_columns: int
    n_rows: int
    dt: float
    n_substeps: int
    n_frames: int
    duration_ms: int
    vorticity_scale: float


# --- math -----------------------------------------------------------------------------
def single_copy() -> core.Point:
    """Single zero-offset periodic copy for an isolated vortex group."""
    return core.mv.vector([[0, 0]]).dual()                                  # [1] Point-


def worldlines(
    centres: core.Point,
    circulations: core.Circulation,
    core_radius: float,
    copies: core.Point,
    dt: float,
    n_steps: int,
) -> core.Point:
    """Worldline positions across `n_steps + 1` instants for each vortex."""
    return stack(
        list(core.trajectory(centres, circulations, core_radius, copies, n_steps + 1, 1, dt))
    )                                                                       # [n_steps + 1, ...] Point-


def braid_worldlines(n_ring_steps: int, n_leapfrog_steps: int) -> list[core.Point]:
    """Worldline arrays for the rigidly rotating 5-vortex ring and the leapfrogging pairs."""
    rotors = (core.mv.xy * (-np.pi * np.arange(n_ring) / n_ring)).exp()     # [n_ring] Even+
    ring_centres = rotors >> (core.mv.x.dual() * ring_radius)               # [n_ring] Point-
    ring_circulations = core.mv.xy * ring_strengths                         # [n_ring] Circulation+
    isolated = single_copy()                                                # [1] Point-
    pair_scene = leapfrog()
    ring_paths = worldlines(
        ring_centres, ring_circulations, ring_core, isolated, worldline_dt, n_ring_steps
    )                                                                       # [n_ring_steps + 1, n_ring] Point-
    pair_paths = worldlines(
        pair_scene.centres,
        pair_scene.circulations,
        pair_scene.core_radius,
        pair_scene.copies,
        worldline_dt,
        n_leapfrog_steps,
    )                                                                       # [n_leapfrog_steps + 1, 4] Point-
    return [ring_paths, pair_paths]


def walkabout_centres() -> core.Point:
    """Initial vortex centres for the symmetry-broken walkabout pairs."""
    return (core.mv.y * walkabout_spans + core.mv.x * walkabout_nudges).dual()   # [4] Point-


def leapfrog() -> Scene:
    """Scene parameters for two coaxial leapfrogging vortex pairs."""
    centres = (core.mv.y * leapfrog_spans).dual().cast(core.Point)          # [4] Point-
    circulations = core.mv.xy * leapfrog_strengths                          # [4] Circulation+
    return Scene(
        name="vortices_leapfrog",
        centres=centres,
        circulations=circulations,
        core_radius=leapfrog_core,
        copies=single_copy(),
        half_width=2.4,
        half_height=1.6,
        n_columns=300,
        n_rows=200,
        dt=1 / 15,
        n_substeps=2,
        n_frames=150,
        duration_ms=frame_duration_ms,
        vorticity_scale=14.0,
    )


def shear_layer() -> Scene:
    """Scene parameters for a periodic Kelvin-Helmholtz shear layer with subharmonic pairing seed."""
    x_positions = (np.arange(n_layer_vortices) + 0.5) / n_layer_vortices * layer_period
    phases = 2 * np.pi * x_positions / layer_period
    y_offsets = layer_amplitude * (np.sin(2 * phases) + layer_seed * np.sin(phases))
    centres = (core.mv.x * (x_positions - layer_period / 2) + core.mv.y * y_offsets).dual()   # [n_layer_vortices] Point-
    strengths = np.full(n_layer_vortices, -layer_jump * layer_period / n_layer_vortices)
    copies = core.mv.x.dual() * (np.arange(-n_layer_copies, n_layer_copies + 1) * layer_period)   # [2 * n_layer_copies + 1] Point-
    return Scene(
        name="vortices_shear_layer",
        centres=centres,
        circulations=core.mv.xy * strengths,
        core_radius=layer_core,
        copies=copies,
        half_width=layer_period,
        half_height=0.4,
        n_columns=300,
        n_rows=120,
        dt=1 / 60,
        n_substeps=2,
        n_frames=240,
        duration_ms=frame_duration_ms,
        vorticity_scale=10.0,
    )


def gas() -> Scene:
    """Scene parameters for a bounded planar gas of vortices of both senses."""
    rng = np.random.default_rng(gas_seed)
    n_gas = n_gas_turning + n_gas_counter
    radii = gas_radius * np.sqrt(rng.uniform(size=n_gas))
    rotors = (core.mv.xy * (-np.pi * rng.uniform(size=n_gas))).exp()        # [n_gas] Even+
    centres = rotors >> (core.mv.x.dual() * radii)                          # [n_gas] Point-
    strengths = rng.permutation(np.repeat([1.0, -1.0], [n_gas_turning, n_gas_counter]))
    return Scene(
        name="vortices_gas",
        centres=centres,
        circulations=core.mv.xy * strengths,
        core_radius=gas_core,
        copies=single_copy(),
        half_width=3.6,
        half_height=3.0,
        n_columns=300,
        n_rows=250,
        dt=1 / 120,
        n_substeps=8,
        n_frames=225,
        duration_ms=frame_duration_ms,
        vorticity_scale=50.0,
    )


# --- plumbing -------------------------------------------------------------------------
def main() -> None:
    from examples.fluids.vortices import render
    from examples.util.animation import save_animation, save_figure

    scenes = (leapfrog(), shear_layer(), gas())
    final_centres: list[core.Point] = []
    for scene in scenes:
        states: list[core.Point] = []

        def recorded_frames(scene: Scene = scene, states: list[core.Point] = states):
            for state in core.trajectory(
                scene.centres, scene.circulations, scene.core_radius, scene.copies, scene.n_frames, scene.n_substeps, scene.dt
            ):
                states.append(state)
                yield state

        save_animation(
            render.animate(
                scene.circulations,
                scene.core_radius,
                scene.copies,
                scene.half_height,
                (scene.n_rows, scene.n_columns),
                scene.vorticity_scale,
                recorded_frames(),
            ),
            scene.name,
            scene.duration_ms,
            n_colours,
        )
        final_centres.append(states[-1])

    walk_centre = (core.mv.x * walkabout_x + core.mv.y * walkabout_y).dual()   # [] Point-
    walk_shape = (
        round(2 * walkabout_half_height * walkabout_pixels_per_unit),
        round(2 * walkabout_half_width * walkabout_pixels_per_unit),
    )
    walk_circulations = core.mv.xy * leapfrog_strengths                     # [4] Circulation+
    isolated = single_copy()                                                # [1] Point-
    walk_frames = core.trajectory(
        walkabout_centres(), walk_circulations, leapfrog_core, isolated, n_walkabout_frames, n_walkabout_substeps, walkabout_dt
    )
    save_animation(
        render.window(
            walk_circulations,
            leapfrog_core,
            isolated,
            walk_centre,
            walkabout_half_height,
            walk_shape,
            walkabout_vorticity_scale,
            walk_frames,
        ),
        "vortices_walkabout",
        frame_duration_ms,
        n_colours,
    )

    ring_paths, pair_paths = braid_worldlines(n_ring_steps, n_leapfrog_steps)
    save_figure(render.braids([ring_paths, pair_paths], braid_height), "vortices_worldlines")

    # checks
    for scene, centres in zip(scenes, final_centres):
        points = centres.mean(axis=-1) + core.grid(scene.half_width, scene.half_height, 12, 8)
        flow_derivative = core.derivative(
            core.gradient(centres, scene.circulations, scene.core_radius, scene.copies, points)
        )
        offsets = core.separations(centres, scene.copies, points)
        blob = (offsets | offsets) + scene.core_radius**2
        vorticity = (scene.circulations[:, None] * scene.core_radius**2 / np.pi / (blob * blob)).sum(axis=-1).sum(axis=-1)
        assert ((flow_derivative - vorticity).norm() < 1e-9).all()
    ring_circulations = core.mv.xy * ring_strengths                         # [n_ring] Circulation+
    for circulations, paths in ((ring_circulations, ring_paths), (scenes[0].circulations, pair_paths)):
        linear_impulse = (paths * circulations).sum(axis=-1)                # [...] Point-
        angular_impulse = (circulations * (paths | paths)).sum(axis=-1)     # [...] Circulation+
        assert ((linear_impulse - linear_impulse[0]).norm() < 1e-9).all()
        assert ((angular_impulse - angular_impulse[0]).norm() < 1e-5).all()
    relative_rotors = ring_paths * ring_paths[0]                            # [n_ring_steps + 1, n_ring] Even+
    assert ((relative_rotors - relative_rotors[:, :1]).norm() < 1e-9).all()


if __name__ == "__main__":
    main()
