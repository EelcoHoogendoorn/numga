# Examples

Examples are grouped by subject. An example that draws is a package of three
modules, as [`geometry/projection/`](geometry/projection/) shows:

| Module | Contains | May import |
| --- | --- | --- |
| `core.py` | the mathematics: GATypes, constructors, and the geometric narrative | numga, numpy |
| `render.py` | all drawing and coordinate read-out | `core`, matplotlib |
| `scenarios.py` | one function per figure: builds the scene, calls `core`, hands the result to `render` | `core`, `render` |

`core.py` never imports `render.py`, so the mathematics has no plotting anywhere
in its import graph. That is a test, not a convention: see
`test_mathematics_does_not_import_plotting`. Run an example through its
scenarios module. An example that draws nothing, such as
[`scenegraph.py`](geometry/scenegraph.py), stays a single file. Older examples
still use a `*_plumbing.py` sibling and are being migrated. Shared animation
helpers live in [`animation.py`](animation.py). Generated figures and animations
go to [`../plots/`](../plots/).

## Writing examples

- The main entry point reads like a tutorial: keep the geometric construction in
  one coherent scope, or a few were legitimate conceptual boundarie exist,
  with inline comments explaining the mathematical steps.
- Keep elegant GA expressions visible. Do not hide them behind helper calls or
  chains of wrappers; helpers are for construction trivia, sampling, numerical
  boundaries, coordinate readout, drawing, and export.
- No default argument values: pass configuration, step sizes, tolerances, seeds,
  and contexts explicitly. Do not hide defaults in function signatures or pad
  docstrings with parameter boilerplate explaining them.
- Construct inputs as named GA objects in the preamble. Do not scatter casts or
  just-in-time GAType conversions through the mathematical narrative.
- Keep plotting and checks out of the mathematics. Build the geometry first and
  hand it to drawing helpers afterward; for animations, yield geometric states
  to a renderer. Put checks in a labelled block at the end or in the example's
  tests, without layers of test-helper calls in the tutorial.
- Keep necessary excursions outside the algebra in minimal functions with named
  GAType inputs and outputs. Wrap only the numerical operation, not the geometry
  around it. Use the library's typed linear algebra methods directly when they
  already provide that boundary; do not wrap an extensor eigensolve again.
- Leave structural selection to static dispatch and numerical validation to the
  backend. Do not add defensive checks, fallback paths, or arbitrary tolerances
  to the demonstration.
- No unnecessary .kernel operations.
- No for loops that can be proper array programming.
- No bare coefficient indexing assuming array layout.
- Where destructuring extensors is required such as when plotting, explicit prior blade layout casting is required.
- No visual noise or text overlays on animations: no intrusive status boxes, iteration labels, or HUD text badges on animated frames; let the clean geometric dynamics speak for itself.
- Mandatory test-time plot regeneration: running an example's module tests must automatically regenerate its canonical plots and animations in `PLOT_DIR`, verify their freshness and non-zero size, and mirror them to the active artifacts directory.

| Topic | Examples |
| --- | --- |
| Geometry | [Planar PGA](geometry/pga2d.py), [projection](geometry/projection/), [scenegraph](geometry/scenegraph.py), [fitting](geometry/fitting.py), [registration](geometry/registration.py), [quadric error metrics (QEM)](geometry/qem/), [epipolar geometry & 3D reconstruction](geometry/epipolar/), [multi-view cone reconstruction & bundle adjustment](geometry/multiview/) |
| Quadrics | [Projective quadrics](quadrics/quadrics.py), [Gaussian fit and 1σ quadric](quadrics/gaussian.py), [spherical quadrics](quadrics/spherical_quadrics.py), [CGA spherical quadrics](quadrics/cga_spherical_quadrics.py), [sphere rendering](quadrics/conformal_elliptical.py), [collision](quadrics/quadric_collision.py), [Cayley–Klein geometry](quadrics/cayley_klein.py), [spherical dynamics](quadrics/spherical_quadric_scenarios.py) |
| Mechanics | [Inertia](mechanics/inertia.py), [simplex inertia](mechanics/simplex.py), [stiffness and normal modes](mechanics/modes/), [tennis racket instability](mechanics/tennis_racket.py), [Lie integrators](mechanics/lie_integrators.py), [rigid body chains (XPBD)](mechanics/xpbd.py) |
| Relativity | [Distributed impulses](relativity/relativistic_impulse.py), [ladder paradox](relativity/ladder_paradox.py), [Bell's spaceships](relativity/bell_spaceships.py), [relativistic aberration](relativity/boosted_quadrics.py), [curvature and gravitational waves](relativity/curvature.py) |
| Electromagnetism | [Maxwell maps](electromagnetism/maxwell.py), [constitutive maps](electromagnetism/constitutive.py) |
| Symmetry | [Heat conduction, a flywheel, and a crystal lattice](sketches/symmetry.py): `heat_conduction()` averages conductivity over a rotation group and plots the allowed heat-flow ellipsoid; `flywheel()` sums point-mass inertia over three rotated arms and plots their mass distribution; `crystal_lattice()` averages axial and face-diagonal bond responses over 24 cube rotations to compare isotropic rank-2 conductivity with anisotropic rank-4 imposed-strain elasticity. `main()` runs all three. |
| Conformal geometry | [CGA quadrics](sketches/cga_quadric.py): sphere/plane constructions, cyclides from the paper, and circle-vortex animations through one shared tracer. |

Run modules from `rewrite/` with `src` and the current directory on the Python
path. Plotting examples use NumPy and Matplotlib; some also use SciPy, and GIF
export uses Pillow. The JAX chain runner additionally requires JAX.

```sh
PYTHONPATH=src:. python -m examples.geometry.projection.scenarios
PYTHONPATH=src:. python -m examples.geometry.qem.scenarios
PYTHONPATH=src:. python -m examples.geometry.epipolar.scenarios
PYTHONPATH=src:. python -m examples.geometry.multiview.scenarios
PYTHONPATH=src:. python -m examples.quadrics.cayley_klein
PYTHONPATH=src:. python -m examples.quadrics.gaussian
PYTHONPATH=src:. python -m examples.mechanics.modes --animate
PYTHONPATH=src:. python -m examples.relativity.curvature
PYTHONPATH=src:. python -m examples.electromagnetism.constitutive
PYTHONPATH=src:. python -m examples.sketches.symmetry
PYTHONPATH=src:. python -c 'from examples.sketches.symmetry import heat_conduction; heat_conduction()'
PYTHONPATH=src:. python -c 'from examples.sketches.symmetry import flywheel; flywheel()'
PYTHONPATH=src:. python -c 'from examples.sketches.symmetry import crystal_lattice; crystal_lattice()'
PYTHONPATH=src:. python -m examples.sketches.cga_quadric
PYTHONPATH=src:. python -m examples.sketches.cga_quadric --scene six_families
PYTHONPATH=src:. python -m examples.sketches.cga_quadric --scene linked_vortex --frames 40 --scale 0.6667
PYTHONPATH=src:. python -m examples.sketches.cga_quadric --scene linked_tori --frames 40 --scale 0.6667
```

Performance comparisons live in [benchmarks/motor_map.py](../benchmarks/motor_map.py).
Each scenario isolates one performance question, prints a small comparison, and
explains the result in its docstring. Run scenarios individually from `rewrite/`
with `PYTHONPATH=src:.`:

```python
from benchmarks.motor_map import jax_map_amortization, jax_unrolling_crossover, numpy_unrolling_cost

jax_map_amortization()
jax_unrolling_crossover()
numpy_unrolling_cost()
```

CGA scenes are selected with `--scene`; `--help` lists them. One frame writes a
PNG; `--frames N` writes a shared-palette GIF. Both use new filenames. Resolution,
supersampling, and final downsampling are controlled by `--width`, `--height`,
`--supersample`, and `--scale`. The gold vortex ring is a torus traced and shaded
alongside the moving surface.

Curvature's GIF export is an argument to `main`:

```sh
PYTHONPATH=src:. python -c 'from examples.relativity.curvature import main; main(animation_path="plots/curvature.gif")'
```

Use `MPLBACKEND=Agg` when rendering without a display. Example tests mirror these
topics under [`../tests/examples/`](../tests/examples/):

```sh
python -m pytest tests/examples
```
