# Examples

Examples are grouped by subject. Each example keeps its supporting
`*_plumbing.py` beside the geometric construction; shared animation helpers live
in [`animation.py`](animation.py). Generated figures and animations go to
[`../plots/`](../plots/).

## Writing examples

- The main entry point reads like a tutorial: keep the geometric construction in
  one coherent scope, with inline comments explaining the mathematical steps.
- Keep elegant GA expressions visible. Do not hide them behind helper calls or
  chains of wrappers; helpers are for construction trivia, sampling, numerical
  boundaries, coordinate readout, drawing, and export.
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

| Topic | Examples |
| --- | --- |
| Geometry | [Planar PGA](geometry/pga2d.py), [projection](geometry/projection.py), [fitting](geometry/fitting.py), [registration](geometry/registration.py) |
| Quadrics | [Projective quadrics](quadrics/quadrics.py), [spherical quadrics](quadrics/spherical_quadrics.py), [sphere rendering](quadrics/conformal_elliptical.py), [collision](quadrics/quadric_collision.py), [Cayley–Klein geometry](quadrics/cayley_klein.py), [spherical dynamics](quadrics/spherical_quadric_scenarios.py) |
| Mechanics | [Inertia](mechanics/inertia.py), [simplex inertia](mechanics/simplex.py), [stiffness and normal modes](mechanics/stiffness.py), [tennis racket instability](mechanics/tennis_racket.py), [rigid body chains](mechanics/rigid_body/) |
| Relativity | [Distributed impulses](relativity/relativistic_impulse.py), [ladder paradox](relativity/ladder_paradox.py), [Bell's spaceships](relativity/bell_spaceships.py), [relativistic aberration](relativity/boosted_quadrics.py), [curvature and gravitational waves](relativity/curvature.py) |
| Electromagnetism | [Maxwell maps](electromagnetism/maxwell.py), [constitutive maps](electromagnetism/constitutive.py) |

Run modules from `rewrite/` with `src` and the current directory on the Python
path. Plotting examples use NumPy and Matplotlib; some also use SciPy, and GIF
export uses Pillow. The JAX chain runner additionally requires JAX.

```sh
PYTHONPATH=src:. python -m examples.geometry.projection
PYTHONPATH=src:. python -m examples.quadrics.cayley_klein
PYTHONPATH=src:. python -m examples.mechanics.stiffness --animate
PYTHONPATH=src:. python -m examples.relativity.curvature
PYTHONPATH=src:. python -m examples.electromagnetism.constitutive
```

Curvature's GIF export is an argument to `main`:

```sh
PYTHONPATH=src:. python -c 'from examples.relativity.curvature import main; main(animation_path="plots/curvature.gif")'
```

Use `MPLBACKEND=Agg` when rendering without a display. Example tests mirror these
topics under [`../tests/examples/`](../tests/examples/):

```sh
python -m pytest tests/examples
```
