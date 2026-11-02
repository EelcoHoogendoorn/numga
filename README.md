# numga

[![CI](https://github.com/EelcoHoogendoorn/numga/actions/workflows/ci.yml/badge.svg)](https://github.com/EelcoHoogendoorn/numga/actions/workflows/ci.yml)

Geometric algebra with Extensors in NumPy, JAX and PyTorch.

Extensors generalize matrices and tensors from vectors to multivectors, and from the inner and tensor products to every product of geometric algebra.

<table>
<tr><td width="50%" align="center"><a href="examples/relativity/closed_universe/"><b>Gravitational waves</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/relativity/closed_universe/waves.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/closed_universe_inside.gif" alt="Crystals of atoms ringing as a gravitational wave passes, seen from inside a closed universe whose great circles are strung with its mesh's vertices" width="100%"></td><td width="50%" align="center"><a href="examples/thermodynamics/snowflake/"><b>Snowflake thermodynamics</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/thermodynamics/snowflake/snowflake.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/snowflake.gif" alt="A seed of ice in undercooled water growing into a six-armed dendrite" width="100%"></td></tr>
<tr><td width="50%" align="center"><a href="examples/geometry/scenegraph/"><b>Scenegraph and camera optics</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/geometry/scenegraph/scenegraph.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/scenegraph.gif" alt="A robot arm in 3D, and its picture on a camera sensor" width="100%"></td><td width="50%" align="center"><a href="examples/mechanics/rattleback/"><b>Spin reversal by rolling contact</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/mechanics/rattleback/rattleback.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/rattleback.gif" alt="A wooden rattleback spun the wrong way, rocking, stopping and spinning back" width="100%"></td></tr>
<tr><td width="50%" align="center"><a href="examples/estimation/multiview/"><b>Summing cones into splats</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/estimation/multiview/multiview_reconstruction.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/multiview_convergence.gif" alt="Cameras aligning on a scene" width="100%"></td><td width="50%" align="center"><a href="examples/quantum/condensate/"><b>Quantized vortices in a Bose–Einstein condensate</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/quantum/condensate/condensate.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/condensate.gif" alt="A rotating condensate settling into a lattice of quantized vortices" width="100%"></td></tr>
</table>

## Extensors

**[An introduction to extensors, and a gallery of what they do](docs/extensors.md)**: from a robot arm and its camera to a snowflake, a dynamo and a closed universe, each example a picture and the few lines of extensor code that compute it.

A short example of the extensor syntax: building unary maps and bilinear forms, and solving for the vibration modes of four point masses on springs:

```python
points = mv.vector([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).dual()   # four unit masses
inertia = (points & points.commutator(Bivector)).sum(axis=0)   # Antibivector <- Bivector

springs = points[:, None] & points[None, :]                    # [points, points] Antibivector: a line through each pair
stiffness = (springs * (springs & Bivector)).sum(axis=(0, 1)) / 2   # Antibivector <- Bivector
# the vibration modes, between two bilinear energy forms
values, modes = (Bivector & stiffness).eigh(Bivector & inertia)
```

The full example: [vibration modes](examples/mechanics/modes/modes.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/mechanics/modes/modes.ipynb)

## Tutorials

* [Geometric algebra](docs/tutorial/geometric-algebra/00-introduction.md): a primer for readers new to geometric algebra.
* [Extensors](docs/tutorial/extensors/00-introduction.md): a primer for readers new to extensors.

## Install and run

Install directly from GitHub with Python 3.11+ and Git:

```sh
python -m pip install "numga[examples] @ git+https://github.com/EelcoHoogendoorn/numga.git@main"
python -m examples.mechanics.modes.scenarios
```

The core library depends only on NumPy; omit `[examples]` to install it. Use `[examples,jax]` or `[examples,torch]` to include those backends.

The API is under active development and may change. Replace `@main` with a commit hash to keep a particular version.
