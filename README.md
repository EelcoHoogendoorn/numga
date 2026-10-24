# numga

[![CI](https://github.com/EelcoHoogendoorn/numga/actions/workflows/ci.yml/badge.svg)](https://github.com/EelcoHoogendoorn/numga/actions/workflows/ci.yml)

Geometric algebra with Extensors in NumPy, JAX and PyTorch.

Extensors generalize matrices and tensors from vectors to multivectors, and from the inner and tensor products to every product of geometric algebra.

<table>
<tr><td width="50%" align="center"><a href="examples/geometry/scenegraph/"><b>Scenegraph and camera optics</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/geometry/scenegraph/scenegraph.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/scenegraph.gif" alt="A robot arm in 3D, and its picture on a camera sensor" width="100%"></td><td width="50%" align="center"><a href="examples/fluids/wing/"><b>Vanishing geometric derivative</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/fluids/wing/wing.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/wing.gif" alt="A wing pitching in a steady stream, its pressure, streamlines and lift" width="100%"></td></tr>
<tr><td width="50%" align="center"><a href="examples/estimation/multiview/"><b>Summing cones into splats</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/estimation/multiview/multiview_reconstruction.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/multiview_convergence.gif" alt="Cameras aligning on a scene" width="100%"></td><td width="50%" align="center"><a href="examples/relativity/twistors/"><b>Twistors and linked light</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/relativity/twistors/twistors.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/twistor_hopfion.gif" alt="The linked electric field lines of a pulse of light, carried along straight light rays" width="100%"></td></tr>
<tr><td width="50%" align="center"><a href="examples/relativity/kerr/"><b>Gravity as a gauge field</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/relativity/kerr/kerr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/kerr.gif" alt="The hot disk of gas around a rotating black hole, bent around its shadow as it orbits" width="100%"></td><td width="50%" align="center"><a href="examples/quantum/kane_mele/"><b>Spin conservation as a type</b></a> <a href="https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/quantum/kane_mele/kane_mele.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a><br><img src="plots/kane_mele.gif" alt="An electron's two spins running round the edge of a honeycomb flake in opposite directions" width="100%"></td></tr>
</table>

* [extensors](docs/extensors.md): a short introduction to extensors, and a showcase of their applications

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

```sh
python -m pip install -e ".[linalg,examples]"     # add ".[jax]" or ".[torch]" for those backends
python -m examples.mechanics.modes.scenarios
```
