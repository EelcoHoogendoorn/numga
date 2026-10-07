# numga

Geometric algebra with Extensors in NumPy, JAX and PyTorch.

Extensors generalize matrices and tensors from vectors to multivectors, and from the inner and tensor products to every product of geometric algebra.

| [**Scenegraph and camera optics**](examples/geometry/scenegraph/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/geometry/scenegraph/scenegraph.ipynb) | [**Vanishing geometric derivative**](examples/mechanics/wing/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/mechanics/wing/wing.ipynb) |
| :---: | :---: |
| <img src="plots/scenegraph.gif" alt="A robot arm in 3D, and its picture on a camera sensor" width="420" /> | <img src="plots/wing.gif" alt="A wing pitching in a steady stream, its pressure, streamlines and lift" width="352" /> |

| [**Summing cones into splats**](examples/estimation/multiview/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/estimation/multiview/multiview_reconstruction.ipynb) | [**Twistors and linked light**](examples/relativity/twistors/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/relativity/twistors/twistors.ipynb) |
| :---: | :---: |
| <img src="plots/multiview_convergence.gif" alt="Cameras aligning on a scene" width="320" /> | <img src="plots/twistor_hopfion.gif" alt="The linked electric field lines of a pulse of light, carried along straight light rays" width="320" /> |

* [extensors](docs/extensors.md): An introduction to extensors in numga
* [examples](examples/README.md): An index of runnable example code.

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

## Install and run

```sh
python -m pip install -e ".[linalg,examples]"     # add ".[jax]" or ".[torch]" for those backends
python -m examples.mechanics.modes.scenarios --animate
```
