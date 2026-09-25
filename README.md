# numga

Geometric algebra with Extensors in NumPy, JAX and PyTorch.

| [**Spring modes**](examples/mechanics/modes/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/mechanics/modes/modes.ipynb) | [**Gravitational waves**](examples/relativity/curvature/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/relativity/curvature/curvature.ipynb) |
| :---: | :---: |
| ![Small motions of spring-supported bodies](plots/modes.gif) | ![Tidal response to gravitational wave packets](plots/curvature.gif) |
| [**Multiview reconstruction**](examples/geometry/multiview/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/geometry/multiview/multiview_reconstruction.ipynb) | [**Spherical quadrics**](examples/quadrics/elliptic_physics/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EelcoHoogendoorn/numga/blob/main/examples/quadrics/elliptic_physics/s2_physics.ipynb) |
| ![Cameras aligning on a scene](plots/multiview_convergence.gif) | ![Quadric bodies colliding on a sphere](plots/spherical_quadric_physics.gif) |

* [extensors](docs/extensors.md): An introduction to extensors in numga
* [examples](examples/README.md): An index of runnable example code.

A short example of the extensor syntax: building unary maps and bilinear forms, transforming an extensor, and solving a problem, the vibration modes of four point masses on springs:

```python
from numga import JaxContext
from numga.algebras import PGA3D

mv = JaxContext(PGA3D).multivector
Bivector = PGA3D.gatype.bivector()

points = mv.vector([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]).dual()   # four unit masses
inertia = (points & points.commutator(Bivector)).sum(axis=0)   # Antibivector <- Bivector

motor = (mv.xy * 0.3 + mv.zw * 0.2).exp()                      # rotation plus translation
world_inertia = motor >> inertia(motor << Bivector)            # move the body

springs = points[[0, 0, 0, 1, 1, 2]] & points[[1, 2, 3, 2, 3, 3]]   # Antibivector
stiffness = (springs * (springs & Bivector)).sum(axis=0)   # Antibivector <- Bivector
# the vibration modes, between two bilinear energy forms
values, modes = (Bivector & stiffness).eigh(Bivector & inertia)
```

## Install and run

```sh
python -m pip install -e ".[linalg,examples]"     # add ".[jax]" or ".[torch]" for those backends
python -m examples.mechanics.modes.scenarios --animate
```
