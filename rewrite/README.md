# numga

**Geometric algebra and numerical linear algebra in one unified language.**

Numga bridges the gap between coordinate-free geometry and numerical computing. In numga, multivectors, linear maps, and multilinear forms are unified into a single abstraction: the **extensor**. Leave an argument open to construct a map; compose it, transform it with a motor sandwich, solve against it, or compute its eigenmodes without ever falling back to coordinate matrices or index swapping.

Numga runs seamlessly on **NumPy** for zero-setup numerical execution and **JAX** for JIT-compiled, autodiff-ready execution on CPU, GPU, and TPU.

| [Spring modes](examples/mechanics/modes/) | [Gravitational wave tidal response](examples/relativity/curvature/) |
| :---: | :---: |
| ![Spring Modes](plots/stiffness.gif) | ![Curvature Response](plots/curvature.gif) |
| [**Multiview camera alignment**](examples/geometry/multiview/) | [**Spherical quadric physics**](examples/quadrics/) |
| ![Multiview Reconstruction](plots/multiview_convergence.gif) | ![Spherical Quadric Physics](plots/spherical_quadric_physics.gif) |

---

## At a Glance

### 1. Natural Vibration Modes without Matrix Inversion

In classical mechanics, finding vibration frequencies requires building coordinate stiffness ($K$) and mass ($M$) matrices, inverting mass, and solving the asymmetric eigenproblem $M^{-1} K v = \omega^2 v$.

In numga, spring geometry produces stiffness, a mass cloud produces inertia, and their contractions with open twists produce coordinate-free energy forms. The natural frequencies and normal modes solve directly between the two forms:

```python
from numga import NumpyContext
from numga.algebras import PGA2D

ctx = NumpyContext(PGA2D)
Twist, Line = PGA2D.gatype.bivector(), PGA2D.gatype.vector()

# Geometric stiffness from spring lines, inertia from mass points: Wrench <- Twist
stiffness = (spring_lines * (spring_lines & Twist) * spring_constants).sum(axis=0)
inertia = ((mass_points & mass_points.commutator(Twist)) * masses).sum(axis=0)

# Contract with open twists into symmetric bilinear energy forms: Scalar <- (Twist, Twist)
pe_form = Twist & stiffness
ke_form = Twist & inertia

# Solve generalized eigenproblem directly between forms (no mass inversion, no coordinates)
frequencies, modes = pe_form.eigh(ke_form)       # [3] Scalar, [3] Twist
```

### 2. Point Cloud to Inertia and Motor Transport

Every mass point contributes momentum under an open rigid motion. Summing those contributions yields the body's inertia map; placing the body with a motor moves the inertia via a sandwich:

```python
from numga import NumpyContext
from numga.algebras import PGA3D

mv = NumpyContext(PGA3D).multivector
Point, Bivector = PGA3D.gatype.antivector(), PGA3D.gatype.bivector()

# Four unit masses represented as homogeneous points
points = mv.antivector([[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])

# Inertia map: Wrench <- Twist
inertia = (points & points.commutator(Bivector)).sum(axis=0)

# Place the entire body with a motor, including its inertia map
motor = (mv.xy * 0.3 + mv.zw * 0.2).exp()
world_inertia = motor >> inertia(motor << Bivector)   # Wrench <- Twist
```

---

## Core Pillars

### 1. Extensors: Multivectors, Maps, and Forms Unified
In standard geometric algebra, multivectors are first-class, but linear maps are awkward external operators. In numga, all geometric and algebraic objects are instances of `Extensor`:
* **Nullary (0 open slots)**: Ordinary geometric elements (points, lines, motors, wrenches).
* **Unary (1 open slot)**: Linear maps (`Plane <- Point`, `Twist <- Line`, `Wrench <- Twist`).
* **Binary (2 open slots)**: Bilinear forms, metrics, and quadrics (`Scalar <- (Point, Point)`).

Open slots create maps; calling them binds arguments. Maps participate in geometric products, motor sandwiches (`motor >> T(motor << In)`), commutators, duals, and traces just like multivectors.

### 2. GAType: Semantic Types and Algebraic Traits
Types in numga encode more than subspace blade masks. They carry algebraic **traits** (`Versor`, `Rotor`, `Normalized`, `Symmetric`, `CoefficientOrthogonal`) that propagate symbolically through products:
* A rotor product preserves `Rotor` and `Normalized`.
* A sandwich with an orthogonal motor automatically certifies output subspaces without runtime casting.
* An orthogonal map's `.inverse()` dispatches at compile-time to coefficient transposition without inspecting numbers.

### 3. Dual Backends: NumPy and JAX
* **Full Array Vectorization**: Batch operations broadcast cleanly across arbitrary leading dimensions (`[batch, n_elements, ...]`).
* **NumPy Backend**: Zero-overhead, instant-execution numerical backend for scripting and interactive research.
* **JAX Backend**: Fully compatible with JAX JIT compilation, autodiff, and GPU/TPU execution. Extensors are native JAX PyTrees (numerical kernels are dynamic arrays; GATypes and contexts are static compile-time metadata).

### 4. Signature Generality $(p, q, r)$
Numga supports any algebra dimension and metric signature:
* **Euclidean**: $R^2, R^3, R^n$
* **Projective Geometric Algebra (PGA)**: 2D PGA $\mathbb{R}_{2,0,1}$, 3D PGA $\mathbb{R}_{3,0,1}$
* **Conformal Geometric Algebra (CGA)**: $\mathbb{R}_{4,1}$
* **Spacetime Algebra (STA)**: $\mathbb{R}_{1,3}$

The same algebraic expressions for kinematics, collision, or quadrics run unmodified across Euclidean, spherical, hyperbolic, or relativistic geometries.

---

## Showcase Applications

The [`examples/`](examples/README.md) directory contains complete, standalone implementations demonstrating numga across domains:

| Domain | Application | Code | Key Algebraic Concept |
| :--- | :--- | :--- | :--- |
| **Mechanics** | Spring Modes | [examples/mechanics/modes/](examples/mechanics/modes/) | Generalized eigensolve between potential & kinetic energy forms: `pe_form.eigh(ke_form)`. |
| **Mechanics** | Inertia & Moments | [examples/mechanics/inertia.py](examples/mechanics/inertia.py) | Recovering second-moment quadrics from inertia via open-slot tensor solve. |
| **Computer Vision** | Multiview Alignment | [examples/geometry/multiview/](examples/geometry/multiview/) | Polarity cone pullbacks, Schur complement marginalization, and gauge-free tangent steps. |
| **Relativity** | Gravitational Waves | [examples/relativity/curvature/](examples/relativity/curvature/) | Spacetime bivector curvature, tidal forces, and coordinate-free Ricci contraction. |
| **Electromagnetism** | Moving Media | [examples/electromagnetism/constitutive/](examples/electromagnetism/constitutive/) | Relativistic constitutive tensors, Lorentz boosts, and wave polarizations. |
| **Robotics & SLAM** | Kalman Filter | [examples/geometry/kalman/](examples/geometry/kalman/) | Pose filtering on motor manifolds with covariance maps `Twist <- Line`. |
| **Optics** | Lens Camera | [examples/optics/lens_camera/](examples/optics/lens_camera/) | Composing optical lens maps and pulling aperture quadrics through the system. |
| **Non-Euclidean** | Spherical Quadrics | [examples/quadrics/](examples/quadrics/) | Dual quadric collisions and motor transport in non-Euclidean spaces. |

---

## Documentation

The documentation is organized into four complementary guides:

1. [**Extensors: Introduction & Mental Models**](docs/extensors.md)
   The high-level introduction to extensors, slot notation, and canonical domain examples.
2. [**Syntax Reference & Extension Methods**](docs/extensor_syntax.md)
   Exhaustive guide to extensor syntax, open-slot lifting rules, currying, and built-in numerical methods.
3. [**Advanced Foundations: The Linear Algebra Rosetta Stone**](docs/extensor_advanced.md)
   Deep theoretical treatment: the absence of transpose, quadric pullbacks, maps vs. forms, gauge and homogeneous variables, traces, and covariance.
4. [**Examples Index & Tutorials**](examples/README.md)
   Detailed walkthroughs, notebooks, and run instructions for all domain examples.

---

## Getting Started

### Installation

Clone the repository and install with optional extras:

```sh
# Core with NumPy backend and linear algebra extensions
pip install -e ".[linalg,test]" matplotlib pillow

# Optional: Install with JAX backend
pip install -e ".[jax]"
```

### Running Examples

Run any example directly from the `rewrite/` directory:

```sh
# Run vibration modes demo
PYTHONPATH=src:. python -m examples.mechanics.modes.scenarios --animate

# Run multiview camera alignment
PYTHONPATH=src:. python -m examples.geometry.multiview.scenarios

# Run gravitational wave curvature simulation
PYTHONPATH=src:. python -m examples.relativity.curvature.scenarios
```

### Running Tests

Run targeted tests using pytest:

```sh
# Test extensor operations and arithmetic
python -m pytest tests/test_extensor_arithmetic.py

# Test linear algebra extensions (solves, eigensolves, SVD)
python -m pytest tests/test_unary_extensions.py tests/test_form_extensions.py
```
