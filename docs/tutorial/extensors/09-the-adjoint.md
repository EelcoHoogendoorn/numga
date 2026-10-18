# 9. The adjoint

Taking the inner product of a fixed vector $c$ with the output of a dyad gives a scalar. For a dyad `a * (b | Vector)`, that scalar is the inner product with $b$, multiplied by `c | a`. It is therefore also the inner product of the input with the vector `(c | a) * b`.

For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
A = a * (b | Vector)
input_vector = (c | a) * b
(input_vector | Vector) - (c | A)
```

```text
   x  y  z
1  0  0  0
```

The two scalar-valued maps agree for every input. Each vector $c$ on the output side has a corresponding vector on the input side that gives the same inner products. Adding dyads adds those corresponding vectors, so the construction extends to every linear map on vectors.

The corresponding vector can also be found by solving. The inner product with both slots open, `Vector | Vector`, takes a vector in its first slot to that vector's inner-product map. Solving it against `c | A` finds the vector giving those scalars. For the dyad of the example:

```python
(Vector | Vector).solve(c | A) - input_vector
```

```text
  x  y  z
  0  0  0
```

Leaving $c$ open makes this correspondence a map from the output space of $A$ to its input space. This map is the adjoint of $A$:

```python
adjoint = (Vector | Vector).solve(Vector | A)
adjoint - A.adjoint()
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

The construction gives `adjoint(c) | v == c | A(v)` for every input vector `v` and every vector `c` in the output space.

Only the inner product is solved, not $A$ itself: the dyad $A$ collapses all of space onto a single line, yet its adjoint exists. A rotation keeps inner products unchanged, so the adjoint of a rotation is its inverse. When a direction squares to zero, the inner product cannot be solved; the [metric and complement discussion](../../extensor_advanced.md#3-the-metric-and-the-complement) covers that case.

In the notation of Hestenes and Sobczyk, which writes the adjoint with an overbar, the identity reads as $\bar{A}(c) \cdot v = c \cdot A(v)$.

In matrix notation, in orthonormal Euclidean coordinates, the adjoint has coefficients $A^T$, and the identity reads as $(A^T c)\cdot v = c\cdot(A v)$.

Example: [a crystal's diffraction peaks under a deformation](../../../examples/optics/crystal_diffraction/crystal_diffraction.ipynb), where the positions move by the deformation and the wavevectors by its adjoint, so that every phase `k | x` is kept.
