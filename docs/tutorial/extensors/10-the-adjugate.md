# 10. The adjugate

Pairing a fixed vector `c` with the output of a dyad on antivectors through the regressive product gives a scalar. For a dyad `p * (b & Antivector)` with an antivector `p`, that scalar is `b & Antivector` multiplied by `c & p`. It is therefore also the regressive product of the input with the vector `(c & p) * b`.

For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
p = a.dual()
A = p * (b & Antivector)
input_vector = (c & p) * b
(input_vector & Antivector) - (c & A)
```

```text
   yz  zx  xy
1   0   0   0
```

The two scalar-valued maps agree for every input antivector. Each vector `c` paired with the output side has a corresponding vector on the input side giving the same regressive pairing. Adding dyads adds those corresponding vectors, so the construction extends to every linear map on antivectors.

The corresponding vector can also be found by solving. The pairing `Vector & Antivector` identifies each vector with its scalar measurement on antivectors, so solving it against `c & A` recovers `input_vector`:

```python
(Vector & Antivector).solve(c & A) - input_vector
```

```text
  x  y  z
  0  0  0
```

The adjugate of `A` maps each vector on the output side to the corresponding vector on the input side:

```python
adjugate = (Vector & Antivector).solve(Vector & A)
adjugate - A.adjugate()
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

The construction gives `adjugate(c) & v == c & A(v)` for every input antivector `v` and every vector `c`. In particular, `c & A(v) == 0` holds if and only if `adjugate(c) & v == 0`: an incidence condition on outputs carries back to an incidence condition on inputs.

For any input and output grades, the adjugate runs from complements of outputs to complements of inputs, with each pairing read extrinsic first, as `c & v` is. Because `&` pairs every basis blade with its complement without using the inner product, the adjugate exists even when basis vectors square to zero and even when `A` itself has no inverse.

In the notation of projective geometric algebra, which writes the regressive product as $\vee$, the identity reads as $\operatorname{adj} A(c) \vee v = c \vee A(v)$.

In matrix notation, with antivectors in the basis $yz$, $zx$, $xy$ and vectors in the paired basis $x$, $y$, $z$, the coefficients of this adjugate read as $A^T$.

Examples: [multi-view reconstruction](../../../examples/estimation/multiview/multiview_reconstruction.ipynb), where a camera's adjugate carries a pixel's disc of uncertainty back into a cone of sight; and [crystal diffraction](../../../examples/optics/crystal_diffraction/crystal_diffraction.ipynb), where solving through a deformation's adjugate carries wavevectors forward while preserving their phases on the moving atoms.

---

Previous: [9. The adjoint](09-the-adjoint.md) · [Contents](00-introduction.md) · Next: [11. Eigenvalues and singular values](11-eigenvalues-and-singular-values.md)
