# 10. The adjugate

The regressive product of a bivector and a vector in three dimensions is a scalar: their signed volume. Pairing a fixed bivector with the output of a map therefore gives a scalar-valued map on its inputs. A bivector on the input side can give those same volumes directly.

The pairing `Bivector & Vector` takes a bivector in its first slot to its regressive-product map. Solving it against `plane & A` finds the corresponding input bivector. For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
A = Vector + a * (b | Vector)
plane = x ^ y
input_plane = (Bivector & Vector).solve(plane & A)
input_plane
```

```text
  yz  zx  xy
  12  15  19
```

Pairing an input with `input_plane` gives the same scalar as pairing its output with `plane`:

```python
(input_plane & Vector) - (plane & A)
```

```text
   x  y  z
1  0  0  0
```

In particular, an output lies in `plane` exactly when its regressive product with `plane` is zero. The corresponding input satisfies `input_plane & v == 0`. A plane condition on the outputs has become a plane condition on the inputs.

Leaving the output bivector open makes the correspondence a map from bivectors to bivectors. This map is the adjugate of $A$:

```python
adjugate = (Bivector & Vector).solve(Bivector & A)
adjugate - A.adjugate()
```

```text
    yz  zx  xy
yz   0   0   0
zx   0   0   0
xy   0   0   0
```

The construction gives, for every input vector $v$ and every bivector $p$ paired with the output:

$$\operatorname{adj} A(p) \vee v = p \vee A(v)$$

For other input and output grades, the regressive pairing uses their complementary grades. The adjugate therefore runs from complements of outputs to complements of inputs. For a rotation, preservation of the regressive product makes the corresponding input bivector the output bivector rotated back.

The regressive pairing pairs every basis blade with its complement, independently of the metric. It remains invertible in a degenerate algebra, and the construction works even when $A$ loses information. The [multiview example](../../../examples/estimation/multiview/core.py) uses a camera's adjugate to find the plane of scene points that project onto an image line.

In matrix notation, with the complementary bivectors named $yz$, $zx$, $xy$, the coefficients of this adjugate read as $A^T$.
