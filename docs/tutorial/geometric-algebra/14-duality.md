# 14. Duality

Each basis blade has a complementary blade: the directions it lacks to fill the whole space. In three dimensions, a vector's complement is a bivector, a bivector's complement a vector. Numga's dual takes an element into that complementary space:

```python
x.dual()
```

```text
yz
```

The sign of a complement matters. In the Euclidean algebra of this tutorial, each unit basis blade followed by its dual in the outer product gives the positive unit volume:

```python
y.dual()
```

```text
zx
```

```python
y ^ y.dual()
```

```text
xyz
```

The dual acts linearly on sums and takes grade $k$ to grade $n-k$ in an algebra of dimension $n$. The complementary blade of $xy$ is $z$, with its sign set by the order $x,y,z$:

```python
(x ^ y).dual()
```

```text
z
```

There is also a change of orientation. The basis vector `x` is extrinsic; `x.dual()` is intrinsic. In three dimensions this intrinsic grade-two element is an antivector. Complementary grade and complementary orientation are both part of its type. [Why orientation](../../why_orientation.md) shows the geometric distinction through reflections.

Use the inverse dual to return to the starting element:

```python
direction = x.dual()
direction.dual_inverse() - x
```

```text
0
```

Complementarity tells us which spaces pair; a dual map chooses how a particular element is sent into the complementary space. In a nondegenerate algebra, numga's `.dual()` is the metric dual, carrying the metric's signs as well as the blade complement. With the positive Euclidean signature used here, it agrees with the volume complement. In the standard projective algebras, whose metric is degenerate, `.dual()` remains invertible as a blade complement; it does not invert the degenerate metric. The [signature chapter](16-signature.md) introduces these other algebras.

The [regressive product](15-the-regressive-product.md) uses volume complements to pair elements without needing a metric.

Example: [a crystal's diffraction peaks under a deformation](../../../examples/optics/crystal_diffraction/crystal_diffraction.ipynb), where positions are intrinsic antivectors and wavevectors are extrinsic vectors, paired into phases.
