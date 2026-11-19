# 14. Duality

Each basis blade has a complementary blade, built from the basis vectors it lacks to fill the whole space. In three dimensions, a vector's complement is a bivector and a bivector's complement is a vector. In numga, `.dual()` maps an element into that complementary space:

```python
x.dual()
```

```text
yz
```

The sign of a complement is fixed by the unit volume. In the Euclidean algebra of this tutorial, each unit basis blade followed by its dual in the outer product gives the positive unit volume $xyz$:

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

The dual acts linearly on sums and sends grade $k$ to grade $n - k$ in an algebra of dimension $n$. For $xy$, the complementary blade is $z$:

```python
(x ^ y).dual()
```

```text
z
```

Taking the dual also flips orientation: the basis vector `x` is extrinsic, while `x.dual()` is intrinsic, an antivector of grade two in three dimensions. Both grade and orientation are tracked in the type ([Orientation in numga](../../orientation.md), [Why orientation](../../../articles/orientation.md)).

The inverse dual returns to the starting element:

```python
direction = x.dual()
direction.dual_inverse() - x
```

```text
0
```

In a nondegenerate algebra, `.dual()` is the metric dual, carrying the metric's signs together with the blade complement. In the positive Euclidean signature used here, it agrees with the volume complement; in projective algebras with a degenerate metric, `.dual()` remains invertible as a blade complement without inverting the metric.

Example: [a crystal's diffraction peaks under a deformation](../../../examples/optics/crystal_diffraction/crystal_diffraction.ipynb), where positions are intrinsic antivectors and wavevectors are extrinsic vectors, paired into phases.

---

Previous: [13. Inverses](13-inverses.md) · [Contents](00-introduction.md) · Next: [15. The regressive product](15-the-regressive-product.md)
