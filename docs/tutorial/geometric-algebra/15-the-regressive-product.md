# 15. The regressive product

An element and an element of complementary grade can together fill the volume. The regressive product reads how much of the unit volume they span.

Numga writes it as `&`:

```python
direction = x.dual()
x & direction
```

```text
1
```

The pairing is a scalar because the grades add to the algebra's dimension. Here an extrinsic vector defines a linear measurement on intrinsic directions; positions and wavevectors use this pairing for phase, and displacements and forces for work.

For other grades the regressive product takes the outer product of the volume complements and takes the inverse complement of the result. If $C$ is the right volume complement, chosen on each basis blade $e$ so that $e \wedge C(e)$ is the positive unit volume, the definition is

$$A \vee B = C^{-1}\big(C(A) \wedge C(B)\big).$$

In this tutorial's positive Euclidean signature, `.dual()` is that complement. For $xy$ and $yz$:

```python
((x ^ y).dual() ^ (y ^ z).dual()).dual_inverse()
```

```text
y
```

The same construction written directly:

```python
(x ^ y) & (y ^ z)
```

```text
y
```

The regressive product of $xy$ and $yz$ is $y$, the basis vector the two share. The same for the bivectors $a \wedge b$ and $b \wedge d$:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
d = 7 * x + 8 * y + 10 * z
(a ^ b) & (b ^ d)
```

```text
-12 x - 15 y - 18 z
```

The result is $-3 b$, a multiple of $b$, the vector the two bivectors share.

The volume complement needs neither lengths nor an inverse metric, so `&` has the same definition in a spacetime or projective algebra. A metric dual can have different signs from the volume complement; the formula using `.dual()` above relies on the Euclidean signature.

Example: [a pose from a point, a line and a plane](../../../examples/geometry/three_point_pose/three_point_pose.ipynb), where joins and meets of points, lines and planes recover a rigid motion.
