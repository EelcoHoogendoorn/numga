# 15. The regressive product

Two elements of complementary grades fill the volume together, and their regressive product reads how much of the unit volume they span. In numga it is written `&`:

```python
direction = x.dual()
x & direction
```

```text
1
```

Because the grades add to the algebra's dimension, the result is a scalar: an extrinsic vector defines a linear measurement on intrinsic antivectors.

For general grades, the regressive product takes the outer product of the volume complements and applies the inverse complement to the result. Writing $C$ for the right volume complement, chosen on each basis blade $e$ so that $e \wedge C(e)$ is the positive unit volume, the definition is

$$A \vee B = C^{-1}\big(C(A) \wedge C(B)\big)$$

In the positive Euclidean signature of this tutorial, `.dual()` agrees with that complement. For $xy$ and $yz$:

```python
((x ^ y).dual() ^ (y ^ z).dual()).dual_inverse()
```

```text
y
```

The same product written directly:

```python
(x ^ y) & (y ^ z)
```

```text
y
```

The regressive product of $xy$ and $yz$ is $y$, the basis vector the two share. For the bivectors $a \wedge b$ and $b \wedge d$:

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

Because the volume complement uses neither lengths nor an inverse metric, `&` keeps the same definition in spacetime and projective algebras alike. A metric dual can differ from the volume complement by signs, so the formula using `.dual()` above relies on the positive Euclidean signature.

Example: [a pose from a point, a line and a plane](../../../examples/geometry/three_point_pose/three_point_pose.ipynb), where the joins of three marker points into a line and a plane recover a rigid motion.

---

Previous: [14. Duality](14-duality.md) · [Contents](00-introduction.md) · Next: [16. Signature](16-signature.md)
