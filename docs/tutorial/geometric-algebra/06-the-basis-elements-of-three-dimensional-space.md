# 6. The basis elements of three-dimensional space

The product of a vector and a bivector follows from the multiplication rules of the geometric product. When the two share a basis vector, the shared pair squares to one:

$$x\, (xy) = x\, x\, y = y$$

When they share none, nothing cancels:

$$x\, (yz) = x\, y\, z$$

The first product is a vector. The second is a product of three different basis vectors, neither a scalar, a vector nor a bivector; in geometric algebra it is called a trivector. Its square follows the same way, moving the second $x$ two places to the left with two sign flips:

$$(xyz)(xyz) = x\, y\, z\, x\, y\, z = x\, x\, y\, z\, y\, z = y\, z\, y\, z = -y\, y\, z\, z = -1$$

Any product of basis vectors reduces in the same way. Exchanging two different neighbours flips the sign, and two equal neighbours square to one:

$$y\, x\, z\, y = -x\, y\, z\, y = x\, y\, y\, z = x\, z$$

What remains is a product of different basis vectors, each at most once, up to sign. In three dimensions there are eight such products, one for each choice of which basis vectors appear: the scalar $1$, the vectors $x$, $y$ and $z$, the bivectors $xy$, $xz$ and $yz$, and the trivector $xyz$. They are the basis elements of the algebra of three-dimensional space.

The same in numga:

```python
x * (x * y)
```

```text
y
```

```python
x * (y * z)
```

```text
xyz
```

```python
(x * y * z) * (x * y * z)
```

```text
-1
```

```python
y * x * z * y
```

```text
xz
```
