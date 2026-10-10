# 13. Higher-order extensors

A map may have any number of slots, built from products as maps with one or two slots are. Binding some slots and leaving others open, with the type in place of a value, gives a map with fewer slots, and everything that applies to such a map applies to it.

For example, a map from three vectors to a vector:

```python
T = 2 * Vector * (Vector | Vector) + 2 * (Vector | Vector) * Vector - (Vector | (Vector ^ Vector))
```

With $z$ bound into its first and third slots, it is a map from vectors to vectors:

```python
T(z, Vector, z)
```

```text
   x  y  z
x  1  0  0
y  0  1  0
z  0  0  4
```

With $x + y$ in the same slots, it is another:

```python
T(x + y, Vector, x + y)
```

```text
   x  y  z
x  5  3  0
y  3  5  0
z  0  0  2
```

Both are symmetric, and have eigenvalues:

```python
T(x + y, Vector, x + y).eigh()[0]
```

```text
[2  2  8]
```

With values in the first two slots, the third stays open:

```python
T(x, y, Vector)
```

```text
   x  y  z
x  0  2  0
y  1  0  0
z  0  0  0
```

Maps of this shape carry the stiffness of a crystal, bound by a heading into the map whose eigenvalues give the speeds of its waves, in the [crystal waves example](../../../examples/mechanics/crystal_waves/core.py); the curvature of spacetime, bound into the tidal stretching it causes, in the [curvature example](../../../examples/relativity/curvature/core.py); and the strain of a laminate with a state slot left open, in the [composite strip example](../../../examples/mechanics/composite_strip/core.py).

In index notation, a map with three vector slots reads as a tensor with four indices, $T^i{}_{jkl}$, and binding a vector into a slot as contracting it with that index.
