# 13. Higher-order extensors

A multilinear map may take any number of arguments. Binding some of them gives a map of the remaining arguments, to which the same algebraic operations apply.

For example, a map from three vectors to a vector:

```python
T = 2 * Vector * (Vector | Vector) + 2 * (Vector | Vector) * Vector - (Vector | (Vector ^ Vector))
```

Binding $z$ as its first and third arguments gives a map from vectors to vectors:

```python
T(z, Vector, z)
```

```text
   x  y  z
x  1  0  0
y  0  1  0
z  0  0  4
```

Binding $x + y$ as the same two arguments gives another map:

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

`T(x, y, Vector)` is the linear map taking $v$ to `T(x, y, v)`:

```python
T(x, y, Vector)
```

```text
   x  y  z
x  0  2  0
y  1  0  0
z  0  0  0
```

In index notation, a map taking three vector arguments reads as a tensor with four indices, $T^i{}_{jkl}$, and binding a vector as contracting it with the corresponding index.

Examples: [elastic waves in crystals](../../../examples/mechanics/crystal_waves/crystal_waves.ipynb), where binding a heading to the stiffness gives the map whose eigenvalues determine wave speeds; [gravitational wave curvature](../../../examples/relativity/curvature/curvature.ipynb), where binding an observer's velocity to spacetime curvature gives the tidal stretching of a ring of beads; and [bending of a composite strip](../../../examples/mechanics/composite_strip/composite_strip.ipynb), where the laminate's strain is a function of its state.
