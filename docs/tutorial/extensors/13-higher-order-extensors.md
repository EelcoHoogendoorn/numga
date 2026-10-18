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

In index notation, a map with three vector slots reads as a tensor with four indices, $T^i{}_{jkl}$, and binding a vector into a slot as contracting it with that index.

Examples: [elastic waves in crystals](../../../examples/mechanics/crystal_waves/crystal_waves.ipynb), where the stiffness, a map with three slots, is bound by a heading into the map whose eigenvalues give the speeds of the waves; [gravitational wave curvature](../../../examples/relativity/curvature/curvature.ipynb), where spacetime curvature is bound with an observer's velocity into the tidal stretching of a ring of beads; and [bending of a composite strip](../../../examples/mechanics/composite_strip/composite_strip.ipynb), where a laminate's strain keeps a state slot open.
