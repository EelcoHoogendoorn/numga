# 11. Eigenvalues and singular values

The map `a ^ Vector` forms an oriented area with $a$. An input parallel to $a$ gives zero:

```python
a = 1 * x + 2 * y + 3 * z
area = a ^ Vector
area(a)
```

```text
0
```

For an input perpendicular to $a$, the area is the input's length times the length of $a$. Two perpendicular input directions therefore have the same scale, and the direction along $a$ has scale zero. These three scales are the singular values of the map:

```python
area.svdvals()
```

```text
[3.7417  3.7417  0]
```

The input and output directions have different grades, vectors and bivectors. Their magnitudes can still be compared.

When a map's input and output belong to the same space, an output can also be a multiple of its input. For example:

```python
stretch = Vector + 2 * z * (z | Vector)
stretch(z)
```

```text
3 z
```

The map leaves $x$ and $y$ unchanged and multiplies $z$ by three. These directions are eigenvectors, and their multipliers are eigenvalues:

```python
stretch.eigh()[0]
```

```text
[1  1  3]
```

A form has a scalar output and two input slots. Pairing a map's output with another vector gives such a form. Solving through the pairing recovers the map:

```python
form = Vector | stretch
metric = Vector | Vector
metric.solve(form) - stretch
```

```text
   x  y  z
x  0  0  0
y  0  0  0
z  0  0  0
```

The eigenvalues of the form relative to that pairing are therefore those of the recovered map. The same construction can use a second positive-definite symmetric form in place of the metric:

```python
weight = Vector | (Vector + z * (z | Vector))
form.eigh(weight)[0]
```

```text
[1  1  1.5]
```

The second form weights the $z$ direction by two, so the relative scale there is three divided by two. These are generalized eigenvalues: the scales of one form relative to another. Without an explicit second form, `eigh` uses the Euclidean inner product of the slot.

In matrix notation, eigenvalues of a map satisfy $M v = \lambda v$, singular values compare orthonormal input and output directions, and eigenvalues relative to a second form satisfy $M v = \lambda W v$.
