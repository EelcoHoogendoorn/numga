# 3. Operating on extensors

A map has the type of its output, so the operations of the algebra apply to the map itself, carrying its open axis along. A product of two maps has the axes of both.

For example, taking the dual of the map `a ^ Vector` takes the dual of its output, the bivector $a \wedge v$, and gives a map from vectors to vectors:

```python
a = 1 * x + 2 * y + 3 * z
(a ^ Vector).dual()
```

```text
    x   y   z
x   0  -3   2
y   3   0  -1
z  -2   1   0
```

Taking the outer product of the map `a ^ Vector` with a further vector $b$ gives a map from vectors to trivectors:

```python
b = 4 * x + 5 * y + 6 * z
(a ^ Vector) ^ b
```

```text
     x   y  z
xyz  3  -6  3
```

Bound to a vector $c$, it gives the same as the outer product written out:

```python
c = 7 * x + 8 * y + 10 * z
((a ^ Vector) ^ b)(c)
```

```text
3 xyz
```

```python
a ^ c ^ b
```

```text
3 xyz
```

The product of the map `a ^ Vector` with itself has two axes, one for each occurrence. Binding $c$ into both gives the square of $a \wedge c$:

```python
((a ^ Vector) * (a ^ Vector))(c, c)
```

```text
-173
```

```python
(a ^ c) * (a ^ c)
```

```text
-173
```

In matrix notation, a linear operation on the output of a map follows the same rule. For an operation $D$ and a map $A$, it reads as the product $D A$: the output of $A$ feeds the input of $D$, and the input of $A$ stays open as the input of the result.

Example: [light's wave surface in a crystal](../../../examples/electromagnetism/fresnel/fresnel.ipynb), where an outer product and a dual act on the output of a material map, `(wavevector ^ medium(field)).dual()`, and leave its field axis open.
