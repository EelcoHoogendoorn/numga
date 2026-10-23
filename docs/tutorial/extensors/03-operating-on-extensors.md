# 3. Operating on extensors

A map has the type of its output, so the operations of the algebra act on that output. A product of two maps takes their arguments independently and multiplies their outputs.

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

The product of the map `a ^ Vector` with itself takes two vector arguments. Evaluating it on $c$ twice gives the square of $a \wedge c$:

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

In matrix notation, a linear operation $D$ on the output of a map $A$ reads as the product $D A$, taking $v$ to $D(A v)$.

Example: [light's wave surface in a crystal](../../../examples/electromagnetism/fresnel/fresnel.ipynb), where `(wavevector ^ medium(field)).dual()` is the wave condition as a function of the field.
