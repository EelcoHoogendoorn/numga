# 6. Composition

An outer product can supply the bivector in another outer product. For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
a ^ (b ^ c)
```

```text
-3 xyz
```

Leaving $c$ open gives the whole calculation as a map from vectors to trivectors:

```python
a ^ (b ^ Vector)
```

```text
      x  y   z
xyz  -3  6  -3
```

The two steps can also be written as maps. The first, `b ^ Vector`, produces a bivector. Its output fits the slot of the second, `a ^ Bivector`. Putting the first map in that slot gives:

```python
composition = (a ^ Bivector)(b ^ Vector)
composition(c)
```

```text
-3 xyz
```

This is composition: the output of one map supplies the input of another, and the first map's input remains open. The intermediate types must match; the result carries the input type of the first map and the output type of the second.

When two maps can be composed in either order, the results may differ. For example:

```python
turn_xy = Vector | (x ^ y)
turn_yz = Vector | (y ^ z)
turn_xy(turn_yz)(z)
```

```text
x
```

```python
turn_yz(turn_xy)(z)
```

```text
0
```

The first sequence sends $z$ to $y$ and then to $x$. The second sends $z$ to zero at its first step.

In matrix notation, feeding $B$ into $A$ reads as $A B$, with the output index of $B$ summed against the input index of $A$.
