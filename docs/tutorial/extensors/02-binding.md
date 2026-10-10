# 2. Binding

Calling a map with a value binds the value into its slot, and the result is the product with that value in place of the type. A map with several slots binds the values in order. A call with fewer values than slots, or with a type in place of a value, leaves the remaining slots open, and the result is a map again.

For example, binding $b$ into the map `a | Vector` gives the inner product of $a$ and $b$, the same as `a | b`:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
(a | Vector)(b)
```

```text
32
```

The map `Vector | Vector` has two slots. Binding $a$ alone fills the first and leaves the second open, and gives the map `a | Vector`:

```python
(Vector | Vector)(a)
```

```text
   x  y  z
1  1  2  3
```

Binding $a$ and $b$ fills both slots:

```python
(Vector | Vector)(a, b)
```

```text
32
```

Passing the type `Vector` to the first slot of `Vector ^ Vector` and binding $b$ into the second gives the map that takes $v$ to $v \wedge b$:

```python
(Vector ^ Vector)(Vector, b)
```

```text
     x   y   z
yz   0   6  -5
zx  -6   0   4
xy   5  -4   0
```
