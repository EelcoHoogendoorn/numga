# 2. Binding

Applying a map evaluates it on its argument. For example, applying `a | Vector` to $b$ gives the inner product `a | b`:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
(a | Vector)(b)
```

```text
32
```

Binding the first argument of a bilinear map gives a linear map. For the inner product, binding $a$ gives `a | Vector`:

```python
(Vector | Vector)(a)
```

```text
   x  y  z
1  1  2  3
```

Evaluating the form on $a$ and $b$ gives their scalar inner product:

```python
(Vector | Vector)(a, b)
```

```text
32
```

Either argument of a bilinear map can be bound independently. Binding $b$ as the second argument of `Vector ^ Vector` gives the map taking $v$ to $v \wedge b$. `Vector` is the identity map on its first argument:

```python
(Vector ^ Vector)(Vector, b)
```

```text
     x   y   z
yz   0   6  -5
zx  -6   0   4
xy   5  -4   0
```

Bilinear maps, linear maps and multivectors are extensors taking two, one and zero arguments respectively.

Example: [stress in an elastic cube](../../../examples/mechanics/stress/stress.ipynb), where binding a face's normal into the stress map gives the traction on that face.
