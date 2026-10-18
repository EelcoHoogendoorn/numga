# 2. Binding

Putting a value into an open slot binds that slot and evaluates the product there. For example, binding $b$ into `a | Vector` gives the inner product `a | b`:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
(a | Vector)(b)
```

```text
32
```

With two open slots, binding one factor leaves a map with one slot. Binding $a$ into the first slot of `Vector | Vector` leaves the second slot open and gives `a | Vector`:

```python
(Vector | Vector)(a)
```

```text
   x  y  z
1  1  2  3
```

Binding both $a$ and $b$ leaves no open slots, and the result is a plain multivector:

```python
(Vector | Vector)(a, b)
```

```text
32
```

Either slot of a two-slot map can be bound while the other stays open. Leaving `Vector` in the first slot of `Vector ^ Vector` and binding $b$ into the second gives the map that takes $v$ to $v \wedge b$:

```python
(Vector ^ Vector)(Vector, b)
```

```text
     x   y   z
yz   0   6  -5
zx  -6   0   4
xy   5  -4   0
```

Maps with several slots, maps with one slot and plain multivectors are the same kind of object with two, one and zero open slots.

Example: [stress in an elastic cube](../../../examples/mechanics/stress/stress.ipynb), where binding a face's normal into the stress map gives the traction on that face.
