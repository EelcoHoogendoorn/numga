# 4. Sums of extensors

Extensors with the same slots add as multivectors do: the sum takes values in its slots to the sum of the outputs. For example:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
(a ^ Vector) + (b ^ Vector) - ((a + b) ^ Vector)
```

```text
    x  y  z
yz  0  0  0
zx  0  0  0
xy  0  0  0
```

The type `Vector` on its own is the identity map, and adds to other maps from vectors to vectors:

```python
Vector + (Vector | (x ^ y))
```

```text
   x   y  z
x  1  -1  0
y  1   1  0
z  0   0  1
```

Outputs of different grades add into their joint space, as with multivectors. The inner and the outer product with both slots open add up to the geometric product with both slots open, `Vector * Vector`:

```python
(Vector | Vector) + (Vector ^ Vector)
```

```text
first slot x:
    x  y   z
1   1  0   0
yz  0  0   0
zx  0  0  -1
xy  0  1   0

first slot y:
     x  y  z
1    0  1  0
yz   0  0  1
zx   0  0  0
xy  -1  0  0

first slot z:
    x   y  z
1   0   0  1
yz  0  -1  0
zx  1   0  0
xy  0   0  0
```

In matrix notation, the sum reads as $A + B$, with $(A + B) v = A v + B v$.
