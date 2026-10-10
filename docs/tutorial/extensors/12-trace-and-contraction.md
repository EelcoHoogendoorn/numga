# 12. Trace and contraction

The trace pairs the output of a map with one of its slots and drops both. It needs no metric: an output and an input of the same type pair through the complement. Slots are numbered with the output as 0 and the inputs from 1, in order.

A contraction pairs two input slots with each other. Two inputs of the same type pair only through the inner product of their slot, the metric, so a contraction is a separate operation from the trace.

For example, the trace of the second moment of three vectors is the sum of its eigenvalues:

```python
a = 1 * x + 2 * y + 3 * z
b = 4 * x + 5 * y + 6 * z
c = 7 * x + 8 * y + 10 * z
moment = a * (a | Vector) + b * (b | Vector) + c * (c | Vector)
moment.trace()
```

```text
304
```

The map `Vector * (Vector | Vector)` has three slots. Pairing its output with its first slot leaves three times the inner product of the other two; pairing it with its second slot leaves the inner product of the first and the third:

```python
(Vector * (Vector | Vector)).trace(0, 1)(a, b)
```

```text
96
```

```python
(Vector * (Vector | Vector)).trace(0, 2)(a, b)
```

```text
32
```

The form `Vector | moment` has two input slots of the same type, and its contraction pairs them through the inner product:

```python
(Vector | moment).contract()
```

```text
304
```

In index notation, the trace reads as an index repeated between an upper and a lower position, $M^i{}_i$, and the contraction of two lower indices as one through the metric, $g^{ij} F_{ij}$.
