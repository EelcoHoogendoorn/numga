import numpy as np
from numga.backend.numpy.context import NumpyContext


def test_basic():
	ga = NumpyContext('x+y+w0')
	V = ga.subspace.vector()
	V = ga.subspace.from_bit_str('100,010,001')
	v = ga.multivector.vector(values=np.random.normal(size=len(V)))
	print(v)
	print(+v)
	print(-v)
	print(v - 1)
	print(1 - v)
	print(v / 1)
	print(1 / v)
	print((1 / v) * v)
