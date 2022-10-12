import numpy as np

from numga.operator.sparse_tensor import SparseTensor
from numga.algebra.algebra import Algebra


def test_sparse():

	a = SparseTensor((np.arange(4),), np.arange(4))
	b = SparseTensor((np.arange(3),), np.arange(3))
	c = a.product(b).product(a)
	d = c.contract([0, 2])
	print()
	print(d.data)
	print(np.einsum('iji->j', c.to_dense_()))

	for a in c.axes:
		print(a)
	print(c.data)
	print(c.to_dense_())
	return


def test_quat():
	r3 = Algebra('x+y+z+')
	quat = r3.subspace.even_grade()
	c, s = r3.product(quat.blades, quat.blades)

	grid = np.indices(c.shape)
	st = SparseTensor(
		(
			quat.blades[grid[0].flatten()],
			quat.blades[grid[1].flatten()],
			c.flatten()
		),
		s.flatten(),
	)
	print()
	print(st.to_dense((quat, quat, quat)))
