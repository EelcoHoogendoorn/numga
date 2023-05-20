"""Sparse matrix types with geometric-algebraic valued types"""
from typing import List, Tuple

import numpy as np
import numpy_indexed as npi


class AbstractGAMatrix:
	shape: Tuple[int, int]
	operator: "Operator"

	def __init__(self, shape, operator):
		self.shape = shape
		self.operator = operator
	@property
	def subspace(self):
		return self.operator.operator.axes[0]
	@property
	def row_subspace(self):
		return self.operator.operator.axes[2]
	@property
	def column_subspace(self):
		return self.operator.operator.axes[1]
	@property
	def context(self):
		return self.operator.context

	def __call__(self, other):
		raise NotImplementedError

	def as_operator(self) -> "LinearOperator":
		from scipy.sparse.linalg import LinearOperator
		r, c = self.shape
		i, j = len(self.row_subspace), len(self.column_subspace)
		def matvec(x):
			*shape, n = x.shape
			x = self.context.multivector(values=x.reshape(shape + [c, j]), subspace=self.column_subspace)
			y = self(x)
			return y.values.reshape(shape + [r * i])
		return LinearOperator(shape=(r*i, c*j), matvec=matvec)


class GA_COO_Matrix(AbstractGAMatrix):
	"""Coordinate Matrix
	to be used as a construction helper"""
	entries: List = []

	def add(self, r, c, v):
		assert v.subspace in self.subspace
		assert 0 <= r < self.shape[0]
		assert 0 <= c < self.shape[1]
		self.entries.append((r, c, v.select_subspace(self.subspace)))

	def as_crs(self):
		return GA_CRS_Matrix(self)


class GA_CRS_Matrix(AbstractGAMatrix):
	"""Compressed row storage matrix;
	precompute some things based on COO matrix for efficient execution"""
	def __init__(self, coo):
		super(GA_CRS_Matrix, self).__init__(coo.shape, coo.operator)

		# sum duplicates and store as sorted dense array
		r, c, V = zip(*coo.entries)
		V = np.array([v.values for v in V])
		(c, r), V = npi.group_by((c, r)).sum(V)

		self.rows = r
		self.columns = c
		self.values = self.context.multivector(self.subspace, V)
		self.indices = npi.as_index(r).start # slices for reduceat
		assert len(self.indices) == self.shape[0]   # need to change indices logic if we want to support empty rows

	def __call__(self, other):
		temp = self.operator(self.values, other[self.columns])
		result = np.add.reduceat(temp.values, self.indices)
		return temp.copy(values=result)

	def as_bound(self):
		return GA_CRS_Bound_Matrix(self)


class GA_CRS_Bound_Matrix(AbstractGAMatrix):
	"""compressed-row-storage matrix with pre-bound values product
	In other words; we reshuffle our operation into a 'matrix form'
	"""
	def __init__(self, crs):
		super(GA_CRS_Bound_Matrix, self).__init__(crs.shape, crs.operator)
		self.rows = crs.rows
		self.columns = crs.columns
		self.indices = crs.indices
		# pre-bind the operator with the sparse entries
		self.bound = self.operator.partial({0: crs.values})

	def __call__(self, other):
		temp = self.bound(other[self.columns])
		result = np.add.reduceat(temp.values, self.indices)
		return temp.copy(values=result)

	def as_dense(self):
		dense = np.zeros(shape=self.shape + self.bound.kernel.shape[1:])
		dense[self.rows, self.columns] = self.bound.kernel
		from einops import rearrange
		return rearrange(dense, 'r c i j -> (r i) (c j)')



def test_quat():
	"""Test quaternionic geometric product matrix"""
	from numga.algebra.algebra import Algebra
	from numga.backend.numpy.context import NumpyContext as Context
	from numga.multivector.test.util import random_subspace

	context = Context(Algebra.from_pqr(3, 0, 0))

	Q = context.subspace.even_grade()
	B = context.subspace.bivector()
	coo = GA_COO_Matrix(
		shape=(3, 3),
		operator=context.operator.product(Q, Q)
	)
	v = random_subspace(context, Q, (10,))
	coo.add(0, 1, v[0])
	coo.add(0, 1, v[1])
	coo.add(0, 2, v[2])
	coo.add(2, 2, v[3])
	coo.add(1, 1, v[4])
	coo.add(0, 0, v[5])
	coo.add(2, 0, v[6])

	crs = coo.as_crs()

	x = random_subspace(context, Q, (3,))
	r = crs(x)
	print(r)

	bm = crs.as_bound()
	# print(bm.bound.kernel)
	r = bm(x)
	print(r)
	r = crs.as_operator()(x.values.flatten()).reshape(x.values.shape)
	print(r)

	print(np.around(bm.as_dense(), 2))


def test_other():
	"""Test some other random things"""
	from numga.algebra.algebra import Algebra
	from numga.backend.numpy.context import NumpyContext as Context
	from numga.multivector.test.util import random_subspace

	context = Context(Algebra.from_pqr(2, 1, 0))

	Q = context.subspace.even_grade()
	B = context.subspace.bivector()
	coo = GA_COO_Matrix(
		shape=(3, 3),
		operator=context.operator.commutator(B, B)
	)
	v = random_subspace(context, B, (10,))
	coo.add(0, 1, v[0])
	coo.add(0, 1, v[1])
	coo.add(0, 2, v[2])
	coo.add(2, 2, v[3])
	coo.add(1, 1, v[4])
	coo.add(0, 0, v[5])
	coo.add(2, 0, v[6])

	crs = coo.as_crs()

	x = random_subspace(context, B, (3,))
	r = crs(x)
	print(r)

	bm = crs.as_bound()
	# print(bm.bound.kernel)
	r = bm(x)
	print(r)
	r = crs.as_operator()(x.values.flatten()).reshape(x.values.shape)
	print(r)

	print(np.around(bm.as_dense(), 2))
