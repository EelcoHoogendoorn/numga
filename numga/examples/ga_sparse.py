"""Sparse matrix types with geometric-algebraic valued types

"""
from typing import List, Tuple, Callable

import scipy.sparse

from numga.util import cached_property

import numpy as np
import numpy_indexed as npi

from numga.subspace.subspace import SubSpace
from numga.multivector.multivector import AbstractMultiVector


class AbstractContainer:
	shape: Tuple[int, int]
	subspace: "Subspace"

	def __init__(self, shape, subspace):
		self.shape = shape
		self.subspace = subspace

	def __getattr__(self, item):
		operator = getattr(self.context.operator, item)
		return MatrixOperator(self, operator)


class Builder(AbstractContainer):
	"""Coordinate Matrix
	to be used as a construction helper"""
	entries: List = []

	def add(self, r, c, v):
		assert v.subspace in self.subspace
		assert 0 <= r < self.shape[0]
		assert 0 <= c < self.shape[1]
		self.entries.append((r, c, v.select_subspace(self.subspace)))

	def finalize(self) -> "MatrixContainer":
		r, c, V = zip(*self.entries)
		context = V[0].context
		V = context.multivector(self.subspace, np.array([v.values for v in V]))
		return MatrixContainer(self.shape, r, c, V)


class MatrixContainer(AbstractContainer):
	"""Plain sparse container type
	it is not yet bound to a product type
	knowledge of how to actually multiply is left to Operator classes
	"""

	@classmethod
	def diag(cls, x: AbstractMultiVector):
		"""construct a diagonal matrix from a vector"""
		l, = x.shape
		idx = np.arange(l)
		return cls(
			(l, l),
			idx,
			idx,
			x
		)
	@classmethod
	def from_scipy(cls, context, sparse):
		sparse = sparse.tocoo()
		return cls(
			sparse.shape,
			sparse.row,
			sparse.col,
			context.multivector.scalar(sparse.data[:, None])
		)
	def copy(self, values):
		return MatrixContainer(
			self.shape,
			self.rows,
			self.columns,
			values
		)

	@property
	def subspace(self):
		return self.values.subspace
	@property
	def context(self):
		return self.values.context
	@cached_property
	def indices(self):
		"""compute indices for reduceat expression, assuming rows idx are in sorted order"""
		# need to change indices logic if we want to support empty rows
		# FIXME: reduceat does not even work with empty slices wtf?
		indices = np.concatenate(([0], np.flatnonzero(np.diff(self.rows)) + 1))
		assert len(indices) == self.shape[0]
		return indices

	def __init__(self, shape, rows, columns, values):
		# sum duplicates and store as sorted dense array
		(columns, rows), V = npi.group_by((columns, rows)).sum(values.values)
		values = values.copy(V)

		self.shape = shape
		self.rows = rows    # given group-by above, rows will be in sorted order
		self.columns = columns
		self.values = values

	def __mul__(self, other: float):
		if isinstance(other, (int, float)):
			# special-case scalar multiplication
			return self.copy(values=self.values * other)
		else:
			return self.product(other)
	def __xor__(self, other):
		return self.wedge(other)
	def __or__(self, other):
		return self.inner(other)
	def __and__(self, other):
		return self.regressive(other)

	def __add__(self, other: "MatrixContainer"):
		assert self.shape == other.shape
		subspace = self.subspace.union(other.subspace)
		return MatrixContainer(
			self.shape,
			np.concatenate([self.rows, other.rows]),
			np.concatenate([self.columns, other.columns]),
			self.values.select_subspace(subspace).concatenate(other.values.select_subspace(subspace), axis=0)
		)

	def __sub__(self, other):
		return self + (-other)
	def __neg__(self):
		return self.copy(values=-self.values)
	def __pos__(self):
		return self

	def __invert__(self):
		return self.reverse()

	def reverse(self):
		return MatrixContainer(
			self.shape[::-1],
			self.columns,
			self.rows,
			self.values.reverse()
		)

	def dual(self):
		return self.copy(values=self.values.dual())


class MatrixOperator:
	"""matrix container bound to an unbound operator"""
	def __init__(self, container, operator: Callable):
		self.container = container
		self.operator = operator

	def matscalar(self, x):
		op = self.operator(self.container.subspace, x.subspace)
		temp = op(self.container.values, x)
		return self.container.copy(values=temp)

	def matvec(self, x):
		assert x.shape[0] == self.container.shape[1]
		op = self.operator(self.container.subspace, x.subspace)
		temp = op(self.container.values, x[self.container.columns])
		result = np.add.reduceat(temp.values, self.container.indices)
		return temp.copy(values=result)

	@staticmethod
	def generate_pair_idx(a, b):
		from collections import defaultdict
		l2_pos = defaultdict(list)
		for (p, k) in enumerate(b): # NOTE: given that b is sorted in our application, could just find its slices rather than defaultdict
			l2_pos[k].append(p)
		return np.array([(p1, p2) for (p1, k) in enumerate(a) for p2 in l2_pos[k]]).T

	def matmat(self, B: MatrixContainer) -> MatrixContainer:
		A = self.container
		assert A.shape[1] == B.shape[0]
		a_idx, b_idx = self.generate_pair_idx(A.columns, B.rows)
		op = self.operator(A.subspace, B.subspace)
		return MatrixContainer(
			(A.shape[0], B.shape[1]),
			A.rows[a_idx], B.columns[b_idx],
			op(A.values[a_idx], B.values[b_idx])
		)

	def __call__(self, x):
		if isinstance(x, SubSpace):
			return BoundMatrixOperator(self.container, self.operator(self.container.subspace, x))
		elif isinstance(x, AbstractMultiVector):
			if x.shape == ():
				return self.matscalar(x)
			return self.matvec(x)
		elif isinstance(x, MatrixContainer):
			return self.matmat(x)
		raise NotImplementedError


class BoundMatrixOperator:
	"""matrix with a specific operator and bound operand subspace
	given knowledge of what product and subspace we are multiplying with,
	we can 'write our GA entries in matrix form',
	which should speed up repeated matrix-vector products
	"""
	def __init__(self, container, operator):
		self.container = container
		self.operator = operator
		# pre-bind the matrix entries; presumably faster execution of repeated products
		self.bound = self.operator.partial({0: container.values})

	def matvec(self, x):
		assert x.shape[0] == self.container.shape[1]
		temp = self.bound(x[self.container.columns])
		result = np.add.reduceat(temp.values, self.container.indices)
		return temp.copy(values=result)

	def __call__(self, other):
		if isinstance(other, AbstractMultiVector):
			return self.matvec(other)
		raise NotImplementedError

	@property
	def row_subspace(self):
		return self.operator.operator.axes[2]
	@property
	def column_subspace(self):
		return self.operator.operator.axes[1]
	@property
	def block_shape(self):
		return len(self.row_subspace), len(self.column_subspace)

	def as_dense(self):
		dense = np.zeros(shape=self.container.shape + self.block_shape)
		dense[self.container.rows, self.container.columns] = self.bound.kernel
		from einops import rearrange
		return rearrange(dense, 'r c i j -> (r i) (c j)')

	def as_operator(self) -> "LinearOperator":
		"""Wrap operator as a flattened scipy linear operator type"""
		from scipy.sparse.linalg import LinearOperator
		r, c = self.container.shape
		i, j = self.block_shape
		def matvec(x):
			*shape, n = x.shape
			x = self.container.context.multivector(values=x.reshape(shape + [c, j]), subspace=self.column_subspace)
			y = self(x)
			return y.values.reshape(shape + [r * i])
		return LinearOperator(shape=(r*i, c*j), matvec=matvec)



def test_dirac():
	"""Test formation of quat matrix from dirac operator"""
	print()
	from numga.algebra.algebra import Algebra
	from numga.backend.numpy.context import NumpyContext as Context
	from numga.multivector.test.util import random_subspace

	context = Context(Algebra.from_pqr(3, 0, 0))

	Q = context.subspace.even_grade()
	B = context.subspace.bivector()
	V = context.subspace.vector()
	I = context.multivector.pseudoscalar([1])

	builder = Builder((2, 3), V)
	v = random_subspace(context, V, (10,))
	builder.add(0, 1, v[0])
	builder.add(0, 1, v[1])
	builder.add(0, 2, v[2])
	# builder.add(2, 2, v[3])
	builder.add(1, 1, v[4])
	builder.add(0, 0, v[5])
	# builder.add(2, 0, v[6])
	D = builder.finalize()

	x = random_subspace(context, V, (3,))

	d = MatrixContainer.diag(x)
	print((d*I).values)
	print(d.dual().values)

	SPD = ~D * D
	# SPD = D * ~D
	r = SPD.product(Q).as_dense()
	print(np.around(r, 2))
	assert SPD.subspace.equals.even_grade()


def test_other():
	"""Test some other random things"""
	from numga.algebra.algebra import Algebra
	from numga.backend.numpy.context import NumpyContext as Context
	from numga.multivector.test.util import random_subspace

	context = Context(Algebra.from_pqr(2, 1, 0))

	Q = context.subspace.even_grade()
	B = context.subspace.bivector()
	builder = Builder((3, 3), B)
	v = random_subspace(context, B, (10,))
	builder.add(0, 1, v[0])
	builder.add(0, 1, v[1])
	builder.add(0, 2, v[2])
	builder.add(2, 2, v[3])
	builder.add(1, 1, v[4])
	builder.add(0, 0, v[5])
	builder.add(2, 0, v[6])
	crs = builder.finalize()

	x = random_subspace(context, B, (3,))
	r = crs.commutator(x)
	print(r)

	bm = crs.commutator(B)
	r = bm.as_operator()(x.values.flatten()).reshape(x.values.shape)
	print(r)

	print(np.around(bm.as_dense(), 2))


def test_matmul():
	"""test sparse matric multiplication for consistency with scipy."""
	from numga.algebra.algebra import Algebra
	from numga.backend.numpy.context import NumpyContext as Context
	context = Context(Algebra.from_pqr(1, 0, 0))
	shape = 100, 80
	e = 1000
	r, c = [np.random.randint(0, s, e) for s in shape]
	v = np.random.normal(size=e)

	As = scipy.sparse.coo_matrix((v,(r,c)), shape)
	A = MatrixContainer.from_scipy(context, As)

	xs = np.random.normal(size=shape[1])
	ys = As * xs
	x = context.multivector.scalar(xs[:,None])
	y = A * x
	assert np.allclose(ys, y.values.flatten())

	vs = (As.T * As).data
	v = (~A*A).values.values
	assert np.allclose(np.sort(vs.flatten()), np.sort(v.flatten()))