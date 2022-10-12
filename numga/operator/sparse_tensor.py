from typing import List, Tuple

import numpy as np
import numpy_indexed as npi

from numga.subspace.subspace import SubSpace


class SparseTensor:
	def __init__(self, axes, data):
		self.axes = tuple(axes)    # these are just coordinate arrays; for instance can be algebraic elements
		self.data = data

	def from_cayley(inputs: List[SubSpace], cayley, sign):
		# FIXME: better interface to pass in input as list of blades? meh
		# FIXME: drop zero terms?
		idx = np.indices(cayley.shape, dtype=cayley.dtype)
		blades = [inp.blades[idx] for inp, idx in zip(inputs, idx)] + [cayley]
		return SparseTensor(
			tuple(b.flatten() for b in blades),
			sign.flatten(),
		)

	def to_dense(self, axes: Tuple[SubSpace]):
		# to dense, given a set of coordinate axes as subspaces
		shape = [len(a) for a in axes]
		dense = np.zeros(shape, dtype=np.int8)
		# translate axes to indices
		idx = (A.to_relative_indices(a) for a, A in zip(self.axes, axes))
		dense[tuple(idx)] = self.data
		return dense

	def drop_zero(self):
		mask = self.data != 0
		return SparseTensor(
			axes=[a[mask] for a in self.axes],
			data=self.data[mask]
		)

	def to_dense_(self):
		# to dense, given a set of coordinate axes as subspaces
		shape = [np.max(a)+1 for a in self.axes]
		dense = np.zeros(shape, dtype=np.int8)
		dense[self.axes] = self.data
		return dense

	def take(self, idx, axis):
		# only keep entries in axis present in idx
		axes = np.copy(self.axes)
		flag = npi.contains(idx, axes[:, axis])
		s = list(self.shape)
		s[axis] = len(idx)
		# translate axes[:, axis] to new zero-based range
		axes = axes[flag]
		# FIXME: might forego this remap, if downstream code 'gets it'
		npi.remap(axes[:, axis], idx, np.arange(len(idx)), inplace=True)
		return SparseTensor(axes, self.data[flag], tuple(s))

	def transpose(self, t):
		assert len(t) == len(self.axes)
		return SparseTensor(
			(self.axes[a] for a in t),
			self.data
		)

	def __add__(self, other):
		# assert self.shape == other.shape
		axes = [np.concatenate([sa, oa], axis=0) for sa, oa in zip(self.axes, other.axes)]
		data = np.concatenate([self.data, other.data], axis=0)
		# FIXME: this can be optional?
		axes, data = npi.group_by(axes).sum(data)
		# FIXME: filter zeros
		return SparseTensor(axes, data)

	def __mul__(self, other):
		# scalar multiply
		return SparseTensor(self.axes, self.data * other)

	def tensorproduct(self, other, sa, oa):
		# FIXME: this is inefficient; as per example of two simple vectors; only case about diag terms
		return self.product(other).contract((sa, len(other.axes)+oa))

	def product(self, other):
		"""non contracting tensor product"""
		n = len(other.data)
		a_axes = [np.repeat(a[:, None], n, axis=1).flatten() for a in self.axes]
		n = len(self.data)
		b_axes = [np.repeat(b[None, :], n, axis=0).flatten() for b in other.axes]
		return SparseTensor(
			a_axes + b_axes,
			np.outer(self.data, other.data).flatten()
		)

	def contract(self, axes):
		l, r = [self.axes[a] for a in axes]

		keep = l == r
		rem = [a[keep] for i, a in enumerate(self.axes) if i not in axes]
		if len(rem):
			rem, data = npi.group_by(tuple(rem)).sum(self.data[keep])
		else:
			# this special case appears to be required for scalar case
			rem = []
			data = self.data[keep].sum()
		return SparseTensor(rem, data)
