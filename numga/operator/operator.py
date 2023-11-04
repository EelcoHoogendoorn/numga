
from typing import Tuple
import numpy as np

from numga.util import cached_property, match

from numga.subspace.subspace import SubSpace
from numga.operator.sparse_tensor import SparseTensor


class Operator:
	"""Multi-Linear symbolic operator"""
	def __init__(self, kernel: np.ndarray, axes: Tuple[SubSpace]):
		# FIXME: if we seek to push dimension limit, need to work with sparse coo tensor here exclusively
		#  in sparse tensor, diags can be spotted efficiently as axes[a]==axes[b]
		#  can use that to optimize compiled kernel

		if isinstance(kernel, SparseTensor):
			kernel = kernel.to_dense(axes)
		self.kernel: np.ndarray = kernel
		self.axes: Tuple[SubSpace] = tuple(axes)

	def copy(self, kernel):
		return type(self)(kernel, self.axes)

	# FIXME: forward all attrs to output subspace; so operator can act as standin of subspace
	#  fails to forward len, actually. so meh.
	# FIXME: cleaner to make operator a subclass of subspace? also feels funny tho?
	#  dunno; maybe it isnt? give them shared parent class?
	#  or just make a as_subspace free function helper, that does getatrr or return self?
	@property
	def subspace(self):
		"""The subspace of an operator; by which we mean its output subspace"""
		return self.output

	@classmethod
	def build(cls, kernel: np.ndarray, subspace_or_operator: Tuple[SubSpace]):
		"""Constructor that accepts either input subspaces or operators as inputs"""
		axes = [a.subspace for a in subspace_or_operator]
		operators = {
			i: a
			for i, a in enumerate(subspace_or_operator)
			if isinstance(a, Operator)
		}
		return cls(kernel, axes).bind(operators)
	def _build(self, subspace_or_operator: Tuple[SubSpace]):
		operators = {
			i: a
			for i, a in enumerate(subspace_or_operator)
			if isinstance(a, Operator)
		}
		return self.bind(operators)


	@property
	def algebra(self):
		return self.output.algebra

	def swapaxes(self, a, b) -> "Operator":
		axes = list(self.axes)
		axes[a], axes[b] = axes[b], axes[a]
		return Operator(
			np.swapaxes(self.kernel, a, b),
			tuple(axes)
		)

	def is_diagonal(self, axes: Tuple[int, int]) -> bool:
		"""Test if a particular axes pair represents a diagonal"""
		sym = self.symmetry(axes, -1)
		return np.all(sym.kernel == 0)

	@cached_property
	def arity(self) -> int:
		return len(self.inputs)
	@cached_property
	def output(self) -> SubSpace:
		return self.axes[-1]
	@cached_property
	def inputs(self) -> Tuple[SubSpace]:
		return self.axes[:-1]

	def add(self, other: "Operator") -> "Operator":
		return Operator(self.kernel + other.kernel, match((self.axes, other.axes)))
	def add_broadcast(self, other: "Operator") -> "Operator":
		axes = [sa + oa for sa, oa in zip(self.axes, other.axes)]
		return self.deslice(axes) + other.deslice(axes)

	def mul(self, s) -> "Operator":
		"""Scalar multiplication"""
		return Operator(self.kernel * s, self.axes)
	def div(self, s) -> "Operator":
		"""Scalar division"""
		return Operator(self.kernel // s, self.axes)

	def __add__(self, other):
		return self.add(other)
	def __sub__(self, other):
		return self.add(-other)
	def __neg__(self):
		return self.mul(-1)
	def __pos__(self):
		return self
	def __mul__(self, s):
		return self.mul(s)
	def __rmul__(self, s):
		return self.mul(s)

	def symmetry(self, axes: Tuple[int, int], sign: int) -> "Operator":
		"""Enforce commutativity symmetry of the given sign on the given axes [a, b]
		That is the output op will have the property
			op(a, b) = sign * op(b, a)

		We may choose to apply this to our operator,
		to encode knowledge about inputs being numerically identical,
		such as encountered in the sandwich product or norm calculations.
		"""
		# FIXME: is it appropriate to view everything as real numbers at this point?
		#  i think so; but i keep confusing myself
		assert self.axes[axes[0]] == self.axes[axes[1]]
		kernel2 = self.kernel + sign * np.swapaxes(self.kernel, *axes)
		kernel = kernel2 // 2
		if not np.all(kernel * 2 == kernel2):
			raise Exception('inappropriate integer division')
		return Operator(kernel, self.axes).squeeze()

	def fuse(self, other: "Operator", axis: int) -> "Operator":
		"""Fuse to operators; feed output of `self` into nth input argument of `other`"""
		assert self.output == other.axes[axis]
		kernel = np.tensordot(self.kernel, other.kernel, axes=(-1, axis))
		# pull forward input axes of other before `axis`,
		# so the input axes of self are inserted in the location of 'axis'
		kernel = np.moveaxis(kernel, range(self.arity, self.arity + axis), range(0, axis))
		axes = other.axes[:axis] + self.inputs + other.axes[axis+1:]
		return Operator(kernel, axes)

	def bind(self, *inputs: Tuple["Operator"]) -> "Operator":
		# FIXME: should abstract operator allow for binding of multivectors?
		"""Bind input operators to operator `self`

		If inputs is a Dict[int, Operator], it is used as a partial apply

		Example
		-------
		Given kernels with named axes self = 'abc', inputs = ('ijk', 'xyz')
		where the last axis denotes the output axis of each kernel,
		binding the arguments will replace the input arguments of self
		with the corresponding input axes of the bound arguments
		'abc' ('ijk', 'xyz') -> 'ijxyc' (k 'fed into' a and z 'fed into' j)
		"""
		# support partial application
		d = inputs[0] if isinstance(inputs[0], dict) else dict(enumerate(inputs))
		# traverse in reversed order; so that in case of insertions of arity > 1,
		# original decreasing indices into self remain valid
		fused = self
		for axis in reversed(sorted(d)):
			fused = d[axis].fuse(fused, axis=axis)
		return fused

	def __getitem__(self, inputs: Tuple[SubSpace]) -> "Operator":
		"""Slice out an operator acting in a subalgebra"""
		inputs = (inputs,) if not isinstance(inputs, tuple) else inputs
		return self.slice(tuple(
			i if isinstance(i, SubSpace) else self.output.algebra.subspace.full()
			for i in inputs)
		)

	def deslice(self, axes: Tuple[SubSpace]) -> "Operator":
		"""inverse of slice; blow up the kernel with zeros"""
		import numpy_indexed as npi
		kernel = np.pad(self.kernel, [(0, 1)]*self.kernel.ndim, 'constant', constant_values=(0, 0))
		for i, axis in enumerate(axes):
			idx = npi.indices(self.axes[i].blades, axis.blades, missing=self.kernel.shape[i])
			kernel = kernel.take(idx, axis=i)
		return Operator(kernel, tuple(axes))

	def slice(self, inputs: Tuple[SubSpace]) -> "Operator":
		kernel = self.kernel
		axes = list(self.axes)

		for i, axis in enumerate(inputs):
			idx = axes[i].to_relative_indices(axis.blades)
			axes[i] = axis
			kernel = kernel.take(idx, axis=i)

		return Operator(kernel, tuple(axes))

	def select_subspace(self, output: SubSpace) -> "Operator":
		"""Remap to new output space; dropping outputs or adding implicit zeros"""
		identity = np.identity(self.algebra.n_blades, self.algebra.blade_dtype)
		# FIXME: construction from dense full identity scales poorly!
		# NOTE: dont squeeze here; broadcasting to zero blades is part of intended functionality
		return Operator.build(
			identity.take(self.subspace.blades, axis=0).take(output.blades, axis=1),
			(self, output)
		)
	def select_grade(self, k: int) -> "Operator":
		"""Remap to new output space; dropping outputs or adding implicit zeros"""
		return self.select_subspace(self.algebra.subspace.k_vector(k))
	def restrict_grade(self, k: int) -> "Operator":
		"""Remap to new output space; dropping outputs"""
		return self.select_subspace(self * self.algebra.subspace.k_vector(k))

	def squeeze(self) -> "Operator":
		"""Drop known zero terms from output subspace"""
		indices = np.flatnonzero(self.kernel.any(axis=tuple(range(self.arity))))
		output: SubSpace = self.output.slice_subspace(indices)

		axes = list(self.axes)
		axes[-1] = output
		kernel = self.kernel.take(indices, axis=-1)
		return Operator(kernel, tuple(axes))

	def grade_selection(self, formula) -> "Operator":
		"""Apply a grade selection formula"""
		def reshape(a, i):
			"""Reshape axis such as to broadcast to whole kernel"""
			s = [1] * self.kernel.ndim
			s[i] = len(a)
			return a.reshape(s).astype(np.int8)
		grades = [reshape(a.grades(), i) for i, a in enumerate(self.axes)]
		mask = formula(*grades)
		return Operator(self.kernel * mask, self.axes).squeeze()

	def sum(self, axis: int) -> "Operator":
		"""Reduce one input axis by summation"""
		return Operator(
			self.kernel.sum(axis=axis),
			tuple(a for i, a in enumerate(self.axes) if i != axis)
		).squeeze()

	@cached_property
	def precompute_einsum_prod(self) -> str:
		"""Little einsum expression to loop-fuse individual product terms"""
		return ',...' * self.arity + '->...'


	def __str__(self) -> str:
		return {1: self.unary_operator_str, 2: self.binary_operator_str}[self.arity]()

	def tokenize(self):
		d = self.algebra.n_dimensions

		axes = [
			{a: np.binary_repr(a, d)[::-1] for a in axis.blades}
			for axis in self.axes
		]
		assert np.all(np.abs(self.kernel).sum(axis=-1) <= 1)

		idx = np.nonzero(self.kernel)
		signmap = {-1: '-', 0: ' ', +1: '+'}
		tokens = {}
		for foo in zip(*idx):
			*bar , baz = tuple(a[oa.blades[f]] for f, a, oa in zip(foo, axes, self.axes))
			s = self.kernel[foo]
			tokens[foo[:-1]] = signmap[s]+baz
		return axes, tokens

	def unary_operator_str(self) -> str:
		"""String table representation of unary operator"""
		assert self.arity == 1

		d = self.algebra.n_dimensions

		axes, tokens = self.tokenize()

		table = np.zeros([5] + [s * 2 + 3 for s in self.kernel.shape[:-1]], object)

		table[1::2, 1::2] = ' ' * (d + 1)
		for k, v in tokens.items():
			table[3, k[0] * 2 + 3] = v

		# signature
		signmap = {-1: '-', 0: '0', +1: '+'}
		table[1, 1] = ' ' + ''.join(signmap[s] for s in self.algebra.description.signature)
		# crosses
		table[::2, ::2] = '\u256c'
		# light
		table[4:-2:2, 4:-2:2] = '\u253c'
		# hbars
		table[0::2, 1::2] = '\u2550' * (d + 1)
		# hbars
		table[4:-2:2, 3::2] = '\u2500' * (d + 1)
		# vbars
		table[1::2, 0::2] = '\u2551'
		# light vbars
		table[3::2, 4:-2:2] = '\u2502'
		# first arg / rows
		table[1, 3::2] = [' ' + v for v in axes[0].values()]

		return '\n'.join(''.join(row) for row in table)

	def binary_operator_str(self) -> str:
		"""String table representation of binary operator"""
		assert self.arity == 2
		d = self.algebra.n_dimensions

		axes, tokens = self.tokenize()

		table = np.zeros([s * 2 + 3 for s in self.kernel.shape[:-1]], object)
		table[1::2, 1::2] = ' ' * (d + 1)
		for k, v in tokens.items():
			table[k[0] * 2 + 3, k[1] * 2 + 3] = v
		# crosses
		# table[2:-2:2, 2:-2:2] = '\u256c'
		table[1, 1] = ' ' + self.algebra.description.signature_str
		table[::2, ::2] = '\u256c'
		# light
		table[4:-2:2, 4:-2:2] = '\u253c'
		# hbars
		table[0::2, 1::2] = '\u2550' * (d + 1)
		# hbars
		table[4:-2:2, 3::2] = '\u2500' * (d + 1)
		# vbars
		table[1::2, 0::2] = '\u2551'
		# light vbars
		table[3::2, 4:-2:2] = '\u2502'
		# first arg / rows
		table[3::2, 1] = [' '+v for v in axes[0].values()]
		# first arg / rows
		table[1, 3::2] = [' '+v for v in axes[1].values()]

		return '\n'.join(''.join(row) for row in table)
