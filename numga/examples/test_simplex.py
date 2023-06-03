"""Some code for calculating the inertia maps of simplices in arbitrary dimensions"""

import numpy as np

from numga.backend.numpy.context import NumpyContext as Context


def simplex_inertia(B, C):
	"""inertia map given [N x n] barycentric weights and n corner points"""
	L = (C * B).sum(axis=1)
	s = np.sqrt(len(B))
	return (L / s).inertia_map().sum(axis=0)


def simplex_inertia_lumped(C):
	"""Inertia map of a simplex described by N corner points described as antivectors as norm 1"""
	N = C.shape[0]          # number of corner points in the simplex
	f = np.sqrt(1/(1+N))    # tested for n=2,3,4
	D = np.eye(N) - 1/N     # barycentric direction vectors from centroid to corners
	B = 1/N + D * f         # centroid offset by f in direction of corners

	return simplex_inertia(B, C)


def simplex_inertia_brute(C):
	"""generate cube of sampling points; and pick sum(C) < 1"""
	# construct barycentric sampling coordinates for a simplex
	n = 300
	N = C.shape[0]
	s = np.linspace(0, 1, n + 1)
	s = (s[1:] + s[:-1]) / 2
	B = np.array(np.meshgrid(*[s]*(N-1),indexing='ij'))
	B = B.T.reshape(-1, (N-1))
	B = B[B.sum(axis=1) < 1]
	B = np.concatenate([B, (1-B.sum(axis=1, keepdims=True))], axis=1)

	return simplex_inertia(B, C)


def test_simplex_line():
	ctx = Context('x+y+w0')
	M = ctx.multivector
	C = (M.x * [1, -1] + M.w).dual()
	print('lumped')
	print(simplex_inertia_lumped(C).operator.axes)
	print(simplex_inertia_lumped(C).kernel)
	print('brute')
	print(simplex_inertia_brute(C).kernel)


def test_simplex_triangle():
	ctx = Context('x+y+w0')
	M = ctx.multivector

	C = M.x * [1, -1, 0] + M.y * [0.5, 0.2, -1]

	C = (C + M.w).dual()
	print('lumped')
	print(simplex_inertia_lumped(C).kernel)
	print('brute')
	print(simplex_inertia_brute(C).kernel)


def test_simplex_tet():
	ctx = Context('x+y+z+w0')
	M = ctx.multivector

	C = M.x * [1, -1, 0, 2] + M.y * [0.5, 0.2, -1, -1] + M.z * [-0.5, 0.2, 0, -1]

	C = (C + M.w).dual()
	print('lumped')
	print(simplex_inertia_lumped(C).kernel)
	print('brute')
	print(simplex_inertia_brute(C).kernel)
