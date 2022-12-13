"""This module writes a large set of operators to the repo, for a large number of common algebras

This is meant to pin the implementation, and avoid accidental unnoticed changes.

Moreover, it can be convenient to have these tables readily available
when comparing definitions and results with other code and papers
"""

import pytest

from numga.algebra.algebra import Algebra

pqrs = [(1, 1, 0), (2, 0, 1), (3, 0, 1), (3, 0, 0), (2, 0, 0), (1, 1, 1), (3, 1, 0)]

unary_operators = [
	'reverse',
	'involute',
	'conjugate',

	'left_dual',
	'right_dual',
	'left_complement',
	'right_complement',
	'left_complement_dual',
	'right_complement_dual',
	'left_hodge',
	'right_hodge',

	'squared',
	'symmetric_reverse_product',
	'degenerate',
	'nondegenerate',
]

binary_operators = [
	'geometric_product',
	'outer_product',
	'inner_product',
	'scalar_product',
	'regressive_product',
	'reverse_product',
	'left_contraction_product',
	'right_contraction_product',
	'left_interior_product',
	'right_interior_product',
	'anti_geometric_product',
	'anti_outer_product',
	'anti_inner_product',
	'anti_scalar_product',
	'anti_reverse_product',
	'anti_left_interior_product',
	'anti_right_interior_product',
	'commutator_product',
	'anti_commutator_product',
	'commutator_anti_product',
	'anti_commutator_anti_product',
]


def write_to_disc(name, operator):
	"""Write operator string to disc"""
	fname = f'{name}.txt'
	import os
	path = os.path.join('operators', operator.algebra.description.pqr_str)
	os.makedirs(path, exist_ok=True)
	with open(os.path.join(path, fname), 'w') as fh:
		fh.write(str(operator))


def read_from_disc(name, operator) -> str:
	"""Read operator string from disc.
	If it exists and deviates, thats a problem.
	"""
	fname = f'{name}.txt'
	import os
	path = os.path.join('operators', operator.algebra.description.pqr_str)
	os.makedirs(path, exist_ok=True)
	if os.path.exists(os.path.join(path, fname)):
		with open(os.path.join(path, fname), 'r') as fh:
			assert str(operator) == fh.read()


@pytest.mark.parametrize('pqr', pqrs)
@pytest.mark.parametrize('name', unary_operators)
def test_print_unary_operator(pqr, name):
	algebra = Algebra.from_pqr(*pqr)
	F = algebra.subspace.full()
	op = getattr(algebra.operator, name)(F)
	read_from_disc(name, op)
	write_to_disc(name, op)


@pytest.mark.parametrize('pqr', pqrs)
@pytest.mark.parametrize('name', binary_operators)
def test_print_binary_operator(pqr, name):
	algebra = Algebra.from_pqr(*pqr)
	F = algebra.subspace.full()
	op = getattr(algebra.operator, name)(F, F)
	read_from_disc(name, op)
	write_to_disc(name, op)
