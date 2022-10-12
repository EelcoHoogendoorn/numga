from numga.backend.python.context import PythonContext


def test_sparse_operator():
	print()
	ga = PythonContext('x+y+z+')
	Q, V = ga.subspace.even_grade(), ga.subspace.vector()
	q, v = ga.multivector(Q), ga.multivector(V)
	print(q)
	print(v)

	output = q.sandwich(v)
	print(output)
