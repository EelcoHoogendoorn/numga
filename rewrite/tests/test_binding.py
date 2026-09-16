from numga.algebra import Algebra
from numga.backend import NumpyContext
from numga.binding import BindingPlan


def test_atomic_bind_preserves_repeated_exact_operand_identity(monkeypatch):
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    factory = algebra.operator
    context = NumpyContext(algebra)
    even = spaces.even()
    target = context.lower(factory.geometric_product(even, even))
    one = factory.build((even,), [1, 0])
    observed = []
    execute_bind = NumpyContext.execute_bind

    def capture_plan(self, receiver, operands, plan):
        observed.append((operands, plan))
        return execute_bind(self, receiver, operands, plan)

    monkeypatch.setattr(NumpyContext, "execute_bind", capture_plan)

    target.bind({0: one, 1: one})

    operands, plan = observed.pop()
    assert operands[0] is operands[1]
    assert plan.equality_groups == ((0, 1),)


def test_repeated_positive_arity_operand_is_not_diagonal_evidence():
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    factory = algebra.operator
    even = spaces.even()
    product = factory.geometric_product(even, even)
    identity = factory.identity(even)

    plan = BindingPlan.build(product.gatype, {0: identity, 1: identity})

    assert plan.result_subspaces == (even, even, even)
    assert plan.equality_groups == ()


def test_sandwich_binds_one_nullary_sandwicher_into_both_slots_atomically(
    monkeypatch,
):
    algebra = Algebra("x+y+")
    spaces = algebra.subspace
    context = NumpyContext(algebra)
    sandwicher = context.extensor(spaces.even(), [1, 0])
    observed = []
    execute_bind = NumpyContext.execute_bind

    def capture_plan(self, receiver, operands, plan):
        observed.append((operands, plan))
        return execute_bind(self, receiver, operands, plan)

    monkeypatch.setattr(NumpyContext, "execute_bind", capture_plan)

    sandwicher.sandwich(spaces.vector())

    assert len(observed) == 1
    operands, plan = observed[0]
    assert operands[0] is operands[2]
    assert plan.equality_groups == ((0, 2),)
