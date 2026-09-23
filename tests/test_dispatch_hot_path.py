"""A warmed extension invocation only looks up its type key and calls code."""

from types import SimpleNamespace

import pytest

from numga import Algebra, ExtensionMethod, Extensor, GATypeDispatch, GATypePattern, NumpyContext
import numga.gatype.dispatch as dispatch_module


class CountingCache(dict):
    lookups = 0

    def __getitem__(self, key):
        self.lookups += 1
        return super().__getitem__(key)


def fail_if_called(*_args, **_kwargs):
    raise AssertionError("a warmed call repeated cold dispatch work")


@pytest.mark.parametrize("operand_count", (1, 2))
@pytest.mark.parametrize("predicate_registration", (False, True))
def test_warmed_extensor_method_does_one_lookup_without_type_analysis(
    monkeypatch, operand_count, predicate_registration,
):
    algebra = Algebra("x+y+")
    mv = NumpyContext(algebra).multivector
    method = ExtensionMethod("hot_path")
    monkeypatch.setattr(Extensor, "hot_path", method, raising=False)
    inputs = tuple(mv.vector([index, 1]) for index in range(operand_count))

    def implementation(*args, **kwargs):
        return args, kwargs

    if predicate_registration:
        predicate = (
            (lambda first: first <= algebra.subspace.vector())
            if operand_count == 1
            else (lambda first, second: first <= second)
        )
        method.register(predicate)(implementation)
    else:
        method.register(*(GATypePattern(arity=0) for _ in inputs))(implementation)

    assert inputs[0].hot_path(*inputs[1:], "warm", scale=2) == (
        (*inputs, "warm"), {"scale": 2},
    )
    dispatch = method._dispatch
    cache = CountingCache(dispatch._cache)
    monkeypatch.setattr(dispatch, "_cache", cache)
    monkeypatch.setattr(GATypeDispatch, "_resolve_uncached", fail_if_called)
    monkeypatch.setattr(GATypeDispatch, "_validate_actual_signature", fail_if_called)
    monkeypatch.setattr(dispatch_module, "_operand_gatype", fail_if_called)
    monkeypatch.setattr(dispatch_module, "_signature_matches", fail_if_called)
    monkeypatch.setattr(dispatch_module, "_signature_strictly_refines", fail_if_called)
    monkeypatch.setattr(
        dispatch, "_predicates",
        [SimpleNamespace(predicate=fail_if_called, implementation=implementation)],
    )

    # Different values share the immutable GAType key; ordinary arguments and
    # keyword options reach the selected implementation unchanged.
    next_inputs = tuple(mv.vector([index + 10, 2]) for index in range(operand_count))
    assert next_inputs[0].hot_path(*next_inputs[1:], "hot", scale=3) == (
        (*next_inputs, "hot"), {"scale": 3},
    )
    assert cache.lookups == 1
    assert dispatch.resolve(*(value.gatype for value in next_inputs)) is implementation
    assert cache.lookups == 2


def test_later_registration_invalidates_a_warmed_bound_method(monkeypatch):
    algebra = Algebra("x+y+")
    value = NumpyContext(algebra).multivector.vector([1, 2])
    method = ExtensionMethod("changing_path")
    monkeypatch.setattr(Extensor, "changing_path", method, raising=False)

    @Extensor.changing_path.register(GATypePattern(arity=0))
    def generic(_value):
        return "generic"

    bound = value.changing_path
    assert bound() == "generic"
    assert method._dispatch.resolution_cache_size == 1

    @Extensor.changing_path.register(lambda gatype: gatype <= algebra.subspace.vector())
    def specialized(_value):
        return "specialized"

    assert method._dispatch.resolution_cache_size == 0
    assert bound() == "specialized"
    assert value.changing_path() == "specialized"


def test_implementation_keyerror_never_reenters_dispatch(monkeypatch):
    algebra = Algebra("x+y+")
    value = NumpyContext(algebra).multivector.vector([1, 2])
    method = ExtensionMethod("raising_path")
    monkeypatch.setattr(Extensor, "raising_path", method, raising=False)
    calls = []

    @Extensor.raising_path.register(GATypePattern(arity=0))
    def implementation(_value):
        calls.append("called")
        raise KeyError("from implementation")

    with pytest.raises(KeyError, match="from implementation"):
        value.raising_path()
    monkeypatch.setattr(GATypeDispatch, "_resolve_uncached", fail_if_called)
    with pytest.raises(KeyError, match="from implementation"):
        value.raising_path()
    assert calls == ["called", "called"]


def test_invalid_calls_still_receive_diagnostics_after_warming(monkeypatch):
    algebra = Algebra("x+y+")
    other = Algebra("x+y+")
    value = NumpyContext(algebra).multivector.vector([1, 2])
    foreign = NumpyContext(other).multivector.vector([1, 2])
    method = ExtensionMethod("checked_path")
    monkeypatch.setattr(Extensor, "checked_path", method, raising=False)

    @Extensor.checked_path.register(GATypePattern(arity=0), GATypePattern(arity=0))
    def implementation(left, right):
        return left, right

    assert value.checked_path(value) == (value, value)
    with pytest.raises(TypeError, match="at least 2 positional operands"):
        value.checked_path()
    with pytest.raises(TypeError, match="at least 2 positional operands"):
        Extensor.checked_path()
    for invalid in (object(), SimpleNamespace(gatype="invalid"), SimpleNamespace(gatype=[])):
        with pytest.raises(TypeError, match="expose one complete GAType"):
            value.checked_path(invalid)
    with pytest.raises(ValueError, match="belong to one algebra"):
        value.checked_path(foreign)
    with pytest.raises(TypeError, match="actual requires 2 GATypes"):
        method._dispatch.resolve(value.gatype)
    with pytest.raises(TypeError, match="every .* actual entry must be a GAType"):
        method._dispatch.resolve(value.gatype, [])


def test_unregistered_descriptor_keeps_its_diagnostic(monkeypatch):
    algebra = Algebra("x+y+")
    value = NumpyContext(algebra).multivector.vector([1, 2])
    monkeypatch.setattr(Extensor, "empty_path", ExtensionMethod("empty_path"), raising=False)

    with pytest.raises(LookupError, match="'empty_path' has no registered implementations"):
        value.empty_path()
