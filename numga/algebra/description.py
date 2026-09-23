"""Immutable descriptions of finite-dimensional diagonal-metric algebras."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Iterable, Sequence


_CHAR_TO_SIGN = {"+": 1, "-": -1, "0": 0}
_SIGN_TO_CHAR = {value: key for key, value in _CHAR_TO_SIGN.items()}


def _metric_signs(signature: str | Iterable[int]) -> tuple[int, ...]:
    if isinstance(signature, str):
        try:
            return tuple(_CHAR_TO_SIGN[character] for character in signature)
        except KeyError as error:
            raise ValueError(
                "a metric signature string may contain only '+', '-', and '0'"
            ) from error

    result: list[int] = []
    for value in signature:
        try:
            sign = index(value)
        except TypeError as error:
            raise TypeError("metric entries must be integers") from error
        if sign not in (-1, 0, 1):
            raise ValueError("metric entries must be -1, 0, or +1")
        result.append(sign)
    return tuple(result)


def _default_basis_names(dimension: int) -> tuple[str, ...]:
    """Return deterministic compact names: a..z, aa..az, ba..."""

    def name(number: int) -> str:
        characters: list[str] = []
        while True:
            number, remainder = divmod(number, 26)
            characters.append(chr(ord("a") + remainder))
            if number == 0:
                return "".join(reversed(characters))
            number -= 1

    return tuple(name(i) for i in range(dimension))


def parse_description(specification: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Parse compact interleaved notation such as ``"x+y+z+w0"``.

    The three metric characters delimit generator names. Structured construction
    via :class:`AlgebraDescription` remains available for names containing one
    of those characters.
    """

    if not isinstance(specification, str):
        raise TypeError("an algebra description must be a string")
    if not specification:
        return (), ()

    names: list[str] = []
    signs: list[int] = []
    start = 0
    for position, character in enumerate(specification):
        if character not in _CHAR_TO_SIGN:
            continue
        name = specification[start:position]
        if not name:
            raise ValueError(
                f"missing generator name before metric character at index {position}"
            )
        names.append(name)
        signs.append(_CHAR_TO_SIGN[character])
        start = position + 1

    if start != len(specification):
        raise ValueError("an algebra description must end in '+', '-', or '0'")

    return tuple(names), tuple(signs)


@dataclass(frozen=True, slots=True)
class AlgebraDescription:
    """The named orthogonal generators and diagonal metric of an algebra."""

    basis_names: tuple[str, ...]
    signature: tuple[int, ...]

    def __post_init__(self) -> None:
        names = tuple(self.basis_names)
        signs = _metric_signs(self.signature)
        object.__setattr__(self, "basis_names", names)
        object.__setattr__(self, "signature", signs)

        if len(names) != len(signs):
            raise ValueError("basis_names and signature must have the same length")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("basis names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("basis names must be unique")
        if "1" in names:
            raise ValueError("'1' is reserved as the scalar blade name")

    @classmethod
    def parse(cls, specification: str) -> "AlgebraDescription":
        return cls(*parse_description(specification))

    @classmethod
    def from_signature(
        cls,
        signature: str | Iterable[int],
        basis_names: Sequence[str] | None = None,
    ) -> "AlgebraDescription":
        signs = _metric_signs(signature)
        names = (
            _default_basis_names(len(signs))
            if basis_names is None
            else tuple(basis_names)
        )
        return cls(names, signs)

    @classmethod
    def from_pqr(
        cls,
        p: int,
        q: int,
        r: int,
        basis_names: Sequence[str] | None = None,
    ) -> "AlgebraDescription":
        counts: list[int] = []
        for value in (p, q, r):
            try:
                count = index(value)
            except TypeError as error:
                raise TypeError("p, q, and r must be integers") from error
            if count < 0:
                raise ValueError("p, q, and r must be non-negative")
            counts.append(count)
        positive, negative, null = counts
        return cls.from_signature(
            (1,) * positive + (-1,) * negative + (0,) * null,
            basis_names=basis_names,
        )

    @property
    def dimension(self) -> int:
        return len(self.signature)

    @property
    def pqr(self) -> tuple[int, int, int]:
        return (
            self.signature.count(1),
            self.signature.count(-1),
            self.signature.count(0),
        )

    @property
    def signature_string(self) -> str:
        return "".join(_SIGN_TO_CHAR[sign] for sign in self.signature)

    def to_compact_string(self) -> str:
        """Return compact notation when all names fit that notation."""

        if any(
            any(character in _CHAR_TO_SIGN for character in name)
            for name in self.basis_names
        ):
            raise ValueError(
                "basis names containing '+', '-', or '0' have no compact representation"
            )
        return "".join(
            name + _SIGN_TO_CHAR[sign]
            for name, sign in zip(self.basis_names, self.signature)
        )

    def __mul__(self, other: object) -> "AlgebraDescription":
        if not isinstance(other, AlgebraDescription):
            return NotImplemented
        overlap = set(self.basis_names).intersection(other.basis_names)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"product algebra has duplicate basis names: {names}")
        return AlgebraDescription(
            self.basis_names + other.basis_names,
            self.signature + other.signature,
        )
