"""numga in WGSL: a shader dialect whose expressions are numga's.

A shader is WGSL in which values of an algebra's types are written as they are in numga: `a & b`,
`a ^ b`, `a | b`, `a * b`, `rotor >> point`, `gauge(direction)`, `x.dual()`, `x.commutator(b)`,
`radius.square_root()`, a basis blade `mv.t`, a type as its identity map, `Antivector - time * (mv.t &
Antivector)`. Operators bind as they do in Python. Every nullary or unary type lowers to a struct of its
coefficients, and every operation on such values to a function whose body is numga's own: numga
evaluates the operation once over coefficients that are polynomials in the operands' coefficients, and
the polynomials are the function. Values known when the shader compiles, blades, types and module
constants, fold into those functions as numbers. `f32` is the algebra's scalar. A struct whose members
are values of the algebra, numbers or vectors of numbers adds, subtracts and scales member by member, as
a state of an integrator does, or the light a pixel's samples average.

A value of the algebra passes from the vertex stage to the fragment stage as a member of the struct between them,
`@location(2) @interpolate(perspective, sample) normal: Plane`: its coefficients cross as vectors of numbers, four to
a location from the one it names, each interpolated as the others are, so the fragment stage reads the value blended
linearly across the triangle.

Constants come from Python by name, as module-scope `const` declarations: an integer as a `u32`, a number as an
`f32`, an array of two to four numbers, such as a colour, as a vector of `f32`, and a value of the algebra folded
into the operations that use it, as `const origin = mv.w.dual();` is. A batch of two to four values of the algebra,
such as a frame's planes, `stack((mv.x, mv.y, mv.z))`, crosses between numbers and the algebra: an operation takes
it value by value, against each component of a vector of as many numbers, and its values gather into such a vector
where they are numbers, `axes & point`, or sum into one value, `(axes.dual() * place).sum()`.

Module-scope `var<uniform>` values gather into one buffer, and storage arrays, textures and samplers are
bound in order of declaration: a shader declares what it reads and writes, not where.
"""

from __future__ import annotations

import operator
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np

from numga.backend.context import Context
from numga.backend.numpy import NumpyContext
from numga.extensor import Extensor
from numga.gatype import GAType


# --- symbolic coefficients -------------------------------------------------------------
class Polynomial:
    """A polynomial in named shader values, the coefficient numga computes with to write a function."""

    __slots__ = ("terms",)

    def __init__(self, terms: Mapping[tuple[str, ...], float]) -> None:
        self.terms = {monomial: coefficient for monomial, coefficient in terms.items() if coefficient != 0}

    @staticmethod
    def variable(name: str) -> Polynomial:
        return Polynomial({(name,): 1.0})

    @staticmethod
    def lift(value: object) -> Polynomial:
        return value if isinstance(value, Polynomial) else Polynomial({(): float(value)})

    def __add__(self, other: object) -> Polynomial:
        terms = dict(self.terms)
        for monomial, coefficient in Polynomial.lift(other).terms.items():
            terms[monomial] = terms.get(monomial, 0.0) + coefficient
        return Polynomial(terms)

    __radd__ = __add__

    def __neg__(self) -> Polynomial:
        return Polynomial({monomial: -coefficient for monomial, coefficient in self.terms.items()})

    def __sub__(self, other: object) -> Polynomial:
        return self + (-Polynomial.lift(other))

    def __rsub__(self, other: object) -> Polynomial:
        return Polynomial.lift(other) - self

    def __mul__(self, other: object) -> Polynomial:
        terms = {}                                                              # dict[tuple[str, ...], float]
        for left, a in self.terms.items():
            for right, b in Polynomial.lift(other).terms.items():
                monomial = tuple(sorted(left + right))
                terms[monomial] = terms.get(monomial, 0.0) + a * b
        return Polynomial(terms)

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> Polynomial:
        return self * (1.0 / float(other))

    def __float__(self) -> float:
        if set(self.terms) - {()}:
            raise TypeError("a polynomial in shader values is not a number")
        return self.terms.get((), 0.0)


class SymbolicContext(NumpyContext):
    """NumPy storage whose coefficients are Python objects: numbers and polynomials."""

    def __init__(self, algebra: Any) -> None:
        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_dtype", np.dtype(object))
        Context.__init__(self, "dense")

    @property
    def key(self) -> tuple[object, ...]:
        return ("symbolic",)

    def prepare_kernel(self, value: object) -> np.ndarray:
        return np.array(value, dtype=object, copy=True)

    def prepare_scalar(self, scalar: object) -> object:
        return scalar


def literal(number: float) -> str:
    """A number as a WGSL float literal."""
    number = float(number)
    if not np.isfinite(number):
        raise ValueError(f"{number} has no WGSL literal")
    text = repr(number)
    return text if any(mark in text for mark in ".e") else text + ".0"


def polynomial_text(coefficient: object) -> str:
    """A coefficient as a WGSL expression in the function's parameters."""
    if not isinstance(coefficient, Polynomial):
        return literal(coefficient)
    text = ""
    for monomial, number in sorted(coefficient.terms.items(), key=lambda term: (len(term[0]), term[0])):
        factors = list(monomial) if abs(number) == 1 else [literal(abs(number))] + list(monomial)
        body = " * ".join(factors) or literal(abs(number))
        text += ((" - " if number < 0 else " + ") if text else ("-" if number < 0 else "")) + body
    return text or "0.0"


# --- tokens ----------------------------------------------------------------------------
_TOKEN = re.compile(r"""
    (?P<space>\s+)
  | (?P<comment>//[^\n]*|/\*.*?\*/)
  | (?P<number>0[xX][0-9a-fA-F]+[iu]?|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fh]?|\d+[eE][+-]?\d+[fh]?|\d+[fhiu]?)
  | (?P<name>[A-Za-z_]\w*)
  | (?P<symbol><<=|>>=|\*\*|->|&&|\|\||==|!=|<=|>=|<<|>>|\+=|-=|\*=|/=|%=|&=|\|=|\^=|\+\+|--|[-+*/%&|^!~<>=(){}\[\],;:.@])
""", re.VERBOSE | re.DOTALL)


@dataclass
class Token:
    kind: str
    text: str
    line: int


# Words WGSL reserves but does not use; a shader may name its values by them, as `null`, and they are
# renamed on the way in.
_RESERVED = set("""NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become
cast catch class co_await co_return co_yield coherent column_major common compile compile_fragment concept
const_cast consteval constexpr constinit crate debugger decltype delete demote demote_to_helper do
dynamic_cast enum explicit export extends extern external fallthrough filter final finally friend from fxgroup
get goto groupshared highp impl implements import inline instanceof interface layout lowp macro macro_rules
match mediump meta mod module move mut mutable namespace new nil noexcept noinline nointerpolation
non_coherent noncoherent noperspective null nullptr of operator package packoffset partition pass patch
pixelfragment precise precision premerge priv protected pub public readonly ref regardless register
reinterpret_cast require resource restrict self set shared sizeof smooth snorm static static_assert
static_cast std subroutine super target template this thread_local throw trait try type typedef typeid
typename typeof union unless unorm unsafe unsized use using varying virtual volatile wgsl where with
writeonly yield""".split())


def tokens(source: str) -> list[Token]:
    found, position, line = [], 0, 1
    while position < len(source):
        match = _TOKEN.match(source, position)
        if match is None:
            raise SyntaxError(f"line {line}: unexpected {source[position]!r}")
        if match.lastgroup not in ("space", "comment"):
            text = match.group() + ("_" if match.group() in _RESERVED else "")
            found.append(Token(match.lastgroup, text, line))
        line += match.group().count("\n")
        position = match.end()
    return found + [Token("end", "", line)]


# --- syntax ----------------------------------------------------------------------------
@dataclass
class Name:
    id: str
    line: int


@dataclass
class Literal:
    text: str
    line: int


@dataclass
class TypeName:
    text: str
    line: int


@dataclass
class Member:
    object: Any
    name: str
    line: int


@dataclass
class Index:
    object: Any
    index: Any
    line: int


@dataclass
class Call:
    callee: Any
    arguments: list
    line: int


@dataclass
class Unary:
    operator: str
    operand: Any
    line: int


@dataclass
class Binary:
    operator: str
    left: Any
    right: Any
    line: int


@dataclass
class Declaration:
    kind: str                    # let, var or const
    name: str
    type: str | None
    value: Any
    line: int
    space: str | None = None     # a module-scope var's address space and access
    attributes: list = field(default_factory=list)


@dataclass
class Assignment:
    target: Any
    operator: str
    value: Any
    line: int


@dataclass
class Step:
    target: Any
    operator: str
    line: int


@dataclass
class If:
    condition: Any
    then: list
    otherwise: Any               # a list of statements, an If, or None
    line: int


@dataclass
class For:
    initial: Any
    condition: Any
    update: Any
    body: list
    line: int


@dataclass
class While:
    condition: Any
    body: list
    line: int


@dataclass
class Loop:
    body: list
    continuing: list | None
    line: int


@dataclass
class Jump:
    keyword: str                 # break, continue, return, discard or `break if`
    value: Any
    line: int


@dataclass
class Evaluate:
    expression: Any
    line: int


@dataclass
class Struct:
    name: str
    members: list                # (attributes, name, type)
    line: int


@dataclass
class Function:
    attributes: list
    name: str
    parameters: list             # (attributes, name, type)
    result_attributes: list
    result: str | None
    body: list
    line: int


@dataclass
class Directive:
    text: str
    line: int


@dataclass
class Alias:
    name: str
    type: str
    line: int


# Binding strength of binary operators: the arithmetic and bitwise ones as in Python, so that numga
# expressions read as they do there; the logical and comparison ones as in WGSL.
_BINARY = {"||": 1, "&&": 2, "==": 3, "!=": 3, "<": 3, ">": 3, "<=": 3, ">=": 3, "|": 4, "^": 5, "&": 6,
           "<<": 7, ">>": 7, "+": 8, "-": 8, "*": 9, "/": 9, "%": 9}
_ASSIGNMENTS = {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
_TEMPLATED = {"vec2", "vec3", "vec4", "array", "bitcast", "ptr", "atomic"} | {
    f"mat{columns}x{rows}" for columns in (2, 3, 4) for rows in (2, 3, 4)}


class Parser:
    def __init__(self, source: str) -> None:
        self.tokens = tokens(source)
        self.position = 0

    def peek(self, offset: int = 0) -> Token:
        return self.tokens[min(self.position + offset, len(self.tokens) - 1)]

    def take(self) -> Token:
        token = self.peek()
        self.position += 1
        return token

    def accept(self, text: str) -> bool:
        if self.peek().text == text and self.peek().kind != "end":
            self.position += 1
            return True
        return False

    def expect(self, text: str) -> Token:
        token = self.peek()
        if token.text == ">" or text != ">" or token.text != ">>":
            if token.text != text:
                raise SyntaxError(f"line {token.line}: expected {text!r}, found {token.text!r}")
            return self.take()
        # A `>>` closing two template lists: take one of its halves.
        token.text = ">"
        return Token("symbol", ">", token.line)

    def name(self) -> str:
        token = self.take()
        if token.kind != "name":
            raise SyntaxError(f"line {token.line}: expected a name, found {token.text!r}")
        return token.text

    # --- module
    def module(self) -> list:
        items = []
        while self.peek().kind != "end":
            line = self.peek().line
            attributes = self.attributes()
            word = self.peek().text
            if word == "struct":
                items.append(self.struct())
            elif word == "fn":
                items.append(self.function(attributes))
            elif word in ("var", "const", "override", "let"):
                declaration = self.declaration()
                declaration.attributes = attributes
                self.expect(";")
                items.append(declaration)
            elif word == "alias":
                self.take()
                name = self.name()
                self.expect("=")
                items.append(Alias(name, self.type(), line))
                self.expect(";")
            elif word in ("enable", "requires", "diagnostic", "const_assert"):
                text = []
                while not self.accept(";"):
                    text.append(self.take().text)
                items.append(Directive(" ".join(text) + ";", line))
            else:
                raise SyntaxError(f"line {line}: unexpected {word!r} at module scope")
        return items

    def attributes(self) -> list:
        found = []
        while self.accept("@"):
            text = "@" + self.name()
            if self.peek().text == "(":
                text += self.balanced()
            found.append(text)
        return found

    def balanced(self) -> str:
        """A parenthesised run of tokens, verbatim."""
        depth, text = 0, ""
        while True:
            token = self.take()
            depth += {"(": 1, ")": -1}.get(token.text, 0)
            text += token.text + (" " if token.text == "," else "")
            if depth == 0:
                return text

    def struct(self) -> Struct:
        line = self.take().line
        name = self.name()
        self.expect("{")
        members = []
        while not self.accept("}"):
            attributes = self.attributes()
            member = self.name()
            self.expect(":")
            members.append((attributes, member, self.type()))
            self.accept(",")
        self.accept(";")
        return Struct(name, members, line)

    def function(self, attributes: list) -> Function:
        line = self.take().line
        name = self.name()
        self.expect("(")
        parameters = []
        while not self.accept(")"):
            parameter_attributes = self.attributes()
            parameter = self.name()
            self.expect(":")
            parameters.append((parameter_attributes, parameter, self.type()))
            self.accept(",")
        result_attributes, result = [], None
        if self.accept("->"):
            result_attributes = self.attributes()
            result = self.type()
        return Function(attributes, name, parameters, result_attributes, result, self.block(), line)

    def type(self) -> str:
        name = self.name()
        if self.peek().text not in ("<", "<<"):
            return normalized_type(name)
        self.expect("<")
        arguments = []
        while True:
            if self.peek().kind == "name":
                arguments.append(self.type())
            else:
                arguments.append(self.take().text)
            if not self.accept(","):
                break
            if self.peek().text in (">", ">>"):
                break
        self.expect(">")
        return normalized_type(f"{name}<{', '.join(arguments)}>")

    # --- statements
    def block(self) -> list:
        self.expect("{")
        statements = []
        while not self.accept("}"):
            statements.append(self.statement())
        return statements

    def declaration(self) -> Declaration:
        token = self.take()
        space = None
        if token.text == "var" and self.peek().text == "<":
            self.take()
            parts = [self.name()]
            while self.accept(","):
                parts.append(self.name())
            self.expect(">")
            space = ", ".join(parts)
        name = self.name()
        declared = self.type() if self.accept(":") else None
        value = self.expression() if self.accept("=") else None
        return Declaration(token.text, name, declared, value, token.line, space)

    def statement(self) -> Any:
        token = self.peek()
        word = token.text if token.kind == "name" else None
        if token.text == "{":
            return If(Literal("true", token.line), self.block(), None, token.line)
        if word in ("let", "var", "const"):
            declaration = self.declaration()
            self.expect(";")
            return declaration
        if word == "if":
            return self.conditional()
        if word == "for":
            self.take()
            self.expect("(")
            initial = None if self.peek().text == ";" else self.simple()
            self.expect(";")
            condition = None if self.peek().text == ";" else self.expression()
            self.expect(";")
            update = None if self.peek().text == ")" else self.simple()
            self.expect(")")
            return For(initial, condition, update, self.block(), token.line)
        if word == "while":
            self.take()
            return While(self.expression(), self.block(), token.line)
        if word == "loop":
            self.take()
            self.expect("{")
            body, continuing = [], None
            while not self.accept("}"):
                if self.accept("continuing"):
                    continuing = self.block()
                else:
                    body.append(self.statement())
            return Loop(body, continuing, token.line)
        if word in ("break", "continue", "return", "discard"):
            self.take()
            if word == "break" and self.accept("if"):
                word = "break if"
            value = None if self.peek().text == ";" else self.expression()
            self.expect(";")
            return Jump(word, value, token.line)
        statement = self.simple()
        self.expect(";")
        return statement

    def conditional(self) -> If:
        line = self.take().line
        condition = self.expression()
        then = self.block()
        otherwise = None
        if self.accept("else"):
            otherwise = self.conditional() if self.peek().text == "if" else self.block()
        return If(condition, then, otherwise, line)

    def simple(self) -> Any:
        """A declaration, assignment, increment or call, as a for loop's clauses hold."""
        token = self.peek()
        if token.text in ("let", "var", "const"):
            return self.declaration()
        if token.text == "_" and self.peek(1).text == "=":
            self.take(), self.take()
            return Assignment(Name("_", token.line), "=", self.expression(), token.line)
        target = self.expression()
        if self.peek().text in _ASSIGNMENTS:
            return Assignment(target, self.take().text, self.expression(), token.line)
        if self.peek().text in ("++", "--"):
            return Step(target, self.take().text, token.line)
        return Evaluate(target, token.line)

    # --- expressions
    def expression(self, level: int = 1) -> Any:
        left = self.unary()
        while True:
            token = self.peek()
            strength = _BINARY.get(token.text) if token.kind == "symbol" else None
            if strength is None or strength < level:
                return left
            self.take()
            left = Binary(token.text, left, self.expression(strength + 1), token.line)

    def unary(self) -> Any:
        token = self.peek()
        if token.kind == "symbol" and token.text in ("-", "!", "~", "&", "*"):
            self.take()
            return Unary(token.text, self.unary(), token.line)
        base = self.postfix()
        if self.peek().text == "**":
            line = self.take().line
            return Binary("**", base, self.unary(), line)
        return base

    def postfix(self) -> Any:
        node = self.primary()
        while True:
            token = self.peek()
            if self.accept("."):
                node = Member(node, self.name(), token.line)
            elif self.accept("["):
                node = Index(node, self.expression(), token.line)
                self.expect("]")
            elif self.accept("("):
                arguments = []
                while not self.accept(")"):
                    arguments.append(self.expression())
                    self.accept(",")
                node = Call(node, arguments, token.line)
            else:
                return node

    def primary(self) -> Any:
        token = self.peek()
        if token.kind == "number":
            self.take()
            return Literal(token.text, token.line)
        if token.kind == "name":
            if token.text in _TEMPLATED and self.peek(1).text == "<":
                return TypeName(self.type(), token.line)
            self.take()
            if token.text in ("true", "false"):
                return Literal(token.text, token.line)
            return Name(token.text, token.line)
        if self.accept("("):
            node = self.expression()
            self.expect(")")
            return node
        raise SyntaxError(f"line {token.line}: unexpected {token.text!r}")


def normalized_type(text: str) -> str:
    """WGSL's shorthand vector and matrix names spelled out: `vec3f` as `vec3<f32>`."""
    shorthand = re.fullmatch(r"(vec[234]|mat[234]x[234])([fhiu])", text)
    if shorthand:
        scalar = {"f": "f32", "h": "f16", "i": "i32", "u": "u32"}[shorthand.group(2)]
        return f"{shorthand.group(1)}<{scalar}>"
    return text


def template(text: str) -> tuple[str, list[str]]:
    """A type's name and its template arguments, split at the top level."""
    if "<" not in text:
        return text, []
    name, inner = text[:text.index("<")], text[text.index("<") + 1:-1]
    arguments, depth, current = [], 0, ""
    for character in inner:
        depth += {"<": 1, ">": -1}.get(character, 0)
        if character == "," and depth == 0:
            arguments.append(current.strip())
            current = ""
        else:
            current += character
    return name, arguments + [current.strip()]


# --- meaning ---------------------------------------------------------------------------
@dataclass
class Value:
    """An expression's WGSL text and type, and, for one known when the shader compiles, its value.

    The type is a GAType, the algebra's scalar standing for f32, or the text of any other WGSL type.
    A value known at compile time is an Extensor, or a number of type `abstract`; it has text only when
    it is a number."""
    text: str | None
    type: Any
    constant: Any = None


_MV = "mv"
_ABSTRACT = "abstract"
# A batch of values known when the shader compiles, or computed from them, held value by value.
_BATCH = "batch"
# Methods of the scalar that are not polynomials, by their WGSL builtin.
_SCALAR_FUNCTIONS = {"square_root": "sqrt", "exp": "exp", "log": "log", "cos": "cos", "sin": "sin",
                     "tan": "tan", "arccos": "acos", "arcsin": "asin", "arctan": "atan", "abs": "abs",
                     "sign": "sign", "floor": "floor", "cosh": "cosh", "sinh": "sinh", "tanh": "tanh"}
# WGSL builtins whose result has the type of their arguments, and those that reduce to a component.
_ELEMENTWISE = {"abs", "sqrt", "exp", "exp2", "log", "log2", "sin", "cos", "tan", "asin", "acos", "atan",
                "atan2", "sinh", "cosh", "tanh", "pow", "min", "max", "clamp", "mix", "smoothstep", "step",
                "floor", "ceil", "fract", "round", "trunc", "sign", "normalize", "select", "inverseSqrt",
                "fma", "saturate", "degrees", "radians",
                "fwidth", "dpdx", "dpdy", "fwidthFine", "dpdxFine", "dpdyFine", "fwidthCoarse", "dpdxCoarse", "dpdyCoarse"}
_REDUCING = {"length", "distance", "dot", "determinant"}
# WGSL builtins the dialect leaves out, as the algebra has the operation, and what to write instead.
_REJECTED = {"cross": "a join or a meet, `a & b` or `a ^ b`", "reflect": "a sandwich, `mirror >> x`"}
# WGSL builtins that measure three or four numbers by their sum of squares, where they are a value of the algebra held
# as numbers, and what to write on the value instead.
_MEASURING = {"normalize": "`x.normalized()`", "length": "`x.norm()`", "distance": "`(a - b).norm()`",
              "dot": "a pairing, `plane & point`, or the metric, `a.dot(b)`"}
_OPERATORS = {"+": operator.add, "-": operator.sub, "*": operator.mul,          # dict[str, Callable]
                                   "&": operator.and_, "|": operator.or_, "^": operator.xor,
                                   ">>": operator.rshift, "<<": operator.lshift}


@dataclass
class Binding:
    """A resource the runner binds: the gathered uniforms, a storage array, a texture or a sampler."""
    index: int
    name: str
    kind: str                    # uniforms, storage, texture or sampler
    type: Any
    access: str = "read"


@dataclass
class Program:
    """A compiled shader: its WGSL, what it binds, how its uniforms lie in their buffer, the types its
    structs stand for, and its compute entry points in order of declaration, with their workgroup sizes."""
    code: str
    bindings: list[Binding]
    uniforms: list[tuple[str, Any]]
    structs: dict[str, list[tuple[str, Any]]]
    gatypes: dict[str, GAType]
    entry_points: list[tuple[str, tuple[int, int, int]]]
    vertex: str | None
    fragment: str | None
    # The colour targets the fragment stage writes, one per location.
    targets: int


class Compiler:
    def __init__(self, algebra: Any, types: Mapping[str, GAType], constants: Mapping[str, Any]) -> None:
        self.algebra = algebra
        self.given = constants
        self.context = SymbolicContext(algebra)
        self.scalar = algebra.gatype.scalar()
        self.names = {}                                                         # dict[GAType, str]
        self.types = {"f32": self.scalar}                                       # dict[str, GAType]
        self.aliases = []                                                       # list[str]
        self.native_aliases = {}                                                # dict[str, str]
        for name, gatype in types.items():
            self.types[name] = gatype
            if gatype == self.scalar:
                continue
            if gatype in self.names:
                self.aliases.append(f"alias {name} = {self.names[gatype]};")
            else:
                self.names[gatype] = name
        self.helpers = {}                                                       # dict[tuple, tuple[str, str, Any]]
        self.structs = {}                                                       # dict[str, list[tuple[str, Any]]]
        self.functions = {}                                                     # dict[str, tuple[list, Any]]
        self.globals = {}                                                       # dict[str, Value]
        self.bindings = []                                                      # list[Binding]
        self.uniforms = []                                                      # list[tuple[str, Any]]
        self.entry_points = []                                                  # list[tuple[str, tuple[int, int, int]]]
        self.stages = {}                                                        # dict[str, str]
        self.targets = 0
        self.member_attributes = {}                                             # dict[str, list[list[str]]]
        # The values of the algebra each struct between shader stages carries, by member.
        self.carried = {}                                                       # dict[str, dict[str, GAType]]

    def error(self, line: int, message: str) -> SyntaxError:
        return SyntaxError(f"line {line}: {message}")

    # --- types
    def resolve(self, text: str | None) -> Any:
        if text is None:
            return None
        if text in self.types:
            return self.types[text]
        if text in self.native_aliases:
            return self.resolve(self.native_aliases[text])
        name, arguments = template(text)
        if arguments:
            # Aliases spelled out inside templates too, so `array<Sample>` names what it holds.
            return f"{name}<{', '.join(self.native_aliases.get(argument, argument) for argument in arguments)}>"
        return text

    def wgsl(self, kind: Any) -> str:
        """The WGSL name of a type, naming a struct for a type of the algebra the shader never named."""
        if isinstance(kind, GAType):
            if kind == self.scalar:
                return "f32"
            if kind not in self.names:
                self.names[kind] = f"GA{len(self.names)}"
            return self.names[kind]
        name, arguments = template(kind)
        if arguments:
            return f"{name}<{', '.join(self.wgsl(self.resolve(argument)) for argument in arguments)}>"
        return kind

    def is_algebra(self, kind: Any) -> bool:
        return isinstance(kind, GAType)

    def is_scalar(self, value: Value) -> bool:
        return value.type == self.scalar or value.type == _ABSTRACT

    def is_state(self, kind: Any) -> bool:
        """A struct that adds and scales member by member: its members values of the algebra, or numbers
        and vectors of numbers."""
        return kind in self.structs and kind not in self.carried and all(
            isinstance(member, GAType) or template(member)[0] in ("vec2", "vec3", "vec4")
            for _, member in self.structs[kind])

    def chunks(self, kind: GAType) -> list[str]:
        """The vectors of numbers a value of the algebra crosses between shader stages as: its coefficients, four to
        a vector, the last holding what remains."""
        size = int(np.prod([len(subspace) for subspace in kind.subspaces]))
        return [f"vec{count}<f32>" if count > 1 else "f32" for count in
                [min(4, size - start) for start in range(0, size, 4)]]

    def lifted(self, kind: GAType, parts: list[str]) -> Value:
        """A value of the algebra from the vectors of numbers it crossed between shader stages as."""
        key = ("lift", kind)
        if key not in self.helpers:
            name = f"ga_lift_{len(self.helpers)}"
            chunks = self.chunks(kind)
            parameters = ", ".join(f"a{index}: {chunk}" for index, chunk in enumerate(chunks))
            components = [f"a{index}" if chunk == "f32" else f"a{index}.{axis}" for index, chunk in enumerate(chunks)
                          for axis in "xyzw"[:int(chunk[3]) if chunk != "f32" else 1]]
            code = (f"fn {name}({parameters}) -> {self.wgsl(kind)} {{\n"
                    f"    return {self.wgsl(kind)}({', '.join(components)});\n}}\n")
            self.helpers[key] = (name, code, kind)
        name, _, _ = self.helpers[key]
        return Value(f"{name}({', '.join(parts)})", kind)

    def lowered(self, kind: GAType, value: str) -> list[str]:
        """WGSL for the vectors of numbers a value of the algebra crosses between shader stages as."""
        chunks, start, found = self.chunks(kind), 0, []
        for chunk in chunks:
            count = int(chunk[3]) if chunk != "f32" else 1
            coefficients = ", ".join(f"{value}.c{index}" for index in range(start, start + count))
            found.append(f"{chunk}({coefficients})")
            start += count
        return found

    def carrying(self, name: str) -> str:
        """The function building a struct between shader stages from its members, values of the algebra as
        they are, each lowered to the vectors of numbers it crosses as."""
        key = ("carry", name)
        if key not in self.helpers:
            helper = f"ga_carry_{len(self.helpers)}"
            members = self.structs[name]
            parameters = ", ".join(f"{member}: {self.wgsl(kind)}" for member, kind in members)
            arguments = ", ".join(", ".join(self.lowered(kind, member)) if member in self.carried[name] else member
                                  for member, kind in members)
            code = f"fn {helper}({parameters}) -> {name} {{\n    return {name}({arguments});\n}}\n"
            self.helpers[key] = (helper, code, name)
        return self.helpers[key][0]

    def element(self, kind: Any) -> Any:
        """The element type of an array, the component type of a vector, a matrix's column."""
        name, arguments = template(kind)
        if name == "array":
            return self.resolve(arguments[0])
        if name.startswith("vec"):
            return self.resolve(arguments[0])
        if name.startswith("mat"):
            return f"vec{name[5]}<{arguments[0]}>"
        raise TypeError(f"{kind} has no elements")

    # --- values
    def constant(self, value: Any) -> Value:
        """A value known at compile time, held in the symbolic context; scalars as plain numbers."""
        if isinstance(value, GAType):
            value = value._identity()
        if isinstance(value, Extensor):
            value = self.context.lower(value)
            if value.gatype.is_scalar and value.gatype.arity == 0:
                return Value(None, _ABSTRACT, float(np.asarray(self.as_scalar(value).kernel).ravel()[0]))
            return Value(None, value.gatype, value)
        return Value(literal(value), _ABSTRACT, float(value))

    def as_scalar(self, value: Extensor) -> Extensor:
        """A scalar of either orientation as the number it is, an `f32`: one stored against minus one
        restated against plus one."""
        sign = -1 if value.gatype.is_reoriented_scalar else 1
        return self.context.extensor(self.scalar, sign * np.asarray(value.kernel))

    def text(self, value: Value) -> str:
        """WGSL for a value, spelling out a compile-time one as its coefficients."""
        if value.text is not None:
            return value.text
        if value.type == _BATCH:
            raise TypeError(f"a batch of {len(value.constant)} values is no WGSL value: gather it into numbers or "
                            f"sum it")
        if value.type == _ABSTRACT:
            return literal(value.constant)
        coefficients = ", ".join(literal(float(Polynomial.lift(c))) for c in np.asarray(value.constant.kernel).ravel())
        return f"{self.wgsl(value.type)}({coefficients})"

    def symbol(self, value: Value, name: str) -> Any:
        """A value as numga computes with it: its compile-time value, or coefficients named for a
        function parameter."""
        if value.text is None and value.type != _ABSTRACT:
            return value.constant
        kind = self.scalar if value.type == _ABSTRACT else value.type
        shape = tuple(len(subspace) for subspace in kind.subspaces)
        kernel = np.empty(shape, dtype=object)
        for index, position in enumerate(np.ndindex(shape)):
            kernel[position] = Polynomial.variable(name if kind == self.scalar else f"{name}.c{index}")
        return self.context.extensor(kind, kernel)

    def operation(self, label: str, compute: Callable, operands: list[Value], line: int) -> Value:
        """An operation of numga's on shader values: folded when every operand is known, otherwise a
        function whose body is numga's result over the operands' named coefficients."""
        if all(operand.text is None or operand.type == _ABSTRACT for operand in operands):
            numbers = [operand.constant if operand.type != _ABSTRACT else operand.constant for operand in operands]
            try:
                return self.constant(compute(*numbers))
            except TypeError:
                pass
        # Numbers enter as parameters, so one function serves every number.
        operands = [Value(self.text(operand), self.scalar) if operand.type == _ABSTRACT else operand
                    for operand in operands]
        signature = tuple(
            ("value", operand.type) if operand.text is not None else
            ("constant", operand.type, tuple(literal(float(Polynomial.lift(c))) for c in np.asarray(operand.constant.kernel).ravel()))
            for operand in operands)
        key = (label, signature)
        if key not in self.helpers:
            try:
                result = compute(*(self.symbol(operand, f"a{index}") for index, operand in enumerate(operands)))
            except Exception as failure:
                described = ", ".join(str(operand.type) for operand in operands)
                raise self.error(line, f"numga cannot {label} on ({described}): {failure}") from failure
            if not isinstance(result, Extensor):
                result = self.context.extensor(self.scalar, np.array([result], dtype=object))
            if result.gatype.is_scalar and result.gatype.arity == 0:
                result = self.as_scalar(result)
            kind = result.gatype
            name = f"ga_{label}_{len(self.helpers)}"
            parameters = ", ".join(f"a{index}: {self.wgsl(operand.type)}"
                                   for index, operand in enumerate(operands) if operand.text is not None)
            components = [polynomial_text(Polynomial.lift(c) if not isinstance(c, Polynomial) else c)
                          for c in np.asarray(result.kernel).ravel()]
            body = components[0] if kind == self.scalar else (
                f"{self.wgsl(kind)}(\n        " + ",\n        ".join(components) + ")")
            code = f"fn {name}({parameters}) -> {self.wgsl(kind)} {{\n    return {body};\n}}\n"
            self.helpers[key] = (name, code, kind)
        name, _, kind = self.helpers[key]
        arguments = ", ".join(operand.text for operand in operands if operand.text is not None)
        return Value(f"{name}({arguments})", kind)

    def coerce(self, value: Value, target: Any, line: int) -> str:
        """WGSL for a value where a type is expected: a value of the algebra embedded into a wider type
        of the same orientation, a number as the scalar."""
        if target is None or value.type == target:
            return self.text(value)
        if isinstance(target, GAType):
            if value.type == _ABSTRACT and target == self.scalar:
                return self.text(value)
            if isinstance(value.type, GAType) and self.embeds(value.type, target):
                return self.text(self.operation("cast", lambda x: self.cast(x, target), [value], line))
            raise self.error(line, f"a {value.type} is not a {target}; cast it with .cast()")
        if isinstance(value.type, GAType) and value.type != self.scalar:
            raise self.error(line, f"a {value.type} is not a {target}")
        return self.text(value)

    def cast(self, value: Extensor, target: GAType) -> Extensor:
        """A value cast to exactly the target type: its layout, and none of the traits a cast drops."""
        return value.context.extensor(target, value.cast(target).kernel)

    def embeds(self, source: GAType, target: GAType) -> bool:
        return (source.arity == target.arity and source.orientation is target.orientation
                and all(a.support_is_subset_of(b) for a, b in zip(source.subspaces, target.subspaces)))

    # --- expressions
    def value(self, node: Any, scope: list[dict]) -> Value:
        if isinstance(node, Literal):
            return self.literal(node)
        if isinstance(node, Name):
            for frame in reversed(scope):
                if node.id in frame:
                    return frame[node.id]
            if node.id in self.globals:
                return self.globals[node.id]
            if node.id == _MV:
                return Value(None, _MV)
            if node.id in self.types and node.id != "f32":
                return self.constant(self.types[node.id])
            raise self.error(node.line, f"unknown name {node.id!r}")
        if isinstance(node, Member):
            return self.member(node, scope)
        if isinstance(node, Index):
            base = self.value(node.object, scope)
            index = self.value(node.index, scope)
            return Value(f"{self.text(base)}[{self.text(index)}]", self.element(base.type))
        if isinstance(node, Call):
            return self.call(node, scope)
        if isinstance(node, Unary):
            operand = self.value(node.operand, scope)
            if node.operator == "-" and operand.type == _BATCH:
                return self.gathered([self.operation("negate", operator.neg, [element], node.line)
                                      for element in operand.constant])
            if node.operator == "-" and (isinstance(operand.type, GAType) and operand.type != self.scalar or operand.text is None):
                return self.operation("negate", operator.neg, [operand], node.line)
            if node.operator == "-" and operand.type == _ABSTRACT:
                return Value(f"(-{operand.text})", _ABSTRACT, -operand.constant)
            if node.operator == "-" and self.is_state(operand.type):
                return self.state(node.line, "*", operand, Value("-1.0", self.scalar))
            return Value(f"({node.operator}{self.text(operand)})", "bool" if node.operator == "!" else operand.type)
        if isinstance(node, Binary):
            return self.binary(node.line, node.operator, self.value(node.left, scope), self.value(node.right, scope))
        if isinstance(node, TypeName):
            return Value(self.wgsl(node.text), "type")
        raise self.error(getattr(node, "line", 0), f"unexpected {node}")

    def literal(self, node: Literal) -> Value:
        text = node.text
        if text in ("true", "false"):
            return Value(text, "bool")
        if text.endswith("u"):
            return Value(text, "u32")
        if text.endswith("i"):
            return Value(text, "i32")
        if text.endswith("f"):
            return Value(text, self.scalar)
        if text.startswith(("0x", "0X")):
            return Value(text, "i32")
        if any(mark in text for mark in ".eE"):
            return Value(text, _ABSTRACT, float(text))
        return Value(text, _ABSTRACT, int(text))

    def member(self, node: Member, scope: list[dict]) -> Value:
        base = self.value(node.object, scope)
        if base.type == _MV:
            blade = getattr(self.context.multivector, node.name)
            return self.constant(blade)
        if isinstance(base.type, GAType):
            raise self.error(node.line, f"read a coefficient of a {base.type} by pairing it, not as .{node.name}")
        if base.type in self.structs:
            members = dict(self.structs[base.type])
            if node.name not in members:
                raise self.error(node.line, f"{base.type} has no member {node.name!r}")
            if node.name in self.carried.get(base.type, {}):
                kind = members[node.name]
                return self.lifted(kind, [f"{self.text(base)}.{node.name}_{index}"
                                          for index in range(len(self.chunks(kind)))])
            return Value(f"{self.text(base)}.{node.name}", members[node.name])
        name, arguments = template(base.type)
        if name.startswith("vec"):
            count = len(node.name)
            component = self.resolve(arguments[0])
            kind = component if count == 1 else f"vec{count}<{arguments[0]}>"
            return Value(f"{self.text(base)}.{node.name}", kind)
        raise self.error(node.line, f"{base.type} has no member {node.name!r}")

    def binary(self, line: int, symbol: str, left: Value, right: Value) -> Value:
        if _BATCH in (left.type, right.type):
            return self.batched(line, symbol, left, right)
        algebra = any(isinstance(side.type, GAType) and (side.type != self.scalar or side.text is None)
                      for side in (left, right))
        if symbol == "**":
            exponent = right.constant
            if right.type == _ABSTRACT and float(exponent).is_integer() and exponent >= 1:
                return self.operation(f"power{int(exponent)}", lambda x: _power(x, int(exponent)), [left], line)
            return Value(f"pow({self.text(left)}, {self.text(right)})", left.type)
        if self.is_state(left.type) or self.is_state(right.type):
            return self.state(line, symbol, left, right)
        if symbol == "/" and (algebra or left.text is None):
            if not self.is_scalar(right) and not (right.text is None and right.type == _ABSTRACT):
                raise self.error(line, f"cannot divide by a {right.type}")
            reciprocal = (self.constant(1 / right.constant) if right.type == _ABSTRACT and right.text is not None
                          else Value(f"(1.0 / {self.text(right)})", self.scalar))
            return self.binary(line, "*", left, reciprocal)
        if algebra and symbol in _OPERATORS:
            return self.operation({"+": "add", "-": "subtract", "*": "product", "&": "regressive",
                                   "|": "inner", "^": "wedge", ">>": "sandwich", "<<": "reverse_sandwich"}[symbol],
                                  _OPERATORS[symbol], [left, right], line)
        if algebra:
            raise self.error(line, f"{symbol!r} does not take a {left.type} and a {right.type}")
        if left.type == _ABSTRACT and right.type == _ABSTRACT and symbol in ("+", "-", "*", "/"):
            number = _OPERATORS.get(symbol, operator.truediv)(left.constant, right.constant)
            return Value(f"({left.text} {symbol} {right.text})", _ABSTRACT, number)
        kind = "bool" if symbol in ("==", "!=", "<", ">", "<=", ">=", "&&", "||") else (
            right.type if left.type == _ABSTRACT or (left.type == self.scalar and str(right.type).startswith("vec"))
            else left.type)
        return Value(f"({self.text(left)} {symbol} {self.text(right)})", kind)

    def batched(self, line: int, symbol: str, left: Value, right: Value) -> Value:
        """An operation on a batch, value by value: against each value of another batch of as many, each component
        of a vector of as many numbers, or one value throughout."""
        count = len((left if left.type == _BATCH else right).constant)

        def elements(side: Value) -> list[Value]:
            if side.type == _BATCH:
                if len(side.constant) != count:
                    raise self.error(line, f"cannot {symbol} batches of {count} and {len(side.constant)} values")
                return side.constant
            name, arguments = template(side.type) if isinstance(side.type, str) else (None, [])
            if name is not None and name.startswith("vec"):
                if int(name[3]) != count:
                    raise self.error(line, f"cannot {symbol} a batch of {count} values and a {side.type}")
                return [Value(f"{self.text(side)}.{axis}", self.resolve(arguments[0])) for axis in "xyzw"[:count]]
            return [side] * count
        return self.gathered([self.binary(line, symbol, a, b) for a, b in zip(elements(left), elements(right))])

    def gathered(self, values: list[Value]) -> Value:
        """Values of a batch: numbers gathered into a vector of them, values of the algebra kept as a batch."""
        if all(self.is_scalar(value) for value in values):
            kind = f"vec{len(values)}<f32>"
            return Value(f"{kind}({', '.join(self.text(value) for value in values)})", kind)
        return Value(None, _BATCH, values)

    def state(self, line: int, symbol: str, left: Value, right: Value) -> Value:
        """Arithmetic on structs of values of the algebra, member by member."""
        if symbol in ("+", "-"):
            if left.type != right.type:
                raise self.error(line, f"cannot {symbol} a {left.type} and a {right.type}")
            kind, operands = left.type, [left, right]
        elif symbol in ("*", "/") and self.is_state(left.type) and self.is_scalar(right):
            kind, operands = left.type, [left, Value(self.text(right), self.scalar)]
        elif symbol == "*" and self.is_state(right.type) and self.is_scalar(left):
            kind, operands = right.type, [right, Value(self.text(left), self.scalar)]
        else:
            raise self.error(line, f"cannot {symbol} a {left.type} and a {right.type}")
        key = ("state", symbol, kind, operands[1].type)
        if key not in self.helpers:
            name = f"ga_state_{len(self.helpers)}"
            first = lambda member, member_type: Value(f"a.{member}", member_type)
            second = lambda member, member_type: (Value(f"b.{member}", member_type) if operands[1].type == kind
                                                  else Value("b", self.scalar))
            members = [self.coerce(self.binary(line, symbol, first(member, member_type), second(member, member_type)),
                                   member_type, line)
                       for member, member_type in self.structs[kind]]
            code = (f"fn {name}(a: {kind}, b: {self.wgsl(operands[1].type)}) -> {kind} {{\n"
                    f"    return {kind}(\n        " + ",\n        ".join(members) + ");\n}\n")
            self.helpers[key] = (name, code, kind)
        name, _, _ = self.helpers[key]
        return Value(f"{name}({self.text(operands[0])}, {self.text(operands[1])})", kind)

    def call(self, node: Call, scope: list[dict]) -> Value:
        callee = node.callee
        if isinstance(callee, Member):
            base = self.value(callee.object, scope)
            if base.type != _MV and (isinstance(base.type, GAType) or base.type in (_ABSTRACT, _BATCH)):
                return self.method(node.line, base, callee.name, callee.arguments if False else node.arguments, scope)
        if isinstance(callee, TypeName):
            arguments = ", ".join(self.text(self.value(argument, scope)) for argument in node.arguments)
            name = template(callee.text)[0]
            kind = self.resolve(template(callee.text)[1][0]) if name == "bitcast" else callee.text
            return Value(f"{self.wgsl(callee.text)}({arguments})", kind)
        if isinstance(callee, Name) and not any(callee.id in frame for frame in scope) and callee.id not in self.globals:
            name = callee.id
            if name in self.functions:
                parameters, result = self.functions[name]
                if len(parameters) != len(node.arguments):
                    raise self.error(node.line, f"{name} takes {len(parameters)} arguments")
                arguments = ", ".join(self.coerce(self.value(argument, scope), parameter, node.line)
                                      for argument, parameter in zip(node.arguments, parameters))
                return Value(f"{name}({arguments})", result)
            if name in self.structs:
                members = self.structs[name]
                arguments = ", ".join(self.coerce(self.value(argument, scope), member, node.line)
                                      for argument, (_, member) in zip(node.arguments, members))
                built = self.carrying(name) if name in self.carried else name
                return Value(f"{built}({arguments})", name)
            if name in self.types and name != "f32":
                # A type applied to a value is its identity map applied: an embedding or a projection.
                return self.operation("apply", lambda kind, x: kind(x),
                                      [self.constant(self.types[name])] + [self.value(a, scope) for a in node.arguments], node.line)
            return self.builtin(node, name, [self.value(argument, scope) for argument in node.arguments])
        mapping = self.value(callee, scope)
        if isinstance(mapping.type, GAType) and mapping.type.arity > 0:
            arguments = [self.value(argument, scope) for argument in node.arguments]
            return self.operation("apply", lambda map_, *xs: map_(*xs), [mapping] + arguments, node.line)
        raise self.error(node.line, f"cannot call a {mapping.type}")

    def method(self, line: int, base: Value, name: str, nodes: list, scope: list[dict]) -> Value:
        if name.endswith("_") and name[:-1] in _RESERVED:
            name = name[:-1]
        if base.type == _BATCH:
            if name == "sum":
                # One operation over the whole batch: its values' coefficients gather in one function.
                return self.operation("sum", lambda *values: sum(values[1:], values[0]), base.constant, line)
            return self.gathered([self.method(line, value, name, nodes, scope) for value in base.constant])
        if name == "cast":
            target = self.types[nodes[0].id]
            if base.text is None:
                return self.constant(self.cast(base.constant, target))
            return self.operation("cast", lambda x: self.cast(x, target), [base], line)
        arguments = [self.value(argument, scope) for argument in nodes]
        if self.is_scalar(base):
            if name in _SCALAR_FUNCTIONS:
                if base.text is None or base.type == _ABSTRACT and base.text is None:
                    return self.constant(getattr(np, {"square_root": "sqrt", "arccos": "arccos", "arcsin": "arcsin",
                                                      "arctan": "arctan"}.get(name, name))(base.constant))
                return Value(f"{_SCALAR_FUNCTIONS[name]}({self.text(base)})", self.scalar)
            if name == "inverse":
                return self.binary(line, "/", Value("1.0", _ABSTRACT, 1.0), base)
            if name == "clip":
                return Value(f"clamp({self.text(base)}, {self.text(arguments[0])}, {self.text(arguments[1])})", self.scalar)
            if name == "inverse_square_root":
                return Value(f"inverseSqrt({self.text(base)})", self.scalar)
        if name == "norm":
            return self.method(line, self.method(line, base, "symmetric_reverse_product", [], scope), "square_root", [], scope)
        if name == "normalized":
            return self.binary(line, "/", base, self.method(line, base, "norm", [], scope))
        if name == "inverse":
            measure = self.method(line, base, "symmetric_reverse_product", [], scope)
            if not self.is_scalar(measure):
                raise self.error(line, f"the shader inverts only values whose reverse product is a scalar, not a {base.type}")
            return self.binary(line, "/", self.method(line, base, "reverse", [], scope), measure)
        if name == "exp":
            return self.exponential(line, base, scope)
        return self.operation(name, lambda x, *xs: getattr(x, name)(*xs), [base] + arguments, line)

    def exponential(self, line: int, base: Value, scope: list[dict]) -> Value:
        """The exponential of a value whose square is a scalar: a turn, a boost or a translation, by
        the trigonometric, hyperbolic or linear series its square selects."""
        square = self.method(line, base, "squared", [], scope)
        if not self.is_scalar(square):
            raise self.error(line, f"the shader exponentiates only values that square to a scalar, not a {base.type}")
        if base.text is None:
            return self.constant(base.constant.exp())
        key = ("exp", base.type)
        if key not in self.helpers:
            name = f"ga_exp_{len(self.helpers)}"
            square_inside = self.method(line, Value("a", base.type), "squared", [], scope)
            size = "sqrt(abs(s))"
            even = "select(select(1.0, cosh(r), s > 0.0), cos(r), s < 0.0)"
            odd = "select(select(1.0, sinh(r) / r, s > 0.0), sin(r) / r, s < 0.0)"
            result = self.binary(line, "+", Value("c", self.scalar), self.binary(line, "*", Value("k", self.scalar), Value("a", base.type)))
            code = (f"fn {name}(a: {self.wgsl(base.type)}) -> {self.wgsl(result.type)} {{\n"
                    f"    let s = {square_inside.text};\n    let r = {size};\n"
                    f"    let c = {even};\n    let k = {odd};\n    return {result.text};\n}}\n")
            self.helpers[key] = (name, code, result.type)
        name, _, kind = self.helpers[key]
        return Value(f"{name}({self.text(base)})", kind)

    def builtin(self, node: Call, name: str, arguments: list[Value]) -> Value:
        if name in _REJECTED:
            raise self.error(node.line, f"{name} is not in the dialect: the algebra has it, as {_REJECTED[name]}")
        measured = [argument.type for argument in arguments if str(argument.type).startswith(("vec3", "vec4"))]
        if name in _MEASURING and measured:
            raise self.error(node.line, f"{name} of a {measured[0]} measures numbers that a value of the algebra would "
                                        f"hold: hold them as one, and write {_MEASURING[name]}")
        if name == "select" and isinstance(arguments[0].type, GAType) and arguments[0].type != self.scalar:
            # A choice between two values of the algebra, of the first one's type, by a condition.
            kind = arguments[0].type
            key = ("select", kind)
            if key not in self.helpers:
                wgsl = self.wgsl(kind)
                code = (f"fn ga_select_{len(self.helpers)}(otherwise: {wgsl}, chosen: {wgsl}, condition: bool) -> {wgsl} {{\n"
                        f"    if (condition) {{\n        return chosen;\n    }}\n    return otherwise;\n}}\n")
                self.helpers[key] = (f"ga_select_{len(self.helpers)}", code, kind)
            helper, _, _ = self.helpers[key]
            return Value(f"{helper}({self.text(arguments[0])}, {self.coerce(arguments[1], kind, node.line)}, "
                         f"{self.text(arguments[2])})", kind)
        for argument in arguments:
            if isinstance(argument.type, GAType) and argument.type != self.scalar:
                raise self.error(node.line, f"{name} does not take a {argument.type}")
        texts = ", ".join(self.text(argument) for argument in arguments)
        concrete = [argument.type for argument in arguments if argument.type != _ABSTRACT]
        if name in _ELEMENTWISE:
            # A choice takes the type of what it chooses between, not of its condition.
            chosen = arguments[:2] if name == "select" else arguments
            kind = next((argument.type for argument in chosen if argument.type not in (_ABSTRACT, "bool")),
                        "i32" if all(isinstance(argument.constant, int) for argument in chosen) else self.scalar)
        elif name in _REDUCING:
            kind = self.element(concrete[0]) if concrete and str(concrete[0]).startswith("vec") else self.scalar
        elif name in ("f32", "u32", "i32", "bool"):
            kind = self.resolve(name)
        elif normalized_type(name) != name:
            kind = normalized_type(name)
        elif name in ("textureSampleLevel", "textureSample", "textureLoad", "unpack4x8unorm"):
            kind = "vec4<f32>"
        elif name == "textureDimensions":
            kind = "vec2<u32>"
        elif name == "arrayLength":
            kind = "u32"
        elif name in ("any", "all"):
            kind = "bool"
        elif name.startswith("atomic"):
            kind = "u32"
        else:
            raise self.error(node.line, f"unknown function {name!r}")
        return Value(f"{name}({texts})", kind)

    # --- statements
    def statements(self, nodes: list, scope: list[dict], result: Any, indent: str) -> list[str]:
        lines = []
        for node in nodes:
            lines.extend(self.statement(node, scope, result, indent))
        return lines

    def statement(self, node: Any, scope: list[dict], result: Any, indent: str) -> list[str]:
        if isinstance(node, Declaration):
            return [indent + line for line in self.declaration(node, scope)]
        if isinstance(node, Assignment):
            return [indent + self.simple(node, scope) + ";"]
        if isinstance(node, (Step, Evaluate)):
            return [indent + self.simple(node, scope) + ";"]
        if isinstance(node, If):
            return self.conditional(node, scope, result, indent)
        if isinstance(node, For):
            scope.append({})
            initial = self.simple(node.initial, scope) if node.initial is not None else ""
            condition = self.text(self.value(node.condition, scope)) if node.condition is not None else ""
            update = self.simple(node.update, scope) if node.update is not None else ""
            body = self.block(node.body, scope, result, indent)
            scope.pop()
            return [f"{indent}for ({initial}; {condition}; {update}) {{"] + body + [indent + "}"]
        if isinstance(node, While):
            condition = self.text(self.value(node.condition, scope))
            return [f"{indent}while ({condition}) {{"] + self.block(node.body, scope, result, indent) + [indent + "}"]
        if isinstance(node, Loop):
            lines = [indent + "loop {"] + self.block(node.body, scope, result, indent)
            if node.continuing is not None:
                lines += [indent + "    continuing {"] + self.block(node.continuing, scope, result, indent + "    ") + [indent + "    }"]
            return lines + [indent + "}"]
        if isinstance(node, Jump):
            if node.value is None:
                return [f"{indent}{node.keyword};"]
            value = self.value(node.value, scope)
            text = self.coerce(value, result, node.line) if node.keyword == "return" else self.text(value)
            return [f"{indent}{node.keyword} {text};"]
        raise self.error(getattr(node, "line", 0), f"unexpected statement {node}")

    def block(self, nodes: list, scope: list[dict], result: Any, indent: str) -> list[str]:
        scope.append({})
        lines = self.statements(nodes, scope, result, indent + "    ")
        scope.pop()
        return lines

    def conditional(self, node: If, scope: list[dict], result: Any, indent: str, prefix: str = "") -> list[str]:
        condition = self.text(self.value(node.condition, scope))
        lines = [f"{indent}{prefix}if ({condition}) {{"] + self.block(node.then, scope, result, indent)
        if isinstance(node.otherwise, If):
            following = self.conditional(node.otherwise, scope, result, indent, "} else ")
            return lines + following
        if node.otherwise is not None:
            lines += [indent + "} else {"] + self.block(node.otherwise, scope, result, indent)
        return lines + [indent + "}"]

    def declaration(self, node: Declaration, scope: list[dict]) -> list[str]:
        declared = self.resolve(node.type)
        value = self.value(node.value, scope) if node.value is not None else None
        if node.kind in ("let", "const") and declared is None and value is not None and value.text is None:
            # Known when the shader compiles: folded into the operations that use it.
            scope[-1][node.name] = value
            return []
        if value is not None and value.type == _ABSTRACT and declared is None:
            declared = self.scalar if isinstance(value.constant, float) else "i32"
        kind = declared if declared is not None else value.type
        if isinstance(kind, str) and kind in ("type", _MV):
            raise self.error(node.line, f"{node.name} names no value")
        scope[-1][node.name] = Value(node.name, kind)
        initial = f" = {self.coerce(value, kind, node.line)}" if value is not None else ""
        keyword = "let" if node.kind == "const" else node.kind
        # WGSL infers what the shader left to inference; a written type is kept.
        annotation = f": {self.wgsl(kind)}" if node.type is not None else ""
        return [f"{keyword} {node.name}{annotation}{initial};"]

    def simple(self, node: Any, scope: list[dict]) -> str:
        if isinstance(node, Declaration):
            return self.declaration(node, scope)[0][:-1]
        if isinstance(node, Assignment):
            if isinstance(node.target, Name) and node.target.id == "_":
                return f"_ = {self.text(self.value(node.value, scope))}"
            if (isinstance(node.target, Member) and node.target.name in
                    self.carried.get(self.value(node.target.object, scope).type, {})):
                raise self.error(node.line, f"build a struct carrying {node.target.name} between stages whole")
            target = self.value(node.target, scope)
            value = self.value(node.value, scope)
            if node.operator != "=" and (isinstance(target.type, GAType) or self.is_state(target.type)):
                value = self.binary(node.line, node.operator[:-1], target, value)
                return f"{target.text} = {self.coerce(value, target.type, node.line)}"
            return f"{target.text} {node.operator} {self.coerce(value, target.type, node.line) if node.operator == '=' else self.text(value)}"
        if isinstance(node, Step):
            return f"{self.text(self.value(node.target, scope))}{node.operator}"
        if isinstance(node, Evaluate):
            return self.text(self.value(node.expression, scope))
        raise self.error(getattr(node, "line", 0), f"unexpected {node}")

    # --- module
    def compile(self, source: str) -> Program:
        items = Parser(source).module()
        directives = [item.text for item in items if isinstance(item, Directive)]
        for item in items:
            if isinstance(item, Alias):
                target = self.resolve(item.type)
                if isinstance(target, GAType):
                    self.types[item.name] = target
                else:
                    self.native_aliases[item.name] = item.type
                directives.append(f"alias {item.name} = {self.wgsl(target)};")
        for item in items:
            if isinstance(item, Struct):
                self.structs[item.name] = [(member, self.resolve(kind)) for _, member, kind in item.members]
                self.member_attributes[item.name] = [attributes for attributes, _, _ in item.members]
                for attributes, (member, kind) in zip(self.member_attributes[item.name], self.structs[item.name]):
                    if (isinstance(kind, GAType) and kind != self.scalar
                            and any(attribute.startswith("@location") for attribute in attributes)):
                        self.carried.setdefault(item.name, {})[member] = kind
            elif isinstance(item, Function):
                self.functions[item.name] = ([self.resolve(kind) for _, _, kind in item.parameters], self.resolve(item.result))
        constants = [line for name, value in self.given.items() for line in self.given_constant(name, value)]
        for item in items:
            if isinstance(item, Declaration):
                if item.name in self.given:
                    raise self.error(item.line, f"{item.name} is a constant given from Python")
                constants.extend(self.global_(item))
        functions = []
        for item in items:
            if isinstance(item, Function):
                functions.append(self.function(item))
        structs = [self.struct_code(name, members) for name, members in self.structs.items()]
        if self.uniforms:
            members = ",\n".join(f"    {name}: {self.wgsl(kind)}" for name, kind in self.uniforms)
            structs.append(f"struct Uniforms {{\n{members},\n}};\n")
            self.bindings.insert(0, Binding(0, "uniforms", "uniforms", "Uniforms"))
            for index, binding in enumerate(self.bindings):
                binding.index = index
        declarations = []
        for binding in self.bindings:
            space = {"uniforms": "<storage, read>", "storage": f"<storage, {binding.access}>"}.get(binding.kind, "")
            declarations.append(f"@group(0) @binding({binding.index}) var{space} {binding.name}: {self.wgsl(binding.type)};")
        helpers = [code for _, code, _ in self.helpers.values()]
        # Types the shader never named take their structs last, once every helper has named them.
        algebra_structs = []
        for kind, name in self.names.items():
            shape = int(np.prod([len(subspace) for subspace in kind.subspaces]))
            fields = "\n".join(f"    c{index}: f32," for index in range(shape))
            algebra_structs.append(f"// {kind}\nstruct {name} {{\n{fields}\n}};\n")
        code = "\n".join(directives + algebra_structs + self.aliases + structs + declarations + constants
                         + helpers + functions)
        return Program(code, self.bindings, self.uniforms, dict(self.structs),
                       {name: kind for kind, name in self.names.items()}, self.entry_points,
                       self.stages.get("vertex"), self.stages.get("fragment"), self.targets)

    def struct_code(self, name: str, members: list) -> str:
        attributes = self.member_attributes.get(name, [[] for _ in members])
        fields, taken = [], set()
        for attribute, (member, kind) in zip(attributes, members):
            if member not in self.carried.get(name, {}):
                fields.append((attribute, member, self.wgsl(kind)))
                continue
            # A value of the algebra takes a location for each vector of numbers it crosses as, from the one it names.
            first = int(next(entry for entry in attribute if entry.startswith("@location"))[len("@location("):-1])
            others = [entry for entry in attribute if not entry.startswith("@location")]
            fields.extend(([f"@location({first + index})"] + others, f"{member}_{index}", chunk)
                          for index, chunk in enumerate(self.chunks(kind)))
        for attribute, member, _ in fields:
            for entry in attribute:
                if entry.startswith("@location"):
                    if entry in taken:
                        raise SyntaxError(f"struct {name}: {member} takes {entry}, which another member holds")
                    taken.add(entry)
        lines = "\n".join(f"    {' '.join(attribute + [member])}: {kind}," for attribute, member, kind in fields)
        return f"struct {name} {{\n{lines}\n}};\n"

    def given_constant(self, name: str, value: Any) -> list[str]:
        """A constant given from Python: a value of the algebra folded where it is used, an integer, a number or a
        vector of numbers declared as a `const`."""
        if isinstance(value, Extensor):
            if value.shape not in ((), (2,), (3,), (4,)):
                raise ValueError(f"the constant {name} is a batch {value.shape}, not one value or two to four")
            self.globals[name] = (self.constant(self.narrowed(value)) if value.shape == () else
                                  Value(None, _BATCH, [self.constant(self.narrowed(value[index]))
                                                       for index in range(value.shape[0])]))
            return []
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"the constant {name} is a truth value, which the shader takes as no number")
        if isinstance(value, (int, np.integer)):
            if value < 0:
                raise ValueError(f"the constant {name} is a negative integer, not a u32")
            self.globals[name] = Value(name, "u32")
            return [f"const {name}: u32 = {int(value)}u;"]
        if isinstance(value, (float, np.floating)):
            self.globals[name] = Value(name, self.scalar)
            return [f"const {name}: f32 = {literal(value)};"]
        numbers = np.asarray(value, dtype=float)
        if numbers.shape not in ((2,), (3,), (4,)):
            raise ValueError(f"the constant {name} of shape {numbers.shape} is no vector of two to four numbers")
        kind = f"vec{len(numbers)}<f32>"
        self.globals[name] = Value(name, kind)
        return [f"const {name}: {kind} = {kind}({', '.join(literal(number) for number in numbers)});"]

    def narrowed(self, value: Extensor) -> Extensor:
        """A value of the algebra given from Python in the type of its blades that are not zero, as a blade such as
        `mv.x` has, so that operations on it compute only with those."""
        names = [name for name, coefficient in zip(value.gatype.subspaces[0].blade_names, np.asarray(value.kernel))
                 if coefficient != 0] if value.gatype.arity == 0 else []
        return value.restrict_subspace(self.algebra.subspace(names)) if names else value

    def global_(self, node: Declaration) -> list[str]:
        if node.kind in ("const", "let") or node.kind == "override":
            value = self.value(node.value, [{}]) if node.value is not None else None
            if node.kind != "override" and value is not None and value.text is None:
                self.globals[node.name] = value
                return []
            declared = self.resolve(node.type)
            kind = declared if declared is not None else (self.scalar if value.type == _ABSTRACT else value.type)
            self.globals[node.name] = Value(node.name, kind)
            initial = f" = {self.coerce(value, kind, node.line)}" if value is not None else ""
            return [f"{node.kind} {node.name}: {self.wgsl(kind)}{initial};"]
        kind = self.resolve(node.type)
        space = node.space or ""
        if space == "uniform":
            self.uniforms.append((node.name, kind))
            self.globals[node.name] = Value(f"uniforms.{node.name}", kind)
            return []
        if space.startswith("storage"):
            access = "read_write" if "read_write" in space else "read"
            self.bindings.append(Binding(0, node.name, "storage", kind, access))
        elif str(kind).startswith("texture"):
            self.bindings.append(Binding(0, node.name, "texture", kind))
        elif kind == "sampler":
            self.bindings.append(Binding(0, node.name, "sampler", kind))
        else:
            self.globals[node.name] = Value(node.name, kind)
            return [f"var<{space}> {node.name}: {self.wgsl(kind)};"]
        self.globals[node.name] = Value(node.name, kind)
        return []

    def function(self, node: Function) -> str:
        for attribute in node.attributes:
            match = re.fullmatch(r"@workgroup_size\((.*)\)", attribute.replace(" ", ""))
            if match and "@compute" in node.attributes:
                sizes = [int(size) for size in match.group(1).split(",")]
                self.entry_points.append((node.name, tuple(sizes + [1] * (3 - len(sizes)))))
        for stage in ("vertex", "fragment"):
            if f"@{stage}" in node.attributes:
                self.stages[stage] = node.name
        if "@fragment" in node.attributes:
            # A struct of outputs writes one target per located member, a single output one.
            located = [any(attribute.startswith("@location") for attribute in attributes)
                       for attributes in self.member_attributes.get(node.result, [])]
            self.targets = sum(located) if node.result in self.structs else 1
        parameters, result = self.functions[node.name]
        frame = {name: Value(name, kind) for (_, name, _), kind in zip(node.parameters, parameters)}
        signature = ", ".join(" ".join(attributes + [f"{name}: {self.wgsl(kind)}"])
                              for (attributes, name, _), kind in zip(node.parameters, parameters))
        returns = f" -> {' '.join(node.result_attributes + [self.wgsl(result)])}" if result is not None else ""
        body = self.statements(node.body, [frame], result, "    ")
        header = " ".join(node.attributes + [f"fn {node.name}({signature}){returns} {{"])
        return "\n".join([header] + body + ["}"]) + "\n"


def _power(value: Any, exponent: int) -> Any:
    result = value
    for _ in range(exponent - 1):
        result = result * value
    return result


def compile(source: str, algebra: Any, types: Mapping[str, GAType], constants: Mapping[str, Any]) -> Program:
    """WGSL for a shader in the dialect, with the algebra's types named as the shader names them, and the constants
    given from Python declared by their names."""
    return Compiler(algebra, types, constants).compile(source)
