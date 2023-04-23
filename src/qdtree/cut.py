from typing import Dict, List, Literal, Optional, Set, Tuple

from .dictionary import Dictionary
from .schema import Schema, SchemaType, SchemaTypeTag
from .range import Range

Operator = Literal["<", ">", "<=", ">="]


class Cut:
    __slots__ = ["_dict", "_attr1", "_op", "_attr2"]

    _op: Operator

    def __init__(
        self,
        dict: Dictionary,
        attr1: str,
        op: Operator,
        attr2: int,
    ):
        self._dict = dict
        self._attr1 = attr1
        self._op = op
        self._attr2 = attr2

    def __str__(self):
        return f"{self._attr1} {self._op} {self.attr2}"

    def __repr__(self):
        return self.__str__()

    @property
    def attr1(self) -> str:
        return self._attr1

    @property
    def op(self) -> Operator:
        return self._op

    @property
    def attr2(self) -> SchemaType:
        return self._dict[self._attr2]

    def evaluate(self, row: Dict[str, SchemaType]) -> bool:
        row_value = row[self._attr1]

        if type(row_value) != type(self.attr2):
            raise ValueError(
                f"Type mismatch on attribute '{self._attr1}'. "
                f"Row: {row_value} ({type(row_value)}) Cut: {self.attr2} ({type(self.attr2)}))"
            )

        if self._op == "<":
            return row_value < self.attr2  # type: ignore
        elif self._op == ">":
            return row_value > self.attr2  # type: ignore
        elif self._op == "<=":
            return row_value <= self.attr2  # type: ignore
        elif self._op == ">=":
            return row_value >= self.attr2  # type: ignore
        else:
            raise ValueError(f"Invalid operator {self._op}")

    def cut_range(self, range: Range) -> Optional[Tuple["Range", "Range"]]:
        if range._dict != self._dict:
            raise ValueError(f"Dictionary mismatch")

        if self.attr2 <= range.left or self.attr2 >= range.right:  # type: ignore
            return None

        # fmt: off
        if self._op == '<' or self._op == '<=':
            # {a, b} => T: {a, min(b, attr2)}, F: {min(b, attr2), b}
            true_range_left = range._left
            true_range_left_open = range._open_left

            true_range_right = self._attr2
            true_range_right_open = self._op == '<'

            false_range_left = self._attr2
            false_range_left_open = self._op == '<='

            false_range_right = range._right
            false_range_right_open = range._open_right

            true_range = Range(
                self._dict,
                true_range_left, true_range_right,
                true_range_left_open, true_range_right_open
            )
            false_range = Range(
                self._dict,
                false_range_left, false_range_right,
                false_range_left_open, false_range_right_open
            )
            return (true_range, false_range)
        elif self._op == '>' or self._op == '>=':
            # {a, b} => T: {max(a, attr2), b}, F: {a, max(a, attr2)}
            true_range_left = self._attr2
            true_range_left_open = self.op == '>'

            true_range_right = range._right
            true_range_right_open = range._open_right

            false_range_left = range._left
            false_range_left_open = range._open_left

            false_range_right = self._attr2
            false_range_right_open = self.op == '>='

            true_range = Range(
                self._dict,
                true_range_left, true_range_right,
                true_range_left_open, true_range_right_open
            )
            false_range = Range(
                self._dict,
                false_range_left, false_range_right,
                false_range_left_open, false_range_right_open
            )
            return (true_range, false_range)
        else:
            raise RuntimeError(f'Unknown operator {self._op}')
        # fmt: on


class CutRepository:
    __slots__ = ["_schema", "_dict", "_cuts"]

    def __init__(
        self,
        schema: Schema,
        dict: Dictionary,
        cuts: Dict[Tuple[str, Operator, str], Cut],
    ):
        self._schema = schema
        self._dict = dict
        self._cuts = cuts

    def __getitem__(self, key: Tuple[str, Operator, str]) -> Cut:
        return self._cuts[key]

    @property
    def schema(self) -> Schema:
        return self._schema

    @property
    def dict(self) -> Dictionary:
        return self._dict

    class Builder:
        __slots__ = ["_cuts", "_values", "_schema"]

        _cuts: List[Tuple[str, Operator, str]]
        _values: Set[Tuple[str, SchemaTypeTag]]

        def __init__(self, schema: Schema):
            self._schema = schema
            self._cuts = []
            self._values = set()

        def add(
            self,
            attr1: str,
            op: Operator,
            attr2: str,
        ):
            if attr1 not in self._schema:
                raise ValueError(f"Invalid attribute {attr1}")

            self._cuts.append((attr1, op, attr2))
            self._values.add((attr2, self._schema[attr1]))

        def build(self) -> "CutRepository":
            dict = Dictionary(self._values)
            cuts = {
                (attr1, op, attr2): Cut(dict, attr1, op, dict.reverse_lookup_str(attr2))
                for attr1, op, attr2 in self._cuts
            }
            return CutRepository(self._schema, dict, cuts)
