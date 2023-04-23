from typing import Dict, List, Literal, Optional, Set, Tuple

from qdtree.dictionary import Dictionary
from qdtree.schema import Schema, SchemaType, SchemaTypeTag
from qdtree.range import Range

Operator = Literal["<", ">", "<=", ">="]


class Cut:
    """A cut is a predicate on a single attribute.

    A cut is a predicate on a single attribute. It is used to split a range
    into two sub-ranges. The cut is defined by an attribute, an operator, and
    a value. Internally, the value is stored as an index into the dictionary.
    """

    __slots__ = ["_dict", "_attr1", "_op", "_attr2"]

    _op: Operator

    def __init__(
        self,
        dict: Dictionary,
        attr1: str,
        op: Operator,
        attr2: int,
    ):
        """Create a new cut.

        Args:
            dict: The dictionary to use for this cut.
            attr1: The attribute to compare.
            op: The operator to use.
            attr2: The attribute to compare against.
        """
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
        """Evaluate the cut on a row.

        Args:
            row: The row to evaluate the cut on.

        Returns:
            True if the cut evaluates to true, False otherwise.
        """
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
            assert False, f"Invalid operator {self._op}"

    def cut_range(self, range: Range) -> Optional[Tuple["Range", "Range"]]:
        """Cut a range.

        Args:
            range: The range to cut.

        Returns:
            A tuple of two ranges. The first range is the range when the cut
            is evaluated to true, the second range is the range when the cut
            is evaluated to false. If the cut cannot be applied to the range,
            None is returned.

        Raises:
            ValueError: If the dictionary of the range does not match the
                dictionary of the cut.
        """
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
    """A repository of cuts.

    A cut repository is a collection of cuts. The repository is immutable and
    is built using a builder. The builder collects all the cuts and then constructs
    a dictionary out of the cuts before creating the repository.
    """

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

    def get(
        self,
        attr1: str,
        op: Operator,
        attr2: str,
    ) -> Cut:
        return self._cuts[(attr1, op, attr2)]

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
            """
            Add a cut to the repository.

            Args:
                attr1: The first attribute.
                op: The operator.
                attr2: The second attribute.

            Raises:
                ValueError: If the attribute is not in the schema.
            """
            if attr1 not in self._schema:
                raise ValueError(f"Invalid attribute {attr1}")

            self._cuts.append((attr1, op, attr2))
            self._values.add((attr2, self._schema[attr1]))

        def build(self) -> "CutRepository":
            """Build the cut repository.

            Returns:
                The cut repository.
            """
            dict = Dictionary(self._values)
            cuts = {
                (attr1, op, attr2): Cut(dict, attr1, op, dict.reverse_lookup_str(attr2))
                for attr1, op, attr2 in self._cuts
            }
            return CutRepository(self._schema, dict, cuts)
