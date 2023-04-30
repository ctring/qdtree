import pandas as pd

from typing import Dict, List, Literal, Optional, Set, Tuple

from qdtree.dictionary import Dictionary
from qdtree.schema import Schema, SchemaType, SchemaTypeTag
from qdtree.range import Range, RangeWithDict

Operator = Literal["<", ">", "<=", ">="]


class Cut:
    """A cut is a predicate on a single attribute.

    A cut is a predicate on a single attribute. It is used to split a range
    into two sub-ranges. The cut is defined by an attribute, an operator, and
    a value. Internally, the value is stored as an index into the dictionary.
    """

    __slots__ = ["_dict", "_attr1", "_op", "_attr2_index"]

    _op: Operator

    def __init__(
        self,
        dict: Dictionary,
        attr1: str,
        op: Operator,
        attr2_index: int,
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
        self._attr2_index = attr2_index

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
        return self._dict[self._attr2_index]

    def eval_data(self, rows: pd.DataFrame) -> pd.Series:
        """Evaluate the cut on a row.

        Args:
            rows: The rows to evaluate the cut on.

        Returns:
            A boolean series, where each element is
            True if the cut evaluates to true, False otherwise.
        """
        col = rows[self._attr1]

        if self._op == "<":
            return col < self.attr2
        elif self._op == ">":
            return col > self.attr2
        elif self._op == "<=":
            return col <= self.attr2
        elif self._op == ">=":
            return col >= self.attr2
        else:
            assert False, f"Invalid operator {self._op}"

    def eval_range(self, range: Range) -> bool:
        """Evaluate the cut on a range.

        Args:
            range: The range to evaluate the cut on.

        Returns:
            True if the cut evaluates to true for at least one value in the
            range, False otherwise.
        """
        init_range = RangeWithDict(self._dict)
        res = self.cut_range(init_range)
        assert res is not None
        pos_range, _ = res
        return pos_range.overlaps(range)

    def cut_range(self, range: RangeWithDict) -> Optional[Tuple["RangeWithDict", "RangeWithDict"]]:
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
            pos_range_left = range._left_index
            pos_range_left_open = range._open_left

            pos_range_right = self._attr2_index
            pos_range_right_open = self._op == '<'

            neg_range_left = self._attr2_index
            neg_range_left_open = self._op == '<='

            neg_range_right = range._right_index
            neg_range_right_open = range._open_right

            pos_range = RangeWithDict(
                self._dict,
                pos_range_left, pos_range_right,
                pos_range_left_open, pos_range_right_open
            )
            neg_range = RangeWithDict(
                self._dict,
                neg_range_left, neg_range_right,
                neg_range_left_open, neg_range_right_open
            )
            return (pos_range, neg_range)
        elif self._op == '>' or self._op == '>=':
            # {a, b} => T: {max(a, attr2), b}, F: {a, max(a, attr2)}
            pos_range_left = self._attr2_index
            pos_range_left_open = self.op == '>'

            pos_range_right = range._right_index
            pos_range_right_open = range._open_right

            neg_range_left = range._left_index
            neg_range_left_open = range._open_left

            neg_range_right = self._attr2_index
            neg_range_right_open = self.op == '>='

            pos_range = RangeWithDict(
                self._dict,
                pos_range_left, pos_range_right,
                pos_range_left_open, pos_range_right_open
            )
            neg_range = RangeWithDict(
                self._dict,
                neg_range_left, neg_range_right,
                neg_range_left_open, neg_range_right_open
            )
            return (pos_range, neg_range)
        else:
            raise RuntimeError(f'Unknown operator {self._op}')
        # fmt: on


class CutRepository:
    """A repository of cuts.

    A cut repository is a collection of cuts. The repository is immutable and
    is built using a builder. The builder collects all the cuts and then constructs
    a dictionary out of the cuts before creating the repository.
    """

    __slots__ = ["_schema", "_dict", "_cuts", "_cut_index"]

    _cuts: List[Cut]

    def __init__(
        self,
        schema: Schema,
        dict: Dictionary,
        cut_index: Dict[Tuple[str, Operator, str], Cut],
    ):
        self._schema = schema
        self._dict = dict
        self._cut_index = cut_index
        self._cuts = list(cut_index.values())

    def __len__(self) -> int:
        return len(self._cuts)

    def __getitem__(self, index: int) -> Cut:
        return self._cuts[index]

    def get(
        self,
        attr1: str,
        op: Operator,
        attr2: str,
    ) -> Cut:
        return self._cut_index[(attr1, op, attr2)]

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
            cut_index = {
                (attr1, op, attr2): Cut(dict, attr1, op, dict.reverse_lookup_str(attr2))
                for attr1, op, attr2 in self._cuts
            }
            return CutRepository(self._schema, dict, cut_index)
