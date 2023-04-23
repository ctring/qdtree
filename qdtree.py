import pprint

import numpy as np
from typing import Dict, List, Literal, Optional, Tuple

Operator = Literal["<", ">", "<=", ">="]


class Cut:
    __slots__ = ["_attr1", "_op", "_attr2"]

    _op: Operator

    def __init__(self, attr1: str, op: Operator, attr2: float):
        self._attr1 = attr1
        self._op = op
        self._attr2 = attr2

    def __str__(self):
        return f"{self._attr1} {self._op} {self._attr2}"

    def __repr__(self):
        return self.__str__()

    @property
    def attr1(self) -> str:
        return self._attr1

    @property
    def op(self) -> Operator:
        return self._op

    @property
    def attr2(self) -> float:
        return self._attr2

    def evaluate(self, row: Dict[str, float]) -> bool:
        if self._op == "<":
            return row[self._attr1] < self._attr2
        elif self._op == ">":
            return row[self._attr1] > self._attr2
        elif self._op == "<=":
            return row[self._attr1] <= self._attr2
        elif self._op == ">=":
            return row[self._attr1] >= self._attr2
        else:
            raise ValueError(f"Invalid operator {self._op}")


class Range:
    __slots__ = ["_left", "_right", "_open_left", "_open_right"]

    _left: float
    _right: float

    def __init__(
        self,
        left: Optional[float] = None,
        right: Optional[float] = None,
        open_left: bool = True,
        open_right: bool = True,
    ):
        self._left = left if left is not None else -np.inf
        self._right = right if right is not None else np.inf
        self._open_left = open_left
        self._open_right = open_right

    def __str__(self):
        left_bracket = "(" if self._open_left else "["
        right_bracket = ")" if self._open_right else "]"
        return f"{left_bracket}{self._left}, {self._right}{right_bracket}"

    def __repr__(self):
        return self.__str__()

    def new_ranges_from_cut(self, cut: Cut) -> Optional[Tuple["Range", "Range"]]:
        if cut.attr2 <= self._left or cut.attr2 >= self._right:
            return None

        # fmt: off
        if cut.op == '<' or cut.op == '<=':
            # {a, b} => T: {a, min(b, attr2)}, F: {min(b, attr2), b}
            true_range_left = self._left
            true_range_left_open = self._open_left

            true_range_right = min(self._right, cut.attr2)
            true_range_right_open = cut.op == '<'

            false_range_left = min(self._right, cut.attr2)
            false_range_left_open = cut.op == '<='

            false_range_right = self._right
            false_range_right_open = self._open_right

            true_range = Range(
                true_range_left, true_range_right,
                true_range_left_open, true_range_right_open
            )
            false_range = Range(
                false_range_left, false_range_right,
                false_range_left_open, false_range_right_open
            )
            return (true_range, false_range)
        elif cut.op == '>' or cut.op == '>=':
            # {a, b} => T: {max(a, attr2), b}, F: {a, max(a, attr2)}
            true_range_left = max(self._left, cut.attr2)
            true_range_left_open = cut.op == '>'

            true_range_right = self._right
            true_range_right_open = self._open_right

            false_range_left = self._left
            false_range_left_open = self._open_left

            false_range_right = max(self._left, cut.attr2)
            false_range_right_open = cut.op == '>='

            true_range = Range(
                true_range_left, true_range_right,
                true_range_left_open, true_range_right_open
            )
            false_range = Range(
                false_range_left, false_range_right,
                false_range_left_open, false_range_right_open
            )
            return (true_range, false_range)
        else:
            raise RuntimeError(f'Unknown operator {cut.op}')
        # fmt: on


class QdTreeNode:
    __slots__ = ["_id", "_cut", "_ranges", "_left", "_right"]

    _left: Optional["QdTreeNode"]
    _right: Optional["QdTreeNode"]

    def __init__(self, id: int, ranges: Dict[str, Range], cut: Optional[Cut] = None):
        self._id = id
        self._cut = cut
        self._ranges = ranges
        self._left = None
        self._right = None

    def __str__(self):
        return pprint.pformat(self.__dict__(), sort_dicts=False)

    def __repr__(self):
        return self.__str__()

    def __dict__(self):
        return {
            "id": self._id,
            "cut": self._cut,
            "ranges": self._ranges,
            "left": self._left.__dict__() if self._left is not None else None,
            "right": self._right.__dict__() if self._right is not None else None,
        }

    @property
    def id(self) -> int:
        return self._id

    @property
    def left(self) -> Optional["QdTreeNode"]:
        return self._left

    @property
    def right(self) -> Optional["QdTreeNode"]:
        return self._right

    def add_cut(self, cut: Cut) -> bool:
        if self._cut is not None:
            raise RuntimeError("Cut already exists")

        assert cut.attr1 in self._ranges, f"Attribute {cut.attr1} not in ranges"

        new_ranges = self._ranges[cut.attr1].new_ranges_from_cut(cut)
        if new_ranges is None:
            return False

        true_range, false_range = new_ranges
        self._left = QdTreeNode(self.id * 2, {**self._ranges, cut.attr1: true_range})
        self._right = QdTreeNode(
            self.id * 2 + 1, {**self._ranges, cut.attr1: false_range}
        )
        self._cut = cut
        return True

    def route_tuple(self, row: Dict[str, float]) -> int:
        if self._cut is None:
            return self._id

        if self._cut.attr1 not in row:
            raise RuntimeError(f"Attribute {self._cut.attr1} not in row")

        assert self._left is not None and self._right is not None
        if self._cut.evaluate(row):
            return self._left.route_tuple(row)
        else:
            return self._right.route_tuple(row)


class QdTree:
    __slots__ = ["_root"]

    _root: QdTreeNode

    def __init__(self, ranges: Dict[str, Range]):
        self._root = QdTreeNode(1, ranges)

    def __str__(self):
        return str(self._root)

    def __repr__(self):
        return self.__str__()

    @property
    def root(self) -> QdTreeNode:
        return self._root

    def route_tuple(self, row: Dict[str, float]) -> int:
        return self._root.route_tuple(row)


if __name__ == "__main__":
    ranges = {
        "x": Range(),
        "y": Range(),
    }
    qdtree = QdTree(ranges)
    assert qdtree.root.add_cut(Cut("x", "<", 0.5)) == True
    assert qdtree.root.left is not None
    assert qdtree.root.left.add_cut(Cut("y", ">=", 10)) == True
    assert qdtree.root.right is not None
    assert qdtree.root.right.add_cut(Cut("y", "<=", 50)) == True
    assert qdtree.root.left.right is not None
    assert qdtree.root.left.right.add_cut(Cut("x", ">", 50)) == False
    assert qdtree.root.left.right.add_cut(Cut("y", ">", 8)) == True

    assert qdtree.route_tuple({"x": 0.2, "y": 10.1}) == 4
    assert qdtree.route_tuple({"x": 0.2, "y": 9.9}) == 10
    assert qdtree.route_tuple({"x": 0.8, "y": 10.1}) == 6

    print(qdtree)
