import pprint

from typing import Dict, Optional

from qdtree.cut import Cut
from qdtree.range import Range
from qdtree.schema import SchemaType


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

    def cut(self, cut: Cut) -> bool:
        """Cut the tree at this node.

        Returns True if the cut was successful, False otherwise.
        """
        if self._cut is not None:
            raise RuntimeError("Cut already exists")

        assert cut.attr1 in self._ranges, f"Attribute {cut.attr1} not in ranges"

        new_ranges = cut.cut_range(self._ranges[cut.attr1])
        if new_ranges is None:
            return False

        true_range, false_range = new_ranges
        self._left = QdTreeNode(self.id * 2, {**self._ranges, cut.attr1: true_range})
        self._right = QdTreeNode(
            self.id * 2 + 1, {**self._ranges, cut.attr1: false_range}
        )
        self._cut = cut
        return True

    def route_tuple(self, row: Dict[str, SchemaType]) -> int:
        """Route a tuple to a leaf node.

        Returns the id of the leaf node.
        """
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

    def route_tuple(self, row: Dict[str, SchemaType]) -> int:
        """Route a tuple to a leaf node.

        Returns the id of the leaf node.
        """
        return self._root.route_tuple(row)


# Example
if __name__ == "__main__":
    from qdtree.schema import Schema
    from qdtree.cut import CutRepository

    schema: Schema = {
        "x": "float",
        "y": "int",
    }

    builder = CutRepository.Builder(schema)
    builder.add("x", "<", "0.5")
    builder.add("y", ">=", "10")
    builder.add("x", ">", "40")
    builder.add("y", "<=", "50")
    builder.add("y", ">", "8")

    repo = builder.build()

    ranges = {
        "x": Range(repo.dict),
        "y": Range(repo.dict),
    }

    qdtree = QdTree(ranges)

    assert qdtree.root.cut(repo.get("x", "<", "0.5")) == True
    assert qdtree.root.left is not None
    assert qdtree.root.left.cut(repo.get("y", ">=", "10")) == True
    assert qdtree.root.right is not None
    assert qdtree.root.right.cut(repo.get("y", "<=", "50")) == True
    assert qdtree.root.left.right is not None
    assert qdtree.root.left.right.cut(repo.get("x", ">", "40")) == False
    assert qdtree.root.left.right.cut(repo.get("y", ">", "8")) == True

    print(qdtree)
