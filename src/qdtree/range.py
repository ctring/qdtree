import numpy as np

from typing import Dict, Optional, Tuple

from qdtree.dictionary import Dictionary
from qdtree.schema import SchemaType


class Range:
    """A range of attribute values.

    A range can be open or closed on either side. 
    """

    __slots__ = ["_left", "_right", "_open_left", "_open_right"]

    _left: SchemaType
    _right: SchemaType

    def __init__(
        self,
        left: Optional[SchemaType] = None,
        right: Optional[SchemaType] = None,
        open_left: bool = True,
        open_right: bool = True,
    ):
        self._left = left if left is not None else float("-inf")
        self._right = right if right is not None else float("inf")
        self._open_left = open_left
        self._open_right = open_right

    def __str__(self):
        left_bracket = "(" if self._open_left else "["
        right_bracket = ")" if self._open_right else "]"
        return f"{left_bracket}{self.left}, {self.right}{right_bracket}"

    def __repr__(self):
        return self.__str__()

    @property
    def left(self) -> SchemaType:
        return self._left

    @property
    def right(self) -> SchemaType:
        return self._right

    @property
    def open_left(self) -> bool:
        return self._open_left

    @property
    def open_right(self) -> bool:
        return self._open_right

    def overlaps(self, other: "Range") -> bool:
        cond1 = (
            self.left < other.right or  # type: ignore
            (self.left == other.right and not self.open_left and not other.open_right)
        )
        cond2 = (
            other.left < self.right or  # type: ignore
            (other.left == self.right and not other.open_left and not self.open_right)
        )
        return cond1 and cond2


class RangeWithDict(Range):
    """A range where the values are dictionary encoded."""

    __slots__ = ["_dict", "_left_index", "_right_index"]

    _left_index: int
    _right_index: int

    def __init__(
        self,
        dict: Dictionary,
        left_index: Optional[int] = None,
        right_index: Optional[int] = None,
        open_left: bool = True,
        open_right: bool = True,
    ):
        super().__init__(None, None, open_left, open_right)
        self._dict = dict
        self._left_index = left_index if left_index is not None else dict.NINF
        self._right_index = right_index if right_index is not None else dict.INF

    @property
    def left(self) -> SchemaType:
        return self._dict[self._left_index]

    @property
    def right(self) -> SchemaType:
        return self._dict[self._right_index]

    @property
    def encoding(self) -> np.ndarray:
        return np.array(
            [self._left_index, self._right_index, int(
                self._open_left), int(self._open_right)]
        )

    @property
    def encoding_space(self) -> Tuple[np.ndarray, np.ndarray]:
        dict_min_index = self._dict.min_index()
        dict_max_index = self._dict.max_index()
        return (
            np.array([dict_min_index, dict_min_index, 0, 0]),
            np.array([dict_max_index, dict_max_index, 1, 1]),
        )


Block = Dict[str, RangeWithDict]
