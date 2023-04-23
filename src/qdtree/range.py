import numpy as np

from typing import List, Optional, Tuple

from qdtree.dictionary import Dictionary
from qdtree.schema import SchemaType


class Range:
    """A range of attribute values.

    A range can be open or closed on either side. The values are dictionary
    encoded based on the values in a cut repository.
    """

    __slots__ = ["_dict", "_left", "_right", "_open_left", "_open_right"]

    _left: int
    _right: int

    def __init__(
        self,
        dict: Dictionary,
        left: Optional[int] = None,
        right: Optional[int] = None,
        open_left: bool = True,
        open_right: bool = True,
    ):
        self._dict = dict
        self._left = left if left is not None else dict.NINF
        self._right = right if right is not None else dict.INF
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
        return self._dict[self._left]

    @property
    def right(self) -> SchemaType:
        return self._dict[self._right]

    def encode(self) -> np.ndarray:
        return np.array(
            [self._left, self._right, int(self._open_left), int(self._open_right)]
        )

    def encode_space(self) -> Tuple[np.ndarray, np.ndarray]:
        dict_min_index = self._dict.min_index()
        dict_max_index = self._dict.max_index()
        return (
            np.array([dict_min_index, dict_min_index, 0, 0]),
            np.array([dict_max_index, dict_max_index, 1, 1]),
        )
