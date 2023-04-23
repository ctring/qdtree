from typing import Optional

from .dictionary import Dictionary
from .schema import SchemaType


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
