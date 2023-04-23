from typing import Dict, List, Tuple, Iterable

from qdtree.schema import SchemaType, SchemaTypeTag, cast_to_type


class Dictionary:
    __slots__ = ["_values", "_values_str"]

    _values: List[SchemaType]
    _values_str: Dict[str, int]

    INF = -1
    NINF = -2

    def __init__(self, values: Iterable[Tuple[str, SchemaTypeTag]]):
        self._values = [cast_to_type(value, typ) for value, typ in values]
        self._values_str = {value: i for i, (value, _) in enumerate(values)}

    def __getitem__(self, index: int) -> SchemaType:
        if index == Dictionary.INF:
            return float("inf")

        if index == Dictionary.NINF:
            return float("-inf")

        return self._values[index]

    def reverse_lookup_str(self, value: str) -> int:
        return self._values_str[value]

    def min_index(self) -> int:
        return min(Dictionary.INF, Dictionary.NINF, 0)

    def max_index(self) -> int:
        return max(Dictionary.INF, Dictionary.NINF, len(self._values) - 1)
