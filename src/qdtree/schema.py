from datetime import datetime
from typing import Dict, Literal, Union

SchemaTypeTag = Literal["int", "float", "date"]
SchemaType = Union[int, float, datetime]
Schema = Dict[str, SchemaTypeTag]


def cast_to_type(value: str, type_tag: SchemaTypeTag) -> SchemaType:
    if type_tag == "int":
        return int(value)
    elif type_tag == "float":
        return float(value)
    elif type_tag == "date":
        return datetime.strptime(value, "%Y-%m-%d")
    else:
        raise ValueError(f"Invalid type tag {type_tag}")
