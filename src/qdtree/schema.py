from typing import Dict, Literal, Union

SchemaTypeTag = Literal["int", "float", "str"]
SchemaType = Union[int, float, str]
Schema = Dict[str, SchemaTypeTag]

def cast_to_type(value: str, type_tag: SchemaTypeTag) -> SchemaType:
    if type_tag == "int":
        return int(value)
    elif type_tag == "float":
        return float(value)
    elif type_tag == "str":
        return value
    else:
        raise ValueError(f"Invalid type tag {type_tag}")