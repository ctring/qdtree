import pandas as pd
from datetime import datetime
from typing import Dict, Literal, Union

SchemaTypeTag = Literal["int", "float", "date"]
SchemaType = Union[int, float]
Schema = Dict[str, SchemaTypeTag]


def cast_to_type(value: str, type_tag: SchemaTypeTag) -> SchemaType:
    if type_tag == "int":
        return int(value)
    elif type_tag == "float":
        return float(value)
    elif type_tag == "date":
        return int(datetime.strptime(value, "%Y-%m-%d").timestamp())
    else:
        raise ValueError(f"Invalid type tag {type_tag}")


def ensure_data_schema(data: pd.DataFrame, schema: Schema):
    ret = pd.DataFrame(index=data.index)
    for col, typ in schema.items():
        if typ == "date":
            ret[col] = data[col].astype("datetime64[ns]").astype("int") // 10**9
        else:
            ret[col] = data[col].astype(typ)

    return ret
