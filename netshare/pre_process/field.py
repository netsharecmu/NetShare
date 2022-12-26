from typing import Tuple

from config_io import Config

from netshare.utils import Field

FieldKey = Tuple[str, str, str]


def field_config_to_key(field: Config) -> FieldKey:
    return (
        field.get("column") or str(field.columns),
        field.type,
        field.get("encoding", ""),
    )


def key_from_field(field: Field) -> FieldKey:
    return field.name, field.__class__.__name__, ""
