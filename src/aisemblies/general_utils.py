from __future__ import annotations

from dataclasses import asdict

from pydantic import BaseModel


def coerce_to_dict(obj) -> dict:
    """
    Coerce an object into a dictionary. Handles:

    - Any object with a `.dict()` method (like Pydantic `BaseModel`).
    - Any dictionary as-is.
    - Potentially a dataclass (using `dataclasses.asdict`).

    Parameters
    ----------
    obj : Any
        The object to convert.

    Returns
    -------
    dict
        A dictionary representation of the input object or an empty dict if conversion
        is not possible.
    """
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, "dict"):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if callable(obj.dict):
            return obj.dict()
    elif hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return {}
