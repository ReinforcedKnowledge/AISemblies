from __future__ import annotations

import importlib
from dataclasses import asdict
from typing import Any, Callable

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


def import_function(module_name: str, func_name: str) -> Callable[..., Any]:
    """Dynamically import a function by module_name and func_name.

    Parameters
    ----------
    module_name : str
        The module where the function is located.
    func_name : str
        The function name.

    Returns
    -------
    Callable[..., Any]
        The imported function.

    Raises
    ------
    TypeError
        If the imported object is not a callable.
    """
    mod = importlib.import_module(module_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise TypeError(f"Object is imported but is not callable: {fn}")
    return fn
